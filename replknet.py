import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    if in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull-requests are welcomed.
        # TODO Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = nn.BatchNorm2d(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = nn.BatchNorm2d(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKNetStage(nn.Module):

    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 small_kernel, dw_ratio=1, ffn_ratio=4,
                 use_checkpoint=False,      # train with torch.utils.checkpoint to save memory
                 small_kernel_merged=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)   # Save training memory
            else:
                x = blk(x)
        return x


class RepLKNet(nn.Module):

    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000,
                 use_checkpoint=False,
                 small_kernel_merged=False):
        super().__init__()

        base_width = channels[0]
        self.use_checkpoint = use_checkpoint
        self.num_stages = len(layers)
        self.stem = nn.ModuleList([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                                  stage_lk_size=large_kernel_sizes[stage_idx],
                                  drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                                  small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                                  use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)

        self.norm = nn.BatchNorm2d(channels[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels[-1], num_classes)


    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(stem_layer, x)     # save memory
            else:
                x = stem_layer(x)
        for stage_idx in range(self.num_stages):
            x = self.stages[stage_idx](x)
            if stage_idx < self.num_stages - 1:
                x = self.transitions[stage_idx](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        #TODO
        pass


def create_RepLKNet31B(num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[128,256,512,1024],
                    drop_path_rate=0.3, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

def create_RepLKNet31L(num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
    return RepLKNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[192,384,768,1536],
                    drop_path_rate=0.3, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
                    small_kernel_merged=small_kernel_merged)

if __name__ == '__main__':
    model = create_RepLKNet31B(small_kernel_merged=False)
    model.eval()
    print('------------------- training-time model -------------')
    print(model)
    x = torch.randn(2, 3, 224, 224)
    origin_y = model(x)
    model.structural_reparam()
    print('------------------- after re-param -------------')
    print(model)
    reparam_y = model(x)
    print('------------------- the difference is ------------------------')
    print((origin_y - reparam_y).abs().sum())


