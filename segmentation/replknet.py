# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmdet.custom_layers.convnet_utils import conv_bn_act, conv_bn, enable_syncbn, bn
from mmdet.custom_layers.lkblock import LargeKernelBlock

from collections import namedtuple
from ..builder import BACKBONES


RepLKCfg = namedtuple('RepLKCfg', ['lk_size', 'drop_path_rate', 'layer_set', 'base_width', 'small_kernels', 'mlp_ratio', 'dw_ratio', 'conv_nonlinear'])

def decode_try_arg(try_arg):

    if 'LS2' in try_arg:
        layer_set = [2, 2, 12, 2]
    elif 'LS3' in try_arg:
        layer_set = [2, 2, 18, 2]
    elif 'LS6' in try_arg:
        layer_set = [2, 4, 6, 2]
    elif 'LS0' in try_arg:
        layer_set = [2, 2, 2, 2]
    elif 'LS7' in try_arg:
        layer_set = [2, 2, 4, 2]
    else:
        layer_set = [2, 2, 6, 2]

    assert layer_set == [2,2,18,2]      # TODO temporarily

    if 'B0' in try_arg:
        lk_size = 3
    elif 'B7' in try_arg:
        lk_size = 7
    elif 'B9' in try_arg:
        lk_size = 9
    elif 'B11' in try_arg:
        lk_size = 11
    elif 'B13x11x9x7' in try_arg:
        lk_size = [13,11,9,7]

    elif 'B13' in try_arg:
        lk_size = 13
    elif 'B17' in try_arg:
        lk_size = 17
    elif 'B23' in try_arg:
        lk_size = 23
    elif 'B27' in try_arg:
        lk_size = 27
    elif 'B27x13' in try_arg:
        lk_size = [27,27,27,13]
    elif 'B19x17x15x13' in try_arg:
        lk_size = [19,17,15,13]
    elif 'B25x21x17x13' in try_arg:
        lk_size = [25,21,17,13]
    elif 'B31x25x19x13' in try_arg:
        lk_size = [31,25,19,13]
    elif 'B31x27x23x19' in try_arg:
        lk_size = [31,27,23,19]
    elif 'B31x29x27x13' in try_arg:
        lk_size = [31,29,27,13]
    elif 'B31all' in try_arg:
        lk_size = [31,31,31,31]
    elif 'B35x31x27x13' in try_arg:
        lk_size = [35,31,27,13]
    elif 'Btwo27x13' in try_arg:
        lk_size = [27, 27, 27, 13]
    elif 'Btwo25x13' in try_arg:
        lk_size = [25, 25, 25, 13]
    elif 'Btwo19x13' in try_arg:
        lk_size = [19, 19, 19, 13]
    else:
        raise ValueError('???')


    if isinstance(lk_size, int):        #TODO different lk_size for each stage
        lk_size = [lk_size] * len(layer_set)

    if 'DP0' in try_arg:
        drop_path_rate = 0
    elif 'DP2' in try_arg:
        drop_path_rate = 0.2
    elif 'DP3' in try_arg:
        drop_path_rate = 0.3
    elif 'DP4' in try_arg:
        drop_path_rate = 0.4
    elif 'DP5' in try_arg:
        drop_path_rate = 0.5
    elif 'DP6' in try_arg:
        drop_path_rate = 0.6
    elif 'DP7' in try_arg:
        drop_path_rate = 0.7
    elif 'DP8' in try_arg:
        drop_path_rate = 0.8
    elif 'DP1' in try_arg:
        drop_path_rate = 0.1
    else:
        assert 0


    if 'WD2' in try_arg:
        base_width = 192
    elif 'WD3' in try_arg:
        base_width = 128
    elif 'WDL' in try_arg:
        base_width = 192
    elif 'WDX' in try_arg:
        base_width = 256
    elif 'WD0' in try_arg:
        base_width = 96
    elif 'WDB' in try_arg:
        base_width = 112
    elif 'WDH' in try_arg:
        base_width = 64
    elif 'WDF' in try_arg:
        base_width = 48
    elif 'WDQ' in try_arg:
        base_width = 32
    elif 'WDM' in try_arg:
        base_width = 16
    else:
        raise ValueError('???')

    assert base_width in [96, 128, 112, 192, 256]

    if 'sma39' in try_arg:
        small_kernels = [3, 9]
    elif 'sma37' in try_arg:
        small_kernels = [3, 7]
    elif 'sma359' in try_arg:
        small_kernels = [3, 5, 9]  # hhb
    elif 'sma3579' in try_arg:
        small_kernels = [3, 5, 7, 9]  # hhb
    elif 'sma0' in try_arg:
        small_kernels = []
    elif 'sma3' in try_arg:
        small_kernels = [3,]
    elif 'sma5' in try_arg:
        small_kernels = [5,]
    elif 'tsm513' in try_arg:
        small_kernels = [5, 13]
    elif 'smahalf' in try_arg:
        small_kernels = 'half'
    else:
        raise ValueError('???')

    if '_mr2' in try_arg:
        mlp_ratio = 2
    elif '_mr3' in try_arg:
        mlp_ratio = 3
    elif '_mr5' in try_arg:
        mlp_ratio = 5
    elif '_mr6' in try_arg:
        mlp_ratio = 6
    else:
        mlp_ratio = 4

    if '_cngelu' in try_arg:
        conv_nonlinear = nn.GELU
    elif '_cnhs' in try_arg:
        conv_nonlinear = nn.Hardswish
    else:
        conv_nonlinear = nn.ReLU

    if '_dr2' in try_arg:
        dw_ratio = 2
    elif '_dr15' in try_arg:
        dw_ratio = 1.5
    elif '_dr3' in try_arg:
        dw_ratio = 3
    elif '_dr4' in try_arg:
        dw_ratio = 4
    elif '_dr05' in try_arg:
        dw_ratio = 0.5
    else:
        dw_ratio = 1
    return RepLKCfg(lk_size=lk_size,
                  drop_path_rate=drop_path_rate, layer_set=layer_set, base_width=base_width, small_kernels=small_kernels,
                  mlp_ratio=mlp_ratio, dw_ratio=dw_ratio, conv_nonlinear=conv_nonlinear)

class ConvMlp(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.,
                 deploy=False):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        if deploy:
            self.fc1 = nn.Conv2d(in_channels, hidden_features, 1, 1, 0)
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        else:
            self.fc1 = conv_bn(in_channels, hidden_features, 1, 1, 0)
            self.fc2 = conv_bn(hidden_features, out_features, 1, 1, 0)
        self.act = act_layer()
        print('nonlinear in ConvMLP is', self.act)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class RepLKBlock(nn.Module):

    #   TODO: note here the "depthwise_layer" should include nonlinearity
    def __init__(self, in_channels, out_channels, block_lk_size, cfg:RepLKCfg,
                 drop=0., drop_path=0.,
                 deploy=False, use_custom_lk=False):
        super().__init__()

        self.p1 = conv_bn_act(in_channels, int(out_channels * cfg.dw_ratio), 1, 1, 0, deploy=deploy,
                              nonlinear_type=cfg.conv_nonlinear)
        self.p2 = conv_bn(int(out_channels * cfg.dw_ratio), out_channels, 1, 1, 0, deploy=deploy)
        dw_layer_nonlinear = cfg.conv_nonlinear

        #   =================================
        if block_lk_size > 3:
            self.depthwise = LargeKernelBlock(in_channels=int(out_channels * cfg.dw_ratio), out_channels=int(out_channels * cfg.dw_ratio),
                                        kernel_size=block_lk_size, stride=1,
                                        padding=block_lk_size // 2, groups=int(out_channels * cfg.dw_ratio), deploy=deploy,
                                        nonlinear=dw_layer_nonlinear(), small_kernels=cfg.small_kernels,
                                              seq_kernels=None, seq_expand=0, use_custom_lk=use_custom_lk)
        else:
            self.depthwise = conv_bn_act(in_channels=int(out_channels * cfg.dw_ratio), out_channels=int(out_channels * cfg.dw_ratio),
                                    kernel_size=3, stride=1, padding=1, groups=int(out_channels * cfg.dw_ratio), deploy=deploy,
                                         nonlinear_type=dw_layer_nonlinear)
        #   =================================

        self.premlp_bn = bn(out_channels)
        self.predwb_bn = bn(out_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print('drop path:', drop_path)
        mlp_hidden_channels = int(out_channels * cfg.mlp_ratio)
        self.mlp = ConvMlp(in_channels=out_channels, hidden_channels=mlp_hidden_channels, drop=drop, deploy=deploy)


    def forward(self, x):
        y = self.p1(self.predwb_bn(x))
        y = self.depthwise(y)
        y = self.p2(y)
        z = x + self.drop_path(y)
        z = z + self.drop_path(self.mlp(self.premlp_bn(z)))
        return z


class RepLKStage(nn.Module):

    def __init__(self,
                 total_stages, stage_idx,
                 in_channels, out_channels, num_blocks, stage_lk_size,
                 cfg:RepLKCfg,
                 deploy,
                 drop=0.,
                 drop_path=0.,
                 use_checkpoint=False,
                 stage_cancel_stride2=None,
                 use_custom_lk=False
                 ):

        super().__init__()
        self.use_checkpoint = use_checkpoint

        blks = []

        for i in range(num_blocks):
            block_in_channels = in_channels if i == 0 else out_channels
            block = RepLKBlock(in_channels=block_in_channels, out_channels=out_channels,
                             block_lk_size=stage_lk_size, cfg=cfg, drop=drop,
                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             deploy=deploy, use_custom_lk=use_custom_lk)
            blks.append(block)
        self.blocks = nn.ModuleList(blks)

        if stage_idx < total_stages - 1:
            actual_stride = 1 if stage_cancel_stride2 else 2
            self.downsample = nn.Sequential(
                conv_bn_act(out_channels, out_channels * 2, 1, 1, 0, deploy=deploy,
                            nonlinear_type=cfg.conv_nonlinear),
                conv_bn_act(in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=3,
                            stride=actual_stride,
                            padding=1, groups=out_channels * 2, deploy=deploy, nonlinear_type=cfg.conv_nonlinear))
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = checkpoint.checkpoint(self.downsample, x)
        return x





@BACKBONES.register_module
class RepLKNet(nn.Module):

    def __init__(self,
                 pretrained,
                 try_arg,
                 deploy,
                 in_channels=3,
                 drop_rate=0.,      #TODO change to dropblock?
                 use_checkpoint=True,
                 cancel_stride2=(False, False, False, False),
                 use_syncbn=False,
                 early_out=False):
        super().__init__()

        self.early_out = early_out

        if use_syncbn:
            enable_syncbn()

        self.pretrained = pretrained
        self.use_checkpoint = use_checkpoint

        cfg = decode_try_arg(try_arg)
        layers = [conv_bn_act(in_channels=in_channels, out_channels=cfg.base_width, kernel_size=3, stride=2, padding=1,
                              deploy=deploy, nonlinear_type=cfg.conv_nonlinear),
                  conv_bn_act(in_channels=cfg.base_width, out_channels=cfg.base_width, kernel_size=3, stride=1,
                              padding=1, groups=cfg.base_width, deploy=deploy, nonlinear_type=cfg.conv_nonlinear),
                  conv_bn_act(in_channels=cfg.base_width, out_channels=cfg.base_width, kernel_size=1, stride=1,
                              padding=0, deploy=deploy, nonlinear_type=cfg.conv_nonlinear),
                  conv_bn_act(in_channels=cfg.base_width, out_channels=cfg.base_width, kernel_size=3, stride=2,
                              padding=1, groups=cfg.base_width, deploy=deploy, nonlinear_type=cfg.conv_nonlinear)]
        if cfg.layer_set[2] <= 12:
            self.conv0 = nn.Sequential(*layers)
        else:
            self.conv0 = nn.ModuleList(layers)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.layer_set))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()

        cur_in_channels = cfg.base_width
        cur_out_channels = cfg.base_width

        for stage_idx in range(len(cfg.layer_set)):
            layer = RepLKStage(total_stages=len(cfg.layer_set), stage_idx=stage_idx,
                             in_channels=cur_in_channels, out_channels=cur_out_channels, num_blocks=cfg.layer_set[stage_idx],
                             stage_lk_size=cfg.lk_size[stage_idx], cfg=cfg, deploy=deploy, drop=drop_rate,
                             drop_path=dpr[sum(cfg.layer_set[:stage_idx]):sum(cfg.layer_set[:stage_idx + 1])],
                             use_checkpoint=use_checkpoint,
                             stage_cancel_stride2=cancel_stride2[stage_idx],
                               use_custom_lk=stage_idx>=0)

            cur_in_channels = 2 * cur_in_channels
            cur_out_channels = cur_in_channels

            self.layers.append(layer)


    def init_weights(self, pretrained=None):
        weights = torch.load(self.pretrained, map_location='cpu')
        if 'model' in weights:
            weights = weights['model']
        weights.pop('norm.weight')
        weights.pop('norm.bias')
        for k in ['norm.running_mean', 'norm.running_var', 'norm.num_batches_tracked']:
            if k in weights:
                weights.pop(k)
        weights.pop('head.weight')
        weights.pop('head.bias')
        self.load_state_dict(weights, strict=True)


    def forward_features(self, x):

        if isinstance(self.conv0, nn.ModuleList):
            x = self.conv0[0](x)
            for i in (1,2,3):
                x = checkpoint.checkpoint(self.conv0[i], x)
        else:
            x = self.conv0(x)
        outs = []


        if self.early_out:

            for i, layer in enumerate(self.layers):

                for blk in layer.blocks:
                    if self.use_checkpoint:
                        x = checkpoint.checkpoint(blk, x)
                    else:
                        x = blk(x)

                outs.append(x)

                if i < len(self.layers) - 1:
                    x = checkpoint.checkpoint(layer.downsample, x)

        else:

            for layer in self.layers:
                x = layer(x)
                outs.append(x)

        return outs



    def forward(self, x):
        x = self.forward_features(x)
        return x