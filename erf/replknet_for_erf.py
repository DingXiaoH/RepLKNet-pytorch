# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

from replknet import RepLKNet

class RepLKNetForERF(RepLKNet):

    def __init__(self, large_kernel_sizes, layers, channels, small_kernel,
                 dw_ratio=1, ffn_ratio=4,
                 small_kernel_merged=False):
        super().__init__(large_kernel_sizes=large_kernel_sizes, layers=layers, channels=channels,
                         drop_path_rate=0, small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                         in_channels=3, num_classes=1000, small_kernel_merged=small_kernel_merged)

    def forward(self, x):
        x = self.forward_features(x)
        return x
        # return self.norm(x)     #   Using the feature maps after the final norm also makes sense. Observed very little difference.

