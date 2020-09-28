import torch as th
import torch.nn as nn

from leibniz.unet.senet import SELayer


# Originally from Kevin Trebing (HansBambel) at https://github.com/HansBambel/SmaAt-UNet/blob/master/models/layers.py
# SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture
# https://arxiv.org/abs/2007.04417v1
# Modified by Mingli Yuan (Mountain)

class SpatialLayer(nn.Module):
    def __init__(self, kernel_size=7, conv=None):
        super(SpatialLayer, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.spatial = conv(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        out = th.cat([avg_out, max_out], dim=1)
        out = self.spatial(out)
        scale = x * th.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, dim, reduction=16, kernel_size=7, conv=None):
        super(CBAM, self).__init__()
        self.se = SELayer(dim, reduction)
        self.sp = SpatialLayer(kernel_size=kernel_size, conv=conv)

    def forward(self, x):
        out = self.se(x)
        out = self.sp(out)
        return out
