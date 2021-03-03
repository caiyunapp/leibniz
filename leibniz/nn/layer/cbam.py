import torch as th
import torch.nn as nn


# Originally from Kevin Trebing (HansBambel) at https://github.com/HansBambel/SmaAt-UNet/blob/master/models/layers.py
# SmaAt-UNet: Precipitation Nowcasting using a Small Attention-UNet Architecture
# https://arxiv.org/abs/2007.04417v1
# Modified by Mingli Yuan (Mountain)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelLayer(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelLayer, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = None
        self.max_pool = None
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio + 1),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio + 1, input_channels)
        )

    def forward(self, x):
        sz = x.size()
        if len(sz) == 3:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool1d(1)
                self.max_pool = nn.AdaptiveMaxPool1d(1)
            # Take the input and apply average and max pooling
            avg_values = self.avg_pool(x)
            max_values = self.max_pool(x)
            out = (self.mlp(avg_values) + self.mlp(max_values)).view(sz[0], sz[1], 1)
        if len(sz) == 4:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1)
            # Take the input and apply average and max pooling
            avg_values = self.avg_pool(x)
            max_values = self.max_pool(x)
            out = (self.mlp(avg_values) + self.mlp(max_values)).view(sz[0], sz[1], 1, 1)
        if len(sz) == 5:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
            # Take the input and apply average and max pooling
            avg_values = self.avg_pool(x)
            max_values = self.max_pool(x)
            out = (self.mlp(avg_values) + self.mlp(max_values)).view(sz[0], sz[1], 1, 1, 1)

        scale = x * th.sigmoid(out)
        return scale


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
        self.ch = ChannelLayer(input_channels=dim, reduction_ratio=reduction)
        self.sp = SpatialLayer(kernel_size=kernel_size, conv=conv)

    def forward(self, x):
        out = self.ch(x)
        out = self.sp(out)
        return out
