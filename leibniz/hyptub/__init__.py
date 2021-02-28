# -*- coding: utf-8 -*-

import torch.nn as nn

from leibniz.hyptub.base import HypTube
from leibniz.unet.hyperbolic import HyperBottleneck


def hyptub(in_channels, hidden_channels, out_channels, block=HyperBottleneck, relu=nn.ReLU(inplace=True), attn=None, normalizor='batch',
                 layers=4, ratio=-2, spatial=(256, 256), vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1], scales=[-2, -2, -2, -2], factors=[1, 1, 1, 1],
                 final_normalized=False):
    return HypTube(in_channels, hidden_channels, out_channels, normalizor=normalizor, spatial=spatial, layers=layers, ratio=ratio,
                vblks=vblks, hblks=hblks, scales=scales, factors=factors, block=block, relu=relu, attn=attn, final_normalized=final_normalized)
