# -*- coding: utf-8 -*-

from leibniz.unet.base import UNet
from leibniz.unet.residual import Basic
from leibniz.nn.activation import Swish
from leibniz.unet.senet import SELayer


def unet4(in_channels, out_channels, spatial=(256, 256)):
    return UNet(in_channels, out_channels, block=None, relu=Swish(), attn=None, layers=4, ratio=0,
                 vblks=[0, 0, 0, 0], hblks=[0, 0, 0, 0],
                 scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1], spatial=spatial)


def unet8(in_channels, out_channels, spatial=(256, 256)):
    return UNet(in_channels, out_channels, block=None, relu=Swish(), attn=None, layers=8, ratio=0,
                 vblks=[0, 0, 0, 0, 0, 0, 0, 0], hblks=[0, 0, 0, 0, 0, 0, 0, 0],
                 scales=[-1, -1, -1, -1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1, 1, 1, 1], spatial=spatial)


def resunet(in_channels, out_channels, block=Basic, relu=Swish(), attn=None, layers=4, ratio=0,
                 vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1],
                 scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1], spatial=(256, 256), normalizor='batch',
                 final_normalized=False):
    return UNet(in_channels, out_channels, block=block, relu=relu, attn=attn, layers=layers, ratio=ratio,
                 vblks=vblks, hblks=hblks, scales=scales, factors=factors, spatial=spatial, normalizor=normalizor,
                 final_normalized=final_normalized)
