# -*- coding: utf-8 -*-

from leibniz.resnet.base import ResNet
from leibniz.unet.residual import Basic
from leibniz.nn.activation import Swish


def resnet(in_channels, out_channels, block=Basic, relu=Swish(), attn=None, layers=4, ratio=0,
                 vblks=[1, 1, 1, 1], scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1],
                 spatial=(256, 256), normalizor='batch'):
    return ResNet(in_channels, out_channels, block=block, relu=relu, attn=attn, layers=layers, ratio=ratio,
                 vblks=vblks, scales=scales, factors=factors, spatial=spatial, normalizor=normalizor)
