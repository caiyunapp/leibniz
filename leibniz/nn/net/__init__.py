# -*- coding: utf-8 -*-

from leibniz.nn.layer.residual import Basic
from leibniz.nn.activation import Swish

from leibniz.nn.net.unet import UNet
from leibniz.nn.net.resnet import ResNet
from nn.net.mlp import MLP2d
from nn.net.hyptube import HypTube, StepwiseHypTube, LayeredHypTube


def mpl2d(in_channels, hidden_channels, out_channels):
    return MLP2d(in_channels, hidden_channels, out_channels)


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


def resnet(in_channels, out_channels, block=Basic, relu=Swish(), attn=None, layers=4, ratio=0,
                 vblks=[1, 1, 1, 1], scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1],
                 spatial=(256, 256), normalizor='batch'):
    return ResNet(in_channels, out_channels, block=block, relu=relu, attn=attn, layers=layers, ratio=ratio,
                 vblks=vblks, scales=scales, factors=factors, spatial=spatial, normalizor=normalizor)


def hyptub(in_channels, hidden_channels, out_channels, encoder=resunet, decoder=resunet, **kwargs):
    return HypTube(in_channels, hidden_channels, out_channels, encoder=encoder, decoder=decoder, **kwargs)


def hyptub_stepwise(in_channels, hidden_channels, out_channels, steps, encoder=resunet, decoder=resunet, **kwargs):
    return StepwiseHypTube(in_channels, hidden_channels, out_channels, steps, encoder=encoder, decoder=decoder, **kwargs)


def hyptub_layered(in_channels, hidden_channels, out_channels, layers, encoder=resunet, decoder=resunet, **kwargs):
    return LayeredHypTube(in_channels, hidden_channels, out_channels, layers, encoder=encoder, decoder=decoder, **kwargs)

