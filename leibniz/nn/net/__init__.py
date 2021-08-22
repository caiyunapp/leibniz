# -*- coding: utf-8 -*-

from leibniz.nn.layer.residual import Basic
from leibniz.nn.activation import Swish

from leibniz.nn.net.unet import UNet
from leibniz.nn.net.unet2 import UNet2
from leibniz.nn.net.resnet import ResNet
from leibniz.nn.net.hyptube import HypTube, StepwiseHypTube, LeveledHypTube
from leibniz.nn.net.conv_lstm import ConvLSTM
from leibniz.nn.net.simple import Linear, Identity, SimpleCNN2d
from leibniz.nn.net.hunet import HUNet


def identical(*args, **kwargs):
    return Identity()


def linear(in_channels, out_channels, **kwargs):
    return Linear(in_channels, out_channels, **kwargs)


def cnn2d(in_channels, out_channels):
    return SimpleCNN2d(in_channels, out_channels)


def unet4(in_channels, out_channels, spatial=(256, 256), ksize_in=7, dropout=0.1):
    return UNet(in_channels, out_channels, block=None, relu=Swish(), attn=None, layers=4, ratio=0, ksize_in=ksize_in, dropout=dropout,
                 vblks=[0, 0, 0, 0], hblks=[0, 0, 0, 0],
                 scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1], spatial=spatial)


def unet8(in_channels, out_channels, spatial=(256, 256), ksize_in=7, dropout=0.1):
    return UNet(in_channels, out_channels, block=None, relu=Swish(), attn=None, layers=8, ratio=0, ksize_in=ksize_in, dropout=dropout,
                 vblks=[0, 0, 0, 0, 0, 0, 0, 0], hblks=[0, 0, 0, 0, 0, 0, 0, 0],
                 scales=[-1, -1, -1, -1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1, 1, 1, 1], spatial=spatial)


def resunet(in_channels, out_channels, block=Basic, relu=Swish(), attn=None, layers=4, ratio=0, ksize_in=7, dropout_prob=0.5,
                 vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1],
                 scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1], spatial=(256, 256), normalizor='batch',
                 enhencer=None, final_normalized=False):
    return UNet(in_channels, out_channels, block=block, relu=relu, attn=attn, layers=layers, ratio=ratio, ksize_in=ksize_in, dropout_prob=dropout_prob,
                 vblks=vblks, hblks=hblks, scales=scales, factors=factors, spatial=spatial, normalizor=normalizor,
                 enhencer=enhencer, final_normalized=final_normalized)


def resunet2(in_channels, out_channels, block=Basic, relu=Swish(), attn=None, layers=4, ratio=0, ksize_in=7, dropout_prob=0.5,
                 vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1],
                 scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1], spatial=(256, 256), normalizor='batch',
                 enhencer=None, final_normalized=False):
    return UNet2(in_channels, out_channels, block=block, relu=relu, attn=attn, layers=layers, ratio=ratio, ksize_in=ksize_in, dropout_prob=dropout_prob,
                 vblks=vblks, hblks=hblks, scales=scales, factors=factors, spatial=spatial, normalizor=normalizor,
                 enhencer=enhencer, final_normalized=final_normalized)


def resnet(in_channels, out_channels, block=Basic, relu=Swish(), attn=None, layers=4, ratio=0, dropout_prob=0.1,
                 vblks=[1, 1, 1, 1], scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1],
                 spatial=(256, 256), normalizor='batch'):
    return ResNet(in_channels, out_channels, block=block, relu=relu, attn=attn, layers=layers, ratio=ratio, dropout_prob=0.5,
                 vblks=vblks, scales=scales, factors=factors, spatial=spatial, normalizor=normalizor)


def hyptub(in_channels, hidden_channels, out_channels, encoder=resunet, decoder=resunet, **kwargs):
    return HypTube(in_channels, hidden_channels, out_channels, encoder=encoder, decoder=decoder, **kwargs)


def hyptub_stepwise(in_channels, hidden_channels, out_channels, steps, encoder=resunet, decoder=resunet, **kwargs):
    return StepwiseHypTube(in_channels, hidden_channels, out_channels, steps, encoder=encoder, decoder=decoder, **kwargs)


def hyptub_leveled(in_channels, hidden_channels, out_channels, levels, encoder=resunet, decoder=resunet, **kwargs):
    return LeveledHypTube(in_channels, hidden_channels, out_channels, levels, encoder=encoder, decoder=decoder, **kwargs)


def lstm(input_dim, hidden_dim, kernel_size, num_layers=1, batch_first=True, bias=True, return_all_layers=False):
    return ConvLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bias=bias,
        return_all_layers=return_all_layers
    )


def hunet(in_channels, out_channels, block=Basic, relu=Swish(), attn=None, layers=4, ratio=0,
                 vblks=[1, 1, 1, 1], scales=[-1, -1, -1, -1], factors=[1, 1, 1, 1],
                 spatial=(256, 256), normalizor='batch'):
    return HUNet(in_channels, out_channels, block=block, relu=relu, attn=attn, layers=layers, ratio=ratio,
                 vblks=vblks, scales=scales, factors=factors, spatial=spatial, normalizor=normalizor)
