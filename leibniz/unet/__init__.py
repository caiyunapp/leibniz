# -*- coding: utf-8 -*-

from unet.base import UNet, Transform
from unet.residual import Basic


def unet(in_channels, num_filters, out_channels, kernel_size=4):
    return UNet(in_channels, num_filters, out_channels, kernel_size=kernel_size, layers=0, block=None)


def resunet(in_channels, num_filters, out_channels, kernel_size=4, layers=8, block=Basic):
    return UNet(in_channels, num_filters, out_channels, kernel_size=kernel_size, layers=layers, block=block)
