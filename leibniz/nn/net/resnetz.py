# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.nn.conv import DepthwiseSeparableConv1d, DepthwiseSeparableConv2d, DepthwiseSeparableConv3d
from leibniz.nn.layer.hyperbolic import Bottleneck

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ResNetZ(nn.Module):
    def __init__(self, in_channels, out_channels,  layers=4, ratio=1, spatial=(256, 256),
                 conv_class=None, relu=None, normalizor=None):
        super().__init__()

        spatial = np.array(spatial, dtype=np.int)
        dim = len(spatial)
        self.dim = dim
        self.ratio = np.power(2, ratio)
        self.layers = layers
        self.in_channels = int(in_channels)
        self.num_filters = int(in_channels * self.ratio)
        self.out_channels = int(out_channels)
        self.spatial = [np.array(spatial, dtype=np.int)]

        logger.info('---------------------------------------')
        logger.info('dim: %f', self.dim)
        logger.info('ratio: %f', self.ratio)
        logger.info('layers: %f', self.layers)
        logger.info('in_channels: %f', self.in_channels)
        logger.info('out_channels: %f', self.out_channels)
        logger.info('num_filters: %f', self.num_filters)
        logger.info('normalizor: %s', normalizor)
        logger.info('---------------------------------------')

        if dim == 1:
            self.bn = nn.BatchNorm1d(self.num_filters, affine=True)
        elif dim == 2:
            self.bn = nn.BatchNorm2d(self.num_filters, affine=True)
        elif dim == 3:
            self.bn = nn.BatchNorm3d(self.num_filters, affine=True)

        if conv_class is None:
            self.conv_class = self.get_conv_class()
        else:
            self.conv_class = conv_class
        if relu is None:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = relu

        if normalizor == 'relu6':
            self.normalizor = nn.ReLU6()
            self.scale = 1.0 / 6.0
            self.bias = 0.0
        elif normalizor == 'sigmoid':
            self.normalizor = nn.Sigmoid()
            self.scale = 1.0
            self.bias = 0.0
        elif normalizor == 'tanh':
            self.normalizor = nn.Tanh()
            self.scale = 1.0
            self.bias = 0.0
        elif normalizor == 'softmax':
            self.normalizor = nn.Softmax()
            self.scale = 1.0
            self.bias = 0.0
        else:
            self.normalizor = None
            self.scale = 1.0
            self.bias = 0.0

        self.iconv = self.conv_class(self.in_channels, self.num_filters, kernel_size=7, padding=3, groups=1)
        self.oconv = self.conv_class(self.num_filters, self.out_channels, kernel_size=3, padding=1, groups=1, bias=False)

        step_length = 1.0 / self.layers
        self.order1 = Bottleneck(self.num_filters, 2 * self.num_filters, step_length, self.relu, self.conv_class, reduction=16)
        self.order2 = Bottleneck(4 * self.num_filters + 1, 2 * self.num_filters, step_length, self.relu, self.conv_class, reduction=16)
        self.order3 = Bottleneck(7 * self.num_filters + 1, 2 * self.num_filters, step_length, self.relu, self.conv_class, reduction=16)

    def get_conv_class(self):
        if self.dim == 1:
            conv = DepthwiseSeparableConv1d
        elif self.dim == 2:
            conv = DepthwiseSeparableConv2d
        elif self.dim == 3:
            conv = DepthwiseSeparableConv3d
        else:
            raise ValueError('dim %d is not supported!' % self.dim)
        return conv

    def forward(self, x):
        x0 = self.bn(self.iconv(x))
        for _ in range(self.layers):
            rslt = self.order1(x0)
            velo = rslt[:, :self.num_filters]
            theta = rslt[:, self.num_filters:]
            du0 = velo * th.cos(theta)
            dv0 = velo * th.sin(theta)
            x1 = x0 * (1 + dv0 / self.layers) + du0 / self.layers

            dd = self.order2(th.cat([x0, x1, du0, dv0, th.ones_like(x0[:, 0:1]) * _ / self.layers], dim=1))
            du1 = dd[:, self.num_filters * 0:self.num_filters * 1]
            dv1 = dd[:, self.num_filters * 1:self.num_filters * 2]

            x2 = x1 * (1 + dv1 / self.layers) + du1 / self.layers

            dd = self.order3(th.cat([x0, x1, x2, du0, dv0, du1, dv1, th.ones_like(x1[:, 0:1]) * _ / self.layers], dim=1))
            du2 = dd[:, self.num_filters * 0:self.num_filters * 1]
            dv2 = dd[:, self.num_filters * 1:self.num_filters * 2]

            x0 = x2 * (1 + dv2 / self.layers) + du2 / self.layers

        out = self.oconv(x0)
        if self.normalizor:
            return self.normalizor(out) * self.scale + self.bias
        else:
            return out * self.scale + self.bias
