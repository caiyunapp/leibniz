# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch.nn as nn

from leibniz.nn.net.unet import Enconv, Transform, Block
from leibniz.nn.conv import DepthwiseSeparableConv1d, DepthwiseSeparableConv2d, DepthwiseSeparableConv3d
from leibniz.nn.layer.cbam import CBAM

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, block=None, attn=None, relu=None, layers=4, ratio=1, dropout_prob=0.5,
                 vblks=None, factors=None, spatial=(256, 256), normalizor='batch', padding=None, final_normalized=True):
        super().__init__()

        extension = block.extension

        spatial = np.array(spatial, dtype=np.int)
        dim = len(spatial)
        self.dim = dim
        Conv = self.get_conv_for_prepare()
        TConv = self.get_conv_for_transform()

        ratio = np.exp2(ratio)
        factors = np.array(factors)
        factors = np.exp2(factors)
        num_filters = int(in_channels * ratio)

        self.final_normalized = final_normalized
        self.ratio = ratio
        self.vblks = vblks
        self.factors = factors
        logger.info('---------------------------------------')
        logger.info('ratio: %f', ratio)
        logger.info('vblks: [%s]', ', '.join(map(str, vblks)))
        logger.info('factors: [%s]', ', '.join(map(str, factors)))
        logger.info('---------------------------------------')

        self.layers = layers
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.out_channels = out_channels

        if relu is None:
            relu = nn.ReLU(inplace=True)

        if attn is None:
            attn = CBAM

        ex = extension
        c0 = int(ex * num_filters)
        accumulated_factors = np.cumprod(factors, axis=0)[-1]
        accumulated_spatial = np.cumprod(spatial, axis=0)[-1]
        pn = int(c0 * accumulated_factors) * accumulated_spatial

        if padding:
            self.conv_padding = 0
            self.iconv = nn.Sequential(
                padding,
                Conv(in_channels, c0, kernel_size=3, padding=self.conv_padding, groups=1),
            )
            self.fc = nn.Sequential(
                padding,
                nn.Linear(pn, out_channels),
            )
        else:
            self.conv_padding = 1
            self.iconv = Conv(in_channels, c0, kernel_size=7, padding=3, groups=1)
            self.fc = nn.Linear(pn, out_channels, bias=False)

        if final_normalized:
            self.relu6 = nn.ReLU6()

        self.enconvs = nn.ModuleList()
        self.transforms = nn.ModuleList()

        self.spatial = [np.array(spatial, dtype=np.int)]
        self.channel_sizes = [np.array(c0, dtype=np.int)]
        for ix in range(layers):
            least_factor = ex
            factor = factors[ix]
            self.channel_sizes.append(np.array(self.channel_sizes[ix] * factor // least_factor * least_factor, dtype=np.int))

            ci, co = self.channel_sizes[ix].item(), self.channel_sizes[ix + 1].item()
            logger.info('%d - ci: %d, co: %d', ix, ci, co)

            self.enconvs.append(Block(Enconv(ci, co, size=spatial, conv=TConv, padding=padding), activation=True, dropout=dropout_prob, relu=relu, attn=attn, dim=self.dim, normalizor=normalizor, conv=TConv))
            self.transforms.append(Transform(co, co, nblks=vblks[ix], block=block, relu=relu, conv=TConv))

    def get_conv_for_prepare(self):
        if self.dim == 1:
            conv = DepthwiseSeparableConv1d
        elif self.dim == 2:
            conv = DepthwiseSeparableConv2d
        elif self.dim == 3:
            conv = DepthwiseSeparableConv3d
        else:
            raise ValueError('dim %d is not supported!' % self.dim)
        return conv

    def get_conv_for_transform(self):
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
        sz = x.size()

        dnt = self.iconv(x)
        for ix in range(self.layers):
            dnt, enc = self.transforms[ix](self.enconvs[ix](dnt))

        return self.fc(dnt.view(sz[0], -1))
