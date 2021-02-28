# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch.nn as nn

from leibniz.unet.base import Enconv, Transform, Block
from leibniz.nn.conv import DepthwiseSeparableConv1d, DepthwiseSeparableConv2d, DepthwiseSeparableConv3d
from leibniz.unet.cbam import CBAM

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, block=None, attn=None, relu=None, layers=4, ratio=2,
                 vblks=None, scales=None, factors=None, spatial=(256, 256), normalizor='batch', padding=None, final_normalized=True):
        super().__init__()

        extension = block.extension
        lrd = block.least_required_dim

        spatial = np.array(spatial, dtype=np.int)
        dim = len(spatial)
        self.dim = dim
        Conv = self.get_conv_for_prepare()
        TConv = self.get_conv_for_transform()

        scales = np.array(scales)
        if scales.shape[0] != layers:
            raise ValueError('scales should have %d layers at dim 0!' % layers)
        if len(scales.shape) == 1:
            scales = scales.reshape([layers, 1])
        if len(scales.shape) != 2:
            raise ValueError('scales should have length 2 to be compatible with spatial dimensions!')

        ratio = np.exp2(ratio)
        factors = np.array(factors + [0.0])
        scales = np.exp2(scales)
        factors = np.exp2(factors)
        num_filters = int(in_channels * ratio)

        self.final_normalized = final_normalized
        self.ratio = ratio
        self.vblks = vblks
        self.scales = scales
        self.factors = factors
        logger.info('---------------------------------------')
        logger.info('ratio: %f', ratio)
        logger.info('vblks: [%s]', ', '.join(map(str, vblks)))
        logger.info('scales: [%s]', ', '.join(map(str, scales)))
        logger.info('factors: [%s]', ', '.join(map(str, factors[0:4])))
        logger.info('---------------------------------------')

        self.exceeded = np.any(np.cumprod(scales, axis=0) * spatial < 1) or np.any((in_channels * ratio * np.cumprod(factors)) < lrd)
        if not self.exceeded:
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
            accumulated_scales = np.cumprod(scales, axis=0)[-1]
            if len(accumulated_scales) == 1:
                accumulated_scales = [accumulated_scales[0] for ix in range(dim)]
            accumulated_scales = np.cumprod(accumulated_scales, axis=0)[-1]
            accumulated_spatial = np.cumprod(spatial, axis=0)[-1]
            pn = int(c0 * accumulated_factors * accumulated_scales * accumulated_spatial)

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
            self.dnforms = nn.ModuleList()

            self.spatial = [np.array(spatial, dtype=np.int)]
            self.channel_sizes = [np.array(c0, dtype=np.int)]
            for ix in range(layers):
                least_factor = ex
                scale, factor = scales[ix], factors[ix]
                self.spatial.append(np.array(self.spatial[ix] * scale, dtype=np.int))
                self.channel_sizes.append(np.array(self.channel_sizes[ix] * factor // least_factor * least_factor, dtype=np.int))

                ci, co = self.channel_sizes[ix].item(), self.channel_sizes[ix + 1].item()
                szi, szo = self.spatial[ix + 1], self.spatial[ix]
                logger.info('%d - ci: %d, co: %d', ix, ci, co)
                logger.info('%d - szi: [%s], szo: [%s]', ix, ', '.join(map(str, szi)), ', '.join(map(str, szo)))

                self.exceeded = self.exceeded or ci < lrd or co < lrd or szi.min() < 1 or szo.min() < 1
                if not self.exceeded:
                    try:
                        dropout_flag = (layers - ix) * 3 < layers
                        self.enconvs.append(Block(Enconv(ci, co, size=szi, conv=TConv, padding=padding), activation=True, dropout=dropout_flag, relu=relu, attn=attn, dim=self.dim, normalizor=normalizor, conv=TConv))
                        self.dnforms.append(Transform(co, co, nblks=vblks[ix], block=block, relu=relu, conv=TConv))
                    except Exception as e:
                        logger.exception(e)
                        self.exceeded = True
                else:
                    logger.error('scales are exceeded!')
                    raise ValueError('scales exceeded!')

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
        if self.exceeded:
            raise ValueError('scales exceeded!')

        sz = x.size()

        dnt = self.iconv(x)
        for ix in range(self.layers):
            dnt, enc = self.dnforms[ix](self.enconvs[ix](dnt))

        return self.fc(dnt.view(sz[0], -1))
