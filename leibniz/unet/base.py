# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
import torch.nn as nn

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Enconv(nn.Module):
    def __init__(self, in_channels, out_channels, size=256):

        super(Enconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale = nn.Upsample(size=size, mode='bilinear')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

    def forward(self, x):
        x = self.scale(x).contiguous()
        x = self.conv(x)

        return x


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, size=256):

        super(Deconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale = nn.Upsample(size=size, mode='bilinear')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

    def forward(self, x):
        x = self.scale(x).contiguous()
        x = self.conv(x)

        return x


class Transform(nn.Module):
    def __init__(self, in_channels, out_channels, nblks=0, block=None, relu=None):

        super(Transform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if relu is None:
            relu = nn.ReLU(inplace=True)

        self.blocks = None
        if nblks > 0 and block is not None:
            blocks = []
            for i in range(nblks - 1):
                blocks.append(block(self.out_channels, step=1.0 / nblks, relu=relu))
                blocks.append(relu)
            blocks.append(block(self.out_channels, step=1.0 / nblks, relu=relu))
            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):

        if self.blocks is not None:
            return self.blocks(x), x
        else:
            return x, x


class Block(nn.Module):
    def __init__(self, transform, activation=True, batchnorm=True, instnorm=False, dropout=False, relu=None):

        super(Block, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.instnorm = instnorm
        self.dropout = dropout
        self.blocks = None

        self.transform = transform

        if self.activation:
            if relu is not None:
                self.lrelu = relu
            else:
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if self.batchnorm:
            self.bn = nn.BatchNorm2d(transform.out_channels, affine=True)

        if self.instnorm:
            self.norm = nn.InstanceNorm2d(transform.out_channels)

        if self.dropout:
            self.drop = nn.Dropout2d(p=0.5)

    def forward(self, *xs):

        x = th.cat(xs, dim=1)

        if self.activation:
            x = self.lrelu(x)

        x = self.transform(x)

        if self.batchnorm:
            x = self.bn(x)
        if self.instnorm:
            x = self.norm(x)

        if self.dropout:
            x = self.drop(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, block=None, relu=None, layers=4, ratio=2,
                 vblks=None, hblks=None, scales=None, factors=None, size=256):
        super().__init__()

        extension = block.extension
        lrd = block.least_required_dim

        ratio = np.exp2(ratio)
        scales = np.array(scales)
        factors = np.array(factors + [0.0])
        scales = np.exp2(scales)
        factors = np.exp2(factors)
        num_filters = int(in_channels * ratio)

        self.ratio = ratio
        self.hblks = hblks
        self.vblks = vblks
        self.scales = scales
        self.factors = factors
        logger.info('---------------------------------------')
        logger.info('ratio: %f', ratio)
        logger.info('vblks: [%s]', ', '.join(map(str, vblks)))
        logger.info('hblks: [%s]', ', '.join(map(str, hblks)))
        logger.info('scales: [%s]', ', '.join(map(str, scales)))
        logger.info('factors: [%s]', ', '.join(map(str, factors[0:4])))
        logger.info('---------------------------------------')

        self.exceeded = np.any(np.cumprod(scales) * size < 1) or np.any((in_channels * ratio * np.cumprod(factors)) < lrd)
        if not self.exceeded:
            self.layers = layers
            self.in_channels = in_channels
            self.num_filters = num_filters
            self.out_channels = out_channels

            if relu is None:
                relu = nn.ReLU(inplace=True)

            ex = extension
            c0 = int(ex * num_filters // ex * ex)
            self.iconv = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1, groups=1)
            self.oconv = nn.Conv2d(c0, out_channels, kernel_size=3, padding=1, bias=False, groups=1)
            self.relu6 = nn.ReLU6()

            self.enconvs = nn.ModuleList()
            self.dnforms = nn.ModuleList()
            self.hzforms = nn.ModuleList()
            self.upforms = nn.ModuleList()
            self.deconvs = nn.ModuleList()

            self.sizes = [int(size)]
            self.channel_sizes = [c0]
            for ix in range(layers):
                least_factor = ex
                scale, factor = scales[ix], factors[ix]
                self.sizes.append(int(self.sizes[ix] * scale))
                self.channel_sizes.append(int(self.channel_sizes[ix] * factor // least_factor * least_factor))

                ci, co = self.channel_sizes[ix], self.channel_sizes[ix + 1]
                szi, szo = self.sizes[ix + 1], self.sizes[ix]
                logger.info('%d - ci: %d, co: %d', ix, ci, co)
                logger.info('%d - szi: %d, szo: %d', ix, szi, szo)

                self.exceeded = self.exceeded or ci < lrd or co < lrd or szi < 1 or szo < 1
                if not self.exceeded:
                    try:
                        self.enconvs.append(Block(Enconv(ci, co, size=szi), activation=True, batchnorm=False, instnorm=True, dropout=False, relu=relu))
                        self.dnforms.append(Transform(co, co, nblks=vblks[ix], block=block, relu=relu))
                        self.hzforms.append(Transform(co, co, nblks=hblks[ix], block=block, relu=relu))
                        self.deconvs.append(Block(Deconv(co * 2, ci, size=szo), activation=True, batchnorm=False, instnorm=True, dropout=False, relu=relu))
                        self.upforms.append(Transform(ci, ci, nblks=vblks[ix], block=block, relu=relu))
                    except Exception:
                        self.exceeded = True

    def forward(self, x):
        if self.exceeded:
            raise ValueError('scales exceeded!')

        dnt = self.iconv(x)
        hzts = []
        for ix in range(self.layers):
            dnt, enc = self.dnforms[ix](self.enconvs[ix](dnt))
            hzt, _ = self.hzforms[ix](enc)
            hzts.append(hzt)

        upt = dnt
        for ix in range(self.layers - 1, -1, -1):
            hzt = hzts[ix]
            upt, dec = self.upforms[ix](self.deconvs[ix](upt, hzt))

        return self.relu6(self.oconv(upt)) / 6
