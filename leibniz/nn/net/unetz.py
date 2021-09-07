# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.nn.conv import DepthwiseSeparableConv1d, DepthwiseSeparableConv2d, DepthwiseSeparableConv3d
from leibniz.nn.layer.cbam import CBAM
from leibniz.nn.layer.hyperbolic import Bottleneck
from leibniz.nn.net.hyptube import HypTube

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Enconv(nn.Module):
    def __init__(self, in_channels, out_channels, size=(256, 256), conv=nn.Conv2d, padding=None):

        super(Enconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size

        if len(size) == 1:
            self.scale = nn.Upsample(size=tuple(size), mode='linear')
        elif len(size) == 2:
            self.scale = nn.Upsample(size=tuple(size), mode='bilinear')
        elif len(size) == 3:
            self.scale = nn.Upsample(size=tuple(size), mode='trilinear')

        self.conv = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

    def forward(self, x):
        ratio = (np.array(x.size())[-len(self.size):].prod()) / (np.array(self.size).prod())
        if ratio < 1.0:
            x = self.scale(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.scale(x)

        return x


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, size=(256,256), conv=nn.Conv2d, padding=None):
        super(Deconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size

        if len(size) == 1:
            self.scale = nn.Upsample(size=tuple(size), mode='linear')
        elif len(size) == 2:
            self.scale = nn.Upsample(size=tuple(size), mode='bilinear')
        elif len(size) == 3:
            self.scale = nn.Upsample(size=tuple(size), mode='trilinear')

        self.conv = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

    def forward(self, x):
        ratio = (np.array(x.size())[-len(self.size):].prod()) / (np.array(self.size).prod())
        if ratio < 1.0:
            x = self.scale(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.scale(x)

        return x


class Transform(nn.Module):
    def __init__(self, num_filters, nblks=5, relu=None, conv=nn.Conv2d):

        super(Transform, self).__init__()

        self.nblks = nblks
        self.num_filters = num_filters
        self.relu = relu

        step_length = 1.0 / self.nblks
        self.order1 = Bottleneck(num_filters, 2 * num_filters, step_length, relu, conv, reduction=16)
        self.order2 = Bottleneck(4 * num_filters + 1, 2 * num_filters, step_length, relu, conv, reduction=16)
        self.order3 = Bottleneck(7 * num_filters + 1, 2 * num_filters, step_length, relu, conv, reduction=16)

    def forward(self, x0):
        rslt = self.order1(x0)
        velo = rslt[:, :self.num_filters]
        theta = rslt[:, self.num_filters:]
        du0 = velo * th.cos(theta)
        dv0 = velo * th.sin(theta)

        for _ in range(self.nblks):
            x1 = x0 * (1 + dv0 / self.nblks) + du0 / self.nblks
            x1 = self.relu(x1)

            dd = self.order2(th.cat([x0, x1, du0, dv0, th.ones_like(x0[:, 0:1]) * _ / self.nblks], dim=1))
            du1 = dd[:, self.num_filters * 0:self.num_filters * 1]
            dv1 = dd[:, self.num_filters * 1:self.num_filters * 2]

            x2 = x1 * (1 + dv1 / self.nblks) + du1 / self.nblks
            x2 = self.relu(x2)

            dd = self.order3(th.cat([x0, x1, x2, du0, dv0, du1, dv1, th.ones_like(x1[:, 0:1]) * _ / self.nblks], dim=1))
            du2 = dd[:, self.num_filters * 0:self.num_filters * 1]
            dv2 = dd[:, self.num_filters * 1:self.num_filters * 2]

            x3 = x2 * (1 + dv2 / self.nblks) + du2 / self.nblks
            x3 = self.relu(x3)

        return x3, x3


class Block(nn.Module):
    def __init__(self, transform, activation=True, dropout=-1, relu=None, attn=CBAM, dim=2, normalizor='batch', conv=None):

        super(Block, self).__init__()
        self.activation = activation
        self.dropout_flag = dropout > 0 and self.training
        self.blocks = None

        self.transform = transform

        if self.activation:
            if relu is not None:
                self.lrelu = relu
            else:
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if dim == 1:
            self.normalizor = nn.BatchNorm1d(transform.out_channels, affine=True)
        elif dim == 2:
            self.normalizor = nn.BatchNorm2d(transform.out_channels, affine=True)
        elif dim == 3:
            self.normalizor = nn.BatchNorm3d(transform.out_channels, affine=True)

    def forward(self, *xs):

        x = th.cat(xs, dim=1)

        if self.activation:
            x = self.lrelu(x)

        x = self.transform(x)

        if self.normalizor:
            x = self.normalizor(x)

        return x


class UNetZ(nn.Module):
    def __init__(self, in_channels, out_channels, relu=None, layers=4, ratio=1, ksize_in=7,
                 vblks=None, hblks=None, scales=None, factors=None, spatial=(256, 256), normalizor='tanh'):
        super().__init__()

        extension = 1
        lrd = 1

        spatial = np.array(spatial, dtype=np.int)
        dim = len(spatial)
        self.dim = dim
        Conv = self.get_conv()

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

        self.exceeded = np.any(np.cumprod(scales, axis=0) * spatial < 1) or np.any((in_channels * ratio * np.cumprod(factors)) < lrd)
        if not self.exceeded:
            self.layers = layers
            self.in_channels = in_channels
            self.num_filters = num_filters
            self.out_channels = out_channels

            if self.dim == 2:
                enhencer = HypTube
            self.enhencer_in = None
            self.enhencer_out = None
            self.enhencer_mid = None

            if relu is None:
                relu = nn.ReLU(inplace=True)

            attn = CBAM

            ex = extension
            c0 = int(ex * num_filters)

            self.conv_padding = 1
            self.iconv = Conv(in_channels, c0, kernel_size=ksize_in, padding=(ksize_in - 1) // 2, groups=1)
            self.oconv = Conv(c0, out_channels, kernel_size=3, padding=self.conv_padding, bias=False, groups=1)

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

            self.enconvs = nn.ModuleList()
            self.dnforms = nn.ModuleList()
            self.hzforms = nn.ModuleList()
            self.upforms = nn.ModuleList()
            self.deconvs = nn.ModuleList()

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
                        self.enconvs.append(Block(Enconv(ci, co, size=szi, conv=Conv, padding=self.conv_padding), activation=True, dropout=False,
                                                  relu=relu, attn=attn, dim=self.dim, normalizor=normalizor, conv=Conv))
                        self.dnforms.append(Transform(co, nblks=vblks[ix], relu=relu, conv=Conv))
                        self.hzforms.append(Transform(co, nblks=hblks[ix], relu=relu, conv=Conv))
                        self.deconvs.append(Block(Deconv(co * 2, ci, size=szo, conv=Conv, padding=self.conv_padding), activation=True, dropout=False,
                                                  relu=relu, attn=attn, dim=self.dim, normalizor=normalizor, conv=Conv))
                        self.upforms.append(Transform(ci, nblks=vblks[ix], relu=relu, conv=Conv))
                    except Exception as e:
                        logger.exception(e)
                        self.exceeded = True
                else:
                    logger.error('scales are exceeded!')
                    raise ValueError('scales exceeded!')

            if self.dim == 2 and enhencer is not None:
                self.enhencer_mid = enhencer(co, (c0 + 1) // 2, co)

    def get_conv(self):
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

        dnt = self.iconv(x)

        hzts = []
        for ix in range(self.layers):
            dnt, enc = self.dnforms[ix](self.enconvs[ix](dnt))
            hzt, _ = self.hzforms[ix](enc)
            hzts.append(hzt)

        if self.enhencer_mid is None:
            upt = dnt
        else:
            upt = self.enhencer_mid(dnt)

        for ix in range(self.layers - 1, -1, -1):
            hzt = hzts[ix]
            upt, dec = self.upforms[ix](self.deconvs[ix](upt, hzt))

        upt = self.oconv(upt)
        if self.normalizor:
            return self.normalizor(upt) * self.scale + self.bias
        else:
            return upt * self.scale + self.bias
