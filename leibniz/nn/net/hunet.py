# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.nn.net.simple import Linear
from leibniz.nn.net.hyptube import HypTube
from leibniz.nn.conv import DepthwiseSeparableConv1d, DepthwiseSeparableConv2d, DepthwiseSeparableConv3d
from leibniz.nn.layer.cbam import CBAM

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Tube(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(Tube, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tube = HypTube(in_channels, in_channels + out_channels, out_channels, encoder=Linear, decoder=Linear)

    def forward(self, x):
        return self.tube(x)


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

        self.padding = padding
        if padding is not None:
            self.conv = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1)
        else:
            self.conv = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

    def forward(self, x):
        ratio = (np.array(x.size())[-len(self.size):].prod()) / (np.array(self.size).prod())
        if ratio < 1.0:
            x = self.scale(x)
            if self.padding is not None:
                x = self.padding(x)
            x = self.conv(x)
        else:
            if self.padding is not None:
                x = self.padding(x)
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

        self.padding = padding
        if padding is not None:
            self.conv = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1)
        else:
            self.conv = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)

    def forward(self, x):
        ratio = (np.array(x.size())[-len(self.size):].prod()) / (np.array(self.size).prod())
        if ratio < 1.0:
            x = self.scale(x)
            if self.padding is not None:
                x = self.padding(x)
            x = self.conv(x)
        else:
            if self.padding is not None:
                x = self.padding(x)
            x = self.conv(x)
            x = self.scale(x)

        return x


class Transform(nn.Module):
    def __init__(self, in_channels, out_channels, nblks=0, block=None, relu=None, conv=nn.Conv2d):
        super(Transform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if relu is None:
            relu = nn.ReLU(inplace=True)

        self.blocks = None
        if nblks > 0 and block is not None:
            blocks = []
            for i in range(nblks - 1):
                blocks.append(block(self.out_channels, step=1.0 / nblks, relu=relu, conv=conv))
                blocks.append(relu)
            blocks.append(block(self.out_channels, step=1.0 / nblks, relu=relu, conv=conv))
            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):

        if self.blocks is not None:
            return self.blocks(x), x
        else:
            return x, x


class Block(nn.Module):
    def __init__(self, transform, activation=True, dropout=False, relu=None, attn=CBAM, dim=2, normalizor='batch', conv=None):

        super(Block, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.blocks = None

        self.transform = transform

        self.attn = attn(transform.out_channels, conv=conv)

        if self.activation:
            if relu is not None:
                self.lrelu = relu
            else:
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.normalizor = None
        if normalizor == 'batch':
            if dim == 1:
                self.normalizor = nn.BatchNorm1d(transform.out_channels, affine=True)
            elif dim == 2:
                self.normalizor = nn.BatchNorm2d(transform.out_channels, affine=True)
            elif dim == 3:
                self.normalizor = nn.BatchNorm3d(transform.out_channels, affine=True)

        elif normalizor == 'instance':
            if dim == 1:
                self.normalizor = nn.InstanceNorm1d(transform.out_channels)
            elif dim == 2:
                self.normalizor = nn.InstanceNorm2d(transform.out_channels)
            elif dim == 3:
                self.normalizor = nn.InstanceNorm3d(transform.out_channels)

        elif normalizor == 'layer':
            self.normalizor = nn.LayerNorm(tuple([transform.out_channels]) + tuple(transform.size))

        if self.dropout:
            if dim == 1:
                self.drop = nn.Dropout(p=0.5)
            elif dim == 2:
                self.drop = nn.Dropout2d(p=0.5)
            elif dim == 3:
                self.drop = nn.Dropout3d(p=0.5)


    def forward(self, *xs):

        x = th.cat(xs, dim=1)

        if self.activation:
            x = self.lrelu(x)

        x = self.transform(x)

        if self.normalizor:
            x = self.normalizor(x)

        if self.dropout:
            x = self.drop(x)

        x = self.attn(x)

        return x


class HUNet(nn.Module):
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
            if padding:
                self.conv_padding = 0
                self.iconv = nn.Sequential(
                    padding,
                    Conv(in_channels, c0, kernel_size=3, padding=self.conv_padding, groups=1),
                )
                self.oconv = nn.Sequential(
                    padding,
                    Conv(c0, out_channels, kernel_size=3, padding=self.conv_padding, bias=False, groups=1),
                )
            else:
                self.conv_padding = 1
                self.iconv = Conv(in_channels, c0, kernel_size=3, padding=self.conv_padding, groups=1)
                self.oconv = Conv(c0, out_channels, kernel_size=3, padding=self.conv_padding, bias=False, groups=1)

            if final_normalized:
                self.relu6 = nn.ReLU6()


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
                        dropout_flag = (layers - ix) * 3 < layers
                        self.enconvs.append(Block(Enconv(ci, co, size=szi, conv=TConv, padding=padding), activation=True, dropout=dropout_flag, relu=relu, attn=attn, dim=self.dim, normalizor=normalizor, conv=TConv))
                        self.dnforms.append(Transform(co, co, nblks=vblks[ix], block=block, relu=relu, conv=TConv))
                        self.hzforms.append(Tube(co, co))
                        self.deconvs.append(Block(Deconv(co * 2, ci, size=szo, conv=TConv, padding=padding), activation=True, dropout=False, relu=relu, attn=attn, dim=self.dim, normalizor=normalizor, conv=TConv))
                        self.upforms.append(Transform(ci, ci, nblks=vblks[ix], block=block, relu=relu, conv=TConv))
                    except Exception as e:
                        logger.exception(e)
                        self.exceeded = True
                else:
                    logger.error('scales are exceeded!')
                    raise ValueError('scales exceeded!')

            self.enhencer = Tube(co, co)

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

        dnt = self.iconv(x)
        hzts = []
        for ix in range(self.layers):
            dnt, enc = self.dnforms[ix](self.enconvs[ix](dnt))
            hzt = self.hzforms[ix](enc)
            hzts.append(hzt)

        upt = self.enhencer(dnt)

        for ix in range(self.layers - 1, -1, -1):
            hzt = hzts[ix]
            upt, dec = self.upforms[ix](self.deconvs[ix](upt, hzt))

        if self.final_normalized:
            return self.relu6(self.oconv(upt)) / 6
        else:
            return self.oconv(upt)
