# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn


def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Enconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4):

        super(Enconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, dilation=1)

    def forward(self, x):

        x = self.conv(x)

        return x


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):

        super(Deconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) // 2

        self.deconv = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):

        x = self.deconv(x)
        x = self.conv(x)

        return x


class Transform(nn.Module):
    def __init__(self, in_channels, out_channels, layers=0, block=None):

        super(Transform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = None
        if layers > 0 and block is not None:
            blocks = []
            for i in range(layers - 1):
                blocks.append(block(self.out_channels, step=1.0/layers))
                blocks.append(nn.ReLU(inplace=True))
            blocks.append(block(self.out_channels, step=1.0/layers))
            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.blocks is not None:
            return self.blocks(x), x
        else:
            return x, x


class Block(nn.Module):
    def __init__(self, transform, activation=True, batchnorm=True, dropout=False):

        super(Block, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.blocks = None

        self.transform = transform

        if self.activation:
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        if self.batchnorm:
            self.bn = nn.BatchNorm2d(transform.out_channels, affine=True)

        if self.dropout:
            self.drop = nn.Dropout2d(p=0.5)

    def forward(self, *xs):

        x = th.cat(xs, dim=1)

        if self.activation:
            x = self.lrelu(x)

        x = self.transform(x)

        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, num_filters, out_channels, kernel_size=4, factor=1, layers=0, block=None, initializer=None):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.out_channels = out_channels

        if initializer is None:
            initializer = lambda x: x
        self.initializer = initializer

        f = factor
        self.enconv1 = Block(Enconv(1 * in_channels * 1, f * num_filters * 1, kernel_size=kernel_size), activation=False, batchnorm=True, dropout=False)
        self.enconv2 = Block(Enconv(f * num_filters * 1, f * num_filters * 2, kernel_size=kernel_size), activation=True, batchnorm=True, dropout=False)
        self.enconv3 = Block(Enconv(f * num_filters * 2, f * num_filters * 4, kernel_size=kernel_size), activation=True, batchnorm=True, dropout=False)
        self.enconv4 = Block(Enconv(f * num_filters * 4, f * num_filters * 8, kernel_size=kernel_size), activation=True, batchnorm=True, dropout=False)
        self.enconv5 = Block(Enconv(f * num_filters * 8, f * num_filters * 8, kernel_size=kernel_size), activation=True, batchnorm=True, dropout=False)
        self.enconv6 = Block(Enconv(f * num_filters * 8, f * num_filters * 8, kernel_size=kernel_size), activation=True, batchnorm=True, dropout=False)
        self.enconv7 = Block(Enconv(f * num_filters * 8, f * num_filters * 8, kernel_size=kernel_size), activation=True, batchnorm=True, dropout=False)
        self.enconv8 = Block(Enconv(f * num_filters * 8, f * num_filters * 8, kernel_size=kernel_size), activation=True, batchnorm=False, dropout=False)

        self.dnform1 = Transform(f * num_filters * 1, f * num_filters * 1, layers=layers, block=block)
        self.dnform2 = Transform(f * num_filters * 2, f * num_filters * 2, layers=layers, block=block)
        self.dnform3 = Transform(f * num_filters * 4, f * num_filters * 4, layers=layers, block=block)
        self.dnform4 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.dnform5 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.dnform6 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.dnform7 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.dnform8 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)

        self.hzform1 = Transform(f * num_filters * 1, f * num_filters * 1, layers=layers, block=block)
        self.hzform2 = Transform(f * num_filters * 2, f * num_filters * 2, layers=layers, block=block)
        self.hzform3 = Transform(f * num_filters * 4, f * num_filters * 4, layers=layers, block=block)
        self.hzform4 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.hzform5 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.hzform6 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.hzform7 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.hzform8 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)

        self.upform1 = Transform(f * num_filters * 1, f * out_channels * 1, layers=layers, block=block)
        self.upform2 = Transform(f * num_filters * 1, f * num_filters * 1, layers=layers, block=block)
        self.upform3 = Transform(f * num_filters * 2, f * num_filters * 2, layers=layers, block=block)
        self.upform4 = Transform(f * num_filters * 4, f * num_filters * 4, layers=layers, block=block)
        self.upform5 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.upform6 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.upform7 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)
        self.upform8 = Transform(f * num_filters * 8, f * num_filters * 8, layers=layers, block=block)

        self.deconv1 = Block(Deconv(f * num_filters * 1 * 2, 1 * out_channels * 1, kernel_size=3), activation=True, batchnorm=False, dropout=False)
        self.deconv2 = Block(Deconv(f * num_filters * 2 * 2, f * num_filters * 1, kernel_size=3), activation=True, batchnorm=True, dropout=False)
        self.deconv3 = Block(Deconv(f * num_filters * 4 * 2, f * num_filters * 2, kernel_size=3), activation=True, batchnorm=True, dropout=False)
        self.deconv4 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 4, kernel_size=3), activation=True, batchnorm=True, dropout=False)
        self.deconv5 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 8, kernel_size=3), activation=True, batchnorm=True, dropout=False)
        self.deconv6 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 8, kernel_size=3), activation=True, batchnorm=True, dropout=True)
        self.deconv7 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 8, kernel_size=3), activation=True, batchnorm=True, dropout=True)
        self.deconv8 = Block(Deconv(f * num_filters * 8 * 2, f * num_filters * 8, kernel_size=3), activation=True, batchnorm=True, dropout=True)

        self.relu6 = nn.ReLU6(inplace=True)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initializer(x)

        dnt1, enc1 = self.dnform1(self.enconv1(x))
        dnt2, enc2 = self.dnform2(self.enconv2(dnt1))
        dnt3, enc3 = self.dnform3(self.enconv3(dnt2))
        dnt4, enc4 = self.dnform4(self.enconv4(dnt3))
        dnt5, enc5 = self.dnform5(self.enconv5(dnt4))
        dnt6, enc6 = self.dnform6(self.enconv6(dnt5))
        dnt7, enc7 = self.dnform7(self.enconv7(dnt6))
        dnt8, enc8 = self.dnform8(self.enconv8(dnt7))

        hzt1, enc1 = self.hzform1(enc1)
        hzt2, enc2 = self.hzform2(enc2)
        hzt3, enc3 = self.hzform3(enc3)
        hzt4, enc4 = self.hzform4(enc4)
        hzt5, enc5 = self.hzform5(enc5)
        hzt6, enc6 = self.hzform6(enc6)
        hzt7, enc7 = self.hzform7(enc7)
        hzt8, enc8 = self.hzform8(enc8)

        upt7, dec7 = self.upform8(self.deconv8(dnt8, hzt8))
        upt6, dec6 = self.upform7(self.deconv7(upt7, hzt7))
        upt5, dec5 = self.upform6(self.deconv6(upt6, hzt6))
        upt4, dec4 = self.upform5(self.deconv5(upt5, hzt5))
        upt3, dec3 = self.upform4(self.deconv4(upt4, hzt4))
        upt2, dec2 = self.upform3(self.deconv3(upt3, hzt3))
        upt1, dec1 = self.upform2(self.deconv2(upt2, hzt2))
        upt0, dec0 = self.upform1(self.deconv1(upt1, hzt1))

        return self.relu6(upt0) / 6.0
