# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from leibniz.nn.pooling import ComplexAvgPool1d, ComplexAvgPool2d, ComplexAvgPool3d
from leibniz.nn.conv import ComplexConv1d, ComplexConv2d, ComplexConv3d
from leibniz.nn.activation import Sigmoid, ComplexReLU, ComplexLinear


def exp(z):
    return 1 + 2 * th.tanh(z / 2) / (1 - th.tanh(z / 2))


class ComplexSELayer(nn.Module):
    def __init__(self, channel, reduction=16, conv=None):
        super(ComplexSELayer, self).__init__()

        self.avg_pool = None
        self.fc = nn.Sequential(
            ComplexLinear(channel, channel // reduction + 1, bias=False),
            ComplexReLU(),
            ComplexLinear(channel // reduction + 1, channel, bias=False),
            Sigmoid()
        )

    def forward(self, x):
        sz = x.size()

        if len(sz) == 3:
            if self.avg_pool is None:
                self.avg_pool = ComplexAvgPool1d()
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1)
        if len(sz) == 4:
            if self.avg_pool is None:
                self.avg_pool = ComplexAvgPool2d()
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1, 1)
        if len(sz) == 5:
            if self.avg_pool is None:
                self.avg_pool = ComplexAvgPool3d()
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, step, relu, conv, reduction=16):
        super(BasicBlock, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.se = ComplexSELayer(out_channel, reduction)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.se(y)
        return y


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, step, relu, conv, reduction=16):
        super(Bottleneck, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(in_channel, in_channel // 4, kernel_size=1, bias=False)
        self.conv2 = conv(in_channel // 4, in_channel // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(in_channel // 4, out_channel, kernel_size=1, bias=False)
        self.se = ComplexSELayer(out_channel, reduction)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.se(y)
        return y


class CmplxHyperBasic(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBasic, self).__init__()
        self.dim = dim
        self.step = step

        if conv == nn.Conv1d:
            conv = ComplexConv1d
        elif conv == nn.Conv2d:
            conv = ComplexConv2d
        elif conv == nn.Conv3d:
            conv = ComplexConv3d

        self.input = BasicBlock(self.dim, 2 * self.dim, step, relu, conv, reduction=reduction)
        self.output = BasicBlock(4 * self.dim, self.dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y1 = (x + th.tan(theta)) * exp(step * th.sin(theta)) - th.tan(theta)
        y2 = (x + th.tan(theta * 1j)) * exp(step * th.cos(theta * 1j)) - th.tan(theta * 1j)
        y3 = (x + th.tan(theta * -1)) * exp(step * th.sin(theta * -1)) - th.tan(theta * -1)
        y4 = (x + th.tan(theta * -1j)) * exp(step * th.cos(theta * -1j)) - th.tan(theta * -1j)
        ys = th.cat((y1, y2, y3, y4), dim=1)

        y = x + self.output(ys)

        return y


class CmplxHyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step

        if conv == nn.Conv1d:
            conv = ComplexConv1d
        elif conv == nn.Conv2d:
            conv = ComplexConv2d
        elif conv == nn.Conv3d:
            conv = ComplexConv3d

        self.input = Bottleneck(self.dim, 2 * self.dim, step, relu, conv, reduction=reduction)
        self.output = Bottleneck(4 * self.dim, self.dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y1 = (x + th.tanh(theta)) * exp(step * th.sin(theta)) - th.tanh(theta)
        y2 = (x + th.tanh(theta * 1j)) * exp(step * th.cos(theta * 1j)) - th.tanh(theta * 1j)
        y3 = (x + th.tanh(theta * -1)) * exp(step * th.cos(theta * -1)) - th.tanh(theta * -1)
        y4 = (x + th.tanh(theta * -1j)) * exp(step * th.cos(theta * -1j)) - th.tanh(theta * -1j)
        ys = th.cat((y1, y2, y3, y4), dim=1)

        y = x + self.output(ys)

        return y
