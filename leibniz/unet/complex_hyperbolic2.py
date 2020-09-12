# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from leibniz.nn.pooling import ComplexAvgPool1d, ComplexAvgPool2d, ComplexAvgPool3d
from leibniz.nn.conv import ComplexConv1d, ComplexConv2d, ComplexConv3d
from leibniz.nn.activation import Sigmoid, ComplexReLU, ComplexLinear


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

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
        self.se = SELayer(out_channel, reduction)

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
        self.se = SELayer(out_channel, reduction)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.se(y)
        return y


class CmplxHyperBasic(nn.Module):
    extension = 2
    least_required_dim = 2

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBasic, self).__init__()
        self.dim = dim // 2
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
        sz = x.size()
        if len(sz) == 3:
            x = x.view(sz[0], sz[1] // 2, 2, sz[2]).permute(0, 1, 3, 2).contiguous()
        elif len(sz) == 4:
            x = x.view(sz[0], sz[1] // 2, 2, sz[2], sz[3]).permute(0, 1, 3, 4, 2).contiguous()
        elif len(sz) == 5:
            x = x.view(sz[0], sz[1] // 2, 2, sz[2], sz[3], sz[4]).permute(0, 1, 3, 4, 5, 2).contiguous()
        x = th.view_as_complex(x)

        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y1 = (x + th.tanh(theta)) * th.exp(step * th.sin(theta)) - th.tanh(theta)
        y2 = (x + th.tanh(theta * 1j)) * th.exp(step * th.cos(theta * 1j)) - th.tanh(theta * 1j)
        y3 = (x + th.tanh(theta * -1)) * th.exp(step * th.sin(theta * -1)) - th.tanh(theta * -1)
        y4 = (x + th.tanh(theta * -1j)) * th.exp(step * th.cos(theta * -1j)) - th.tanh(theta * -1j)
        ys = th.cat((y1, y2, y3, y4), dim=1)

        y = th.view_as_real(x + self.output(ys))

        if len(sz) == 3:
            y = y.permute(0, 1, 3, 2).reshape(*sz)
        elif len(sz) == 4:
            y = y.permute(0, 1, 4, 2, 3).reshape(*sz)
        elif len(sz) == 5:
            y = y.permute(0, 1, 5, 2, 3, 4).reshape(*sz)
        return y


class CmplxHyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 2

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBottleneck, self).__init__()
        self.dim = dim // 2
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
        sz = x.size()
        if len(sz) == 3:
            x = x.view(sz[0], sz[1] // 2, 2, sz[2]).permute(0, 1, 3, 2).contiguous()
        elif len(sz) == 4:
            x = x.view(sz[0], sz[1] // 2, 2, sz[2], sz[3]).permute(0, 1, 3, 4, 2).contiguous()
        elif len(sz) == 5:
            x = x.view(sz[0], sz[1] // 2, 2, sz[2], sz[3], sz[4]).permute(0, 1, 3, 4, 5, 2).contiguous()
        x = th.view_as_complex(x)

        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y1 = (x + th.tanh(theta)) * th.exp(step * th.sin(theta)) - th.tanh(theta)
        y2 = (x + th.tanh(theta * 1j)) * th.exp(step * th.cos(theta * 1j)) - th.tanh(theta * 1j)
        y3 = (x + th.tanh(theta * -1)) * th.exp(step * th.cos(theta * -1)) - th.tanh(theta * -1)
        y4 = (x + th.tanh(theta * -1j)) * th.exp(step * th.cos(theta * -1j)) - th.tanh(theta * -1j)
        ys = th.cat((y1, y2, y3, y4), dim=1)

        y = th.view_as_real(x + self.output(ys))

        if len(sz) == 3:
            y = y.permute(0, 1, 3, 2).reshape(*sz)
        elif len(sz) == 4:
            y = y.permute(0, 1, 4, 2, 3).reshape(*sz)
        elif len(sz) == 5:
            y = y.permute(0, 1, 5, 2, 3, 4).reshape(*sz)
        return y
