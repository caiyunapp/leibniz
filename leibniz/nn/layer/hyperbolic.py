# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.nn.layer.simam import SimAM


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, step, relu, conv, reduction=16):
        super(BasicBlock, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.simam = SimAM(out_channel, reduction=reduction, conv=conv)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.simam(y)
        return y


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, step, relu, conv, reduction=16):
        super(Bottleneck, self).__init__()
        self.step = step
        self.relu = relu
        hidden = max(in_channel, out_channel) // 4

        self.conv1 = conv(in_channel, hidden, kernel_size=1, bias=False)
        self.conv2 = conv(hidden, hidden, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(hidden, out_channel, kernel_size=1, bias=False)
        self.simam = SimAM(out_channel, reduction=reduction, conv=conv)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.simam(y)
        return y


class HyperBasic(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBasic, self).__init__()
        self.dim = dim
        self.step = step

        self.input = BasicBlock(dim, 2 * dim, step, relu, conv, reduction=reduction)
        self.output = BasicBlock(4 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x) * self.step
        u = input[:, :self.dim]
        v = input[:, self.dim:]

        y1 = x * v + u
        y2 = x * u - v
        y3 = - x * v - u
        y4 = - x * u + v
        ys = th.cat((y1, y2, y3, y4), dim=1)

        return x + self.output(ys)


class HyperBottleneck(nn.Module):
    extension = 2
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step

        self.input = Bottleneck(dim, 2 * dim, step, relu, conv, reduction=reduction)
        self.output = Bottleneck(4 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x) * self.step
        u = input[:, :self.dim]
        v = input[:, self.dim:]

        y1 = x * v + u
        y2 = x * u - v
        y3 = - x * v - u
        y4 = - x * u + v
        ys = th.cat((y1, y2, y3, y4), dim=1)

        return x + self.output(ys)
