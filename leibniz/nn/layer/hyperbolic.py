# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from leibniz.nn.layer.senet import SELayer


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
        hidden = min(in_channel, out_channel) // 4 + 1

        self.conv1 = conv(in_channel, hidden, kernel_size=1, bias=False)
        self.conv2 = conv(hidden, hidden, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(hidden, out_channel, kernel_size=1, bias=False)
        self.se = SELayer(out_channel, reduction)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.se(y)
        return y


class HyperBasic(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBasic, self).__init__()
        self.dim = dim
        self.step = step

        self.input = BasicBlock(dim, 2 * dim, step, relu, conv, reduction=reduction)
        self.output = BasicBlock(2 * dim, 2 * dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        r = self.input(x) * self.step
        u = r[:, :self.dim]
        v = r[:, self.dim:]

        y1 = x * (1 + v) + u
        ys = th.cat([y1, x], dim=1)
        r = self.output(ys) * self.step
        u = r[:, :self.dim]
        v = r[:, self.dim:]

        return y1 * (1 + v) + u


class HyperBottleneck(nn.Module):
    extension = 2
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step

        self.input = Bottleneck(dim, 2 * dim, step, relu, conv, reduction=reduction)
        self.output = Bottleneck(2 * dim, 2 * dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        r = self.input(x) * self.step
        u = r[:, :self.dim]
        v = r[:, self.dim:]

        y1 = x * (1 + v) + u
        ys = th.cat([y1, x], dim=1)
        r = self.output(ys) * self.step
        u = r[:, :self.dim]
        v = r[:, self.dim:]

        return y1 * (1 + v) + u
