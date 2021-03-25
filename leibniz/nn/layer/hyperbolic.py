# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.nn.layer.cbam import CBAM


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, step, relu, conv, reduction=16):
        super(BasicBlock, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(out_channel, reduction=reduction, conv=conv)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.cbam(y)
        return y


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, step, relu, conv, reduction=16):
        super(Bottleneck, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(in_channel, in_channel // 4, kernel_size=1, bias=False)
        self.conv2 = conv(in_channel // 4, in_channel // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(in_channel // 4, out_channel, kernel_size=1, bias=False)
        self.cbam = CBAM(out_channel, reduction=reduction, conv=conv)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.cbam(y)
        return y


class HyperBasic(nn.Module):
    extension = 3
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBasic, self).__init__()
        self.dim = dim // 3
        self.step = step
        self.output = BasicBlock(7 * self.dim, 3 * self.dim, step, relu, conv, reduction=reduction)

    def forward(self, xs):
        x, cs, ss = xs[:, :self.dim], xs[:, self.dim:self.dim*2], xs[:, self.dim*2:]

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss, x), dim=1)

        out = self.output(ys)
        dx, dcs, dss = out[:, :self.dim], out[:, self.dim:self.dim*2], out[:, self.dim*2:]
        x, cs, ss = x + dx, cs + dcs, ss + dss

        return th.cat((x, cs, ss), dim=1)


class HyperBottleneck(nn.Module):
    extension = 3
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBottleneck, self).__init__()
        self.dim = dim // 3
        self.step = step
        self.output = Bottleneck(7 * self.dim, 3 * self.dim, step, relu, conv, reduction=reduction)

    def forward(self, xs):
        x, cs, ss = xs[:, :self.dim], xs[:, self.dim:self.dim*2], xs[:, self.dim*2:]

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss, x), dim=1)

        out = self.output(ys)
        dx, dcs, dss = out[:, :self.dim], out[:, self.dim:self.dim*2], out[:, self.dim*2:]
        x, cs, ss = x + dx, cs + dcs, ss + dss

        return th.cat((x, cs, ss), dim=1)
