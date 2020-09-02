# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.unet.senet import SEBasicBlock, SEBottleneck, SELayer


class HyperBasic(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBasic, self).__init__()
        self.dim = dim
        self.step = step
        self.relu = relu

        self.theta = SEBasicBlock(dim, step, relu, conv, reduction=reduction)
        self.velo = SEBasicBlock(dim, step, relu, conv, reduction=reduction)
        self.conv1 = conv(6 * dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.se = SELayer(dim, reduction)

    def forward(self, x):
        velo = self.velo(x)
        theta = self.theta(x)

        cs = self.step * velo * th.cos(theta * np.pi * 6)
        ss = self.step * velo * th.sin(theta * np.pi * 6)

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss), dim=1)

        y = self.conv1(ys)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.se(y)
        y = x + y

        return y


class HyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step
        self.relu = relu

        self.theta = SEBottleneck(dim, step, relu, conv, reduction=reduction)
        self.velo = SEBottleneck(dim, step, relu, conv, reduction=reduction)
        self.conv1 = conv(6 * dim, 3 * dim // 2, kernel_size=1, bias=False)
        self.conv2 = conv(3 * dim // 2, 3 * dim // 2, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(3 * dim // 2, dim, kernel_size=1, bias=False)
        self.se = SELayer(dim, reduction)

    def forward(self, x):
        velo = self.velo(x)
        theta = self.theta(x)

        cs = self.step * velo * th.cos(theta * np.pi * 6)
        ss = self.step * velo * th.sin(theta * np.pi * 6)

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss), dim=1)

        y = self.conv1(ys)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.se(y)
        y = x + y

        return y
