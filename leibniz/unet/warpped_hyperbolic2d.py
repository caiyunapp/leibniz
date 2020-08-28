# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.unet.warp import WarpLayer


class HyperBasic(nn.Module):
    extension = 2
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBasic, self).__init__()
        self.dim = dim
        self.step = step

        if relu is None:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = relu

        if conv is None:
            self.conv = nn.Conv2d
        else:
            self.conv = conv

        self.conv0 = self.conv(dim, dim, kernel_size=3, padding=1)
        self.conv1 = self.conv(dim, dim // 2, kernel_size=3, padding=1)
        self.conv2 = self.conv(3 * dim, dim, kernel_size=3, padding=1)
        self.conv3 = self.conv(dim, dim, kernel_size=3, padding=1)
        self.warp = WarpLayer(dim)

    def forward(self, y):
        x, theta = y[:, 0:self.dim // 2], y[:, self.dim // 2:self.dim]
        u = self.conv0(y)
        u = self.relu(u)
        u = self.conv1(u)

        cs = self.step * u * th.cos(theta * np.pi * 6)
        ss = self.step * u * th.sin(theta * np.pi * 6)

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss), dim=1)

        dy = self.conv2(ys)
        dy = self.relu(dy)
        dy = self.conv3(dy)

        y = y + self.warp(dy)

        return y


class HyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step

        if relu is None:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = relu

        if conv is None:
            self.conv = nn.Conv2d
        else:
            self.conv = conv

        self.conv0 = self.conv(dim, dim, kernel_size=3, padding=1)
        self.conv1 = self.conv(dim, dim, kernel_size=3, padding=1)
        self.conv2 = self.conv(dim, dim, kernel_size=3, padding=1)
        self.conv3 = self.conv(dim, dim, kernel_size=3, padding=1)
        self.conv4 = self.conv(6 * dim, 6 * dim // 4, kernel_size=1, bias=False)
        self.conv5 = self.conv(6 * dim // 4, dim, kernel_size=3, bias=False, padding=1)
        self.conv6 = self.conv(dim, dim, kernel_size=1, bias=False)
        self.warp = WarpLayer(dim, reduction)

    def forward(self, y):
        u = self.conv0(y)
        u = self.relu(u)
        u = self.conv1(u)

        theta = self.conv2(y)
        theta = self.relu(theta)
        theta = self.conv3(theta)

        cs = self.step * u * th.cos(theta * np.pi * 6)
        ss = self.step * u * th.sin(theta * np.pi * 6)

        y1 = (1 + ss) * y + cs
        y2 = (1 + cs) * y - ss
        y3 = (1 - cs) * y + ss
        y4 = (1 - ss) * y - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss), dim=1)

        dy = self.conv4(ys)
        dy = self.relu(dy)
        dy = self.conv5(dy)
        dy = self.relu(dy)
        dy = self.conv6(dy)

        y = y + self.warp(dy)

        return y
