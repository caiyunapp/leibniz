# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.nn.conv import conv3x3


class HyperBasic(nn.Module):
    def __init__(self, dim, step):
        super(HyperBasic, self).__init__()

        self.dim = dim

        self.step = step
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(3 * dim, dim)
        self.conv2 = conv3x3(dim, dim)

    def forward(self, y):

        x, theta = y[:, 0:self.dim], y[:, self.dim:2 * self.dim]

        cs = self.step * th.cos(theta * np.pi * 6)
        ss = self.step * th.sin(theta * np.pi * 6)

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss), dim=1)

        dy = self.conv1(ys)
        dy = self.relu(dy)
        dy = self.conv2(dy)
        y = y + dy

        return y


class HyperBottleneck(nn.Module):
    def __init__(self, dim, step):
        super(HyperBottleneck, self).__init__()

        self.dim = dim

        self.step = step
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3 * dim, 3 * dim // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(3 * dim // 4, dim // 2, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(dim // 2, dim, kernel_size=1, bias=False)

    def forward(self, y):

        x, theta = y[:, 0:self.dim // 2], y[:, self.dim // 2:self.dim]

        cs = self.step * th.cos(theta * np.pi * 6)
        ss = self.step * th.sin(theta * np.pi * 6)

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss), dim=1)

        dy = self.conv1(ys)
        dy = self.relu(dy)
        dy = self.conv2(dy)
        dy = self.relu(dy)
        dy = self.conv3(dy)

        y = y + dy * self.step

        return y
