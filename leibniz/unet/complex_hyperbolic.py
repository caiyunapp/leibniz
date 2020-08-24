# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.unet.senet import SELayer


class CmplxHyperBasic(nn.Module):
    extension = 2
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBasic, self).__init__()
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

        self.conv0 = self.conv(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv1 = self.conv(dim, dim // 2, kernel_size=3, padding=1, bias=False)
        self.conv2 = self.conv(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv3 = self.conv(dim, dim // 2, kernel_size=3, padding=1, bias=False)
        self.se = SELayer(dim, reduction)

    def forward(self, z):
        x, y = z[:, 0:self.dim // 2], z[:, self.dim // 2:self.dim]

        u = self.conv0(z)
        u = self.relu(u)
        u = self.conv1(u)

        theta = self.conv2(z)
        theta = self.relu(theta)
        theta = self.conv3(theta)

        cs = self.step * u * th.cos(theta * np.pi * 6)
        ss = self.step * u * th.sin(theta * np.pi * 6)

        dx = ss * x - cs * y + cs
        dy = ss * y + cs * x + ss
        dz = th.cat([dx, dy], dim=1)

        z = z + self.se(dz)

        return z


class CmplxHyperBottleneck(nn.Module):
    extension = 6
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBottleneck, self).__init__()
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
        self.conv1 = self.conv(dim, dim // 3, kernel_size=3, padding=1)
        self.conv2 = self.conv(10 * dim // 3, 10 * dim // 12, kernel_size=1, bias=False)
        self.conv3 = self.conv(10 * dim // 12, dim // 3, kernel_size=3, bias=False, padding=1)
        self.conv4 = self.conv(dim // 3, dim, kernel_size=1, bias=False)
        self.se = SELayer(dim, reduction)

    def forward(self, z):
        x, y, theta = z[:, 0:self.dim // 3], z[:, self.dim // 3:self.dim * 2 // 3], z[:, self.dim * 2 // 3:self.dim]
        u = self.conv0(z)
        u = self.relu(u)
        u = self.conv1(u)

        cs = self.step * u * th.cos(theta * np.pi * 6)
        ss = self.step * u * th.sin(theta * np.pi * 6)

        x1 = (1 + ss) * x - cs * y + cs
        y1 = (1 + ss) * y + cs * x + ss
        x2 = (1 + cs) * x + ss * y - ss
        y2 = (1 + cs) * y - ss * x + cs
        x3 = (1 - ss) * x + cs * y - cs
        y3 = (1 - ss) * y - cs * x - ss
        x4 = (1 - cs) * x - ss * y + ss
        y4 = (1 - cs) * y + ss * x - cs
        zs = th.cat((x1, y1, x2, y2, x3, y3, x4, y4, cs, ss), dim=1)

        dz = self.conv2(zs)
        dz = self.relu(dz)
        dz = self.conv3(dz)
        dz = self.relu(dz)
        dz = self.conv4(dz)

        z = z + self.se(dz)

        return z
