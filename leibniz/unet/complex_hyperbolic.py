# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.unet.hyperbolic import BasicBlock, Bottleneck


class CmplxHyperBasic(nn.Module):
    extension = 2
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBasic, self).__init__()
        self.dim = dim
        self.step = step

        self.input = BasicBlock(dim, 4 * dim, step, relu, conv, reduction=reduction)
        self.output = BasicBlock(8 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, z):
        input = self.input(z)
        x = input[:, 0 * self.dim:1 * self.dim]
        y = input[:, 1 * self.dim:2 * self.dim]
        v = input[:, 2 * self.dim:3 * self.dim]
        t = input[:, 3 * self.dim:4 * self.dim]

        cs = self.step * v * th.cos(t * np.pi * 6)
        ss = self.step * v * th.sin(t * np.pi * 6)

        x1 = (1 + ss) * x - cs * y + cs
        y1 = (1 + ss) * y + cs * x + ss
        x2 = (1 + cs) * x + ss * y - ss
        y2 = (1 + cs) * y - ss * x + cs
        x3 = (1 - ss) * x + cs * y - cs
        y3 = (1 - ss) * y - cs * x - ss
        x4 = (1 - cs) * x - ss * y + ss
        y4 = (1 - cs) * y + ss * x - cs
        zs = th.cat((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)

        return z + self.output(zs)


class CmplxHyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step

        self.input = Bottleneck(dim, 4 * dim, step, relu, conv, reduction=reduction)
        self.output = Bottleneck(8 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, z):
        input = self.input(z)
        x = input[:, 0 * self.dim:1 * self.dim]
        y = input[:, 1 * self.dim:2 * self.dim]
        v = input[:, 2 * self.dim:3 * self.dim]
        t = input[:, 3 * self.dim:4 * self.dim]

        cs = self.step * v * th.cos(t * np.pi * 6)
        ss = self.step * v * th.sin(t * np.pi * 6)

        x1 = (1 + ss) * x - cs * y + cs
        y1 = (1 + ss) * y + cs * x + ss
        x2 = (1 + cs) * x + ss * y - ss
        y2 = (1 + cs) * y - ss * x + cs
        x3 = (1 - ss) * x + cs * y - cs
        y3 = (1 - ss) * y - cs * x - ss
        x4 = (1 - cs) * x - ss * y + ss
        y4 = (1 - cs) * y + ss * x - cs
        zs = th.cat((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)

        return z + self.output(zs)
