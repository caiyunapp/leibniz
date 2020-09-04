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

        self.input = BasicBlock(dim, 5 * dim, step, relu, conv, reduction=reduction)
        self.output = BasicBlock(8 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, z):
        input = self.input(z)
        x     = input[:, 0 * self.dim:1 * self.dim]
        y     = input[:, 1 * self.dim:2 * self.dim]
        velo  = input[:, 2 * self.dim:3 * self.dim]
        theta = input[:, 3 * self.dim:4 * self.dim]
        phi   = input[:, 4 * self.dim:5 * self.dim]

        cs = self.step * velo * th.cos(theta * np.pi * 6)
        ss = self.step * velo * th.sin(theta * np.pi * 6)
        cp = th.cos(phi * np.pi * 6)
        sp = th.sin(phi * np.pi * 6)

        # z1 = (1 + ss * (cp + 1j * sp)) * (x + 1j * y) + cs
        x1 = x + ss * cp * x - ss * sp * y + cs
        y1 = y + ss * sp * x + ss * cp * y
        # z2 = (1 + cs * (cp + 1j * sp)) * (x + 1j * y) - ss
        x2 = x + cs * cp * x - cs * sp * y - ss
        y2 = y + cs * sp * x + cs * cp * y
        # z3 = (1 - cs * (cp + 1j * sp)) * (x + 1j * y) + ss
        x3 = x - cs * cp * x + cs * sp * y + ss
        y3 = y - cs * sp * x - cs * cp * y
        # z4 = (1 - ss * (cp + 1j * sp)) * (x + 1j * y) - cs
        x4 = x - ss * cp * x + ss * sp * y - cs
        y4 = y - ss * sp * x + ss * sp * y
        zs = th.cat((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)

        return z + self.output(zs)


class CmplxHyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(CmplxHyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step

        self.input = Bottleneck(dim, 5 * dim, step, relu, conv, reduction=reduction)
        self.output = Bottleneck(8 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, z):
        input = self.input(z)
        x     = input[:, 0 * self.dim:1 * self.dim]
        y     = input[:, 1 * self.dim:2 * self.dim]
        velo  = input[:, 2 * self.dim:3 * self.dim]
        theta = input[:, 3 * self.dim:4 * self.dim]
        phi   = input[:, 4 * self.dim:5 * self.dim]

        cs = self.step * velo * th.cos(theta * np.pi * 6)
        ss = self.step * velo * th.sin(theta * np.pi * 6)
        cp = th.cos(phi * np.pi * 6)
        sp = th.sin(phi * np.pi * 6)

        # z1 = (1 + ss * (cp + 1j * sp)) * (x + 1j * y) + cs
        x1 = x + ss * cp * x - ss * sp * y + cs
        y1 = y + ss * sp * x + ss * cp * y
        # z2 = (1 + cs * (cp + 1j * sp)) * (x + 1j * y) - ss
        x2 = x + cs * cp * x - cs * sp * y - ss
        y2 = y + cs * sp * x + cs * cp * y
        # z3 = (1 - cs * (cp + 1j * sp)) * (x + 1j * y) + ss
        x3 = x - cs * cp * x + cs * sp * y + ss
        y3 = y - cs * sp * x - cs * cp * y
        # z4 = (1 - ss * (cp + 1j * sp)) * (x + 1j * y) - cs
        x4 = x - ss * cp * x + ss * sp * y - cs
        y4 = y - ss * sp * x + ss * sp * y
        zs = th.cat((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)

        return z + self.output(zs)
