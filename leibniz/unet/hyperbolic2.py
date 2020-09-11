# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.unet.hyperbolic import BasicBlock, Bottleneck


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
        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y1 = (x + 1.0 / th.tan(theta)) * th.exp(step * th.sin(theta)) - 1.0 / th.tan(theta)
        y2 = (x + th.tan(theta)) * th.exp(- step * th.cos(theta)) - th.tan(theta)
        y3 = (x - 1.0 / th.tan(theta)) * th.exp(- step * th.sin(theta)) + 1.0 / th.tan(theta)
        y4 = (x - th.tan(theta)) * th.exp(- step * th.cos(theta)) + th.tan(theta)
        ys = th.cat((y1, y2, y3, y4), dim=1)

        return x + self.output(ys)


class HyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step

        self.input = Bottleneck(dim, 2 * dim, step, relu, conv, reduction=reduction)
        self.output = Bottleneck(4 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y1 = (x + 1.0 / th.tan(theta)) * th.exp(step * th.sin(theta)) - 1.0 / th.tan(theta)
        y2 = (x + th.tan(theta)) * th.exp(- step * th.cos(theta)) - th.tan(theta)
        y3 = (x - 1.0 / th.tan(theta)) * th.exp(- step * th.sin(theta)) + 1.0 / th.tan(theta)
        y4 = (x - th.tan(theta)) * th.exp(- step * th.cos(theta)) + th.tan(theta)
        ys = th.cat((y1, y2, y3, y4), dim=1)

        return x + self.output(ys)
