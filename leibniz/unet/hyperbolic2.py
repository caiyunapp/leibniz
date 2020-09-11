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
        self.output = BasicBlock(2 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y2 = (x + th.tanh(theta)) * th.exp(step * th.cosh(theta)) - th.tanh(theta)
        y4 = (x - th.tanh(theta)) * th.exp(step * th.cosh(theta)) + th.tanh(theta)
        ys = th.cat((y2, y4), dim=1)

        return x + self.output(ys)


class HyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(HyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step

        self.input = Bottleneck(dim, 2 * dim, step, relu, conv, reduction=reduction)
        self.output = Bottleneck(2 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y2 = (x + th.tanh(theta)) * th.exp(step * th.cosh(theta)) - th.tanh(theta)
        y4 = (x - th.tanh(theta)) * th.exp(step * th.cosh(theta)) + th.tanh(theta)
        ys = th.cat((y2, y4), dim=1)

        return x + self.output(ys)
