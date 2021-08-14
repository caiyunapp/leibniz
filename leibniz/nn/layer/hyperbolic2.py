# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from leibniz.nn.layer.hyperbolic import BasicBlock, Bottleneck


class HyperBasic(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, ix, tx, relu, conv, reduction=16):
        super(HyperBasic, self).__init__()
        self.dim = dim
        self.step = step
        self.ix = ix
        self.tx = tx

        self.input = BasicBlock(dim, 2 * dim, step, relu, conv, reduction=reduction)
        self.output = BasicBlock(4 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y1 = (x + th.tanh(theta)) * th.exp(step * th.sin(theta)) - th.tanh(theta)
        y2 = (x + th.tanh(theta)) * th.exp(- step * th.cos(theta)) - th.tanh(theta)
        y3 = (x - th.tanh(theta)) * th.exp(step * th.sin(theta)) + th.tanh(theta)
        y4 = (x - th.tanh(theta)) * th.exp(- step * th.cos(theta)) + th.tanh(theta)
        ys = th.cat((y1, y2, y3, y4), dim=1)

        return x + self.output(ys)


class HyperBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, ix, tx, relu, conv, reduction=16):
        super(HyperBottleneck, self).__init__()
        self.dim = dim
        self.step = step
        self.ix = ix
        self.tx = tx

        self.input = Bottleneck(dim, 2 * dim, step, relu, conv, reduction=reduction)
        self.output = Bottleneck(4 * dim, dim, step, relu, conv, reduction=reduction)

    def forward(self, x):
        input = self.input(x)
        velo = input[:, :self.dim]
        theta = input[:, self.dim:]

        step = self.step * velo

        y1 = (x + th.tanh(theta)) * th.exp(step * th.sin(theta)) - th.tanh(theta)
        y2 = (x + th.tanh(theta)) * th.exp(- step * th.cos(theta)) - th.tanh(theta)
        y3 = (x - th.tanh(theta)) * th.exp(step * th.sin(theta)) + th.tanh(theta)
        y4 = (x - th.tanh(theta)) * th.exp(- step * th.cos(theta)) + th.tanh(theta)
        ys = th.cat((y1, y2, y3, y4), dim=1)

        return x + self.output(ys)
