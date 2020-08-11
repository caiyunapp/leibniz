# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn


class Swish(th.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class Mish(th.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return x * self.tanh(self.softplus(x))


class CappingRelu(th.nn.Module):
    def __init__(self):
        super(CappingRelu, self).__init__()
        self.leaky = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return th.clamp(self.leaky(x), max=10)


class Atanh(nn.Module):
    def __init__(self):
        super(Atanh, self).__init__()

    def forward(self, x):
        return th.log1p(2 * x / (1 - x)) / 2
