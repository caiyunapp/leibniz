# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn
from torch import nn as nn


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
        if x.dtype == th.cfloat or x.dtype == th.cdouble:
            return th.clamp(self.leaky(x.real), max=10) + 1j * th.clamp(self.leaky(x.imag), max=10)
        return th.clamp(self.leaky(x), max=10)


class Sigmoid(th.nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return (1 + th.tanh(x)) / 2.0


class Atanh(nn.Module):
    def __init__(self):
        super(Atanh, self).__init__()

    def forward(self, x):
        return th.log1p(2 * x / (1 - x)) / 2


class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        return self.relu(input.real) + 1j * self.relu(input.imag)


class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ComplexLinear, self).__init__()
        self.rfc = nn.Linear(in_channels, out_channels, bias)
        self.ifc = nn.Linear(in_channels, out_channels, bias)

    def forward(self, input):
        return self.rfc(input.real) - self.ifc(input.imag) + 1j * (self.rfc(input.imag) + self.ifc(input.real))
