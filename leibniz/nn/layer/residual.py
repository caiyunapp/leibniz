# -*- coding: utf-8 -*-

import torch as th


class Basic(th.nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, ix, tx, relu, conv):
        super(Basic, self).__init__()
        self.ix = ix
        self.tx = tx

        self.step = step
        self.relu = relu
        self.conv1 = conv(dim, dim, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.conv2 = conv(dim, dim, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = x + y

        return y


class Bottleneck(th.nn.Module):
    extension = 1
    least_required_dim = 4

    def __init__(self, dim, step, ix, tx, relu, conv):
        super(Bottleneck, self).__init__()
        self.ix = ix
        self.tx = tx

        self.step = step
        self.relu = relu
        self.conv1 = conv(dim, dim // 4, kernel_size=1, bias=False)
        self.conv2 = conv(dim // 4, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(dim // 4, dim, kernel_size=1, bias=False)

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = x + y

        return y
