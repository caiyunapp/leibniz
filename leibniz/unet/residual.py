# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from leibniz.unet.base import conv3x3


class Basic(th.nn.Module):
    def __init__(self, dim, step):
        super(Basic, self).__init__()

        self.step = step
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        self.conv2 = conv3x3(dim, dim)

        nn.init.normal_(self.conv1.weight, 0.0, 0.04)
        nn.init.normal_(self.conv2.weight, 0.0, 0.04)

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = x + y

        return y


class Bottleneck(th.nn.Module):
    def __init__(self, dim, step):
        super(Bottleneck, self).__init__()

        self.step = step
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False)

        nn.init.normal_(self.conv1.weight, 0.0, 0.04)
        nn.init.normal_(self.conv2.weight, 0.0, 0.04)

    def forward(self, x):

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = x + y

        return y
