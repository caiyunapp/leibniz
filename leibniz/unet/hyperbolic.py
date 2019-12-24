# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
import torch.nn as nn

from leibniz.unet.base import conv3x3


logger = logging.getLogger('hyperbolic')
logger.setLevel(logging.INFO)


class HyperBasic(nn.Module):
    def __init__(self, dim, step):
        super(HyperBasic, self).__init__()

        self.dim = dim

        self.step = step
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(3 * dim, dim)
        self.conv2 = conv3x3(dim, dim)

    def forward(self, y):

        x, theta = y[:, 0:self.dim], y[:, self.dim:2 * self.dim]

        cs = self.step * th.cos(theta * np.pi * 6)
        ss = self.step * th.sin(theta * np.pi * 6)

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss), dim=1)

        dy = self.conv1(ys)
        dy = self.relu(dy)
        dy = self.conv2(dy)
        y = y + dy

        #if logger.level == logging.INFO:
        #    xval = x.cpu().detach().numpy()
        #    logger.info(f'x: {xval.min():0.8f}, {xval.max():0.8f}, {xval.mean():0.8f}')
        #
        #    theta = th.fmod(theta, 1.0)
        #    tval = theta.cpu().detach().numpy()
        #    logger.info(f'θ: {tval.min():0.8f}, {tval.max():0.8f}, {tval.mean():0.8f}')

        return y


class HyperBottleneck(nn.Module):
    def __init__(self, dim, step):
        super(HyperBottleneck, self).__init__()

        self.dim = dim

        self.step = step
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3 * dim, 3 * dim // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(3 * dim // 4, dim // 2, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(dim // 2, dim, kernel_size=1, bias=False)

    def forward(self, y):

        x, theta = y[:, 0:self.dim // 2], y[:, self.dim // 2:self.dim]

        cs = self.step * th.cos(theta * np.pi * 6)
        ss = self.step * th.sin(theta * np.pi * 6)

        y1 = (1 + ss) * x + cs
        y2 = (1 + cs) * x - ss
        y3 = (1 - cs) * x + ss
        y4 = (1 - ss) * x - cs
        ys = th.cat((y1, y2, y3, y4, cs, ss), dim=1)

        dy = self.conv1(ys)
        dy = self.relu(dy)
        dy = self.conv2(dy)
        dy = self.relu(dy)
        dy = self.conv3(dy)

        y = y + dy * self.step

        #if logger.level == logging.INFO:
        #    xval = x.cpu().detach().numpy()
        #    logger.info(f'x: {xval.min():0.8f}, {xval.max():0.8f}, {xval.mean():0.8f}')
        #
        #    theta = th.fmod(theta, 1.0)
        #    tval = theta.cpu().detach().numpy()
        #    logger.info(f'θ: {tval.min():0.8f}, {tval.max():0.8f}, {tval.mean():0.8f}')

        return y
