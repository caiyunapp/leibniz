# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import leibniz as lbnz

from cached_property import cached_property
from torch import Tensor
from leibniz import cast


class RegularGrid:
    def __init__(self, basis, L=2, W=2, H=2, east=1, west=-1, north=1, south=-1, upper=1, lower=-1):
        self.basis = basis

        self.L = L
        self.W = W
        self.H = H
        self.shape = (1, 1, self.L, self.W, self.H)

        self.east = east
        self.west = west
        self.north = north
        self.south = south
        self.upper = upper
        self.lower = lower

        self.default_device = -1

    def get_device(self):
        if self.default_device == -1:
            return 'cpu'
        return self.default_device

    def set_device(self, ix):
        self.default_device = ix

    def mk_zero(self) -> Tensor:
        return cast(np.zeros(self.shape), device=self.default_device)

    @cached_property
    def zero(self) -> Tensor:
        return self.mk_zero()

    def mk_one(self) -> Tensor:
        return cast(np.ones(self.shape), device=self.default_device)

    @cached_property
    def one(self) -> Tensor:
        return self.mk_one()

    def boundary(self) -> Tensor:
        foreward = self.mk_zero()
        backward = self.mk_zero()
        backward[-1, :, :] = 1
        foreward[0, :, :] = 1

        top = self.mk_zero()
        bottom = self.mk_zero()
        top[:, :, -1] = 1
        bottom[:, :, 0] = 1

        left = self.mk_zero()
        right = self.mk_zero()
        right[:, -1, :] = 1
        left[:, 0, :] = 1

        return th.cat([
            top, bottom, foreward, backward, left, right
        ], dim=1).view(1, 6, self.W, self.L, self.H)

    def data(self) -> Tensor:
        return cast(np.meshgrid(
            np.linspace(self.west, self.east, num=self.L),
            np.linspace(self.south, self.north, num=self.W),
            np.linspace(self.lower, self.upper, num=self.H), indexing='ij'
        ), device=self.default_device).view(1, 3, self.L, self.W, self.H)

    def delta(self) -> Tensor:
        return th.cat([
            (self.east - self.west) / (self.L - 1) * self.one,
            (self.north - self.south) / (self.W - 1) * self.one,
            (self.upper - self.lower) / (self.H - 1) * self.one
        ], dim=1).view(1, 3, self.L, self.W, self.H)

    def random(self) -> Tensor:
        d = lbnz.get_device()
        r = th.rand(1, 1, self.L, self.W, self.H, dtype=th.float64)
        if th.cuda.is_available():
            return r.cuda(device=d)
        else:
            return r
