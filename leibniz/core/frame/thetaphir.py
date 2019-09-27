# -*- coding: utf-8 -*-

import torch as th
import leibniz as lbnz

from cached_property import cached_property
from torch import Tensor
from typing import Tuple


class ThetaPhiRFrame:
    def __init__(self, grid):
        self.default_device = -1
        self.grid = grid
        lbnz.use('theta,phi,r')

        self.thx = - th.sin(lbnz.theta)
        self.thy = th.cos(lbnz.theta)
        self.thz = - lbnz.zero

        self.phx = - th.sin(lbnz.phi) * th.cos(lbnz.theta)
        self.phy = - th.sin(lbnz.phi) * th.sin(lbnz.theta)
        self.phz = th.cos(lbnz.phi)

        self.rx = th.cos(lbnz.phi) * th.cos(lbnz.theta)
        self.ry = th.cos(lbnz.phi) * th.sin(lbnz.theta)
        self.rz = th.sin(lbnz.phi)

    def get_device(self):
        return self.default_device

    def set_device(self, ix):
        self.thx = self.thx.cuda(device=ix)
        self.thy = self.thy.cuda(device=ix)
        self.thz = self.thz.cuda(device=ix)
        self.phx = self.phx.cuda(device=ix)
        self.phy = self.phy.cuda(device=ix)
        self.phz = self.phz.cuda(device=ix)
        self.rx = self.rx.cuda(device=ix)
        self.ry = self.ry.cuda(device=ix)
        self.rz = self.rz.cuda(device=ix)
        self.default_device = ix

    @cached_property
    def phi(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.phx, self.phy, self.phz

    @cached_property
    def theta(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.thx, self.thy, self.thz

    @cached_property
    def r(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.rx, self.ry, self.rz


_name_ = 'thetaphir'
_clazz_ = ThetaPhiRFrame
