# -*- coding: utf-8 -*-

import leibniz as iza

from cached_property import cached_property
from torch import Tensor
from typing import Tuple


class XYZFrame:
    def __init__(self, grid):
        self.grid = grid
        self.default_device = -1
        iza.use('x,y,z')

    def get_device(self):
        return self.default_device

    def set_device(self, ix):
        self.grid.set_device(ix)
        self.default_device = ix

    @cached_property
    def x(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.grid.one, self.grid.zero, self.grid.zero

    @cached_property
    def y(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.grid.zero, self.grid.one, self.grid.zero

    @cached_property
    def z(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.grid.zero, self.grid.zero, self.grid.one


_name_ = 'xyz'
_clazz_ = XYZFrame
