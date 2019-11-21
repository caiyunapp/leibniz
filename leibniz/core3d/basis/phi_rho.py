# -*- coding: utf-8 -*-

import torch as th


_name_ = 'phi,rho'
_params_ = ('phi', 'rho')


def transform(phi, rho, **kwargs):
    x = rho * th.cos(phi)
    y = rho * th.sin(phi)
    return x, y


def dtransform(phi, rho, dphi, drho, **kwargs):
    dx = - rho * th.sin(phi) * dphi + th.cos(phi) * drho
    dy = - rho * th.cos(phi) * dphi + th.sin(phi) * drho

    return dx, dy


def inverse(x, y, **kwargs):
    rho = th.sqrt(x * x + y * y)
    phi = th.atan2(y, x)
    return phi, rho


def dinverse(x, y, dx, dy, **kwargs):
    rhosq = x * x + y * y
    rho = th.sqrt(rhosq)

    drho = (x * dx + y * dy) / rho
    dphi = (x * dy - y * dx) / rhosq

    return dphi, drho

