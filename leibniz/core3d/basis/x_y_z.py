# -*- coding: utf-8 -*-


_name_ = 'x,y,z'
_params_ = ('x', 'y', 'z')


def transform(x, y, z, **kwargs):
    return x, y, z


def dtransform(x, y, z, dx, dy, dz, **kwargs):
    return dx, dy, dz


def inverse(x, y, z, **kwargs):
    return x, y, z


def dinverse(x, y, z, dx, dy, dz, **kwargs):
    return dx, dy, dz

