# -*- coding: utf-8 -*-


_name_ = 'x,y'
_params_ = ('x', 'y')


def transform(x, y, **kwargs):
    return x, y


def dtransform(x, y, dx, dy, **kwargs):
    return dx, dy


def inverse(x, y, **kwargs):
    return x, y


def dinverse(x, y, dx, dy, **kwargs):
    return dx, dy

