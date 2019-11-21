# -*- coding: utf-8 -*-

import torch as th


def add(x, y):
    x1, x2 = x
    y1, y2 = y

    return x1 + y1, x2 + y2


def sub(x, y):
    x1, x2 = x
    y1, y2 = y

    return x1 - y1, x2 - y2


def mult(x, y):
    x1, x2 = x
    y1, y2 = y

    return x1 * y1, x2 * y2


def div(x, y):
    x1, x2 = x
    y1, y2 = y

    return x1 / y1, x2 / y2


def dot(x, y):
    x1, x2 = x
    y1, y2 = y

    return x1 * y1 + x2 * y2


def cross(x, y):
    x1, x2 = x
    y1, y2 = y

    return x1 * y2 - x2 * y1


def norm(v):
    return th.sqrt(th.sum(dot(v, v), dim=1, keepdim=True))


def normsq(v):
    return th.sum(dot(v, v), dim=1, keepdim=True)


def normalize(v):
    x, y = v
    r = th.sqrt(x * x + y * y)

    return x / r, y / r


def det(transform):
    a, b, c, d = transform[0, 0], transform[0, 1], transform[1, 0], transform[1, 1]
    return a * d - b * c
