# -*- coding: utf-8 -*-

import torch as th


def add(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 + y1, x2 + y2, x3 + y3


def sub(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 - y1, x2 - y2, x3 - y3


def mult(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 * y1, x2 * y2, x3 * y3


def div(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 / y1, x2 / y2, x3 / y3


def dot(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return x1 * y1 + x2 * y2 + x3 * y3


def cross(x, y):
    x1, x2, x3 = x
    y1, y2, y3 = y

    return (
        x2 * y3 - x3 * y2,
        x3 * y1 - x1 * y3,
        x1 * y2 - x2 * y1,
    )


def norm(v):
    return th.sqrt(th.sum(dot(v, v), dim=1, keepdim=True))


def normsq(v):
    return th.sum(dot(v, v), dim=1, keepdim=True)


def normalize(v):
    x, y, z = v
    r = th.sqrt(x * x + y * y + z * z)

    return x / r, y / r, z / r


def box(a, b, c):
    return dot(a, cross(b, c))


def det(transform):
    return box(transform[0:1, 0:3], transform[1:2, 0:3], transform[2:3, 0:3])
