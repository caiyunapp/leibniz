# -*- coding: utf-8 -*-

import torch as th


class LocalOps:
    def __init__(self, grid, elem, diff):
        self.grid = grid
        self.elem = elem
        self.diff = diff

    def grad(self, scalar):
        dL1, dL2, dL3 = self.elem.dL

        g_1, g_2, g_3 = self.diff(scalar)
        g_1, g_2, g_3 = g_1 / dL1, g_2 / dL2, g_3 / dL3

        return g_1, g_2, g_3

    def div(self, F):
        F1, F2, F3 = F
        dS1, dS2, dS3 = self.elem.dS

        val_1, _, _ = self.diff(F1 * dS1)
        _, val_2, _ = self.diff(F2 * dS2)
        _, _, val_3 = self.diff(F3 * dS3)

        return (val_1 + val_2 + val_3) / self.elem.dV

    def div_(self, F):
        F1, F2, F3 = F
        dL1, dL2, dL3 = self.elem.dL

        val_1, _, _ = self.diff(F1)
        _, val_2, _ = self.diff(F2)
        _, _, val_3 = self.diff(F3)

        return val_1 / dL1 + val_2 / dL2 + val_3 / dL3

    def curl(self, F):
        F1, F2, F3 = F
        dL1, dL2, dL3 = self.elem.dL
        dS1, dS2, dS3 = self.elem.dS

        _, a, b = self.diff(F1 * dL1)
        c, _, d = self.diff(F2 * dL2)
        e, f, _ = self.diff(F3 * dL3)

        return (f - d) / dS1, (b - e) / dS2, (c - a) / dS3

    def laplacian(self, f):
        return self.div(self.grad(f))

    def vlaplacian(self, F):
        a, b, c = self.grad(self.div(F))
        d, e, f = self.curl(self.curl(F))

        return a - d, b - e, c - f

    def adv(self, wind, data, filter=None):
        u, v, w = wind
        a, b, c = self.grad(data)

        val = a * u + b * v + c * w
        if filter is None:
            return val
        else:
            return filter(val)

    def vadv(self, wind, vector, filter=None):
        u, v, w = wind
        v1, v2, v3 = vector
        a, b, c = self.grad(v1)
        d, e, f = self.grad(v2)
        g, h, i = self.grad(v3)

        val1 = a * u + b * v + c * w
        val2 = d * u + e * v + f * w
        val3 = g * u + h * v + i * w
        if filter is None:
            return val1, val2, val3
        else:
            return filter(val1), filter(val2), filter(val3)

    def upwind(self, wind, data, filter=None, upwindp=None, upwindm=None):
        # upwind scheme using speed-splitting techniques.
        # Referring to https://web.stanford.edu/group/frg/course_work/AA214B/CA-AA214B-Ch6.pdf
        # and https://en.wikipedia.org/wiki/Upwind_scheme
        # Param :
        #   points: points to be used in upwind scheme

        # grad part
        dL1, dL2, dL3 = self.elem.dL

        g_p1, g_p2, g_p3 = upwindp(data)
        g_m1, g_m2, g_m3 = upwindm(data)

        g_p1, g_p2, g_p3 = g_p1 / dL1, g_p2 / dL2, g_p3 / dL3
        g_m1, g_m2, g_m3 = g_m1 / dL1, g_m2 / dL2, g_m3 / dL3

        # adv part
        u, v, w = wind
        zero = self.grid.zero
        u_p, u_m = th.where(u > zero, u, zero), th.where(u < zero, u, zero)
        v_p, v_m = th.where(v > zero, v, zero), th.where(v < zero, v, zero)
        w_p, w_m = th.where(w > zero, w, zero), th.where(w < zero, w, zero)

        val = g_p1 * u_m + g_m1 * u_p + g_p2 * v_m + g_m2 * v_p + g_p3 * w_m + g_m3 * w_p
        if filter is None:
            return val
        else:
            return filter(val)
