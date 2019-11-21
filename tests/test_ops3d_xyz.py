# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch as th
import leibniz as lbnz

from torch import Tensor
from leibniz.core3d.gridsys.regular3 import RegularGrid


class TestOpsXYZ(unittest.TestCase):

    def setUp(self):
        lbnz.bind(RegularGrid(
            basis='x,y,z',
            L=51, W=51, H=51,
            east=6.0, west=1.0,
            north=6.0, south=1.0,
            upper=6.0, lower=1.0
        ))
        lbnz.use('xyz')
        lbnz.use('theta,phi,r')
        lbnz.use('thetaphir')

    def tearDown(self):
        lbnz.clear()

    def magnitude(self, var):
        # return magnitude of variable to test accuracy
        if th.is_tensor(var):
            var = var.cpu().numpy()
        if isinstance(var, np.ndarray):
            return 10 ** int(np.log10(np.abs(var.max())) + 1)
        else:
            return 10 ** int(np.log10(np.abs(var)) + 1)

    def assertAlmostEqualWithMagnitude(self, excepted, test, magnitude=6):
        if (isinstance(excepted, Tensor) and th.abs(excepted).max().cpu().numpy() == 0.0) or \
                (isinstance(excepted, float) and excepted == 0.0):
            return self.assertAlmostEqual(0.0, test.max().cpu().numpy(), magnitude)
        else:
            mag = self.magnitude(excepted)
            err = th.abs(excepted - test) / mag
            return self.assertAlmostEqual(0.0, err.max().cpu().numpy(), magnitude)

    def assertAlmostEqualWithMagExceptBoundary(self, excepted, test, magnitude=6):
        if isinstance(excepted, float):
            test = test[:, :, 1:-1, 1:-1, 1:-1]
            return self.assertAlmostEqualWithMagnitude(excepted, test, magnitude)
        else:
            excepted, test = excepted[:, :, 1:-1, 1:-1, 1:-1], test[:, :, 1:-1, 1:-1, 1:-1]
            return self.assertAlmostEqualWithMagnitude(excepted, test, magnitude)

    def test_grad0(self):
        dLx, dLy, dLz = lbnz.dL

        g0, g1, g2 = lbnz.grad(lbnz.x)

        self.assertAlmostEqualWithMagExceptBoundary(5.0, g0 * dLx * (lbnz.W - 1), 6)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, g2, 6)

        g0, g1, g2 = lbnz.grad(lbnz.y)

        self.assertAlmostEqualWithMagExceptBoundary(0.0, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(5.0, g1 * dLy * (lbnz.L - 1), 6)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, g2, 6)

        g0, g1, g2 = lbnz.grad(lbnz.z)

        self.assertAlmostEqualWithMagExceptBoundary(0.0, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(5.0, g2 * dLz * (lbnz.H - 1), 6)

    def test_grad1(self):
        # f = x * y * y + z * z * z
        # expect that grad_f = y * y, 2 * x * y, 3 * z * z

        fld = lbnz.x * lbnz.y * lbnz.y + lbnz.z * lbnz.z * lbnz.z
        e0, e1, e2 = lbnz.y * lbnz.y, 2 * lbnz.x * lbnz.y, 3 * lbnz.z * lbnz.z
        g0, g1, g2 = lbnz.grad(fld)

        self.assertAlmostEqualWithMagExceptBoundary(e0, g0, 3)
        self.assertAlmostEqualWithMagExceptBoundary(e1, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(e2, g2, 4)

    def test_div1(self):
        # F = (0, phi * theta, 0);
        # expect that div_F = theta * (-z * y / (r ** 2 * sqrt(r ** 2 - z ** 2))) + phi * x / (x ** 2 + y ** 2)

        fld = lbnz.zero, lbnz.phi * lbnz.theta, lbnz.zero
        expected = lbnz.theta * (-lbnz.z * lbnz.y / (lbnz.r ** 2 * th.sqrt(lbnz.r ** 2 - lbnz.z ** 2))) + lbnz.phi * lbnz.x / \
            (lbnz.x ** 2 + lbnz.y ** 2)
        test = lbnz.div(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 2)

    def test_div2(self):
        # F = (-y, x * y, z);
        # expect that div_F = x + 1

        fld = - lbnz.y, lbnz.x * lbnz.y, lbnz.z
        expected = lbnz.x + 1
        test = lbnz.div(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 6)

    def test_div3(self):
        # F = y * y, 2 * x * y, 3 * z * z
        # expect that div_F = 2 * x + 6 * z

        fld = lbnz.y * lbnz.y, 2 * lbnz.x * lbnz.y, 3 * lbnz.z * lbnz.z
        expected = 2 * lbnz.x + 6 * lbnz.z
        test = lbnz.div(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 6)

    def test_lapacian(self):
        # f = x * y * y + z * z * z
        # expect that grad_f = y * y, 2 * x * y, 3 * z * z

        fld = lbnz.x * lbnz.y * lbnz.y + lbnz.z * lbnz.z * lbnz.z
        e0, e1, e2 = lbnz.y * lbnz.y, 2 * lbnz.x * lbnz.y, 3 * lbnz.z * lbnz.z
        g0, g1, g2 = lbnz.grad(fld)

        self.assertAlmostEqualWithMagExceptBoundary(e0, g0, 3)
        self.assertAlmostEqualWithMagExceptBoundary(e1, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(e2, g2, 4)

        # F = y * y, 2 * x * y, 3 * z * z = grad(f)
        # expect that div_F = 2 * x + 6 * z

        expected = 2 * lbnz.x + 6 * lbnz.z
        test = lbnz.div((g0, g1, g2))

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 2)

        # f = x * y * y + z * z * z
        # expect that lapacian_f = 2 * x + 6 * z

        fld = lbnz.x * lbnz.y * lbnz.y + lbnz.z * lbnz.z * lbnz.z
        expected = 2 * lbnz.x + 6 * lbnz.z
        test = lbnz.laplacian(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expected, test, 2)

    def test_div_sensibility(self):
        # F = y * y, 2 * x * y, 3 * z * z
        fld = lbnz.y * lbnz.y, 2 * lbnz.x * lbnz.y, 3 * lbnz.z * lbnz.z

        noise = th.randn_like(fld[0]) * th.std(fld[0]) * 1e-6, \
            th.randn_like(fld[1]) * th.std(fld[1]) * 1e-6, \
            th.randn_like(fld[2]) * th.std(fld[2]) * 1e-6
        fld_ = fld[0] + noise[0], fld[1] + noise[1], fld[2] + noise[2]
        self.assertAlmostEqualWithMagExceptBoundary(fld[0], fld_[0], 5)
        self.assertAlmostEqualWithMagExceptBoundary(fld[1], fld_[1], 5)
        self.assertAlmostEqualWithMagExceptBoundary(fld[2], fld_[2], 5)

        div1 = lbnz.div(fld)
        div2 = lbnz.div(fld_)
        self.assertAlmostEqualWithMagExceptBoundary(div1, div2, 4)

        diff_div1_0, diff_div1_1, diff_div1_2 = lbnz.diff(div1)
        diff_div2_0, diff_div2_1, diff_div2_2 = lbnz.diff(div2)
        self.assertAlmostEqualWithMagExceptBoundary(diff_div1_0, diff_div2_0, 2)
        self.assertAlmostEqualWithMagExceptBoundary(0.0, diff_div2_1, 2)
        self.assertAlmostEqualWithMagExceptBoundary(diff_div1_2, diff_div2_2, 2)

    def test_curl1(self):
        # F = (y, -x, 0)
        # expect that curl_F = (0, 0, -2)

        fld = lbnz.y, - lbnz.x, lbnz.zero
        expt_x, expt_y, expt_z = lbnz.zero, lbnz.zero, -2 * lbnz.one
        test_x, test_y, test_z = lbnz.curl(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expt_x, test_x, 6)
        self.assertAlmostEqualWithMagExceptBoundary(expt_y, test_y, 6)
        self.assertAlmostEqualWithMagExceptBoundary(expt_z, test_z, 6)

    def test_curl2(self):
        # F = (0, - x**2, 0)
        # expect that curl_F = (0, 0, -2 * x)

        fld = lbnz.zero, - lbnz.x * lbnz.x, lbnz.zero
        expt_x, expt_y, expt_z = lbnz.zero, lbnz.zero, -2 * lbnz.x
        test_x, test_y, test_z = lbnz.curl(fld)

        self.assertAlmostEqualWithMagExceptBoundary(expt_x, test_x, 6)
        self.assertAlmostEqualWithMagExceptBoundary(expt_y, test_y, 6)
        self.assertAlmostEqualWithMagExceptBoundary(expt_z, test_z, 6)

        self.assertAlmostEqualWithMagnitude(expt_x, test_x, 6)
        self.assertAlmostEqualWithMagnitude(expt_y, test_y, 6)
        self.assertAlmostEqualWithMagnitude(expt_z, test_z, 6)

    def test_zero_curl(self):
        g0, g1, g2 = lbnz.curl(lbnz.grad(lbnz.x))

        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g2, 6)

        g0, g1, g2 = lbnz.curl(lbnz.grad(lbnz.y))

        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g2, 6)

        g0, g1, g2 = lbnz.curl(lbnz.grad(lbnz.z))

        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g2, 6)

        g0, g1, g2 = lbnz.curl(lbnz.grad(lbnz.z * lbnz.z + lbnz.x * lbnz.y))

        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g0, 6)
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g1, 6)
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g2, 6)

    def test_zero_div(self):
        g = lbnz.div(lbnz.curl(lbnz.thetaphir.phi))
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g, 6)

        g = lbnz.div(lbnz.curl(lbnz.thetaphir.theta))
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g, 6)

        g = lbnz.div(lbnz.curl(lbnz.thetaphir.r))
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g, 6)

        phx, phy, phz = lbnz.thetaphir.phi
        thx, thy, thz = lbnz.thetaphir.theta
        rx, ry, rz = lbnz.thetaphir.r
        g = lbnz.div(lbnz.curl((rx * rx + phx * thx, ry * ry + phy * thy, rz * rz + phz * thz)))
        self.assertAlmostEqualWithMagExceptBoundary(lbnz.zero, g, 6)
