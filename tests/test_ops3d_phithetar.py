# -*- coding: utf-8 -*-

import unittest

import numpy as np
import torch as th

import leibniz as lbnz

from torch import Tensor
from leibniz.core3d.gridsys.regular3 import RegularGrid


class TestOpsPhiThetaR(unittest.TestCase):

    def setUp(self):
        lbnz.bind(RegularGrid(
            basis='lng,lat,alt',
            L=51, W=51, H=51,
            east=119.0, west=114.0,
            north=42.3, south=37.3,
            upper=16000.0, lower=0.0
        ), r0=6371000)
        lbnz.use('theta,phi,r')
        lbnz.use('thetaphir')

    def tearDown(self):
        lbnz.clear()

    def magnitude(self, var):
        # return magnitude of variable to test accuracy
        if th.is_tensor(var):
            var = var.cpu().numpy()
        if isinstance(var, np.ndarray):
            return 10 ** int(np.log10(np.abs(var.max()) + 1))
        else:
            return 10 ** int(np.log10(np.abs(var) + 1))

    def assertAlmostEqualWithMagnitude(self, excepted, test, magnitude=6):
        if isinstance(excepted, Tensor) and th.abs(excepted).max().cpu().numpy() == 0.0 or isinstance(excepted, float) and excepted == 0.0:
            return self.assertAlmostEqual(0.0, test.max().cpu().numpy(), magnitude)
        else:
            mag = self.magnitude(excepted)
            err = th.abs(excepted - test) / mag
            return self.assertAlmostEqual(0.0, err.max().cpu().numpy(), magnitude)

    def test_grad(self):
        dLth, dLph, dLr = lbnz.dL

        g0, g1, g2 = lbnz.grad(lbnz.lng)

        self.assertAlmostEqualWithMagnitude(5.0, g0 * dLth * (lbnz.L - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0.0, g2, 6)

        g0, g1, g2 = lbnz.grad(lbnz.lat)

        self.assertAlmostEqualWithMagnitude(0.0, g0, 6)
        self.assertAlmostEqualWithMagnitude(5.0, g1 * dLph * (lbnz.W - 1), 6)
        self.assertAlmostEqualWithMagnitude(0.0, g2, 6)

        g0, g1, g2 = lbnz.grad(lbnz.alt)

        self.assertAlmostEqualWithMagnitude(0.0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0.0, g1, 6)
        self.assertAlmostEqualWithMagnitude(16000.0, g2 * dLr * (lbnz.H - 1), 6)

    def test_zero_curl(self):
        g0, g1, g2 = lbnz.curl(lbnz.grad(lbnz.lat))

        self.assertAlmostEqualWithMagnitude(0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0, g2, 6)

        g0, g1, g2 = lbnz.curl(lbnz.grad(lbnz.lng))

        self.assertAlmostEqualWithMagnitude(0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0, g2, 6)

        g0, g1, g2 = lbnz.curl(lbnz.grad(lbnz.alt))

        self.assertAlmostEqualWithMagnitude(0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0, g2, 6)

        g0, g1, g2 = lbnz.curl(lbnz.grad(lbnz.alt * lbnz.alt + lbnz.lng * lbnz.lat))

        self.assertAlmostEqualWithMagnitude(0, g0, 6)
        self.assertAlmostEqualWithMagnitude(0, g1, 6)
        self.assertAlmostEqualWithMagnitude(0, g2, 6)

    def test_zero_div(self):
        g = lbnz.div(lbnz.curl(lbnz.thetaphir.phi))
        self.assertAlmostEqualWithMagnitude(0, g, 6)

        g = lbnz.div(lbnz.curl(lbnz.thetaphir.theta))
        self.assertAlmostEqualWithMagnitude(0, g, 6)

        g = lbnz.div(lbnz.curl(lbnz.thetaphir.r))
        self.assertAlmostEqualWithMagnitude(0, g, 6)

        phx, phy, phz = lbnz.thetaphir.phi
        thx, thy, thz = lbnz.thetaphir.theta
        rx, ry, rz = lbnz.thetaphir.r
        g = lbnz.div(lbnz.curl((rx * rx + phx * thx, ry * ry + phy * thy, rz * rz + phz * thz)))
        self.assertAlmostEqualWithMagnitude(0, g, 6)

    def test_div1(self):
        # F = (-theta, phi * theta, r);
        # expect that div_F = theta * tan(phi) / r + phi / (r * cos(phi)) + 3

        fld = - lbnz.theta, lbnz.phi * lbnz.theta, lbnz.r

        expected = lbnz.theta * th.tan(lbnz.phi) / lbnz.r + lbnz.phi / (lbnz.r * th.cos(lbnz.phi)) + 3

        test = lbnz.div(fld)

        self.assertAlmostEqualWithMagnitude(expected, test, 3)

    def test_div2(self):
        # phi0 = phi[:, :, W//2, L//2, H//2], theta0 = theta[:, :, W//2, L//2, H//2], r0 = r[:, :, W//2, L//2, H//2]
        # F = (-(theta - theta0), (phi - phi0) * (theta-theta0), (r-r0));
        # expect that div_F = (theta - theta0) * tan(phi) / r + (phi - phi0) / (r * cos(phi)) + 3 - 2 * r0 / r

        phi0 = lbnz.phi[:, :, lbnz.W // 2, lbnz.L // 2, lbnz.H // 2]
        theta0 = lbnz.theta[:, :, lbnz.W // 2, lbnz.L // 2, lbnz.H // 2]
        r0 = lbnz.r[:, :, lbnz.W // 2, lbnz.L // 2, lbnz.H // 2]

        Fx = -(lbnz.theta - theta0)
        Fy = (lbnz.phi - phi0) * (lbnz.theta - theta0)
        Fz = (lbnz.r - r0)

        test = lbnz.div((Fx, Fy, Fz))

        expected = (lbnz.theta - theta0) * th.tan(lbnz.phi) / lbnz.r +\
                   (lbnz.phi - phi0) / (lbnz.r * th.cos(lbnz.phi)) + 3 - 2 * r0 / lbnz.r

        self.assertAlmostEqualWithMagnitude(expected, test, 3)

    def test_curl1(self):
        # F = (-theta, phi * theta, r);
        # expect that curl_F = (-phi*theta/r, -theta/r, (theta-phi*theta*tan(phi)+1/cos(phi))/r)

        fld = - lbnz.theta, lbnz.phi * lbnz.theta, lbnz.r

        expected_ph = -lbnz.phi * lbnz.theta / lbnz.r
        expected_th = -lbnz.theta / lbnz.r
        expected_r = (lbnz.theta - lbnz.phi * lbnz.theta * th.tan(lbnz.phi) + 1 / th.cos(lbnz.phi)) / lbnz.r

        test_ph, test_th, test_r = lbnz.curl(fld)

        self.assertAlmostEqualWithMagnitude(expected_ph, test_ph, 6)
        self.assertAlmostEqualWithMagnitude(expected_th, test_th, 6)
        self.assertAlmostEqualWithMagnitude(expected_r, test_r, 6)

    def test_curl2(self):
        # F = (0, cos(phi), 0)
        # expect that curl = (-cos(phi)/r, 0, -2*sin(phi)/r)
        fld = 0, th.cos(lbnz.phi), 0

        expected_ph = -th.cos(lbnz.phi) / lbnz.r
        expected_th = lbnz.zero
        expected_r = -2 * th.sin(lbnz.phi) / lbnz.r

        test_ph, test_th, test_r = lbnz.curl(fld)

        self.assertAlmostEqualWithMagnitude(expected_ph, test_ph, 6)
        self.assertAlmostEqualWithMagnitude(expected_th, test_th, 6)
        self.assertAlmostEqualWithMagnitude(expected_r, test_r, 6)
