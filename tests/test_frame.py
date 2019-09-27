# -*- coding: utf-8 -*-

import unittest

import numpy as np
import leibniz as lbnz

from leibniz.core.vector.v3 import box
from leibniz.core.gridsys.regular3 import RegularGrid


class TestFrame(unittest.TestCase):

    def setUp(self):
        lbnz.bind(RegularGrid(
            basis='lng,lat,alt',
            W=51, L=51, H=51,
            east=119.0, west=114.0,
            north=42.3, south=37.3,
            upper=16000.0, lower=0.0
        ))
        lbnz.use('thetaphir')
        lbnz.use('xyz')
        lbnz.use('x,y,z')

    def tearDown(self):
        lbnz.clear()

    def test_basis_ortho(self):
        phx, phy, phz = lbnz.thetaphir.phi
        phx = phx.cpu().numpy()
        phy = phy.cpu().numpy()
        phz = phz.cpu().numpy()
        self.assertAlmostEqual(1, np.max(phx * phx + phy * phy + phz * phz))
        self.assertAlmostEqual(1, np.min(phx * phx + phy * phy + phz * phz))

        thx, thy, thz = lbnz.thetaphir.theta
        thx = thx.cpu().numpy()
        thy = thy.cpu().numpy()
        thz = thz.cpu().numpy()
        val = thx * thx + thy * thy + thz * thz
        self.assertAlmostEqual(1, np.max(val))
        self.assertAlmostEqual(1, np.min(val))

        rx, ry, rz = lbnz.thetaphir.r
        rx = rx.cpu().numpy()
        ry = ry.cpu().numpy()
        rz = rz.cpu().numpy()
        self.assertAlmostEqual(1, np.max(rx * rx + ry * ry + rz * rz))
        self.assertAlmostEqual(1, np.min(rx * rx + ry * ry + rz * rz))

        self.assertAlmostEqual(0, np.max(phx * thx + phy * thy + phz * thz))
        self.assertAlmostEqual(0, np.min(phx * rx + phy * ry + phz * rz))
        self.assertAlmostEqual(0, np.min(thx * rx + thy * ry + thz * rz))

    def test_basis_righthand(self):
        d = box(lbnz.thetaphir.theta, lbnz.thetaphir.phi, lbnz.thetaphir.r).cpu().numpy()
        self.assertAlmostEqual(1, np.mean(d))
        self.assertAlmostEqual(1, np.min(d))
        self.assertAlmostEqual(1, np.max(d))

        d = box(lbnz.xyz.x, lbnz.xyz.y, lbnz.xyz.z).cpu().numpy()
        self.assertAlmostEqual(1, np.mean(d))
        self.assertAlmostEqual(1, np.min(d))
        self.assertAlmostEqual(1, np.max(d))

    def test_basis_diff(self):
        from leibniz.core.basis.theta_phi_r import transform

        x, y, z = transform(lbnz.theta + 0.0001, lbnz.phi, lbnz.r)
        thx, thy, thz = lbnz.normalize((x - lbnz.x, y - lbnz.y, z - lbnz.z))
        a = lbnz.dot((thx, thy, thz), lbnz.thetaphir.theta)
        self.assertAlmostEqual(1, np.mean(a.cpu().numpy()))

        x, y, z = transform(lbnz.theta, lbnz.phi + 0.0001, lbnz.r)
        phx, phy, phz = lbnz.normalize((x - lbnz.x, y - lbnz.y, z - lbnz.z))
        a = lbnz.dot((phx, phy, phz), lbnz.thetaphir.phi)
        self.assertAlmostEqual(1, np.mean(a.cpu().numpy()))

        x, y, z = transform(lbnz.theta, lbnz.phi, lbnz.r + 0.0001)
        rx, ry, rz = lbnz.normalize((x - lbnz.x, y - lbnz.y, z - lbnz.z))
        a = lbnz.dot((rx, ry, rz), lbnz.thetaphir.r)
        self.assertAlmostEqual(1, np.mean(a.cpu().numpy()))



