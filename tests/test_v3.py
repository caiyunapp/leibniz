# -*- coding: utf-8 -*-

import unittest

import numpy as np

import leibniz as lbnz

from leibniz.core3d.vec3 import norm, cross, sub


class TestV3(unittest.TestCase):

    def setUp(self):
        self.zero = lbnz.cast(np.zeros([1, 1, 2, 2, 2]))
        self.one = lbnz.cast(np.ones([1, 1, 2, 2, 2]))

    def tearDown(self):
        pass

    def test_cross(self):
        i = self.one, self.zero, self.zero
        j = self.zero, self.one, self.zero
        k = self.zero, self.zero, self.one

        z = norm(sub(cross(i, j), k)).cpu().numpy()
        self.assertAlmostEqual(0, np.max(z))
        self.assertAlmostEqual(0, np.min(z))

        z = norm(sub(cross(j, k), i)).cpu().numpy()
        self.assertAlmostEqual(0, np.max(z))
        self.assertAlmostEqual(0, np.min(z))

        z = norm(sub(cross(k, i), j)).cpu().numpy()
        self.assertAlmostEqual(0, np.max(z))
        self.assertAlmostEqual(0, np.min(z))
