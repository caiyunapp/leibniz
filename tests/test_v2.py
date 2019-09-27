# -*- coding: utf-8 -*-

import unittest

import numpy as np

import leibniz as lbnz

from leibniz.core.vector.v2 import norm, cross, sub


class TestV2(unittest.TestCase):

    def setUp(self):
        self.zero = lbnz.cast(np.zeros([1, 1, 2, 2]))
        self.one = lbnz.cast(np.ones([1, 1, 2, 2]))

    def tearDown(self):
        pass
