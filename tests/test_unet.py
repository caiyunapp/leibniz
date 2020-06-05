# -*- coding: utf-8 -*-

import unittest

from leibniz.unet import resunet


class TestUnet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1D(self):
        resunet(1, 1, spatial=(16,))

    def test2D(self):
        resunet(1, 1, spatial=(16, 16))
        resunet(1, 1, spatial=(16, 32))
        resunet(1, 1, spatial=(32, 16))
        resunet(1, 1, spatial=(32, 16), scales=[[0, 1], [0, 1], [0, 1], [0, 1]])

    def test3D(self):
        resunet(1, 1, spatial=(16, 16, 16))
        resunet(1, 1, spatial=(32, 16, 16))
        resunet(1, 1, spatial=(16, 32, 16))
        resunet(1, 1, spatial=(16, 16, 32))
        resunet(1, 1, spatial=(11, 16, 32))
        resunet(1, 1, spatial=(11, 16, 32), scales=[[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]])
