# -*- coding: utf-8 -*-

import unittest
import torch as th

from leibniz.unet import resunet


class TestUnet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1D(self):
        net = resunet(1, 1, spatial=(16,))
        net(th.rand(1, 1, 16))

    def test2D(self):
        resunet(1, 1, spatial=(16, 16))
        resunet(1, 1, spatial=(16, 32))
        resunet(1, 1, spatial=(32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]])
        net(th.rand(1, 1, 32, 16))

    def test3D(self):
        resunet(1, 1, spatial=(16, 16, 16))
        resunet(1, 1, spatial=(32, 16, 16))
        resunet(1, 1, spatial=(16, 32, 16))
        resunet(1, 1, spatial=(16, 16, 32))
        resunet(1, 1, spatial=(11, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]])
        net(th.rand(1, 1, 4, 16, 32))
