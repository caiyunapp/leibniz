# -*- coding: utf-8 -*-

import unittest
import torch as th

from leibniz.nn.net import resunet
from leibniz.nn.layer.hyperbolic import HyperBasic
from leibniz.nn.layer.hyperbolic import HyperBottleneck
from leibniz.nn.layer.senet import SEBasicBlock, SEBottleneck
from leibniz.nn.layer.hyperbolic2 import HyperBasic as HyperBasic2, HyperBottleneck as HyperBottleneck2


class TestUnet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test2D(self):
        resunet(1, 1, spatial=(16, 16))
        resunet(1, 1, spatial=(16, 32))
        resunet(1, 1, spatial=(32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]])
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance')
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer')
        net(th.rand(1, 1, 32, 16))
