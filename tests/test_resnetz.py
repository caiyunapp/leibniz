# -*- coding: utf-8 -*-

import unittest
import torch as th

from leibniz.nn.net import resnetz
from leibniz.nn.layer.hyperbolic import HyperBasic
from leibniz.nn.layer.hyperbolic import HyperBottleneck


class Testresnetz(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1D(self):
        net = resnetz(1, 1, spatial=(32,))
        net(th.rand(1, 1, 32))
        net = resnetz(1, 1, spatial=(32,), normalizor='relu6')
        net(th.rand(1, 1, 32))
        net = resnetz(1, 1, spatial=(32,), normalizor='tanh')
        net(th.rand(1, 1, 32))
        net = resnetz(1, 1, spatial=(32,), normalizor='softmax')
        net(th.rand(1, 1, 32))
        net = resnetz(1, 1, spatial=(32,), normalizor='sigmoid')
        net(th.rand(1, 1, 32))
        net(th.rand(2, 1, 32))
        net = resnetz(2, 1, spatial=(32,), ratio=0)
        net(th.rand(1, 2, 32))
        net(th.rand(2, 2, 32))

    def test2D(self):
        resnetz(1, 1, spatial=(16, 16))
        resnetz(1, 1, spatial=(16, 32))
        resnetz(1, 1, spatial=(32, 16))
        net = resnetz(1, 1, spatial=(32, 16))
        net(th.rand(1, 1, 32, 16))
        net = resnetz(1, 1, spatial=(32, 16), normalizor='relu6')
        net(th.rand(1, 1, 32, 16))
        net = resnetz(1, 1, spatial=(32, 16), normalizor='tanh')
        net(th.rand(1, 1, 32, 16))
        net = resnetz(1, 1, spatial=(32, 16), normalizor='softmax')
        net(th.rand(1, 1, 32, 16))
        net = resnetz(1, 1, spatial=(32, 16), normalizor='sigmoid')
        net(th.rand(1, 1, 32, 16))
        net(th.rand(2, 1, 32, 16))
        net = resnetz(1, 1, spatial=(32, 64))
        net(th.rand(1, 1, 32, 64))
        net = resnetz(1, 1, spatial=(16, 64))
        net(th.rand(1, 1, 32, 64))

    def test3D(self):
        resnetz(1, 1, spatial=(16, 16, 16))
        resnetz(1, 1, spatial=(32, 16, 16))
        resnetz(1, 1, spatial=(16, 32, 16))
        resnetz(1, 1, spatial=(16, 16, 32))
        resnetz(1, 1, spatial=(11, 16, 32))
        net = resnetz(1, 1, spatial=(4, 16, 32))
        net(th.rand(1, 1, 4, 16, 32))
        net = resnetz(1, 1, spatial=(4, 16, 32), normalizor='relu6')
        net(th.rand(1, 1, 4, 16, 32))
        net = resnetz(1, 1, spatial=(4, 16, 32), normalizor='tanh')
        net(th.rand(1, 1, 4, 16, 32))
        net = resnetz(1, 1, spatial=(4, 16, 32), normalizor='sofmax')
        net(th.rand(1, 1, 4, 16, 32))
        net = resnetz(1, 1, spatial=(4, 16, 32), normalizor='sigmoid')
        net(th.rand(1, 1, 4, 16, 32))
        net(th.rand(2, 1, 4, 16, 32))
