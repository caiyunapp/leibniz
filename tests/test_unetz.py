# -*- coding: utf-8 -*-

import unittest
import torch as th

from leibniz.nn.net import resunetz


class TestResUNetZ(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1D(self):
        net = resunetz(2, 1, spatial=(32,))
        net(th.rand(1, 2, 32))
        net = resunetz(1, 2, spatial=(32,), normalizor='relu6')
        net(th.rand(1, 1, 32))
        net = resunetz(1, 2, spatial=(32,), normalizor='tanh')
        net(th.rand(1, 1, 32))
        net = resunetz(1, 1, spatial=(32,), normalizor='softmax')
        net(th.rand(1, 1, 32))
        net = resunetz(1, 1, spatial=(32,), normalizor='sigmoid')
        net(th.rand(1, 1, 32))
        net(th.rand(2, 1, 32))
        net = resunetz(2, 1, spatial=(32,), ratio=0)
        net(th.rand(1, 2, 32))
        net(th.rand(2, 2, 32))

    def test2D(self):
        resunetz(1, 1, spatial=(16, 16))
        resunetz(1, 1, spatial=(16, 32))
        resunetz(1, 1, spatial=(32, 16))
        net = resunetz(1, 2, spatial=(32, 16))
        net(th.rand(1, 1, 32, 16))
        net = resunetz(2, 1, spatial=(32, 16), normalizor='relu6')
        net(th.rand(1, 2, 32, 16))
        net = resunetz(1, 1, spatial=(32, 16), normalizor='tanh')
        net(th.rand(1, 1, 32, 16))
        net = resunetz(1, 1, spatial=(32, 16), normalizor='softmax')
        net(th.rand(1, 1, 32, 16))
        net = resunetz(1, 1, spatial=(32, 16), normalizor='sigmoid')
        net(th.rand(1, 1, 32, 16))
        net(th.rand(2, 1, 32, 16))
        net = resunetz(1, 1, spatial=(32, 64))
        net(th.rand(1, 1, 32, 64))
        net = resunetz(1, 1, spatial=(16, 64))
        net(th.rand(1, 1, 32, 64))

    def test3D(self):
        resunetz(1, 1, spatial=(16, 16, 16))
        resunetz(1, 1, spatial=(32, 16, 16))
        resunetz(1, 1, spatial=(16, 32, 16))
        resunetz(1, 1, spatial=(16, 16, 32))
        resunetz(1, 1, spatial=(11, 16, 32))
        net = resunetz(1, 1, spatial=(16, 16, 16))
        net(th.rand(2, 1, 16, 16, 16))
        net = resunetz(1, 1, spatial=(16, 16, 16), normalizor='relu6')
        net(th.rand(2, 1, 16, 16, 16))
        net = resunetz(1, 1, spatial=(16, 16, 16), normalizor='tanh')
        net(th.rand(2, 1, 16, 16, 16))
        net = resunetz(1, 1, spatial=(16, 16, 16), normalizor='softmax')
        net(th.rand(2, 1, 16, 16, 16))
        net = resunetz(1, 1, spatial=(16, 16, 16), normalizor='sigmoid')
        net(th.rand(2, 1, 16, 16, 16))
        net(th.rand(2, 1, 16, 16, 16))
