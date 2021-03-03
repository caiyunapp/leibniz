# -*- coding: utf-8 -*-

import unittest
import torch as th

from leibniz.nn.net import resnet
from leibniz.nn.layer.hyperbolic import HyperBasic
from leibniz.nn.layer.hyperbolic import HyperBottleneck


class TestResNet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1D(self):
        net = resnet(1, 1, spatial=(32,))
        net(th.rand(1, 1, 32))
        net = resnet(1, 1, spatial=(32,), normalizor='instance')
        net(th.rand(1, 1, 32))
        net = resnet(1, 1, spatial=(32,), normalizor='layer')
        net(th.rand(1, 1, 32))
        net(th.rand(2, 1, 32))
        net = resnet(2, 1, spatial=(32,), ratio=0)
        net(th.rand(1, 2, 32))
        net(th.rand(2, 2, 32))

    def test2D(self):
        resnet(1, 1, spatial=(16, 16))
        resnet(1, 1, spatial=(16, 32))
        resnet(1, 1, spatial=(32, 16))
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]])
        net(th.rand(1, 1, 32, 16))
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance')
        net(th.rand(1, 1, 32, 16))
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer')
        net(th.rand(1, 1, 32, 16))
        net(th.rand(2, 1, 32, 16))
        net = resnet(1, 1, spatial=(32, 64))
        net(th.rand(1, 1, 32, 64))
        net = resnet(1, 1, spatial=(16, 64))
        net(th.rand(1, 1, 32, 64))

    def test3D(self):
        resnet(1, 1, spatial=(16, 16, 16))
        resnet(1, 1, spatial=(32, 16, 16))
        resnet(1, 1, spatial=(16, 32, 16))
        resnet(1, 1, spatial=(16, 16, 32))
        resnet(1, 1, spatial=(11, 16, 32))
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]])
        net(th.rand(1, 1, 4, 16, 32))
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance')
        net(th.rand(1, 1, 4, 16, 32))
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer')
        net(th.rand(1, 1, 4, 16, 32))
        net(th.rand(2, 1, 4, 16, 32))

    def testHyp1D(self):
        net = resnet(1, 1, spatial=(32,), block=HyperBasic)
        net(th.rand(1, 1, 16))
        net = resnet(1, 1, spatial=(32,), normalizor='instance', block=HyperBasic)
        net(th.rand(1, 1, 16))
        net = resnet(1, 1, spatial=(32,), normalizor='layer', block=HyperBasic)
        net(th.rand(1, 1, 16))
        net = resnet(1, 1, spatial=(32,), block=HyperBottleneck)
        net(th.rand(1, 1, 16))
        net = resnet(1, 1, spatial=(32,), normalizor='instance', block=HyperBottleneck)
        net(th.rand(1, 1, 16))
        net = resnet(1, 1, spatial=(32,), normalizor='layer', block=HyperBottleneck)
        net(th.rand(1, 1, 16))
        net(th.rand(2, 1, 16))

    def testHyp2D(self):
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resnet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net(th.rand(2, 1, 32, 16))

    def testHyp3D(self):
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=HyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=HyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=HyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=HyperBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=HyperBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net = resnet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=HyperBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net(th.rand(2, 1, 4, 16, 32))
