# -*- coding: utf-8 -*-

import unittest
import torch as th

from leibniz.unet import resunet
from leibniz.unet.hyperbolic import HyperBasic
from leibniz.unet.hyperbolic import HyperBottleneck
from leibniz.unet.senet import SEBasicBlock
from leibniz.unet.senet import SEBottleneck


class TestUnet(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test1D(self):
        net = resunet(1, 1, spatial=(32,))
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance')
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer')
        net(th.rand(1, 1, 16))

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

    def test3D(self):
        resunet(1, 1, spatial=(16, 16, 16))
        resunet(1, 1, spatial=(32, 16, 16))
        resunet(1, 1, spatial=(16, 32, 16))
        resunet(1, 1, spatial=(16, 16, 32))
        resunet(1, 1, spatial=(11, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]])
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance')
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer')
        net(th.rand(1, 1, 4, 16, 32))

    def testHyp1D(self):
        net = resunet(1, 1, spatial=(32,), block=HyperBasic)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance', block=HyperBasic)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer', block=HyperBasic)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), block=HyperBottleneck)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance', block=HyperBottleneck)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer', block=HyperBottleneck)
        net(th.rand(1, 1, 16))

    def testHyp2D(self):
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))

    def testHyp3D(self):
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=HyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=HyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=HyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=HyperBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=HyperBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=HyperBottleneck)
        net(th.rand(1, 1, 4, 16, 32))

    def testSE1D(self):
        net = resunet(1, 1, spatial=(32,), block=SEBasicBlock)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance', block=SEBasicBlock)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer', block=SEBasicBlock)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), block=SEBottleneck)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance', block=SEBottleneck)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer', block=SEBottleneck)
        net(th.rand(1, 1, 16))

    def testSE2D(self):
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=SEBasicBlock)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=SEBasicBlock)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=SEBasicBlock)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=SEBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=SEBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=SEBottleneck)
        net(th.rand(1, 1, 32, 16))

    def testSE3D(self):
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=SEBasicBlock)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=SEBasicBlock)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=SEBasicBlock)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=SEBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=SEBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=SEBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
