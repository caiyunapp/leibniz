# -*- coding: utf-8 -*-

import unittest
import torch as th

from leibniz.unet import resunet
from leibniz.unet.hyperbolic import HyperBasic
from leibniz.unet.hyperbolic import HyperBottleneck
from leibniz.unet.complex_hyperbolic import CmplxHyperBasic
from leibniz.unet.complex_hyperbolic import CmplxHyperBottleneck
from leibniz.unet.senet import SEBasicBlock, SEBottleneck, SELayer
from leibniz.unet.warp import WarpBasicBlock, WarpBottleneck, WarpLayer
from leibniz.unet.hyperbolic2 import HyperBasic as HyperBasic2, HyperBottleneck as HyperBottleneck2


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

    def testCmplxHyp1D(self):
        net = resunet(1, 1, spatial=(32,), block=CmplxHyperBasic)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance', block=CmplxHyperBasic)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer', block=CmplxHyperBasic)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), block=CmplxHyperBottleneck)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance', block=CmplxHyperBottleneck)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer', block=CmplxHyperBottleneck)
        net(th.rand(1, 1, 16))

    def testCmplxHyp2D(self):
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=CmplxHyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=CmplxHyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=CmplxHyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=CmplxHyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=CmplxHyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=CmplxHyperBottleneck)
        net(th.rand(1, 1, 32, 16))

    def testCmplxHyp3D(self):
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=CmplxHyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=CmplxHyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=CmplxHyperBasic)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=CmplxHyperBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=CmplxHyperBottleneck)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=CmplxHyperBottleneck)
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

    def testHyp2DGroupNorm(self):
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='group', block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='group', block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='group', block=HyperBasic)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='group', block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='group', block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='group', block=HyperBottleneck)
        net(th.rand(1, 1, 32, 16))

    def testWP2D(self):
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=WarpBasicBlock, attn=SELayer)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=WarpBasicBlock, attn=WarpLayer)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=WarpBasicBlock, attn=SELayer)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=WarpBottleneck, attn=SELayer)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=WarpBottleneck, attn=WarpLayer)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=WarpBottleneck, attn=SELayer)
        net(th.rand(1, 1, 32, 16))

    def testHyp1D2(self):
        net = resunet(1, 1, spatial=(32,), block=HyperBasic2)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance', block=HyperBasic2)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer', block=HyperBasic2)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), block=HyperBottleneck2)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='instance', block=HyperBottleneck2)
        net(th.rand(1, 1, 16))
        net = resunet(1, 1, spatial=(32,), normalizor='layer', block=HyperBottleneck2)
        net(th.rand(1, 1, 16))

    def testHyp2D2(self):
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=HyperBasic2)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=HyperBasic2)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=HyperBasic2)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], block=HyperBottleneck2)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance', block=HyperBottleneck2)
        net(th.rand(1, 1, 32, 16))
        net = resunet(1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer', block=HyperBottleneck2)
        net(th.rand(1, 1, 32, 16))

    def testHyp3D2(self):
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=HyperBasic2)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=HyperBasic2)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=HyperBasic2)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], block=HyperBottleneck2)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='instance', block=HyperBottleneck2)
        net(th.rand(1, 1, 4, 16, 32))
        net = resunet(1, 1, spatial=(4, 16, 32), scales=[[0, -1, -1], [-1, -1, -1], [0, -1, -1], [-1, -1, -1]], normalizor='layer', block=HyperBottleneck2)
        net(th.rand(1, 1, 4, 16, 32))
