# -*- coding: utf-8 -*-

import unittest
import torch as th

from leibniz.nn.net import hyptub, resunet
from leibniz.nn.net.hyptube import StepwiseHypTube, LeveledHypTube


class TestHyptub(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test2D(self):
        hyptub(1, 1, 1, spatial=(16, 16))
        hyptub(1, 1, 1, spatial=(16, 32))
        hyptub(1, 1, 1, spatial=(32, 16))
        net = hyptub(1, 1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]])
        net(th.rand(1, 1, 32, 16))
        net = hyptub(1, 1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='instance')
        net(th.rand(1, 1, 32, 16))
        net = hyptub(1, 1, 1, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], normalizor='layer')
        net(th.rand(1, 1, 32, 16))
        net(th.rand(2, 1, 32, 16))
        net = hyptub(1, 1, 1, spatial=(32, 64))
        net(th.rand(1, 1, 32, 64))
        net = hyptub(1, 1, 1, spatial=(32, 64))
        net(th.rand(1, 1, 32, 64))

    def testStepwise2D(self):
        StepwiseHypTube(1, 1, 1, 2, spatial=(16, 16), encoder=resunet, decoder=resunet)
        StepwiseHypTube(1, 1, 1, 2, spatial=(16, 32), encoder=resunet, decoder=resunet)
        StepwiseHypTube(1, 1, 1, 2, spatial=(32, 16), encoder=resunet, decoder=resunet)
        net = StepwiseHypTube(1, 1, 1, 2, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], encoder=resunet, decoder=resunet)
        net(th.rand(1, 1, 32, 16))
        net = StepwiseHypTube(1, 1, 1, 2, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], encoder=resunet, decoder=resunet, normalizor='instance')
        net(th.rand(1, 1, 32, 16))
        net = StepwiseHypTube(1, 1, 1, 2, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], encoder=resunet, decoder=resunet, normalizor='layer')
        net(th.rand(1, 1, 32, 16))
        net(th.rand(2, 1, 32, 16))
        net = StepwiseHypTube(1, 1, 1, 2, spatial=(32, 64), encoder=resunet, decoder=resunet)
        net(th.rand(1, 1, 32, 64))
        net = StepwiseHypTube(1, 1, 1, 2, spatial=(32, 64), encoder=resunet, decoder=resunet)
        net(th.rand(1, 1, 32, 64))

    def testLayered2D(self):
        LeveledHypTube(1, 1, 1, 2, spatial=(16, 16), encoder=resunet, decoder=resunet)
        LeveledHypTube(1, 1, 1, 2, spatial=(16, 32), encoder=resunet, decoder=resunet)
        LeveledHypTube(1, 1, 1, 2, spatial=(32, 16), encoder=resunet, decoder=resunet)
        net = LeveledHypTube(1, 1, 1, 2, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], encoder=resunet, decoder=resunet)
        net(th.rand(1, 1, 32, 16))
        net = LeveledHypTube(1, 1, 1, 2, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], encoder=resunet, decoder=resunet, normalizor='instance')
        net(th.rand(1, 1, 32, 16))
        net = LeveledHypTube(1, 1, 1, 2, spatial=(32, 16), scales=[[0, -1], [0, -1], [0, -1], [0, -1]], encoder=resunet, decoder=resunet, normalizor='layer')
        net(th.rand(1, 1, 32, 16))
        net(th.rand(2, 1, 32, 16))
        net = LeveledHypTube(1, 1, 1, 2, spatial=(32, 64), encoder=resunet, decoder=resunet)
        net(th.rand(1, 1, 32, 64))
        net = LeveledHypTube(1, 1, 1, 2, spatial=(32, 64), encoder=resunet, decoder=resunet)
        net(th.rand(1, 1, 32, 64))
