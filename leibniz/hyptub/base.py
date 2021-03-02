import logging

import torch as th
import torch.nn as nn

from leibniz.unet import resunet
from leibniz.unet.hyperbolic import HyperBottleneck

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class HypTube(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, block=HyperBottleneck, relu=nn.ReLU(inplace=True), attn=None, normalizor='batch',
                 layers=4, ratio=-2, spatial=(256, 256), vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1], scales=[-2, -2, -2, -2], factors=[1, 1, 1, 1],
                 final_normalized=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.enc = resunet(in_channels, 6 * hidden_channels, normalizor=normalizor, spatial=spatial, layers=layers, ratio=ratio,
                vblks=vblks, hblks=hblks, scales=scales, factors=factors, block=block, relu=relu, attn=attn, final_normalized=False)

        self.dec = resunet(hidden_channels, out_channels, normalizor=normalizor, spatial=spatial, layers=layers, ratio=ratio,
                vblks=vblks, hblks=hblks, scales=scales, factors=factors, block=block, relu=relu, attn=attn, final_normalized=final_normalized)

    def forward(self, input):
        b, c, w, h = input.size()
        hc = self.hidden_channels

        flow = self.enc(input)
        flow, uparam, vparam = flow[:, 2*hc:], flow[:, 0:hc], flow[:, hc:2*hc]
        flow = flow.view(-1, hc, 2, 2, w, h)

        output = th.zeros(b, hc, w, h, device=input.device)
        for ix in range(2):
            aparam = flow[:, :, ix, 0]
            mparam = flow[:, :, ix, 1]
            output = (output + aparam * uparam) * (1 + mparam * vparam)

        return self.dec(output)


class StepwiseHypTube(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, steps, block=HyperBottleneck, relu=nn.ReLU(inplace=True), attn=None, normalizor='batch',
                 layers=4, ratio=-2, spatial=(256, 256), vblks=[1, 1, 1, 1], hblks=[1, 1, 1, 1], scales=[-2, -2, -2, -2], factors=[1, 1, 1, 1],
                 final_normalized=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.steps = steps

        self.enc = resunet(in_channels, (4 * steps + 2) * hidden_channels, normalizor=normalizor, spatial=spatial, layers=layers, ratio=ratio,
                vblks=vblks, hblks=hblks, scales=scales, factors=factors, block=block, relu=relu, attn=attn, final_normalized=False)

        self.dec = resunet(hidden_channels, out_channels, normalizor=normalizor, spatial=spatial, layers=layers, ratio=ratio,
                vblks=vblks, hblks=hblks, scales=scales, factors=factors, block=block, relu=relu, attn=attn, final_normalized=final_normalized)

    def forward(self, input):
        b, c, w, h = input.size()
        hc = self.hidden_channels

        flow = self.enc(input)
        flow, uparam, vparam = flow[:, 2*hc:], flow[:, 0:hc], flow[:, hc:2*hc]
        flow = flow.view(-1, hc, self.steps, 2, 2, w, h)

        result = []
        for jx in range(self.steps):
            output = th.zeros(b, hc, w, h, device=input.device)
            for ix in range(2):
                aparam = flow[:, :, jx, ix, 0]
                mparam = flow[:, :, jx, ix, 1]
                output = (output + aparam * uparam) * (1 + mparam * vparam)
            result.append(self.dec(output))

        return th.cat(result, dim=1).view(-1, self.steps, self.out_channels, w, h)
