import logging

import torch as th
import torch.nn as nn

from leibniz.nn.net.simple import Linear, SimpleCNN2d

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class HypTube(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder=Linear, decoder=Linear, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.enc = encoder(in_channels, 6 * hidden_channels, **kwargs)
        self.dec = decoder(hidden_channels, out_channels, **kwargs)

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
    def __init__(self, in_channels, hidden_channels, out_channels, steps, encoder, decoder, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.steps = steps

        self.enc = encoder(in_channels, (4 * steps + 2) * hidden_channels, **kwargs)
        self.dec = decoder(hidden_channels, out_channels, **kwargs)

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


class LeveledHypTube(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, levels, encoder, decoder, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.levels = levels

        self.enc = encoder(in_channels, 6 * hidden_channels, **kwargs)
        self.dec = decoder(hidden_channels, out_channels, **kwargs)
        self.leveled = nn.ModuleList()
        for ix in range(levels):
            self.leveled.append(SimpleCNN2d(6 * hidden_channels, 6 * hidden_channels))

    def forward(self, input):
        b, c, w, h = input.size()
        hc = self.hidden_channels

        flow = self.enc(input)
        output = th.zeros(b, hc, w, h, device=input.device)
        for jx in range(self.levels):
            flow = self.leveled[jx](flow)
            params, uparam, vparam = flow[:, 2 * hc:], flow[:, 0:hc], flow[:, hc:2 * hc]
            params = params.view(-1, hc, 2, 2, w, h)
            for ix in range(2):
                aparam = params[:, :, ix, 0]
                mparam = params[:, :, ix, 1]
                output = (output + aparam * uparam) * (1 + mparam * vparam)

        return self.dec(output)
