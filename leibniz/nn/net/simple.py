import torch.nn as nn
from torch import nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Linear, self).__init__()
        params = dict(kernel_size=3, padding=1)
        params.update(kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, **params)

    def forward(self, x):
        return self.conv(x)


class SimpleCNN2d(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SimpleCNN2d, self).__init__()
        channels_hidden = channels_in + channels_out
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(channels_hidden, channels_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
