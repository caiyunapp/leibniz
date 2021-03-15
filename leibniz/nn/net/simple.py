import torch.nn as nn


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