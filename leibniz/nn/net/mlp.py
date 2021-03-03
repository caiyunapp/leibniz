import torch.nn as nn


class MLP1d(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(MLP1d, self).__init__()
        channels_hidden = channels_in + channels_out
        self.layers = nn.Sequential(
            nn.Linear(channels_in, channels_hidden),
            nn.ReLU(),
            nn.Linear(channels_hidden, channels_out)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class MLP2d(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(MLP2d, self).__init__()
        channels_hidden = channels_in + channels_out
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(channels_hidden, channels_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
