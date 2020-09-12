from torch import nn as nn


class ComplexAvgPool1d(nn.Module):
    def __init__(self):
        super(ComplexAvgPool1d, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input):
        return self.pool(input.real) + 1j * self.pool(input.imag)


class ComplexAvgPool2d(nn.Module):
    def __init__(self):
        super(ComplexAvgPool2d, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        return self.pool(input.real) + 1j * self.pool(input.imag)


class ComplexAvgPool3d(nn.Module):
    def __init__(self):
        super(ComplexAvgPool3d, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, input):
        return self.pool(input.real) + 1j * self.pool(input.imag)
