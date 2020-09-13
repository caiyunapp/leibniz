import torch.nn as nn
import torch.nn.functional as F


class ComplexUpsample1d(nn.Module):
    def __init__(self, size=None):
        super(ComplexUpsample1d, self).__init__()
        self.size = size
        self.upsample = nn.Upsample(size=self.size, mode='linear')

    def forward(self, input):
        return self.upsample(input.real) + 1j * self.upsample(input.imag)


class ComplexUpsample2d(nn.Module):
    def __init__(self, size=None):
        super(ComplexUpsample2d, self).__init__()
        self.size = size
        self.upsample = nn.Upsample(size=self.size, mode='bilinear')

    def forward(self, input):
        return self.upsample(input.real) + 1j * self.upsample(input.imag)


class ComplexUpsample3d(nn.Module):
    def __init__(self, size=None):
        super(ComplexUpsample3d, self).__init__()
        self.size = size
        self.upsample = nn.Upsample(size=self.size, mode='trilinear')

    def forward(self, input):
        return self.upsample(input.real) + 1j * self.upsample(input.imag)
