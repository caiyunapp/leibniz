import torch.nn as nn
import torch.nn.functional as F


class ComplexDropout1d(nn.Module):
    def __init__(self, p=0.5, inplace=True):
        super(ComplexDropout1d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input.real, p=self.p, training=self.training, inplace=self.inplace) +\
               1j * F.dropout(input.imag, p=self.p, training=self.training, inplace=self.inplace)


class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.5, inplace=True):
        super(ComplexDropout2d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout2d(input.real, p=self.p, training=self.training, inplace=self.inplace) +\
               1j * F.dropout2d(input.imag, p=self.p, training=self.training, inplace=self.inplace)


class ComplexDropout3d(nn.Module):
    def __init__(self, p=0.5, inplace=True):
        super(ComplexDropout3d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout3d(input.real, p=self.p, training=self.training, inplace=self.inplace) +\
               1j * F.dropout3d(input.imag, p=self.p, training=self.training, inplace=self.inplace)
