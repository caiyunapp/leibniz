import torch as th
import torch.nn as nn

'''
Original by ZjjConan
from https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py
modified by Mingli Yuan (Mountain)
'''


class SimAM(nn.Module):
    def __init__(self, dim, reduction=16, kernel_size=7, conv=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        sz = x.size()

        if len(sz) == 3:
            b, c, w = x.size()
            n = w - 1
            dims = [2]
        if len(sz) == 4:
            b, c, h, w = x.size()
            n = w * h - 1
            dims = [2, 3]
        if len(sz) == 5:
            b, c, l, h, w = x.size()
            n = w * h * l - 1
            dims = [2, 3, 4]

        x_minus_mu_square = (x - x.mean(dim=dims, keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=dims, keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
