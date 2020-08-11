import torch.nn as nn

from torchpwl.pwl import BasePointPWL, MonoPointPWL, MonoSlopedPWL


class InversedMonoPointPWL(BasePointPWL):
    def __init__(self, peer):
        super(InversedMonoPointPWL, self).__init__(peer.num_channels, peer.num_breakpoints, num_x_points=peer.num_breakpoints+1)
        self.peer = peer

    def _reset_params(self):
        self.peer._reset_params()

    def get_y_positions(self):
        return self.peer.get_sorted_x_positions()

    def get_x_positions(self):
        xs = self.peer.get_sorted_x_positions()
        data = xs.transpose(1, 0)
        return self.peer(data).transpose(1, 0)


class PWLNormalizorApp(nn.Module):
    def __init__(self, num_channels, num_breakpoints, monotonicity=1):
        super(PWLNormalizorApp, self).__init__()
        self.pwl = MonoPointPWL(num_channels, num_breakpoints, monotonicity=monotonicity)

    def forward(self, x):
        return self.pwl(x)


class PWLNormalizorInv(nn.Module):
    def __init__(self, app):
        super(PWLNormalizorInv, self).__init__()
        self.pwl = InversedMonoPointPWL(app.pwl)

    def forward(self, x):
        return self.pwl(x)


class PWLNormalizor(nn.Module):
    def __init__(self, num_channels, num_breakpoints, monotonicity=1):
        super(PWLNormalizor, self).__init__()
        self.app = PWLNormalizorApp(num_channels, num_breakpoints, monotonicity=monotonicity)
        self.inv = PWLNormalizorInv(self.app)

    def forward(self, x):
        return self.app(x)

    def inverse(self, x):
        return self.inv(x)


