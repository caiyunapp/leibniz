import torch as th
import torch.nn as nn

from torchpwl.pwl import BasePointPWL, MonoPointPWL


class InversedMonoPointPWL(BasePointPWL):
    def __init__(self, peer):
        super(InversedMonoPointPWL, self).__init__(peer.num_channels, peer.num_breakpoints, num_x_points=peer.num_breakpoints+1)
        self.peer = peer

    def _reset_params(self):
        self.peer._reset_params()

    def get_y_positions(self):
        return th.sort(self.peer.get_x_positions(), dim=1, descending=self.peer.mono<0)[0]

    def get_x_positions(self):
        return th.sort(self.peer.get_y_positions(), dim=1, descending=self.peer.mono<0)[0]


class PWLNormalizorApp(nn.Module):
    def __init__(self, num_channels, num_breakpoints, monotonicity=1):
        super(PWLNormalizorApp, self).__init__()
        self.pwl = MonoPointPWL(num_channels, num_breakpoints, monotonicity=monotonicity)
        self.pwl.mono = monotonicity

    def forward(self, x):
        sz1 = list(x.size())
        l, c = len(sz1), sz1[1]
        shift = [0] + list(range(2, l)) + [1]
        unshift = [0, l - 1] + list(range(1, l - 1))
        sz2 = [sz1[0]] + list([sz1[i] for i in range(2, l)]) + [sz1[1]]
        x = x.permute(*shift).reshape(-1, c)
        return self.pwl(x).reshape(*sz2).permute(*unshift).reshape(*sz1)


class PWLNormalizorInv(nn.Module):
    def __init__(self, app):
        super(PWLNormalizorInv, self).__init__()
        self.pwl = InversedMonoPointPWL(app.pwl)

    def forward(self, x):
        sz1 = list(x.size())
        l, c = len(sz1), sz1[1]
        shift = [0] + list(range(2, l)) + [1]
        unshift = [0, l - 1] + list(range(1, l - 1))
        sz2 = [sz1[0]] + list([sz1[i] for i in range(2, l)]) + [sz1[1]]
        x = x.permute(*shift).reshape(-1, c)
        return self.pwl(x).reshape(*sz2).permute(*unshift).reshape(*sz1)


class PWLNormalizor(nn.Module):
    def __init__(self, num_channels, num_breakpoints, monotonicity=1, std=1.0, mean=0.0):
        super(PWLNormalizor, self).__init__()
        self.app = PWLNormalizorApp(num_channels, num_breakpoints, monotonicity=monotonicity)
        self.inv = PWLNormalizorInv(self.app)
        self.mean = mean
        self.std = std

    def forward(self, x):
        return self.app((x - self.mean) / self.std)

    def inverse(self, x):
        return self.inv(x) * self.std + self.mean


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm1d, self).__init__()
        self.rbn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.ibn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.rbn(input.real) + 1j * self.ibn(input.imag)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.rbn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.ibn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.rbn(input.real) + 1j * self.ibn(input.imag)


class ComplexBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm3d, self).__init__()
        self.rbn = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
        self.ibn = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.rbn(input.real) + 1j * self.ibn(input.imag)
