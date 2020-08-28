import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DenseGridGen(nn.Module):
    def __init__(self):
        super(DenseGridGen, self).__init__()
        self.grids = {}

    def forward(self, x):
        x = x.transpose(1, 2).transpose(2, 3)
        xs = x.size()

        if xs[0] not in self.grids:
            if x.get_device() < 0:
                g0 = th.linspace(-1, 1, x.size(2), requires_grad=False).unsqueeze(0).repeat(x.size(1), 1)
                g1 = th.linspace(-1, 1, x.size(1), requires_grad=False).unsqueeze(1).repeat(1,x.size(2))
            else:
                g0 = th.linspace(-1, 1, x.size(2), device=x.get_device(), requires_grad=False).unsqueeze(0).repeat(x.size(1), 1)
                g1 = th.linspace(-1, 1, x.size(1), device=x.get_device(), requires_grad=False).unsqueeze(1).repeat(1, x.size(2))
            grid = th.cat([g0.unsqueeze(-1), g1.unsqueeze(-1)], -1)
            gs = grid.size()
            self.grids[xs[0]] = grid.unsqueeze(0).expand(xs[0], gs[0], gs[1], gs[2])

        grid = self.grids[xs[0]]
        return grid - x


class BilinearWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super(BilinearWarpingScheme, self).__init__()
        self.grid = DenseGridGen()
        self.padding_mode = padding_mode

    def forward(self, im, ws):
        return F.grid_sample(im, self.grid(ws), padding_mode=self.padding_mode, mode='bilinear')


class WarpLayer(nn.Module):
    def __init__(self, channel, reduction=16, wchannels=[0, 1]):
        super(WarpLayer, self).__init__()
        self.wchannels = wchannels

        self.avg_pool = None
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction + 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction + 1, channel, bias=False),
            nn.Sigmoid()
        )

        self.warp = BilinearWarpingScheme()

    def forward(self, x):
        sz = x.size()
        u = x[:, self.wchannels[0]:self.wchannels[0]+1]
        v = x[:, self.wchannels[1]:self.wchannels[1]+1]
        ws = th.cat((u, v), dim=1)
        if len(sz) == 3:
            raise Exception('Unimplemented')
        if len(sz) == 4:
            x = self.warp(x, ws)
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1, 1)
        if len(sz) == 5:
            raise Exception('Unimplemented')

        return x * y.expand_as(x)


class WarpBasicBlock(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(WarpBasicBlock, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.wp = WarpLayer(dim, reduction)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.wp(y)
        y = x + y

        return y


class WarpBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(WarpBottleneck, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(dim, dim // 4, kernel_size=1, bias=False)
        self.conv2 = conv(dim // 4, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(dim // 4, dim, kernel_size=1, bias=False)
        self.wp = WarpLayer(dim, reduction)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.wp(y)
        y = x + y

        return y

