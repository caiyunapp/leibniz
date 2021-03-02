import torch as th
import torch.nn as nn
import torch.nn.functional as F

from leibniz.unet.senet import SELayer


class BilinearWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super(BilinearWarpingScheme, self).__init__()
        self.padding_mode = padding_mode
        self.grids = {}

    def forward(self, im, ws):
        b, c, h, w = im.size()
        key = '%d-%s' % (b, im.device)
        if key not in self.grids:
            g0 = th.linspace(-1, 1, h, device=im.device, requires_grad=False)
            g1 = th.linspace(-1, 1, w, device=im.device, requires_grad=False)
            grid = th.cat(th.meshgrid([g0, g1]), dim=1).reshape(1, 2, h, w)
            self.grids[key] = grid.repeat(b * c, 1, 1, 1)

        grid = self.grids[key]
        shift = grid.reshape(b * c, 2, h, w) - ws.reshape(b * c, 2, h, w)
        shift = shift.permute(0, 2, 3, 1).view(b * c, h, w, 2)
        return F.grid_sample(im, shift, padding_mode=self.padding_mode, mode='bilinear').reshape(b, c, h, w)


class WarpLayer(nn.Module):
    def __init__(self, channel, conv=None):
        super(WarpLayer, self).__init__()
        self.warp = BilinearWarpingScheme()
        self.se = SELayer(channel // 2, conv=conv)

    def forward(self, x):
        sz = x.size()
        u = x[:, 0::4]
        v = x[:, 1::4]
        y = x[:, 2::4]
        z = x[:, 3::4]
        ws = th.cat((u, v), dim=1)
        ds = th.cat((y, z), dim=1)
        if len(sz) == 3:
            raise Exception('Unimplemented')
        if len(sz) == 4:
            pst = self.warp(ds, ws)
            att = self.se(pst)
        if len(sz) == 5:
            raise Exception('Unimplemented')

        return th.cat([ws, ds * att], dim=1)


class WarpBasicBlock(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv):
        super(WarpBasicBlock, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.wp = WarpLayer(dim)

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
        self.wp = WarpLayer(dim)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.wp(y)
        y = x + y

        return y

