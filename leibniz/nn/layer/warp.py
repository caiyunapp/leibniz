import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BilinearWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros'):
        super(BilinearWarpingScheme, self).__init__()
        self.padding_mode = padding_mode
        self.grids = {}

    def forward(self, im, ws):
        b = ws.size()[0]
        c = ws.size()[1]
        h = ws.size()[2]
        w = ws.size()[3]

        key = '%d-%s' % (b, im.get_device())
        if key not in self.grids:
            if im.get_device() < 0:
                g0 = th.linspace(-1, 1, h, requires_grad=False)
                g1 = th.linspace(-1, 1, w, requires_grad=False)
            else:
                g0 = th.linspace(-1, 1, h, device=im.get_device(), requires_grad=False)
                g1 = th.linspace(-1, 1, w, device=im.get_device(), requires_grad=False)
            grid = th.cat(th.meshgrid([g0, g1]), dim=1).reshape(1, 2, h, w)
            self.grids[key] = grid.repeat(b * c // 2, 1, 1, 1)

        grid = self.grids[key]
        shift = grid.reshape(-1, 2, h, w) - ws.reshape(-1, 2, h, w)
        shift = shift.permute(0, 2, 3, 1)
        return F.grid_sample(im.reshape(-1, 2, h, w), shift, padding_mode=self.padding_mode, mode='bilinear').reshape(-1, c, h, w)
