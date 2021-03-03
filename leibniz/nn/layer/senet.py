import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, conv=None):
        super(SELayer, self).__init__()

        self.avg_pool = None
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction + 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction + 1, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        sz = x.size()

        if len(sz) == 3:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool1d(1)
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1)
        if len(sz) == 4:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1, 1)
        if len(sz) == 5:
            if self.avg_pool is None:
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
            y = self.avg_pool(x)
            y = y.view(sz[0], sz[1])
            y = self.fc(y).view(sz[0], sz[1], 1, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.se = SELayer(dim, reduction)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.se(y)
        y = x + y

        return y


class SEBottleneck(nn.Module):
    extension = 4
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(SEBottleneck, self).__init__()
        self.step = step
        self.relu = relu

        self.conv1 = conv(dim, dim // 4, kernel_size=1, bias=False)
        self.conv2 = conv(dim // 4, dim // 4, kernel_size=3, bias=False, padding=1)
        self.conv3 = conv(dim // 4, dim, kernel_size=1, bias=False)
        self.se = SELayer(dim, reduction)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.se(y)
        y = x + y

        return y

