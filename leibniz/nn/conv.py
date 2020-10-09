from torch import nn as nn


def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True, kernels_per_layer=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv1d(in_channels, in_channels * kernels_per_layer, kernel_size, padding=padding,
                                   stride=stride, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels * kernels_per_layer, output_channels, 1, padding=0,
                                   stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True, kernels_per_layer=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size, padding=padding,
                                   stride=stride, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, 1, padding=0,
                                   stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels * kernels_per_layer, kernel_size, padding=padding,
                                   stride=stride, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv3d(in_channels * kernels_per_layer, output_channels, 1, padding=0,
                                   stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ComplexConv1d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.rconv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.iconv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return self.rconv(input.real) - self.iconv(input.imag) + 1j * (self.rconv(input.imag) + self.iconv(input.real))


class ComplexConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.rconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.iconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return self.rconv(input.real) - self.iconv(input.imag) + 1j * (self.rconv(input.imag) + self.iconv(input.real))


class ComplexConv3d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv3d, self).__init__()
        self.rconv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.iconv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return self.rconv(input.real) - self.iconv(input.imag) + 1j * (self.rconv(input.imag) + self.iconv(input.real))
