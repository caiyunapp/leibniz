# -*- coding: utf-8 -*-

import logging
import numpy as np
import leibniz as lbnz

from torch.nn.functional import conv3d, pad
from leibniz import cast

logger = logging.getLogger('leibniz')

default_device = -1


def get_device():
    return default_device


def set_device(ix):
    global default_device

    logger.info('reset device from %d to %d', default_device, ix)

    if ix != default_device:
        global centralx, centraly, centralz
        centralx = centralx.cuda(device=ix)
        centraly = centraly.cuda(device=ix)
        centralz = centralz.cuda(device=ix)

        global central5x, central5y, central5z
        central5x = central5x.cuda(device=ix)
        central5y = central5y.cuda(device=ix)
        central5z = central5z.cuda(device=ix)

        global sobel3x, sobel3y, sobel3z
        sobel3x = sobel3x.cuda(device=ix)
        sobel3y = sobel3y.cuda(device=ix)
        sobel3z = sobel3z.cuda(device=ix)

        global sobel5x, sobel5y, sobel5z
        sobel5x = sobel5x.cuda(device=ix)
        sobel5y = sobel5y.cuda(device=ix)
        sobel5z = sobel5z.cuda(device=ix)

        global sharr3x, sharr3y, sharr3z
        sobel3x = sobel3x.cuda(device=ix)
        sharr3y = sharr3y.cuda(device=ix)
        sharr3z = sharr3z.cuda(device=ix)

        global upwind_p2x, upwind_p2y, upwind_p2z
        upwind_p2x = upwind_p2x.cuda(device=ix)
        upwind_p2y = upwind_p2y.cuda(device=ix)
        upwind_p2z = upwind_p2z.cuda(device=ix)

        global upwind_m2x, upwind_m2y, upwind_m2z
        upwind_m2x = upwind_m2x.cuda(device=ix)
        upwind_m2y = upwind_m2y.cuda(device=ix)
        upwind_m2z = upwind_m2z.cuda(device=ix)

        global upwind_p3x, upwind_p3y, upwind_p3z
        upwind_p3x = upwind_p3x.cuda(device=ix)
        upwind_p3y = upwind_p3y.cuda(device=ix)
        upwind_p3z = upwind_p3z.cuda(device=ix)

        global upwind_m3x, upwind_m3y, upwind_m3z
        upwind_m3x = upwind_m3x.cuda(device=ix)
        upwind_m3y = upwind_m3y.cuda(device=ix)
        upwind_m3z = upwind_m3z.cuda(device=ix)

        global upwind_p4x, upwind_p4y, upwind_p4z
        upwind_p4x = upwind_p4x.cuda(device=ix)
        upwind_p4y = upwind_p4y.cuda(device=ix)
        upwind_p4z = upwind_p4z.cuda(device=ix)

        global upwind_m4x, upwind_m4y, upwind_m4z
        upwind_m4x = upwind_m4x.cuda(device=ix)
        upwind_m4y = upwind_m4y.cuda(device=ix)
        upwind_m4z = upwind_m4z.cuda(device=ix)

    default_device = ix


def fill_boundary_central(f):
    # Filling a variable to maintain 2nd-order accuracy at boundary points,
    # while compatible with conv3d using central-diff.
    f = pad(f, (1, 1, 1, 1, 1, 1), mode='replicate')  # 3D zero-padding
    f[:, :, 0, :, :] = 3 * f[:, :, 1, :, :] - 3 * f[:, :, 2, :, :] + f[:, :, 3, :, :]
    f[:, :, -1, :, :] = 3 * f[:, :, -2, :, :] - 3 * f[:, :, -3, :, :] + f[:, :, -4, :, :]
    f[:, :, :, 0, :] = 3 * f[:, :, :, 1, :] - 3 * f[:, :, :, 2, :] + f[:, :, :, 3, :]
    f[:, :, :, -1, :] = 3 * f[:, :, :, -2, :] - 3 * f[:, :, :, -3, :] + f[:, :, :, -4, :]
    f[:, :, :, :, 0] = 3 * f[:, :, :, :, 1] - 3 * f[:, :, :, :, 2] + f[:, :, :, :, 3]
    f[:, :, :, :, -1] = 3 * f[:, :, :, :, -2] - 3 * f[:, :, :, :, -3] + f[:, :, :, :, -4]

    return f


centralx = cast([[[
    [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
]]], device=lbnz.get_device()) / 2
centraly = cast([[[
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, -1, 0], [0, 0, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
]]], device=lbnz.get_device()) / 2
centralz = cast([[[
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
]]], device=lbnz.get_device()) / 2


def central(f):
    f = fill_boundary_central(f)

    return (
        conv3d(f, centralx),
        conv3d(f, centraly),
        conv3d(f, centralz),
    )


sobel3x = -cast([[[
    [[+1, +2, +1], [+2, +4, +2], [+1, +2, +1]],
    [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]],
    [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
]]], device=lbnz.get_device()) / 32
sobel3y = -cast([[[
    [[+1, +2, +1], [ 0,  0,  0], [-1, -2, -1]],
    [[+2, +4, +2], [ 0,  0,  0], [-2, -4, -2]],
    [[+1, +2, +1], [ 0,  0,  0], [-1, -2, -1]]
]]], device=lbnz.get_device()) / 32
sobel3z = -cast([[[
    [[+1,  0, -1], [+2,  0, -2], [+1,  0, -1]],
    [[+2,  0, -2], [+4,  0, -4], [+2,  0, -2]],
    [[+1,  0, -1], [+2,  0, -2], [+1,  0, -1]]
]]], device=lbnz.get_device()) / 32


def sobel3(f):
    f = fill_boundary_central(f)

    return (
        conv3d(f, sobel3x),
        conv3d(f, sobel3y),
        conv3d(f, sobel3z),
    )


sharr3x = -cast([[[
    [[+47, +162, +47], [+94, +324, +94], [+47, +162, +47]],
    [[  0,    0,   0], [  0,    0,   0], [  0,    0,   0]],
    [[-47, -162, -47], [-94, -324, -94], [-47, -162, -47]]
]]], device=lbnz.get_device()) / 2048
sharr3y = -cast([[[
    [[+ 47, + 94, + 47], [ 0,  0,  0], [- 47, - 94, - 47]],
    [[+162, +324, +162], [ 0,  0,  0], [-162, -324, -162]],
    [[+ 47, + 94, + 47], [ 0,  0,  0], [- 47, - 94, - 47]]
]]], device=lbnz.get_device()) / 2048
sharr3z = -cast([[[
    [[+ 47,  0, - 47], [+ 94,  0, - 94], [+ 47,  0, - 47]],
    [[+162,  0, -162], [+324,  0, -324], [+162,  0, -162]],
    [[+ 47,  0, - 47], [+ 94,  0, - 94], [+ 47,  0, - 47]]
]]], device=lbnz.get_device()) / 2048


def sharr3(f):
    f = fill_boundary_central(f)

    return (
        conv3d(f, sharr3x),
        conv3d(f, sharr3y),
        conv3d(f, sharr3z),
    )


def get_five_point_central_kernels():
    # 4th-order, kernel size = 5x5x5; diff_f = (-f_(i+2) + 8*f_(i+1) - 8*f_(i-1) + f_(i-2)) / 12
    five_point_diff = np.array([1, -8, 0, 8, -1]) / 12  # 5-point differential
    kernelx = np.zeros((1, 1, 5, 5, 5))
    kernelx[0, 0, :, 2, 2] = five_point_diff
    kernely = np.swapaxes(kernelx, 2, 3)
    kernelz = np.swapaxes(kernelx, 2, 4)

    return cast(kernelx, device=lbnz.get_device()), cast(kernely, device=lbnz.get_device()), cast(kernelz, device=lbnz.get_device())


central5x, central5y, central5z = get_five_point_central_kernels()


def central5(f):
    f = fill_boundary_central(fill_boundary_central(f))

    return (
        conv3d(f, central5x),
        conv3d(f, central5y),
        conv3d(f, central5z)
    )


def get_five_point_sobel_kernels():
    # get 5x5 sobel-way built kernels
    smooth_5_point = np.array([[1, 4, 6, 4, 1]])  # 1-D 5-point smoothing
    smooth_5x5 = (np.transpose(smooth_5_point) * smooth_5_point).reshape((1, 5, 5))  # 2-D 5x5 smoothing
    five_point_diff = (np.array([1, -8, 0, 8, -1]) / 12).reshape(5, 1, 1)  # 5-point differential
    # add the denominator to normalize the kernel parameters
    sobel_5x5_x = (smooth_5x5 * five_point_diff / 256).reshape(1, 1, 5, 5, 5)
    sobel_5x5_y = np.swapaxes(sobel_5x5_x, 2, 3)
    sobel_5x5_z = np.swapaxes(sobel_5x5_x, 2, 4)

    return cast(sobel_5x5_x, device=lbnz.get_device()), cast(sobel_5x5_y, device=lbnz.get_device()), cast(sobel_5x5_z, device=lbnz.get_device())


sobel5x, sobel5y, sobel5z = get_five_point_sobel_kernels()


def sobel5(f):

    # this filling way can somehow bring errors, remains to be revised.
    f = fill_boundary_central(fill_boundary_central(f))

    return (
        conv3d(f, sobel5x),
        conv3d(f, sobel5y),
        conv3d(f, sobel5z),
    )


def fill_boundary_upwind(f, direction):
    if direction == 'p':
        f = pad(f, (0, 1, 0, 1, 0, 1))  # one-side zero-padding
        f[:, :, -1, :, :] = 2 * f[:, :, -2, :, :] - f[:, :, -3, :, :]
        f[:, :, :, -1, :] = 2 * f[:, :, :, -2, :] - f[:, :, :, -3, :]
        f[:, :, :, :, -1] = 2 * f[:, :, :, :, -2] - f[:, :, :, :, -3]
    elif direction == 'm':
        f = pad(f, (1, 0, 1, 0, 1, 0))  # one-side zero-padding
        f[:, :, 0, :, :] = 2 * f[:, :, 1, :, :] - f[:, :, 2, :, :]
        f[:, :, :, 0, :] = 2 * f[:, :, :, 1, :] - f[:, :, :, 2, :]
        f[:, :, :, :, 0] = 2 * f[:, :, :, :, 1] - f[:, :, :, :, 2]

    return f


def get_two_point_upwind_kernels(direction):
    # 'p' denotes 'plus', i.e. diff_u = (u_i+1) - u_i
    # 'm' denotes 'minus', i.e. diff_u = u_i - (u_i-1)

    upwind_x = np.zeros((1, 1, 2, 2, 2))
    upwind_diff = np.array([-1, 1])

    if direction == 'p':
        upwind_x[0, 0, :, 0, 0] = upwind_diff
    elif direction == 'm':
        upwind_x[0, 0, :, 1, 1] = upwind_diff

    upwind_y = np.swapaxes(upwind_x, 2, 3)
    upwind_z = np.swapaxes(upwind_x, 2, 4)

    return cast(upwind_x), cast(upwind_y), cast(upwind_z)


upwind_p2x, upwind_p2y, upwind_p2z = get_two_point_upwind_kernels('p')
upwind_m2x, upwind_m2y, upwind_m2z = get_two_point_upwind_kernels('m')


def upwind_p2(f):
    f = fill_boundary_upwind(f, 'p')
    return (
        conv3d(f, upwind_p2x),
        conv3d(f, upwind_p2y),
        conv3d(f, upwind_p2z),
    )


def upwind_m2(f):
    f = fill_boundary_upwind(f, 'm')

    return (
        conv3d(f, upwind_m2x),
        conv3d(f, upwind_m2y),
        conv3d(f, upwind_m2z),
    )


def get_three_point_upwind_kernels(direction):
    # 'p' denotes 'plus', i.e. diff_u = -(u_i+2) + 4 * (u_i+1) - 3 * u_i
    # 'm' denotes 'minus', i.e. diff_u = 3 * u_i - 4 * (u_i-1) + (u_i-2)

    upwind_x = np.zeros((1, 1, 3, 3, 3))

    if direction == 'p':
        upwind_diff = np.array([-3, 4, -1]) / 2
        upwind_x[0, 0, :, 0, 0] = upwind_diff
    elif direction == 'm':
        upwind_diff = np.array([1, -4, 3]) / 2
        upwind_x[0, 0, :, -1, -1] = upwind_diff

    upwind_y = np.swapaxes(upwind_x, 2, 3)
    upwind_z = np.swapaxes(upwind_x, 2, 4)

    return cast(upwind_x, device=lbnz.get_device()), cast(upwind_y, device=lbnz.get_device()), cast(upwind_z, device=lbnz.get_device())


upwind_p3x, upwind_p3y, upwind_p3z = get_three_point_upwind_kernels('p')
upwind_m3x, upwind_m3y, upwind_m3z = get_three_point_upwind_kernels('m')


def upwind_p3(f):

    # this filling way can somehow bring errors, remains to be revised.
    f = fill_boundary_upwind(fill_boundary_upwind(f, 'p'), 'p')

    return (
        conv3d(f, upwind_p3x),
        conv3d(f, upwind_p3y),
        conv3d(f, upwind_p3z),
    )


def upwind_m3(f):

    # this filling way can somehow bring errors, remains to be revised.
    f = fill_boundary_upwind(fill_boundary_upwind(f, 'm'), 'm')

    return (
        conv3d(f, upwind_m3x),
        conv3d(f, upwind_m3y),
        conv3d(f, upwind_m3z),
    )


def get_four_point_upwind_kernels(direction):
    # 'p' denotes 'plus', i.e. diff_u = -(u_i+2) + 4 * (u_i+1) - 3 * u_i
    # 'm' denotes 'minus', i.e. diff_u = 3 * u_i - 4 * (u_i-1) + (u_i-2)

    upwind_x = np.zeros((1, 1, 4, 4, 4))

    if direction == 'p':
        upwind_diff = np.array([-2, -3, 6, -1]) / 6
        upwind_x[0, 0, :, 1, 1] = upwind_diff
    elif direction == 'm':
        upwind_diff = np.array([1, -6, 3, 2]) / 6
        upwind_x[0, 0, :, -2, -2] = upwind_diff

    upwind_y = np.swapaxes(upwind_x, 2, 3)
    upwind_z = np.swapaxes(upwind_x, 2, 4)

    return cast(upwind_x, device=lbnz.get_device()), cast(upwind_y, device=lbnz.get_device()), cast(upwind_z, device=lbnz.get_device())


upwind_p4x, upwind_p4y, upwind_p4z = get_four_point_upwind_kernels('p')
upwind_m4x, upwind_m4y, upwind_m4z = get_four_point_upwind_kernels('m')


def upwind_p4(f):

    # this filling way can somehow bring errors, remains to be revised.
    f = fill_boundary_upwind(fill_boundary_upwind(fill_boundary_upwind(f, 'm'), 'p'), 'p')

    return (
        conv3d(f, upwind_p4x),
        conv3d(f, upwind_p4y),
        conv3d(f, upwind_p4z),
    )


def upwind_m4(f):

    # this filling way can somehow bring errors, remains to be revised.
    f = fill_boundary_upwind(fill_boundary_upwind(fill_boundary_upwind(f, 'p'), 'm'), 'm')

    return (
        conv3d(f, upwind_m4x),
        conv3d(f, upwind_m4y),
        conv3d(f, upwind_m4z),
    )
