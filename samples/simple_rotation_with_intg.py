# -*- coding: utf-8 -*-

import os
import torch as th

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers

from diffeq import odeint as odeint


def binary(tensor):
    return th.where(tensor > iza.zero, iza.one, iza.zero)


def upwind_adv(wind, data, points):
    # upwind scheme using speed-splitting techniques.
    # Referring to https://web.stanford.edu/group/frg/course_work/AA214B/CA-AA214B-Ch6.pdf
    # and https://en.wikipedia.org/wiki/Upwind_scheme
    # Param :
    #   points: points to be used in upwind scheme

    # grad part
    dL1, dL2, dL3 = iza.dL
    if points == '2':
        g_p1, g_p2, g_p3 = upwind_p2(data)
        g_m1, g_m2, g_m3 = upwind_m2(data)
    elif points == '3':
        g_p1, g_p2, g_p3 = upwind_p3(data)
        g_m1, g_m2, g_m3 = upwind_m3(data)
    elif points == '4':
        g_p1, g_p2, g_p3 = upwind_p4(data)
        g_m1, g_m2, g_m3 = upwind_m4(data)
    g_p1, g_p2, g_p3 = g_p1 / dL1, g_p2 / dL2, g_p3 / dL3
    g_m1, g_m2, g_m3 = g_m1 / dL1, g_m2 / dL2, g_m3 / dL3

    # adv part
    u, v, w = wind
    u_p, u_m = th.where(u > iza.zero, u, iza.zero), th.where(u < iza.zero, u, iza.zero)
    v_p, v_m = th.where(v > iza.zero, v, iza.zero), th.where(v < iza.zero, v, iza.zero)
    w_p, w_m = th.where(w > iza.zero, w, iza.zero), th.where(w < iza.zero, w, iza.zero)

    return g_p1 * u_m + g_m1 * u_p + g_p2 * v_m + g_m2 * v_p + g_p3 * w_m + g_m3 * w_p


def draw_3D_plot(x, y, fld, fig_name='test.png'):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    fld = fld.cpu().numpy()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    c = ax.plot_surface(x, y, fld, rstride=1, cstride=1,
                        cmap='binary', edgecolor='none', vmin=-2, vmax=2)
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_zlim(-2, 2)

    fig.colorbar(c)
    plt.savefig(os.path.join('results', fig_name))
    plt.close()


def draw_animation(sequence, x, y, gif_name='test.gif', frames=51, mode='3d'):
    gif_name = os.path.join('results', gif_name)
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    sequence = [item.cpu().numpy() for item in sequence]

    fig = plt.figure()

    def update(n):
        print('drawing %2dth image in animation' % n)
        plt.clf()
        ax = fig.gca(projection='3d')
        c = ax.plot_surface(x, y, sequence[n], rstride=1, cstride=1,
                            cmap='binary', edgecolor='none', vmin=-2, vmax=2)
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_zlim(-2, 2)
        fig.colorbar(c)

    ani = FuncAnimation(fig, update, frames=frames, interval=1000.0 / 4, save_count=100, repeat_delay=1000, repeat=True)

    # save it into gif (need imagemagick)
    ani.save(gif_name[:-4] + mode + '.gif', writer='imagemagick')
    plt.close()


if __name__ == '__main__':
    import isaac as iza
    from isaac.gridsys.regular3 import RegularGrid
    from isaac.diffr.regular3 import upwind_p2, upwind_m2, upwind_p3, upwind_m3, upwind_p4, upwind_m4

    iza.bind(RegularGrid(
        basis='x,y,z',
        W=151, L=151, H=15,
        east=16.0, west=-16.0,
        north=16.0, south=-16.0,
        upper=6.0, lower=1.0
    ))
    iza.use('x,y,z')

    dt = 1 / 120
    t_iter = 1000

    init = th.exp(-(((iza.x - 1) * 2) ** 2 + ((iza.y - 1) * 2) ** 2))  # Gaussian

    r = th.sqrt(iza.x ** 2 + iza.y ** 2)
    theta = th.atan2(iza.y, iza.x)
    wind = - r * th.sin(theta), r * th.cos(theta), iza.zero

    def derivitive(t, clouds):
        # return - iza.adv(wind, clouds)
        return - upwind_adv(wind, clouds, '4')

    pred = odeint(derivitive, init, th.arange(0, 8, 1 / 200), method='rk4')
    sequence = [pred[i].squeeze()[..., iza.H // 2] for i in range(1600) if i % 20 == 0]
    draw_animation(sequence, iza.x.squeeze()[..., 0], iza.y.squeeze()[..., 0], 'simple_rotation(upwind4).gif',
                   mode='3d', frames=len(sequence))
