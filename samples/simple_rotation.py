# -*- coding: utf-8 -*-

import os
import torch as th
import leibniz as lbnz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers


def binary(tensor):
    return th.where(tensor > lbnz.zero, lbnz.one, lbnz.zero)


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


def test_Gaussian_rotation():
    fld = th.exp(-(((lbnz.x - 1) * 2) ** 2 + ((lbnz.y - 1) * 2) ** 2))  # Gaussian

    r = th.sqrt(lbnz.x ** 2 + lbnz.y ** 2)
    theta = th.atan2(lbnz.y, lbnz.x)
    wind = - r * th.sin(theta), r * th.cos(theta), 0

    sequence, test = [fld], fld
    for t in range(1, t_iter + 1):
        adv_increment = lbnz.adv(wind, test)
        if t == 1:
            test = sequence[t - 1] - adv_increment * dt
        else:
            test = sequence[t - 2] - adv_increment * 2 * dt
        sequence.append(test)

    sequence = [sequence[i].squeeze()[..., lbnz.H // 2] for i in range(0, t_iter, 10)]
    draw_animation(sequence, lbnz.x.squeeze()[..., 0], lbnz.y.squeeze()[..., 0], 'simple_rotation1.gif', mode='3d',
                   frames=len(sequence))


def test_cube_rotation():
    fld = binary((lbnz.x - 2) * (3 - lbnz.x)) * \
          binary((lbnz.y - 2) * (3 - lbnz.y)) * \
          binary((lbnz.z - 1) * (6 - lbnz.z))

    r = th.sqrt(lbnz.x ** 2 + lbnz.y ** 2)
    theta = th.atan2(lbnz.y, lbnz.x)
    wind = - r * th.sin(theta), r * th.cos(theta), 0

    sequence, test = [fld], fld
    for t in range(1, t_iter + 1):
        adv_increment = lbnz.adv(wind, test)
        if t == 1:
            test = sequence[t - 1] - adv_increment * dt
        else:
            test = sequence[t - 2] - adv_increment * 2 * dt
        sequence.append(test)

    sequence = [sequence[i].squeeze()[..., lbnz.H // 2] for i in range(0, t_iter, 10)]
    draw_animation(sequence, lbnz.x.squeeze()[..., 0], lbnz.y.squeeze()[..., 0], 'simple_rotation2.gif', mode='3d',
                   frames=len(sequence))


if __name__ == '__main__':
    from leibniz.core3d.gridsys.regular3 import RegularGrid

    lbnz.bind(RegularGrid(
        basis='x,y,z',
        W=151, L=151, H=15,
        east=16.0, west=-16.0,
        north=16.0, south=-16.0,
        upper=6.0, lower=1.0
    ))
    lbnz.use('x,y,z')

    dt = 1 / 120
    t_iter = 1000

    # test_Gaussian_rotation()
    test_cube_rotation()
