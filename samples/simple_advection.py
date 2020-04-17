# -*- coding: utf-8 -*-

import os
import matplotlib
matplotlib.use('Agg')

import torch as th
from leibniz.diffeq import odeint as odeint

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.animation import FuncAnimation


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
    ax.set_xlim(1, 16)
    ax.set_ylim(1, 6)
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
        ax.set_xlim(1, 16)
        ax.set_ylim(1, 6)
        ax.set_zlim(-2, 2)
        fig.colorbar(c)

    ani = FuncAnimation(fig, update, frames=frames, interval=1000.0 / 4, save_count=100, repeat_delay=1000, repeat=True)

    # save it into gif (need imagemagick)
    ani.save(gif_name[:-4] + mode + '.gif', writer='imagemagick')
    plt.close()


if __name__ == '__main__':
    import leibniz as lbnz
    from leibniz.core3d.gridsys.regular3 import RegularGrid

    lbnz.bind(RegularGrid(
        basis='x,y,z',
        W=51, L=151, H=51,
        east=16.0, west=1.0,
        north=6.0, south=1.0,
        upper=6.0, lower=1.0
    ))
    lbnz.use('x,y,z')

    # dt = 0.05
    # t_iter = 50
    fld = binary((lbnz.x - 8) * (9 - lbnz.x)) * \
          binary((lbnz.y - 3) * (4 - lbnz.y)) * \
          binary((lbnz.z - 3) * (4 - lbnz.z))

    wind = lbnz.one, lbnz.zero, lbnz.zero

    def derivitive(t, clouds):
        return - lbnz.upwind(wind, clouds)

    pred = odeint(derivitive, fld, th.arange(0, 7, 1 / 100), method='rk4')
    #sequence = [pred[i].squeeze()[..., lbnz.H // 2] for i in range(700) if i % 10 == 0]
    #draw_animation(sequence, lbnz.x.squeeze()[..., 0], lbnz.y.squeeze()[..., 0], 'simple_advection(upwind4).gif',
    #               mode='3d', frames=len(sequence))
