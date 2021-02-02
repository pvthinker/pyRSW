import numpy as np
import os
import matplotlib

matplotlib.use('TkAgg')
font = {'size': 16}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from movietools import Movie

plt.ion()

class Figure(object):
    def __init__(self, param, state, time):
        self.param = param
        Lx, Ly = param.Lx, param.Ly
        self.plotvar = param.plotvar
        self.var = state.get(self.plotvar)
        z2d = self.var.view("i")[-1]

        self.fig = plt.figure(figsize=(12, 12))
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.im = self.ax1.imshow(z2d, extent=[0, Lx, 0, Ly], cmap='RdBu_r',
                                  interpolation='nearest', origin='lower')
        plt.colorbar(self.im)
        self.ax1.set_title('%s / t=%.2f' % (self.var.name, time))
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.text(0.02*Lx, 0.95*Ly, 'Rd=%.2f' % (np.sqrt(param.g*param.H)/param.f0))
        #self.ax1.text(0.02*Lx, 0.9*Ly, 'Ro=%.2f' % rossby)
        self.ax1.text(0.02*Lx, 0.85*Ly, r'$\sqrt{gH}$=%.2f' % (np.sqrt(param.g*param.H)))
        self.fig.show()
        self.fig.canvas.draw()
        if param.generate_mp4:
            datadir = os.path.expanduser(param.datadir)
            moviename = f"{datadir}/{param.expname}/movie_{param.plotvar}"
            print(moviename)
            self.mov = Movie(self.fig, name=moviename)
        self.update(time)

    def update(self, time):
        z2d = self.var.view("i")[-1]
        self.im.set_array(z2d)
        if self.param.colorscheme == 'imposed':
            vmin, vmax = self.param.cax
        elif self.param.colorscheme == 'auto':
            vmin, vmax = np.min(z2d), np.max(z2d)
        else:
            raise ValueError('colorscheme should be "imposed" or "auto"')

        self.ax1.set_title('%s / t=%.2f' % (self.plotvar, time))
        self.im.set_clim(vmin=vmin, vmax=vmax)
        self.fig.canvas.draw()
        if self.param.generate_mp4:
            self.mov.addframe()

    def finalize(self):
        if self.param.generate_mp4:
            self.mov.finalize()
