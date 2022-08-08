from movietools import Movie
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
import warnings

matplotlib.use('TkAgg')
font = {'size': 16}
matplotlib.rc('font', **font)

plt.ion()

# to disable the warning when no topography
warnings.filterwarnings("ignore")


class Figure(object):
    def __init__(self, param, grid, state, time):
        self.param = param
        self.grid = grid
        Lx, Ly = param.Lx, param.Ly
        self.tu = self.param.timeunit
        self.plotvar = param.plotvar
        var = state.get(self.plotvar)
        z2d = var.getproperunits(self.grid)[-1]

        topocolor = "#556b2f"  # topography contours are green

        self.titlestr = "%s / t=%.2f"
        if param.nz > 1:
            self.titlestr += " / layer="+f"{self.param.nz-1}"

        self.fig = plt.figure(figsize=(12, 12))
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        if param.plot_type == "imshow":
            self.im = self.ax1.imshow(z2d, extent=[0, Lx, 0, Ly], cmap=param.cmap,
                                      interpolation='nearest', origin='lower')
        elif param.plot_type == "pcolormesh":
            ye, xe = grid.ye, grid.xe
            self.im = self.ax1.pcolormesh(xe, ye, z2d, cmap=param.cmap)
        else:
            raise ValueError

        if param.coordinates != "spherical":
            hb = grid.arrays.hb.view("i")
            self.ax1.contour(grid.xc, grid.yc, hb, 3, colors=topocolor)

        plt.colorbar(self.im)
        self.ax1.set_title(self.titlestr % (var.name, time/self.tu))
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        H = param.H
        if isinstance(H, list):
            H = sum(H)
        self.ax1.text(0.02*Lx, 0.95*Ly, 'Rd=%.2f' %
                      (np.sqrt(param.g*H)/param.f0))
        #self.ax1.text(0.02*Lx, 0.9*Ly, 'Ro=%.2f' % rossby)
        self.ax1.text(0.02*Lx, 0.85*Ly,
                      r'$\sqrt{gH}$=%.2f' % (np.sqrt(param.g*H)))
        self.fig.show()
        self.fig.canvas.draw()
        if param.generate_mp4:
            datadir = os.path.expanduser(param.datadir)
            expname = param.expname
            moviename = f"{datadir}/{expname}/{expname}_{param.plotvar}"
            print(f"generate: {moviename}.mp4")
            self.mov = Movie(self.fig, name=moviename)
        self.update(time, state)

    def update(self, time, state):
        var = state.get(self.plotvar)
        z2d = var.getproperunits(self.grid)[-1]
        if self.param.plot_type == "imshow":
            self.im.set_array(z2d)
        else:
            self.im.set_array(z2d.ravel())

        if self.param.colorscheme == 'imposed':
            vmin, vmax = self.param.cax
        elif self.param.colorscheme == 'auto':
            vmin, vmax = np.min(z2d), np.max(z2d)
        else:
            raise ValueError('colorscheme should be "imposed" or "auto"')

        self.ax1.set_title(self.titlestr % (self.plotvar, time/self.tu))
        self.im.set_clim(vmin=vmin, vmax=vmax)
        self.fig.canvas.draw()
        plt.pause(1e-4)
        if self.param.generate_mp4:
            self.mov.addframe()

    def finalize(self):
        if self.param.generate_mp4:
            self.mov.finalize()
