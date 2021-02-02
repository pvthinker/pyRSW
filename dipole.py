""" Dipole-Wall collision experiment

    with no-slip condition

    one layer case

    The strength of the dipole interaction strongly depends on the ratio
    Rossby deformation radius / dipole distance (which is sqrt(Burger) )
"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW

param = Param()

param.nz = 1
param.ny = 64*2
param.nx = 64*2
param.Lx = 1.
param.Ly = 1.
param.auto_dt = False
param.geometry = "closed"
param.cfl = 0.25
param.dt = 1e-2/4
param.tend = 6
param.plotvar = "vor"
param.freq_plot = 8
param.freq_his = 0.1
param.plot_interactive = False
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = True
param.linear = False
param.timestepping = "RK3_SSP"
param.f0 = 1.
param.noslip = True

grid = Grid(param)

model = RSW(param, grid)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye

h = model.state.h
hb = model.state.hb
u = model.state.ux
v = model.state.uy

h0 = param.H
g = param.g
f = param.f0

# setup initial conditions

d = 0.07  # vortex radius
dsep = d*1.1  # half distance between the two vortices
# the vortex amplitude controls the Froude number
amp = -0.05
hmin = 1e-3
x0 = 0.5

def vortex(x1, y1, x0, y0, d):
    xx, yy = np.meshgrid(x1, y1)
    d2 = (xx-x0)**2 + (yy-y0)**2
    return np.exp(-d2/(2*d**2))



h[0] = h0
#h[0] += amp*dambreak(xc, yc, 0.5, 0.5-dsep, d)
h[0] -= amp*vortex(xc, yc, x0, 0.5-dsep, d)
h[0] += amp*vortex(xc, yc, x0, 0.5+dsep, d)
#h[0] = np.maximum(hmin, h[0])


# to set initial geostropic adjustement
# define exactly the same height but at corner cells...
# trick: we use the vorticity array because it has the good shape
# this array will be overwritten with the true vorticity
# once the model is launched
hF = model.state.vor
hF[0] = h0
#hF[0] += amp*dambreak(xe, ye, 0.5, 0.5-dsep, d)
hF[0] -= amp*vortex(xe, ye, x0, 0.5-dsep, d)
hF[0] += amp*vortex(xe, ye, x0, 0.5+dsep, d)
#hF[0] = np.maximum(hmin, hF[0])


def grad(phi, dphidx, dphidy):
    phi.setview("i")
    dphidx.setview("i")
    dphidx[:] = phi[..., 1:]-phi[..., :-1]
    phi.setview("j")
    dphidy.setview("j")
    dphidy[:] = phi[..., 1:]-phi[..., :-1]

u[:] = 0.
v[:] = 0.
# then take the rotated gradient of it
grad(hF, v, u)
u[:] *= -(g/f)
v[:] *= +(g/f)
hF[:] = 0.
model.run()