""" Vortex merging experiment

    one layer case
"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW

param = Param()

reso = 2
param.expname = "merging_linear"
param.nz = 1
param.ny = 64*reso
param.nx = 64*reso
param.Lx = 1.
param.Ly = 1.
param.auto_dt = False
param.geometry = "closed"
param.cfl = 0.25
param.dt = 0.8e-2/reso
param.tend = 5

param.VF_linear = True
param.MF_linear = True
param.MF_order = 1

param.plotvar = "pv"
param.freq_plot = 20
param.freq_his = 0.1
param.plot_interactive = True
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = False
param.timestepping = "RK3_SSP"
param.f0 = 5.
param.var_to_save = ["h", "vor", "pv"]


grid = Grid(param)


grid.finalize()

model = RSW(param, grid)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye
area = grid.arrays.vol.view("i")

h = model.state.h.view("i")
u = model.state.ux
v = model.state.uy

h0 = param.H
g = param.g
f = param.f0

# setup initial conditions

d = 0.07  # vortex radius
dsep = d*1.4  # half distance between the two vortices
# the vortex amplitude controls the Froude number
amp = 0.2
x0 = 0.5


def vortex(xx, yy, x0, y0, d):
    d2 = (xx-x0)**2 + (yy-y0)**2
    return np.exp(-d2/(2*d**2))


h[0] = h0
h[0] += amp*vortex(xc, yc, x0, 0.5-dsep, d)
h[0] += amp*vortex(xc, yc, x0, 0.5+dsep, d)

# convert height "h" to a volume form, i.e. multiply with the cell area
h[0] *= area
# topography
#hb[:] = 0.4*vortex(xc, yc, x0+dsep, 0.5, d)


# to set initial geostropic adjustement
# define exactly the same height but at corner cells...
# trick: we use the vorticity array because it has the good shape
# this array will be overwritten with the true vorticity
# once the model is launched
hF = model.state.vor
hF[0] = h0
hF[0] += amp*vortex(xe, ye, x0, 0.5-dsep, d)
hF[0] += amp*vortex(xe, ye, x0, 0.5+dsep, d)


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
u[:] *= -(g/param.f0)
v[:] *= +(g/param.f0)

u = model.state.ux.view("i")
v = model.state.uy.view("j")

msk = grid.arrays.msk.view("i")
h = model.state.h.view("i")
msku = grid.msku()
mskv = grid.mskv()

for k in range(param.nz):
    u[k] *= msku
    v[k] *= mskv
    h[k][msk == 0] = param.H*area[msk == 0]

hF[:] = 0.
model.run()
