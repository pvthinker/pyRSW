""" Annulus geometry

    one layer case

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW

param = Param()

reso = 2
param.expname = "annulus"
param.nz = 1
param.ny = 128*reso
param.nx = 128*reso
param.Lx = 2.
param.Ly = 1.
param.auto_dt = False
param.geometry = "perio_y"
param.coordinates = "cylindrical"
param.cfl = 0.2
param.dt = 0.8e-2/reso
param.tend = 5  # 0*param.dt
param.plotvar = "h"
param.freq_plot = 50
param.freq_his = 0.1
param.plot_interactive = True
param.plot_type = "pcolormesh"
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = False
param.timestepping = "RK3_SSP"
param.f0 = 5.
param.var_to_save = ["h", "vor"]


def vortex(xx, yy, **kwargs):
    """
    analytical function that defines the domain

    fmsk < 0 : solid
    fmsk == 0: boundary
    fmsk > 0 : fluid
    """
    x0 = kwargs["x0"]
    y0 = kwargs["y0"]
    d = kwargs["d"]
    vtype = kwargs["vtype"]
    if "ratio" in kwargs:
        ratio = kwargs["ratio"]
    else:
        ratio = 1.
    d2 = (xx-x0)**2*ratio + (yy-y0)**2
    if vtype == "cosine":
        d0 = np.sqrt(d2)
        m = np.cos(d0/d*(np.pi/2))
        m[d0 > d] = 0.
    else:
        m = np.exp(-d2/(2*d**2))
    return m


grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye


grid.finalize()

model = RSW(param, grid)


h = model.state.h
area = grid.arrays.vol.view("i")
u = model.state.ux
v = model.state.uy

h0 = param.H
g = param.g
f = param.f0

# setup initial conditions

d = 0.1  # vortex radius
dsep = -d*1.1  # half distance between the two vortices
# the vortex amplitude controls the Froude number
amp = 0.1


y0 = 0.5
sigma = 0.08

h[0] = h0-amp*(0.5+0.5*np.tanh((yc-y0)/sigma))


# convert height "h" to a volume form, i.e. multiply with the cell area
h[0] *= area

# for k in range(param.nz):
#     h[k] *= msk

# to set initial geostropic adjustement
# define exactly the same height but at corner cells...
# trick: we use the vorticity array because it has the good shape
# this array will be overwritten with the true vorticity
# once the model is launched
hF = model.state.vor
hF[0] = h0-amp*(0.5+0.5*np.tanh((ye-y0)/sigma))


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
# grad(hF, v, u)
# u[:] *= -(g/param.f0)
# v[:] *= +(g/param.f0)


u = model.state.ux.view("i")
v = model.state.uy.view("j")

msku = grid.msku()
mskv = grid.mskv()

for k in range(param.nz):
    u[k] *= msku
    v[k] *= mskv

hF[:] = 0.
model.run()
