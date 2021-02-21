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

reso = 4
param.expname = "dipole"
param.nz = 1
param.ny = 25*reso
param.nx = 50*reso
param.Lx = 2.
param.Ly = 1.
param.auto_dt = False
param.geometry = "closed"
param.cfl = 0.2
param.dt = 1e-2*2/reso
param.tend = 30  # 100*param.dt
param.plotvar = "h"
param.freq_plot = 100
param.freq_his = 0.1
param.plot_interactive = True
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = False
param.linear = False
param.timestepping = "RK3_SSP"
param.f0 = 10.
param.noslip = False
param.var_to_save = ["h", "vor", "pv"]


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
    if "vtype" in kwargs:
        vtype = kwargs["vtype"]
    else:
        vtype = "gaussian"
    if "ratio" in kwargs:
        ratio = kwargs["ratio"]
    else:
        ratio = 1.
    d2 = (xx-x0)**2*ratio + (yy-y0)**2
    if vtype == "cosine":
        d0 = np.sqrt(d2)
        m = np.zeros_like(d0)
        m[d0 <= d] = 1
        m[d0 > d] = -1
        d2 = (xx-x0)**2 + (yy-y0)**2
        d0 = np.sqrt(d2)
        m[d0 < 0.1] = -1

    else:
        m = np.exp(-d2/(2*d**2))
    return m


def fulldomain(xx, yy, **kwargs):
    return np.ones(xx.shape)


def sliced(xx, yy, **kwargs):
    return -param.Ly*0.1+(yy-(xx-param.Lx/2)*.1)


grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye
hb = grid.arrays.hb.view("i")

kwargs = {"ratio": 0.5, "x0": param.Lx*0.5, "y0": param.Ly *
          0.5, "d": param.Ly*0.5-grid.dy, "vtype": "cosine"}

#grid.boundary = {"fbry": vortex, "kwargs": kwargs}

kwargs = {"x0": param.Lx*0.4, "y0": param.Ly*0.4, "d": 0.1}
htopo = 0.6
hb[:] = htopo*vortex(xc, yc, **kwargs)



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
amp = 0.3
vtype = "gaussian"
x0 = param.Lx/2
y0 = 2*param.Ly/3


h[0] = h0-hb
h[0] -= amp*vortex(xc, yc, **{"x0": x0-dsep, "y0": y0, "d": d, "vtype": vtype})
h[0] += amp*vortex(xc, yc, **{"x0": x0+dsep, "y0": y0, "d": d, "vtype": vtype})

# convert height "h" to a volume form, i.e. multiply with the cell area
h[0] *= area
# topography also needs to be multiplied with area (cf montgomery computation)
hb *= area

# to set initial geostropic adjustement
# define exactly the same height but at corner cells...
# trick: we use the vorticity array because it has the good shape
# this array will be overwritten with the true vorticity
# once the model is launched
hF = model.state.vor
hF[0] = h0-htopo*vortex(xe, ye, **kwargs)
#hF[0] += amp*dambreak(xe, ye, 0.5, 0.5-dsep, d)
hF[0] -= amp*vortex(xe, ye, **{"x0": x0-dsep,
                               "y0": y0, "d": d, "vtype": vtype})
hF[0] += amp*vortex(xe, ye, **{"x0": x0+dsep,
                               "y0": y0, "d": d, "vtype": vtype})


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
# f=1e1/4
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
