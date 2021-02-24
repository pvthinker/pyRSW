""" Dipole-Wall collision experiment

    one layer case, inviscid

    submesoscale regime (Rd=10)

    Rossby number = 10 !

    The initial velocity is set in geostrophic balance, but
    given the Rossby number, it's far from being well balanced
    therefore gravity waves are excited at t=0
   
"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW
import geostrophy as geos

param = Param()

reso = 2
param.expname = "dipole"
param.nz = 1
param.ny = 25*reso
param.nx = 50*reso
param.Lx = 2.
param.Ly = 1.
param.partialcell = True
param.auto_dt = False
param.geometry = "closed"
param.cfl = 0.2
param.dt = 1e-2/reso
param.tend = 10.
param.plotvar = "h"
param.freq_plot = 100
param.freq_his = 0.05
param.freq_diag = 0.01
param.plot_interactive = True
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = False
param.linear = False
param.timestepping = "RK3_SSP"
param.f0 = 1e-1
param.noslip = False
param.var_to_save = ["h", "vor", "pv", "u"]


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
    elif vtype == "tanh":
        sigma = kwargs["sigma"]
        d0 = np.sqrt(d2)
        m = 0.5-0.5*np.tanh( (d-d0)/sigma)
    else:
        m = np.exp(-d2/(2*d**2))
    return m


def fulldomain(xx, yy, **kwargs):
    return np.ones(xx.shape)


def sliced(xx, yy, **kwargs):
    return param.Ly*0.9-(yy-(xx-param.Lx/2)*.5)


grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye


kwargs = {"ratio": 0.25, "x0": param.Lx*0.5, "y0": param.Ly *
          0.5, "d": param.Ly*0.5-grid.dy, "vtype": "cosine"}

grid.boundary = {"fbry": vortex, "kwargs": kwargs}

# for topography
kwargs = {"ratio": 0.25, "x0": param.Lx*0.5, "y0": param.Ly *
          0.5, "d": param.Ly*0.5-grid.dy, "sigma": 0.02, "vtype": "tanh"}


grid.finalize()

model = RSW(param, grid)


h = model.state.h
hb = grid.arrays.hb.view("i")
area = grid.arrays.vol.view("i")
u = model.state.ux
v = model.state.uy

h0 = param.H
g = param.g


htopo = 0.8
#hb[:] = htopo*vortex(xc,yc,**kwargs)

# setup initial conditions

d = 0.12  # vortex radius
dsep = -d*1.1  # half distance between the two vortices
# the vortex amplitude controls the Froude number
amp = -0.006

vtype = "gaussian"
x0 = param.Lx/2
y0 = param.Ly/2


# h[0] -= amp*vortex(xc, yc, x0-dsep, y0, d)
# h[0] += amp*vortex(xc, yc, x0+dsep, y0, d)
h[0] -= amp*vortex(xc, yc, **{"x0": x0-dsep, "y0": y0, "d": d, "vtype": vtype})
h[0] += amp*vortex(xc, yc, **{"x0": x0+dsep, "y0": y0, "d": d, "vtype": vtype})

h[0] += h0-hb
# convert height "h" to a volume form, i.e. multiply with the cell area
h[0] *= area
hb *= area

# set the geostrophic balance (et voilà!)
geos.set_balance(model)
    
model.run()
