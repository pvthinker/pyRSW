"""
 Dam-break in the elliptical pool with rotation

See the impact of param.f0 on the deformation radius

Play with the amplitude of the initial jump

Look how the Kelvin waves propagate along the boundary

Look how the current evolves in time. It looks like a snake
whose length growth until its head eats its tail ...

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW
import geostrophy as geos

param = Param()

reso = 8
param.expname = "dambreak_elliptical"
param.nz = 1
param.npx = 1
param.ny = 25*reso
param.nx = 50*reso
param.Lx = 2.
param.Ly = 1.
param.partialcell = True
param.auto_dt = False
param.geometry = "closed"
param.cfl = 0.2
param.dt = 1./(param.nx*param.npx)
param.tend = 5.
param.plotvar = "h"
param.singlefile = True
param.freq_plot = 5
param.freq_his = 0.04
param.plot_interactive = True
param.colorscheme = "imposed"
param.cax = [0.78, 1.22]
param.f0 = 10.
param.noslip = False
param.var_to_save = ["h", "u", "pv"]


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
        # uncomment to have an island at the center
        # d2 = (xx-x0)**2 + (yy-y0)**2
        # d0 = np.sqrt(d2)
        # m[d0 < 0.1] = -1

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
          0.5, "d": param.Ly*0.5, "vtype": "cosine"}

grid.boundary = {"fbry": vortex, "kwargs": kwargs}


grid.finalize()

model = RSW(param, grid)


h = model.state.h.view("i")
hb = grid.arrays.hb.view("i")
area = grid.arrays.vol.view("i")

h0 = param.H
g = param.g
f = param.f0

# setup initial conditions

# amplitude of the initial jump
amp = 0.2


h[0] = h0
h[0] += amp*np.tanh( (xc-1.)/0.1)

# convert height "h" to a volume form, i.e. multiply with the cell area
h *= area


model.run()
