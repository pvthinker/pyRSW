"""
Geostrophic adjustement of a circular shaped pertubation

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW
import geostrophy as geos

param = Param()

reso = 4
param.expname = "geos_adj"
param.nz = 1
param.ny = 25*reso
param.nx = 50*reso
param.Lx = 2.
param.Ly = 1.
param.auto_dt = False
param.geometry = "closed"
param.cfl = 0.2
param.dt = 1e-2*2/reso
param.tend = 2
param.plotvar = "h"
param.freq_plot = 8
param.freq_his = 0.05
param.plot_interactive = True
param.colorscheme = "imposed"
param.cmap = "RdBu_r"
param.generate_mp4 = True
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

domainshapes = [fulldomain, sliced, vortex]
domainshape = fulldomain
assert domainshape in domainshapes

grid.boundary = {"fbry": domainshape, "kwargs": kwargs}


grid.finalize()

model = RSW(param, grid)


h = model.state.h
hb = grid.arrays.hb
area = grid.arrays.vol.view("i")
u = model.state.ux
v = model.state.uy

h0 = param.H
g = param.g
f = param.f0

# setup initial conditions

d = 0.1  # vortex radius
dsep = -d*1.1  # half distance between the two vortices / dipole
# the vortex amplitude controls the Froude number
amp = -0.5
vtype = "gaussian"

configs = ["onevortex", "dipole"]
config = "onevortex"

assert config in configs

param.cax = np.asarray([1-abs(amp), 1+abs(amp)])


if config == "onevortex":
    x0 = param.Lx/2
    y0 = param.Ly/2

    h[0] = h0
    h[0] += amp*vortex(xc, yc, **{"x0": x0, "y0": y0, "d": d, "vtype": vtype})

elif config == "dipole":
    x0 = param.Lx/2
    y0 = 2*param.Ly/3


    h[0] = h0
    h[0] -= amp*vortex(xc, yc, **{"x0": x0-dsep, "y0": y0, "d": d, "vtype": vtype})
    h[0] += amp*vortex(xc, yc, **{"x0": x0+dsep, "y0": y0, "d": d, "vtype": vtype})

# convert height "h" to a volume form, i.e. multiply with the cell area
h[0] *= area


# uncomment to start from a geostrophic state
# use nite = 2 to initialize a cyclogeostrophic state
# geos.set_balance(model, nite=1)

model.run()
