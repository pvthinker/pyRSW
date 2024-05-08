""" Heton in the two layer case

    A heton is a dipole for which each vortex is in a different layer

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW
import geostrophy as geos

param = Param()

reso = 2
param.expname = "heton"
param.nz = 2
param.rho = [1., 0.9]
param.Hlayers = [0.5, 0.5]
param.ny = 25*reso
param.nx = 50*reso
param.Lx = 2.
param.Ly = 1.
param.auto_dt = False
param.geometry = "closed"
param.cfl = 0.2
param.dt = 1e-2/reso
param.tend = 10  # 100*param.dt
param.plotvar = "h"
param.freq_plot = 100
param.freq_his = 0.02
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

# uncomment to have the elliptical domain
# grid.boundary = {"fbry": vortex, "kwargs": kwargs}

kwargs = {"x0": param.Lx*0.4, "y0": param.Ly*0.4, "d": 0.1}

htopo = 0.
hb[:] = htopo*vortex(xc, yc, **kwargs)


grid.finalize()

model = RSW(param, grid)


h = model.state.h

area = grid.arrays.vol.view("i")
u = model.state.ux
v = model.state.uy

h0 = param.H*0.5
g = param.g
f = param.f0

# setup initial conditions

d = 0.1  # vortex radius
dsep = -d*1.1  # half distance between the two vortices
# the vortex amplitude controls the Froude number
amp = 0.15
vtype = "gaussian"
x0 = param.Lx/2
y0 = 2*param.Ly/3

h[0] -= amp*vortex(xc, yc, **{"x0": x0-dsep, "y0": y0, "d": d, "vtype": vtype})
h[1] += amp*vortex(xc, yc, **{"x0": x0+dsep, "y0": y0, "d": d, "vtype": vtype})

# convert height "h" to a volume form, i.e. multiply with the cell area
#h[1] = -h[0]

h[0] += h0-hb
h[1] += h0
h[:] *= area
# topography also needs to be multiplied with area (cf montgomery computation)
hb *= area

geos.set_balance(model)

model.run()
