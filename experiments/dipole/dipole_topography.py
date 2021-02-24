""" Dipole interacting with a topography

    one layer case

    For the initial state, you may either decide that

    a) the h+hb is constant over the topography (flat interface)

    b) h=H over the topography (bumped interface)

    Look at the PV evolution to understand the differences!

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW
import geostrophy as geos

param = Param()

reso = 4
param.expname = "dipole_topo"
param.nz = 1
param.ny = 25*reso
param.nx = 50*reso
param.Lx = 2.
param.Ly = 1.
param.partialcell = True
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
param.generate_mp4 = False
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

    else:
        m = np.exp(-d2/(2*d**2))
    return m


grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye

kwargs = {"ratio": 0.25, "x0": param.Lx*0.5, "y0": param.Ly *
          0.5, "d": param.Ly*0.5, "vtype": "cosine"}

# uncomment to have the elliptical domain
grid.boundary = {"fbry": vortex, "kwargs": kwargs}


grid.finalize()

model = RSW(param, grid)

hb = grid.arrays.hb.view("i")
kwargstopo = {"x0": param.Lx*0.4, "y0": param.Ly*0.4, "d": 0.1}
htopo = 0.6
hb[:] = htopo*vortex(xc, yc, **kwargstopo)

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

# Choice a) or b) uncomment
h[0] = h0-hb  # a)
# h[0] = h0  # b)
h[0] -= amp*vortex(xc, yc, **{"x0": x0-dsep, "y0": y0, "d": d, "vtype": vtype})
h[0] += amp*vortex(xc, yc, **{"x0": x0+dsep, "y0": y0, "d": d, "vtype": vtype})

# convert height "h" to a volume form, i.e. multiply with the cell area
h[0] *= area
# topography also needs to be multiplied with area (cf montgomery computation)
hb *= area

# with nite=1 we use the geostrophic balance
# with nite=2 we use a cyclogeostrophic balance (more balanced, less gravity waves)
geos.set_balance(model, nite=2)

model.run()
