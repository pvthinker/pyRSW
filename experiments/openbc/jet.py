""" Jet with open boundary conditions

    The jet passes near a seamount that triggers
    a down-wind instability

    In this well balanced QG regime (Bu=1, Ro=0.1),
    the open boundaries behave quite well. PV is fluxed out
    without too much bouncing back on the boundary

    Open boundary conditions are treated with a forcing
    that acts in the halo. To force a halo

    param.geometry = "perio_x"

    To avoid copy the halo 

    param.openbc = True
   
"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW
from numba import jit
import geostrophy

param = Param()

reso = 2
param.expname = "jet"
param.nz = 1
param.ny = 25*reso
param.nx = 100*reso
param.Lx = 4.
param.Ly = 1.
param.partialcell = False
param.auto_dt = False
param.geometry = "perio_x"
param.openbc = True
param.cfl = 0.2
param.dt = 1e-2*2/reso
param.tend = 180  # 100*param.dt
param.plotvar = "pv"
param.freq_plot = 100
param.freq_his = 1.
param.plot_interactive = True
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = False
param.linear = False
param.timestepping = "RK3_SSP"
param.f0 = 10.
param.noslip = False
param.var_to_save = ["h", "vor", "pv", "u"]
param.forcing = True


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
        m = np.exp(-d2/(2*d**2))-0.7
    return m


grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye


hb = grid.arrays.hb.view("i")
kwargs = {"x0": 0.8, "y0": 0.4, "d": 0.1}
htopo = 0.05
hb[:] = htopo*vortex(xc, yc, **kwargs)


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
dsep = -d*1.1  # half distance between the two vortices
# the vortex amplitude controls the Froude number
amp = +0.1
vtype = "gaussian"
x0 = param.Lx/2
y0 = param.Ly/2


h[0] = h0-hb[:]
h[0] += amp*np.tanh(-(yc+0e-2*np.cos(xc*np.pi*2)-y0)/d)

# convert height "h" to a volume form, i.e. multiply with the cell area
h[0] *= area
hb[:] *= area

geostrophy.set_balance(model)


class Forcing():
    def __init__(self, param, grid, state):
        self.param = param
        self.grid = grid
        nh = param.nh
        # copy the initial state on the left into buffers
        u = state.ux.view("i")
        self.u = u[..., nh+1].copy()
        v = state.uy.view("i")
        self.v = v[..., nh].copy()
        h = state.h.view("i")
        self.h = h[..., nh].copy()

    def add(self, state, dstate, time):
        nh = self.param.nh
        u = state.ux.view("i")
        du = dstate.ux.view("i")
        for j in range(u.shape[1]):
            # on the left: copy the buffer
            u[..., j, :nh+1] = self.u[..., j]
            # on the right: copy the rightmost point
            u[..., j, -nh:] = u[..., j, -nh-1]
            # set the tendency to zero
            du[..., j, :nh+1] = 0.
            du[..., j, -nh:] = 0.

        v = state.uy.view("i")
        dv = dstate.uy.view("i")
        for j in range(v.shape[1]):
            v[..., j, :nh] = self.v[..., j]
            v[..., j, -nh:] = v[..., j, -nh-1]
            dv[..., j, :nh] = 0.
            dv[..., j, -nh:] = 0.

        h = state.h.view("i")
        dh = dstate.h.view("i")
        for j in range(h.shape[1]):
            h[..., j, :nh] = self.h[..., j]
            h[..., j, -nh:] = h[..., j, -nh-1]
            dh[..., j, :nh] = 0.
            dh[..., j, -nh:] = 0.


model.forcing = Forcing(param, grid, model.state)

model.run()
