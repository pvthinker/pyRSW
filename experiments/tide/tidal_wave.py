"""Tidal wave

Play with:

  - the deformation radius (via param.f0)
  - the tidal pulsation (via omega)
  - the tide forcing wavelength (via wavelength)
  - the damping coefficient

the linear damping on momentum is an important parameter to reach a
clean oscillatory state

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW

param = Param()

reso = 2
param.expname = "tide"
param.nz = 1
param.npx = 1
param.ny = 25*reso
param.nx = 50*reso
param.Lx = 2.
param.Ly = 1.
param.auto_dt = False
param.partialcell = True
param.geometry = "closed"
param.cfl = 0.2
param.dt = 1./(param.nx*param.npx)
param.tend = 20.
param.plotvar = "h"
param.singlefile = True
param.freq_plot = 5
param.freq_his = 0.04
param.plot_interactive = True
param.colorscheme = "imposed"
param.cax = [0.9, 1.1]
param.f0 = 2.
param.noslip = False
param.var_to_save = ["h", "u", "pv"]
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
area = grid.arrays.vol.view("i")
h[0] = param.H
h *= area

# perimeter of the ellipsis
# https://www.mathsisfun.com/geometry/ellipse-perimeter.html
a = param.Lx/2
b = param.Ly/2
h = (a-b)**2/(a+b)**2
perimeter = np.pi*(a+b)*(1+3*h/(10+np.sqrt(4-3*h)))

param.perimeter = perimeter
print(f"perimeter : {perimeter:.3f}")


class Forcing():
    """

    Add a moving a tidal force (via the pressure gradient) whose
    spatial shape is a sine

    the forcing moves along x at speed wavelength/period

    """

    def __init__(self, param, grid):
        self.param = param
        self.grid = grid

        # user parameters
        amp = 2e-5
        wavelength = param.perimeter/3
        period = wavelength/1.2
        self.damping = period/2

        self.omega = 2*np.pi/period
        kx = 2*np.pi/(wavelength)
        xc = grid.xc
        self.tide = amp*np.exp(1j*kx*xc)*grid.arrays.msk.view("i")

    def add(self, state, dstate, time):
        dh = dstate.h.view("i")
        dh += np.real(self.tide*np.exp(-1j*self.omega*time))

        ux = state.ux.view("i")
        uy = state.uy.view("i")
        dux = dstate.ux.view("i")
        duy = dstate.uy.view("i")
        dux -= self.damping*ux
        duy -= self.damping*uy


model.forcing = Forcing(param, grid)


model.run()
