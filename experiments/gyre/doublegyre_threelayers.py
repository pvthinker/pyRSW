"""
Double gyre experiment

the three layers case

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW, check_nan


Omega = 7.292e-5
f = 9.375e-5
beta = 1.754e-11

g = 9.81
day = 86_400.
hour = 3_600.
year = 365*day

nlayers = 3

Hlayers = [350., 750., 2900.]

gprimes = [0.025, 0.0125]


def get_rhos_from_grimes(g, gprimes):
    rhos = [1]
    rho0 = rhos[0]
    for gprime in gprimes:
        rhos.append((gprime/g)*rho0+rhos[-1])
    return rhos


rho = get_rhos_from_grimes(g, gprimes)

param = Param()

dx = 100e3

param.expname = "doublegyre"
param.Lx = 5_000e3
param.Ly = 4_000e3

param.ny = int(param.Ly/dx)
param.nx = int(param.Lx/dx)
param.Lx = param.nx*dx
param.Ly = param.ny*dx

param.nz = nlayers
param.geometry = "closed"

H = sum(Hlayers)
c = np.sqrt(g*H)

param.g = g
param.H = H
param.Hlayers = Hlayers
param.f0 = f
param.rho = rho


dt_barotropic = 0.5*dx/c

# use either a small barotropic time step
param.dt = dt_barotropic

# or a baroclinic one (that requires barotropic filtering)
param.dt = hour

param.barotropicfilter = (param.dt > dt_barotropic)

if param.barotropicfilter:
    print("activate Barotropic Filter")

param.tend = 5*year

param.freq_his = day

param.timeunit = day # for the print during integration

param.timestepping = "RK3_SSP"

param.noslip = False
param.var_to_save = ["h", "vor", "pv", "u"]

param.forcing = True
param.tau0 = 0.02/10

param.plotvar = "h"

def set_beta_plane(param, grid):
    f = grid.arrays.f.view("i")
    areav = grid.arrays.volv.view("i")
    ye = grid.ye
    y0 = param.Ly/2

    f[:] = param.f0 + beta*(ye-y0)
    f *= areav

grid = Grid(param)

# this is unfortunate to have to set the beta plane
# in the user script. It should be done in Grid...

set_beta_plane(param, grid)

grid.finalize()

model = RSW(param, grid)
h = model.state.h
area = grid.arrays.vol.view("i")

# layer thicknesses are set there !
for k in range(nlayers):
    h[k] = Hlayers[k]
    h[k] *= area


class Forcing():
    def __init__(self, param, grid, has_bottom_friction=True):
        self.param = param
        self.grid = grid
        y = grid.coord.y(grid.jc, grid.ie)
        x = grid.coord.x(grid.jc, grid.ie)
        y0 = param.Ly/2.
        self.tau = param.tau0*np.cos(np.pi*(y-y0)/param.Ly)
        self.tau[:, 0] = 0.
        self.tau[:, -1] = 0.

        # the grid.dx is because tau is a 1-form
        self.tau *= grid.dx/param.Hlayers[0]

        self.has_bottom_friction = has_bottom_friction
        self.damping_coef = 0.e-9/(hour)


    def add(self, state, dstate, time):

        du = dstate.ux.view("i")

        ramp = min(1, (time/(5*86_400)))
        ramp = 0.5-0.5*np.cos(np.pi*ramp)

        du[0] += self.tau # *ramp

        if self.has_bottom_friction:
            u = state.ux.view("i")
            v = state.uy.view("i")
            dv = dstate.uy.view("i")

            du[-1] -= u[-1]*self.damping_coef
            dv[-1] -= v[-1]*self.damping_coef


model.forcing = Forcing(param, grid)
model.run()
