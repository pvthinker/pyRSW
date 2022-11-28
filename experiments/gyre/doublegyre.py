"""
Double gyre experiment

the three layers case

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW


Omega = 7.292e-5
f = 9.375e-5
beta = 1.754e-11

g = 9.81
day = 86_400.

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

dx = 50e3

param.expname = "doublegyre"
param.Lx = 5_000e3
param.Ly = 4_000e3

param.ny = int(param.Ly/dx)
param.nx = int(param.Lx/dx)
param.nz = nlayers
param.geometry = "closed"

H = sum(Hlayers)
c = np.sqrt(g*H)

param.g = g
param.H = H
param.f0 = f
param.rho = rho

param.dt = 0.5*dx/c * 25
param.tend = 50*day

param.freq_his = day/2

param.timeunit = day # for the print during integration

param.timestepping = "RK3_SSP"

param.noslip = True
param.var_to_save = ["h", "vor", "pv", "u"]

param.forcing = True
param.tau0 = 0.02


def set_beta_plane(grid):
    f = grid.arrays.f.view("i")
    areav = grid.arrays.volv.view("i")
    ye = grid.ye

    f[:] = param.f0 + beta*ye
    f *= areav


grid = Grid(param)

# this is unfortunate to have to set the beta plane
# in the user script. It should be done in Grid...

set_beta_plane(grid)

grid.finalize()

model = RSW(param, grid)
h = model.state.h
area = grid.arrays.vol.view("i")

# layer thicknesses are set there !
for k in range(nlayers):
    h[k] = Hlayers[k]
    h[k] *= area


class Forcing():
    def __init__(self, param, grid, Hlayers):
        self.param = param
        self.grid = grid
        y = grid.coord.y(grid.jc, grid.ie)
        x = grid.coord.x(grid.jc, grid.ie)
        y0 = param.Ly/2.
        self.tau = param.tau0*np.cos(np.pi*(y-y0)/param.Ly)
        sigma = 100e3
        self.tau *= (np.tanh(x/sigma)+np.tanh((param.Lx-x)/sigma)-1)
        #self.tau[:, 0] = 0.
        #self.tau[:, -1] = 0.

        # the grid.dx is because tau is a 1-form
        self.tau *= grid.dx/Hlayers[0]

    def add(self, state, dstate, time):

        du = dstate.ux.view("i")

        ramp = min(1, (time/(10*86_400))**2)

        du[0] += self.tau*ramp

        # if you want bottom stress, add it in this
        # function
        # on du[-1] and dv[-1] the bottom layers


model.forcing = Forcing(param, grid, Hlayers)
model.run()
