"""
Double gyre experiment

the one layer case

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW

pi = np.pi
sin = np.sin
cos = np.cos
arccos = np.arccos

a = 6.37122e6
Omega = 7.292e-5
g = 9.860616
day = 86400.

param = Param()

reso = 2
param.expname = "gyre"
param.phi = [0, pi/2]
param.theta = [0.2*pi/2, 0.6*pi/2]
param.nz = 1
param.ny = 25*reso
param.nx = 50*reso
param.geometry = "closed"
param.coordinates = "spherical"
param.timeunit = day
param.dt = 200./reso
param.tend = 50*day  # 100*param.dt
param.plotvar = "pv"
param.freq_plot = 100
param.freq_his = day
param.plot_interactive = True
param.colorscheme = "auto"
param.timestepping = "RK3_SSP"
param.g = g
param.H = 4000.
param.Omega = Omega
param.sphere_radius = a
param.noslip = False
param.var_to_save = ["h", "vor", "pv", "u"]

param.forcing = True
param.tau0 = 0.01/1e3

grid = Grid(param)
grid.finalize()

model = RSW(param, grid)
h = model.state.h
area = grid.arrays.vol.view("i")

h[0] = param.H
h[0] *= area


class Forcing():
    def __init__(self, param, grid):
        self.param = param
        self.grid = grid
        lat = grid.coord.theta(grid.jc, grid.ie)
        deltalat = param.theta[1]-param.theta[0]
        latc = 0.5*(param.theta[1]+param.theta[0])
        self.tau = param.tau0*np.cos(np.pi*(lat-latc)/(deltalat))
        self.tau[:, 0] = 0.
        self.tau[:, -1] = 0.

        dx = 1/np.sqrt(grid.arrays.invdx.view("i"))
        self.tau *= dx

    def add(self, state, dstate, time):
        du = dstate.ux.view("i")
        du += self.tau


model.forcing = Forcing(param, grid)
model.run()
