"""
 Testcase 5 of  Williamson et al. 1992

  onal Flow Over an Isolated Mountain

  theta : latitude
  lambd : longitude

  here with alpha = 0

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
R = a/3
u0 = 2*pi*a/(12*day)

h0 = 5400.

lambdac, thetac = 3*pi/2, pi/6

alpha = 0.

param = Param()

reso = 4
param.expname = "test5"
param.phi = [0, 2*pi]
param.theta = [-pi/6, 0.8*pi/2]
param.nz = 1
param.ny = 25*reso
param.nx = 50*reso
param.geometry = "perio_x"
param.coordinates = "spherical"
param.timeunit = day
param.dt = day/100/reso
param.tend = 15*day  # 100*param.dt
param.plotvar = "pv"
param.freq_plot = 100
param.freq_his = day/4
param.plot_interactive = True
param.colorscheme = "auto"
param.timestepping = "RK3_SSP"
param.g = g
param.Omega = Omega
param.sphere_radius = a
param.noslip = False
param.var_to_save = ["h", "vor", "pv", "u"]

grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye

ic, jc = grid.ic, grid.jc
ie, je = grid.ie, grid.je

grid.finalize()

model = RSW(param, grid)


h = model.state.h
area = grid.arrays.vol.view("i")
hb = grid.arrays.hb.view("i")
u = model.state.ux

hs0 = 2000.
R = pi/9.
theta = grid.coord.theta(jc, ic)
lambd = grid.coord.phi(jc, ic)
r = np.sqrt((lambd-lambdac)**2+(theta-thetac)**2)
r[r > R] = R
hb[:] = hs0*(1-r/R)
hb *= area

theta = grid.coord.theta(jc, ie)
lambd = grid.coord.phi(jc, ie)
u[0] = u0*(cos(theta)*cos(alpha) + sin(theta)*cos(lambd)*sin(alpha))

dx = 1/np.sqrt(grid.arrays.invdx.view("i"))
u[0] *= dx


theta = grid.coord.theta(jc, ic)
lambd = grid.coord.phi(jc, ic)
h[0] = h0-(1/g)*(a*Omega*u0+u0**2/2)*(-cos(lambd) *
                                      cos(theta)*sin(alpha)+sin(theta)*cos(alpha))**2

h[0] *= area

model.run()
