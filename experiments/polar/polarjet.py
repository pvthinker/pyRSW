"""
Polar jet

North Pole is set at theta=0 and phi=0, with param.lat_pole_shift

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
param.expname = "polar"
param.phi = [-pi/4, pi/4]
param.theta = [-pi/4, pi/4]
param.nz = 1
param.ny = 50*reso
param.nx = 50*reso
param.geometry = "closed"
param.coordinates = "spherical"
param.lat_pole_shift = -pi/2.
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
param.u0 = 0.2
param.r0 = 0.95
param.amp = 0.05
param.sigma = 0.05
param.Omega = Omega
param.sphere_radius = a
param.noslip = False
param.var_to_save = ["h", "vor", "pv", "u"]


grid = Grid(param)
ic, jc = grid.ic, grid.jc
ie, je = grid.ie, grid.je
grid.finalize()

model = RSW(param, grid)
h = model.state.h
u = model.state.ux
v = model.state.uy
area = grid.arrays.vol.view("i")

alpha = param.lat_pole_shift
h0 = param.H
u0 = param.u0

def hprofile(r, r0,sigma):
    """ gaussian profile
    
    r is a radial coordinate from the domain center
    
    watch out r = 1 at the origin and decreases with the distance
    """
    return np.exp( -(r-r0)**2/(2*sigma**2))

theta = grid.coord.theta(jc, ic)
lambd = grid.coord.phi(jc, ic)
r = cos(lambd)*cos(theta)

h[0] = h0*(1+param.amp*hprofile(r, param.r0, param.sigma))
h[0] *= area

theta = grid.coord.theta(je, ie)
lambd = grid.coord.phi(je, ie)
r = cos(lambd)*cos(theta)
hF = model.state.vor
hF[0] = h0*(1+param.amp*hprofile(r, param.r0, param.sigma))

def grad(phi, dphidx, dphidy):
    phi.setview("i")
    dphidx.setview("i")
    dphidx[:] = phi[..., 1:]-phi[..., :-1]
    phi.setview("j")
    dphidy.setview("j")
    dphidy[:] = phi[..., 1:]-phi[..., :-1]


u[:] = 0.
v[:] = 0.
# then take the rotated gradient of it
grad(hF, v, u)
theta = grid.coord.theta(jc, ie)
lambd = grid.coord.phi(jc, ie)
r = cos(lambd)*cos(theta)
u = model.state.ux.view("i")
u[0] *= -g/(param.Omega*2*r)

theta = grid.coord.theta(je, ic)
lambd = grid.coord.phi(je, ic)
r = cos(lambd)*cos(theta)
u = model.state.uy.view("i")
v[0] *= +g/(param.Omega*2*r)


u = model.state.ux.view("i")
v = model.state.uy.view("j")

msk = grid.arrays.msk.view("i")
h = model.state.h.view("i")
msku = grid.msku()
mskv = grid.mskv()

for k in range(param.nz):
    u[k] *= msku
    v[k] *= mskv
    h[k][msk == 0] = param.H*area[msk == 0]

hF[:] = 0.
model.run()
