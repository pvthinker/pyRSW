""" Unstable baroclinic jet


"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW
import geostrophy as geos

param = Param()

reso = 4
param.expname = "bc_jet"
param.nz = 2
param.rho = [1., 0.5]
param.Hlayers = [0.5, 0.5]
param.ny = 25*reso
param.nx = 50*reso
param.Lx = 2.
param.Ly = 1.
param.auto_dt = False
param.geometry = "perio_x"
param.cfl = 0.2
param.dt = 2e-2/reso
param.tend = 15  # 100*param.dt
param.plotvar = "uy"
param.freq_plot = 20
param.freq_his = .5
param.plot_interactive = True
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = False
param.linear = False
param.timestepping = "RK3_SSP"
param.f0 = 5.
param.noslip = False
param.var_to_save = ["h", "pv", "u"]


grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye


grid.finalize()


model = RSW(param, grid)


h = model.state.h
u = model.state.ux
area = grid.arrays.vol.view("i")

g = param.g
f = param.f0

# setup initial conditions

amp = 0.2
eps = 1e-3

y0 = param.Ly/2
sigma = 0.05*param.Ly

f = lambda y : np.tanh((y-y0)/sigma)
perturb = lambda x:  np.sin(x*6*np.pi/param.Lx)
noise = np.random.normal(size=xc.shape)
#noise = perturb(xc)

h[0] = param.Hlayers[0]-amp*f(yc+eps*noise)
h[1] = param.Hlayers[1]+amp*f(yc-eps*noise)

# convert height "h" to a volume form, i.e. multiply with the cell area

h[:] *= area

geos.set_balance(model)

u[:,0,:] = u[:,1,:]
u[:,-1,:] = u[:,-2,:]

model.run()
