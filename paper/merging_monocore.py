""" Vortex merging experiment

    one layer case

    You may observe the sensitivity to the interpolation
    scheme for the vorticity flux and/or the mass flux
    
    You can use linear or weno interpolation
    You can use 1st, 3rd or 5th order
    
"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW
import geostrophy as geos

param = Param()

perfs = {}
npx = 1
for nxglo in [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]:

    param.expname = f"merging_{nxglo}_{npx}"
    param.nz = 1
    param.ny = nxglo//npx
    param.nx = nxglo//npx
    param.npx = npx
    param.npy = npx
    param.Lx = 1.
    param.Ly = 1.
    param.auto_dt = False
    param.geometry = "closed"
    param.cfl = 0.25
    param.dt = 0.5/nxglo
    param.tend = 1000*param.dt

    # the choice below was the default in the old rsw code
    # param.VF_linear = True  # default: False
    # param.MF_linear = True  # default: False
    # param.VF_order = 5  # default: 5
    # param.MF_order = 1  # default: 5

    param.plotvar = "pv"
    param.freq_plot = 20
    param.freq_his = 1000*param.dt
    param.freq_diag = 0.1
    param.plot_interactive = False
    param.timestepping = "RK3_SSP"
    param.f0 = 5.
    param.var_to_save = ["h", "vor", "pv"]


    grid = Grid(param)


    grid.finalize()

    model = RSW(param, grid)

    xc, yc = grid.xc, grid.yc
    xe, ye = grid.xe, grid.ye
    area = grid.arrays.vol.view("i")

    h = model.state.h.view("i")
    u = model.state.ux
    v = model.state.uy

    h0 = param.H
    g = param.g
    f = param.f0

    # setup initial conditions

    d = 0.07  # vortex radius
    dsep = d*1.4  # half distance between the two vortices
    # the vortex amplitude controls the Froude number
    amp = 0.2
    x0 = 0.5


    def vortex(xx, yy, x0, y0, d):
        d2 = (xx-x0)**2 + (yy-y0)**2
        return np.exp(-d2/(2*d**2))


    h[0] = h0
    h[0] += amp*vortex(xc, yc, x0, 0.5-dsep, d)
    h[0] += amp*vortex(xc, yc, x0, 0.5+dsep, d)

    # convert height "h" to a volume form, i.e. multiply with the cell area
    h[0] *= area
    # topography
    #hb[:] = 0.4*vortex(xc, yc, x0+dsep, 0.5, d)

    geos.set_balance(model)

    stats = model.run(timing=True)
    perfs[(nxglo, npx)] = stats
    print(stats)

import pickle
with open("perfs.pkl", "wb") as fid:
    pickle.dump(perfs, fid)
    
