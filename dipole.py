""" Dipole-Wall collision experiment

    with no-slip condition

    one layer case

    The strength of the dipole interaction strongly depends on the ratio
    Rossby deformation radius / dipole distance (which is sqrt(Burger) )
"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW

param = Param()

param.expname = "dipole"
param.nz = 1
param.ny = 50*2
param.nx = 100*2
param.Lx = 2.
param.Ly = 1.
param.auto_dt = True
param.geometry = "closed"
param.cfl = 0.2
param.dt = 1e-2/4
param.tend = 40  # 0*param.dt
param.plotvar = "h"
param.freq_plot = 50
param.freq_his = 0.5
param.plot_interactive = True
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = False
param.linear = False
param.timestepping = "RK3_SSP"
param.f0 = 5.
param.noslip = False
param.var_to_save = ["h", "pv", "u", "vor"]


def vortex(x1, y1, **kwargs):
    xx, yy = np.meshgrid(x1, y1)
    return vortex2(xx, yy, **kwargs)


def vortex2(xx, yy, **kwargs):
    """
    analytical function that defines the domain

    fmsk < 0 : solid
    fmsk == 0: boundary
    fmsk > 0 : fluid
    """
    x0 = kwargs["x0"]
    y0 = kwargs["y0"]
    d = kwargs["d"]
    vtype = kwargs["vtype"]
    if "ratio" in kwargs:
        ratio = kwargs["ratio"]
    else:
        ratio = 1.
    d2 = (xx-x0)**2*ratio + (yy-y0)**2
    if vtype == "cosine":
        d0 = np.sqrt(d2)
        m = np.cos(d0/d*(np.pi/2))
        m[d0 > d] = 0.
    else:
        m = np.exp(-d2/(2*d**2))-0.7
    return m


def fulldomain(xx, yy, **kwargs):
    return np.ones(xx.shape)


def sliced(xx, yy, **kwargs):
    return param.Ly*0.9-(yy+(xx-param.Lx/2)*.1)


grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye


def compute_area(grid, fbry, **kwargs):
    """
    fbry function fbry(x,y,**kwargs) that returns a float: >0 fluid, <=0 solid
    """
    print("Compute cell areas")
    msk = grid.arrays.msk.view("i")
    ny, nx = msk.shape
    area = grid.arrays.vol.view("i")
    areav = grid.arrays.volv.view("i")

    npergridcell = 20
    x1 = (np.arange(npergridcell)+0.5)/npergridcell*grid.dx
    y1 = (np.arange(npergridcell)+0.5)/npergridcell*grid.dy
    cff = grid.dx*grid.dy
    xx, yy = np.meshgrid(x1, y1)
    xe, ye = grid.xe, grid.ye
    for j in range(ny):
        for i in range(nx):
            if msk[j, i] == 1:
                if (i > 0 and i < nx-1 and j > 0 and j < ny-1) and (
                        msk[j-1, i]+msk[j, i-1]+msk[j+1, i]+msk[j, i+1] < 4):

                    m = fbry(xx+xe[j, i], yy+ye[j, i], **kwargs)
                    fraction = np.count_nonzero(m > 0)/npergridcell**2
                    if fraction < 0.25:
                        print(f"i, j = {i:3}, {j:3} fraction = {fraction:.2}")

                    area[j, i] = fraction*grid.dx*grid.dy
                else:
                    area[j, i] = grid.dx*grid.dy
            else:
                area[j, i] = grid.dx*grid.dy

    print("Compute areas at corners")
    npergridcell = 20
    x1 = (np.arange(npergridcell//2)+0.5)/npergridcell*grid.dx
    y1 = (np.arange(npergridcell//2)+0.5)/npergridcell*grid.dy
    xx, yy = np.meshgrid(x1, y1)
    xc, yc = grid.xc, grid.yc
    areav[:] = grid.dx*grid.dy
    cff = grid.dx*grid.dy/npergridcell**2
    for j in range(ny):
        for i in range(nx):
            if msk[j, i] == 1:
                if (i > 0 and i < nx-1 and j > 0 and j < ny-1) and (
                        msk[j-1, i]+msk[j, i-1]+msk[j+1, i]+msk[j, i+1] < 4):
                    m = fbry(xx+xe[j, i], yy+ye[j, i], **kwargs)
                    areav[j, i] = np.count_nonzero(m > 0)*cff
                    m = fbry(xx+xc[j, i], yy+ye[j, i], **kwargs)
                    areav[j, i+1] = np.count_nonzero(m > 0)*cff
                    m = fbry(xx+xe[j, i], yy+yc[j, i], **kwargs)
                    areav[j+1, i] = np.count_nonzero(m > 0)*cff
                    m = fbry(xx+xc[j, i], yy+yc[j, i], **kwargs)
                    areav[j+1, i+1] = np.count_nonzero(m > 0)*cff
                else:
                    areav[j, i] = grid.dx*grid.dy
            else:
                areav[j, i] = grid.dx*grid.dy

    areav[:] = 0.
    for j in range(ny):
        for i in range(nx):
            #area[j,i] = 0.25*(areav[j,i]+areav[j+1,i]+areav[j,i+1]+areav[j+1,i+1])
            a = 0.25*area[j, i]
            areav[j, i] += a
            areav[j, i+1] += a
            areav[j+1, i] += a
            areav[j+1, i+1] += a


def compute_lame(grid, direc, fbry, **kwargs):
    print(f"Compute Lame coefficient in direction {direc}")
    assert direc in "ij"
    msk = grid.arrays.msk.view(direc)
    ny, nx = msk.shape
    area = grid.arrays.vol.view(direc)
    npergridcell = 50
    if direc == "i":
        dy = grid.dy
        idx2 = grid.arrays.invdx.view(direc)
        idx2[:] = 1/grid.dx**2
        x1 = np.zeros((npergridcell))
        y1 = (np.arange(npergridcell)+0.5)/npergridcell*grid.dy
    else:
        dy = grid.dx
        idx2 = grid.arrays.invdy.view(direc)
        idx2[:] = 1/grid.dy**2
        y1 = np.zeros((npergridcell))
        x1 = (np.arange(npergridcell)+0.5)/npergridcell*grid.dx

    cff = 1/npergridcell

    xe = grid.xe
    ye = grid.ye

    for j in range(ny):
        for i in range(1, nx):
            if msk[j, i-1]+msk[j, i] == 2:
                if direc == "i":
                    xx, yy = xe[j, i]+x1, ye[j, i]+y1
                else:
                    xx, yy = xe[i, j]+x1, ye[i, j]+y1
                m = fbry(xx, yy, **kwargs)
                fraction = np.count_nonzero(m > 0)*cff
                idx2[j, i] = (dy*fraction)**2/(area[j, i-1]*area[j, i])


kwargs = {"ratio": 0.25, "x0": param.Lx*0.5,
          "y0": param.Ly*0.5, "d": param.Ly*0.5, "vtype": "cosine"}


def set_mask(grid, fbry, **kwargs):
    xx, yy = grid.xc, grid.yc  # np.meshgrid(grid.xc, grid.yc)
    m = fbry(xx, yy, **kwargs)
    msk = grid.arrays.msk.view("i")
    msk[:] = m > 0

#m = vortex((xc-param.Lx/2)*.9, yc-0.2, 0., param.Ly/2, param.Lx/2)
#m = vortex(xc, yc, **kwargs)
#msk[:] = m > 0

#msk[ny//3:ny//3+4,nx//3:2*nx//3] = 0
# msk[-1,:] = 0
# msk[:,-2:] = 0
# msk[:,:2] = 0


#msk[-2:,:] = 0
# msk[0,:] = 0
fbry = sliced  # vortex2#fulldomain
xc, yc = grid.xc, grid.yc
xx, yy = np.meshgrid(xc, yc)
grid.xc = xx
grid.yc = yy

xe, ye = grid.xe, grid.ye
ny = len(ye)
nx = len(xe)
xx, yy = np.meshgrid(xe, ye)
grid.xe = xx
grid.ye = yy
print(grid.xe.shape, grid.ye.shape)
if True:
    set_mask(grid, fbry, **kwargs)
    compute_area(grid, fbry, **kwargs)
    compute_lame(grid, "i", fbry, **kwargs)
    compute_lame(grid, "j", fbry, **kwargs)

else:
    area = grid.arrays.vol.view("i")
    area[:] = grid.dx*grid.dy
    area = grid.arrays.volv.view("i")
    area[:] = grid.dx*grid.dy

    idx2 = grid.arrays.invdx.view("i")
    idx2[:] = 1/grid.dx**2
    idy2 = grid.arrays.invdy.view("i")
    idy2[:] = 1/grid.dy**2

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
amp = -0.2
vtype = "gaussian"
x0 = param.Lx/2
y0 = 2*param.Ly/3


h[0] = h0
# h[0] -= amp*vortex(xc, yc, x0-dsep, y0, d)
# h[0] += amp*vortex(xc, yc, x0+dsep, y0, d)
h[0] -= amp*vortex(xc, yc, **{"x0": x0-dsep, "y0": y0, "d": d, "vtype": vtype})
h[0] += amp*vortex(xc, yc, **{"x0": x0+dsep, "y0": y0, "d": d, "vtype": vtype})

# convert height "h" to a volume form, i.e. multiply with the cell area
h[0] *= area

# for k in range(param.nz):
#     h[k] *= msk

# to set initial geostropic adjustement
# define exactly the same height but at corner cells...
# trick: we use the vorticity array because it has the good shape
# this array will be overwritten with the true vorticity
# once the model is launched
hF = model.state.vor
hF[0] = h0
#hF[0] += amp*dambreak(xe, ye, 0.5, 0.5-dsep, d)
hF[0] -= amp*vortex(xe, ye, **{"x0": x0-dsep,
                               "y0": y0, "d": d, "vtype": vtype})
hF[0] += amp*vortex(xe, ye, **{"x0": x0+dsep,
                               "y0": y0, "d": d, "vtype": vtype})


def grad(phi, dphidx, dphidy):
    phi.setview("i")
    dphidx.setview("i")
    dphidx[:] = phi[..., 1:]-phi[..., :-1]
    phi.setview("j")
    dphidy.setview("j")
    dphidy[:] = phi[..., 1:]-phi[..., :-1]


idx2 = grid.arrays.invdx.view("j")
idy2 = grid.arrays.invdy.view("i")

# dx = 1/idx2
# dx[idx2==0]=0.

# dy = 1/idy2
# dy[idy2==0]=0.


u[:] = 0.
v[:] = 0.
# then take the rotated gradient of it
grad(hF, v, u)
u[:] *= -(g/f)
v[:] *= +(g/f)


u = model.state.ux.view("i")
v = model.state.uy.view("j")

msku = grid.msku()
mskv = grid.mskv()

for k in range(param.nz):
    u[k] *= msku
    v[k] *= mskv

hF[:] = 0.
model.run()
