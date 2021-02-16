""" Annulus geometry

    one layer case

"""
import numpy as np
from parameters import Param
from grid import Grid
from rsw import RSW

param = Param()

param.expname = "annulus"
param.nz = 1
param.ny = 64*2
param.nx = 128*2
param.Lx = 2.
param.Ly = 1.
param.auto_dt = False
param.geometry = "perio_x"
param.cfl = 0.2
param.dt = 1e-2/2
param.tend = 20#0*param.dt
param.plotvar = "h"
param.freq_plot = 10
param.freq_his = 0.1
param.plot_interactive = True
param.colorscheme = "auto"
param.cax = np.asarray([-2e-4, 12e-4])/2
param.generate_mp4 = False
param.linear = False
param.timestepping = "RK3_SSP"
param.f0 = 5.
param.noslip = False
param.var_to_save = ["h", "pv"]


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
        m[d0>d] = 0.
    else:
        m = np.exp(-d2/(2*d**2))-0.7
    return m

grid = Grid(param)

xc, yc = grid.xc, grid.yc
xe, ye = grid.xe, grid.ye


def compute_area(grid, **kwargs):
    print("Compute cell areas")
    msk = grid.arrays.msk.view("i")
    ny, nx = msk.shape
    area = grid.arrays.vol.view("i")
    areav = grid.arrays.volv.view("i")
    
    npergridcell = 20
    x1 = (np.arange(npergridcell)+0.5)/npergridcell*grid.dx
    y1 = (np.arange(npergridcell)+0.5)/npergridcell*grid.dy
    cff = grid.dx*grid.dy/npergridcell**2
    xx, yy = np.meshgrid(x1, y1)
    xe, ye = grid.xe, grid.ye
    for j in range(ny):
        for i in range(nx):
            if msk[j,i] == 1:
                if (i>0 and i<nx-1 and j>0 and j<ny-1) and (
                    msk[j-1,i]+msk[j,i-1]+msk[j+1,i]+msk[j,i+1] < 4):
                        
                    m = vortex2(xx+xe[i], yy+ye[j], **kwargs)
                    fraction = np.count_nonzero(m>0)
                    if fraction <0.25:
                        print(f"i, j = {i:3}, {j:3} fraction = {fraction:.2}")

                    area[j,i] = fraction*cff
                else:
                    area[j,i] = grid.dx*grid.dy
            else:
                area[j,i] = grid.dx*grid.dy

    print("Compute areas at corners")
    npergridcell = 20
    x1 = (np.arange(npergridcell//2)+0.5)/npergridcell*grid.dx
    y1 = (np.arange(npergridcell//2)+0.5)/npergridcell*grid.dy
    xx, yy = np.meshgrid(x1, y1)
    xc, yc = grid.xc, grid.yc
    for j in range(ny):
        for i in range(nx):
            if msk[j,i] == 1:
                if (i>0 and i<nx-1 and j>0 and j<ny-1) and (
                    msk[j-1,i]+msk[j,i-1]+msk[j+1,i]+msk[j,i+1] < 4):
                    m = vortex2(xx+xe[i], yy+ye[j], **kwargs)
                    areav[j,i] = np.count_nonzero(m>0)*cff
                    m = vortex2(xx+xc[i], yy+ye[j], **kwargs)
                    areav[j,i+1] = np.count_nonzero(m>0)*cff
                    m = vortex2(xx+xe[i], yy+yc[j], **kwargs)
                    areav[j+1,i] = np.count_nonzero(m>0)*cff
                    m = vortex2(xx+xc[i], yy+yc[j], **kwargs)
                    areav[j+1,i+1] = np.count_nonzero(m>0)*cff
                else:
                    areav[j,i] = grid.dx*grid.dy
            else:
                areav[j,i] = grid.dx*grid.dy

def compute_lame(grid, direc, **kwargs):
    print(f"Compute Lame coefficient in direction {direc}")    
    assert direc in "ij"
    msk = grid.arrays.msk.view(direc)
    ny, nx = msk.shape
    area = grid.arrays.vol.view(direc)
    npergridcell = 50
    if direc == "i":
        dy = grid.dy
        idx2 = grid.arrays.invdx.view(direc)
        x1 = 0.
        y1 = (np.arange(npergridcell)+0.5)/npergridcell*grid.dy
    else:
        dy = grid.dx
        idx2 = grid.arrays.invdy.view(direc)
        y1 = 0.
        x1 = (np.arange(npergridcell)+0.5)/npergridcell*grid.dx

    cff = dy/npergridcell

    xe = grid.xe
    ye = grid.ye

    for j in range(ny):
        for i in range(1, nx):
            if msk[j,i-1]+msk[j,i] == 2:
                if direc == "i":
                    xx, yy = xe[i]+x1, ye[j]+y1
                else:
                    xx, yy = xe[j]+x1, ye[i]+y1
                m = vortex2(xx, yy, **kwargs)
                dy0 = np.count_nonzero(m>0)*cff
                idx2[j,i] = dy0**2/(area[j,i-1]*area[j,i])
                
msk = grid.arrays.msk.view("i")
ny, nx = param.ny, param.nx
kwargs = {"ratio": 0.25, "x0":param.Lx*0.5, "y0":param.Ly*0.5, "d":param.Ly*0.5, "vtype": "cosine"}

area = grid.arrays.vol.view("i")
idx2 = grid.arrays.invdx.view("i")
idy2 = grid.arrays.invdy.view("i")


def get_coord(stagg=""):
    if "y" in stagg:
        r = (np.arange(ny+1))*dr+r0
    else:
        r = (np.arange(ny)+0.5)*dr+r0

    if "x" in stagg:
        t = np.arange(nx+1)*dt
    else:
        t = (np.arange(nx)+0.5)*dt

    tt, rr = np.meshgrid(t, r)
    return tt, rr#rr*np.cos(tt), rr*np.sin(tt)

r0 = 0.5
r1 = 2.
dr = (r1-r0)/param.ny
dt = 2*np.pi/param.nx



ny, nx = area.shape
yc= np.ones((ny,))
xc = np.ones((nx,))
ye= np.ones((ny+1,))
xe = np.ones((nx+1,))
r = (np.arange(ny)+0.5)*dr+r0
area[:] = (r*dr*dt)[:, np.newaxis] * xc[np.newaxis, :]
idx2[:] = ( (r*dt)[:,np.newaxis] * xe[np.newaxis, :])**-2
idy2[:] = ( (dr*ye)[:,np.newaxis] * xc[np.newaxis, :])**-2

tc, rc = get_coord()
grid.xc = rc*np.cos(tc)
grid.yc = rc*np.sin(tc)

te, re = get_coord("xy")
grid.xe = re*np.cos(te)
grid.ye = re*np.sin(te)


#m = vortex((xc-param.Lx/2)*.9, yc-0.2, 0., param.Ly/2, param.Lx/2)
#m = vortex(xc, yc, **kwargs)
#msk[:] = m > 0
#msk[-1,:] = 0
# msk[:,-2:] = 0
# msk[:,:2] = 0

#msk[-2:,:] = 0
# msk[0,:] = 0
#compute_area(grid, **kwargs)
#compute_lame(grid, "i", **kwargs)
#compute_lame(grid, "j", **kwargs)

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
amp = 0.2
vtype = "gaussian"
x0 = param.Lx/2
y0 = param.Ly/2

y0 = 1.25
sigma = 0.08

tc, rc = get_coord()
h[0] = h0+amp*(np.tanh( (rc-y0+0.5*np.cos(tc*3))/sigma)-1)*0.5


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
te, re = get_coord("xy")
hF[0] = h0+amp*(np.tanh( (re-y0+0.5*np.cos(te*3))/sigma)-1)*0.5

def grad(phi, dphidx, dphidy):
    phi.setview("i")
    dphidx.setview("i")
    dphidx[:] = phi[..., 1:]-phi[..., :-1]
    phi.setview("j")
    dphidy.setview("j")
    dphidy[:] = phi[..., 1:]-phi[..., :-1]

idx2 = grid.arrays.invdx.view("j")
idy2 = grid.arrays.invdy.view("i")
u[:] = 0.
v[:] = 0.
# then take the rotated gradient of it
grad(hF, v, u)
u[:] *= -(g/f)#/idx2**0.5
v[:] *= +(g/f)#/idy2**0.5


u = model.state.ux.view("i")
v = model.state.uy.view("j")

msku = grid.msku()
mskv = grid.mskv()

for k in range(param.nz):
    u[k] *= msku
    v[k] *= mskv

hF[:] = 0.
model.run()
