import numpy as np
from math import exp
from numba.pycc import CC
from numba.pycc import compiler
import numba.types as types
from numba import jit, generated_jit
from numba import sigutils, typing
from numba import decorators

# @jit
# def interp1d(q:types.float32[:], qp:types.float32[:], qm:types.float32[:]):
#     nx = len(q)
#     for i in range(nx):
#         qp[i] = exp(q[i])
#         qm[i] = exp(q[i])


# to be able to use interp1d within a compiled function
# we need to either 1) define it in this script with @jit
# or 2) to import it and jit it here

# option 2) import it and jit it here
#from interpolation import interp1d as interp1d_raw
from weno import wenoflux_edge2center
#import weno
wrapper = decorators._jit(None,
        locals={}, target="cpu", cache=False, targetoptions={})
interp1d_etoc = wrapper(wenoflux_edge2center)
#interp1d_ctoe = wrapper(weno.wenoflux_center2edge)

cc = CC("finitediff")
cc.verbose = True


# option 1) define and jit it in this script
# @jit
# def interp1d(q:types.float32[:], qp:types.float32[:], qm:types.float32[:]):
#     nx = len(q)
#     for i in range(nx):
#         qp[i] = exp(q[i])
#         qm[i] = exp(q[i])


# to have interp1d inside this module uncomment this block
# ------>
# exported_name = "interp1d"
# sig = "void(f8[:], f8[:], f8[:])"
# func = interp1d

# fn_args, fn_retty = sigutils.normalize_signature(sig)
# sig = typing.signature(fn_retty, *fn_args)


# entry = compiler.ExportEntry(exported_name, sig, func)
# cc._exported_functions[exported_name] = entry
# -------<

@cc.export("bernoulli",
           "void(f8[:, :, :], f8[:, :, :], f8[:, :, :], boolean)")
def bernoulli(ke, p, u, linear):
    shape = p.shape
    if linear:
        # linear case, only pressure, no kinetic energy
        for k in range(shape[0]):
            for j in range(shape[1]):
                for i in range(1, shape[2]):
                    u[k, j, i] += p[k, j, i]-p[k, j, i-1]
    else:
        # nonlinear case
        for k in range(shape[0]):
            for j in range(shape[1]):
                for i in range(1, shape[2]):
                    u[k, j, i] += p[k, j, i]-p[k, j, i-1]+ke[k, j, i]-ke[k, j, i-1]


@cc.export("curl",
           "void(f8[:, :, :], f8[:, :, :], i4)")
def curl(v, vort, sign):
    shape = v.shape
    if sign == 1:
        for k in range(shape[0]):
            for j in range(1, shape[1]-1):
                for i in range(1, shape[2]-1):
                    vort[k, j, i] += v[k, j, i]-v[k, j, i-1]
    elif sign == -1:
        for k in range(shape[0]):
            for j in range(1, shape[1]-1):
                for i in range(1, shape[2]-1):
                    vort[k, j, i] -= v[k, j, i]-v[k, j, i-1]


@cc.export("compke",
           "void(f8[:, :, :], f8[:, :, :], f8[:, :, :], i4)")
def compke(u, U, ke, sign):
    shape = ke.shape
    if sign == 0:
        for k in range(shape[0]):
            for j in range(shape[1]):
                ke0 = u[k, j, 0]*U[k, j, 0]
                for i in range(shape[2]):
                    ke1 = u[k, j, i+1]*U[k, j, i+1]
                    ke[k, j, i] = (ke0+ke1)/4
                    ke0 = ke1
    elif sign == 1:
        for k in range(shape[0]):
            for j in range(shape[1]):
                ke0 = u[k, j, 0]*U[k, j, 0]
                for i in range(shape[2]):
                    ke1 = u[k, j, i+1]*U[k, j, i+1]
                    ke[k, j, i] += (ke0+ke1)/4
                    ke0 = ke1


@cc.export("montgomery",
           "void(f8[:, :, :], f8[:, :], f8[:, :, :], f8, f8, f8)")
def montgomery(h, hb, p, rho0, rho1, g):
    shape = h.shape

    cff0 = rho1*g
    cff1 = rho0*g
    for k in range(shape[0]):
        for j in range(shape[1]):
            for i in range(shape[2]):
                p0 = cff1*h[1, j, i]
                p[k, j, i] = p0

    for k in range(shape[0]):
        if k == 0:
            cff = cff0
        else:
            cff = cff1
        for j in range(shape[1]):
            for i in range(shape[2]):
                z = hb[j, i]+h[0, j, i]
                p[k, j, i] = cff*z


@cc.export("vortex_force",
           "void(f8[:, :, :], f8[:, :], f8[:, :, :], f8[:, :, :], i4)")
def vortex_force(U, f, vor, dv, sign):
    """
    computes du = du + (sign) * (vor+f)*V

    1) along-x component of du

    du_x += \interp_{y}(vor+f)*\bar_x\bar_y{U_y}

    interpolation is done along y !...
    so we use the convention [k, i, j]
    
    2) along-y component of du

    du_y -= \interp_{x}(vor+f)*\bar_x\bar_y{U_x}

    interpolation is done along x !...
    so we use the convention [k, j, i]

    The code is written for case 2), thus the name
    of the functions parameters

    """
    nz, ny, nx = vor.shape
    q = np.zeros((nx,))
    Um = np.zeros((nx-1,))
    flux = np.zeros((nx-1,))    
    

    for k in range(nz):
        for j in range(1, ny-1):
            for i in range(nx):
                q[i] = vor[k,j,i]+f[j,i]

            U0 = U[k,j,0] + U[k,j+1,0]
            for i in range(1, nx):
                U1 = U[k,j,i] + U[k,j+1,i]
                Um[i] = (U0+U1)/4
                U0 = U1

            interp1d_etoc(q, Um, flux)

            if sign == 1:
                for i in range(nx-1):
                    dv[k, j, i] += flux[i]
            elif sign == -1:
                for i in range(nx-1):
                    dv[k, j, i] -= flux[i]

cc.compile()
