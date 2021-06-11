import finitediff as fd
import numpy as np
from timing import timeit

def comppv(state, grid):
    vor = state.vor.view("i")
    f = grid.arrays.f.view("i")
    h = state.h.view("i")
    pv = state.pv.view("i")
    fd.comppv(vor, f, h, pv)


def comppv_c(state, grid):
    vor = state.vor.view("i")
    f = grid.arrays.f.view("i")
    h = state.h.view("i")
    pv = state.pv.view("i")
    k0, k1, j0, j1, i0, i1 = state.h.domainindices
    fd.comppv_c(vor, f, h, pv, j0, j1, i0, i1)

@timeit
def vorticity(state, grid, noslip):
    if True:
        if noslip:
            m0 = 1
        else:
            m0 = 4
        # + dv/di
        v = state.uy.view("i")
        vor = state.vor.view("i")
        mskv = grid.arrays.mskv.view("i")
        fd.curl(v, vor, mskv, 1, m0)

        # - du/dj
        u = state.ux.view("j")
        vor = state.vor.view("j")
        mskv = grid.arrays.mskv.view("j")
        fd.curl(u, vor, mskv, -1, m0)
    else:
        u = state.ux.view("i")
        v = state.uy.view("i")
        vor = state.vor.view("i")
        mskv = grid.arrays.mskv.view("i")
        fd.totalcurl(u, v, vor, mskv)


@timeit
def kinenergy(state, param):
    u = state.u["i"].view("i")
    U = state.U["i"].view("i")
    ke = state.ke.view("i")

    # ke[:] = u*U/2
    fd.compke(u, U, ke, 0)

    v = state.u["j"].view("j")
    V = state.U["j"].view("j")
    ke = state.ke.view("j")

    # ke += v*V/2
    fd.compke(v, V, ke, 1)


@timeit
def montgomery(state, grid, param):
    rho = param["rho"]
    g = param["g"]

    h = state.h.view("i")
    p = state.p.view("i")

    hb = grid.arrays.hb.view("i")
    area = grid.arrays.vol.view("i")
    msk = grid.arrays.msk.view("i")

    fd.montgomery(h, hb, p, area, rho, g, msk)

# def montgomery(state, param):
#     #rho0 = param["rho"]
#     g = param["g"]
#     h = state.h.view("i")
#     hb = state.hb.view("i")
#     p = state.p.view("i")
#     p[:] = g*(h+hb)


@timeit
def vortex_force(state, dstate, param, grid):
    du = dstate.u["i"].view("j")
    V = state.U["j"].view("j")
    vor = state.vor.view("j")
    f = grid.arrays.f.view("j")
    # du/dt += (vor+f) * V
    order = grid.arrays.vpordery.view("j")
    nx = vor.shape[2]
    q = np.zeros((nx,))
    fd.vortex_force(V, f, vor, du, q, order, +1, param.VF_linear)

    dv = dstate.u["j"].view("i")
    U = state.U["i"].view("i")
    vor = state.vor.view("i")
    f = grid.arrays.f.view("i")
    # dv/dt -= (vor+f) * U
    order = grid.arrays.vporderx.view("i")
    nx = vor.shape[2]
    q = np.zeros((nx,))
    fd.vortex_force(U, f, vor, dv, q, order, -1, param.VF_linear)


@timeit
def bernoulli(state, dstate, param, grid):
    du = dstate.u["i"].view("i")
    p = state.p.view("i")
    ke = state.ke.view("i")
    msk = grid.arrays.msk.view("i")
    fd.bernoulli(ke, p, du, msk, param.linear)

    dv = dstate.u["j"].view("j")
    p = state.p.view("j")
    ke = state.ke.view("j")
    msk = grid.arrays.msk.view("j")
    fd.bernoulli(ke, p, dv,  msk, param.linear)
