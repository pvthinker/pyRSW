import numpy as np
from math import exp
from numba.pycc import CC
from numba.pycc import compiler
#import numba.types as types
#from numba import jit, generated_jit
#from numba import sigutils, typing
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

def compile(verbose=False):
    print("Compile the finite difference module with numba")

    #from weno import wenoflux_edge2center
    from weno import weno5 as w5
    from weno import weno3 as w3
    from weno import wenoflux_center2edge
    from weno import linearflux_center2edge
    from weno import linearflux_edge2center
    #import weno
    wrapper = decorators._jit(None,
                              locals={}, target="cpu", cache=False, targetoptions={})
    # interp1d_etoc = wrapper(wenoflux_edge2center)
    weno5 = wrapper(w5)
    weno3 = wrapper(w3)
    # interp1d_ctoe = wrapper(wenoflux_center2edge)
    # interp1d_etoc_linear = wrapper(linearflux_edge2center)
    # interp1d_ctoe_linear = wrapper(linearflux_center2edge)

    #from wenos import weno5

    cc = CC("finitediff")
    cc.verbose = verbose

    # @cc.export("interp1d_etoc",
    #            "void(f8[:], f8[:], f8[:], i8[:], i8[:])")
    # def wenoflux_edge2center(q, U, flux, porder, morder):
    #     """ compute flux = U * q

    #     staggering:

    #     center: U, flux
    #     edge: q

    #     U and flux have the same shape
    #     q has one more element
    #     q[i] is on the *left* of U[i]
    #     """
    #     n = U.shape[0]

    #     # porder = np.zeros((n,), dtype=int)
    #     # morder = np.zeros((n,), dtype=int)

    #     porder[0] = 1
    #     porder[1] = 3
    #     porder[2:-1] = 5
    #     porder[-1] = 3

    #     morder[0] = 3
    #     morder[1:-2] = 5
    #     morder[-2] = 3
    #     morder[-1] = 1

    #     # third order linear
    #     c1 = -1./6.
    #     c2 = 5./6.
    #     c3 = 2./6.
    #     #porder = 5
    #     #morder = 5
    #     qi = 0.
    #     for k in range(n):
    #         if U[k] > 0:
    #             if porder[k] == 5:
    #                 #qi = d1*q[k-2]+d2*q[k-1]+d3*q[k]+d4*q[k+1]+d5*q[k+2]

    #                 qi = weno5(q[k-2], q[k-1], q[k], q[k+1], q[k+2], 1)
    #             elif porder[k] == 3:
    #                 #qi = c1*q[k-1]+c2*q[k]+c3*q[k+1]
    #                 qi = weno3(q[k-1], q[k], q[k+1], 1)
    #             else:
    #                 # porder[k] == 1:
    #                 qi = q[k]
    #         else:
    #             if morder[k] == 5:
    #                 # qi = d5*q[k-1]+d4*q[k]+d3*q[k+1]+d2*q[k+2]+d1*q[k+3]
    #                 qi = weno5(q[k-1], q[k], q[k+1], q[k+2], q[k+3], -1)
    #             elif morder[k] == 3:
    #                 # qi = c3*q[k]+c2*q[k+1]+c1*q[k+2]
    #                 qi = weno3(q[k], q[k+1], q[k+2], -1)
    #             else:
    #                 # morder[k] == 1:
    #                 qi = q[k+1]

    #         flux[k] = U[k]*qi

    @cc.export("bernoulli",
               "void(f8[:, :, :], f8[:, :, :], f8[:, :, :], i1[:,:], boolean)")
    def bernoulli(ke, p, du, msk, linear):
        nz, ny, nx = p.shape
        assert du.shape == (nz, ny, nx+1)
        if linear:
            # linear case, only pressure, no kinetic energy
            for k in range(nz):
                for j in range(ny):
                    du[k, j, 0] = 0.
                    for i in range(1, nx):
                        if msk[j, i]+msk[j, i-1] == 2:
                            du[k, j, i] -= p[k, j, i]-p[k, j, i-1]
                        else:
                            du[k, j, i] = 0.
                    du[k, j, -1] = 0.
        else:
            # nonlinear case
            for k in range(nz):
                for j in range(ny):
                    du[k, j, 0] = 0.
                    for i in range(1, nx):
                        if msk[j, i]+msk[j, i-1] == 2:
                            du[k, j, i] -= p[k, j, i]-p[k, j, i-1] + \
                                ke[k, j, i]-ke[k, j, i-1]
                        else:
                            du[k, j, i] = 0.
                    du[k, j, -1] = 0.

    @cc.export("comppv",
               "void(f8[:, :, :], f8[:, :], f8[:, :, :], f8[:, :, :])")
    def comppv(vort, f, h, pv):
        """ PV at cell corners
        """
        shape = vort.shape
        for k in range(shape[0]):
            for j in range(1, shape[1]-1):
                h0 = h[k, j-1, 0]+h[k, j, 0]
                for i in range(1, shape[2]-1):
                    h1 = h[k, j-1, i]+h[k, j, i]
                    hm = (h1+h0)*.25
                    pv[k, j, i] = (vort[k, j, i]+f[j, i])/hm
                    h0 = h1

    @cc.export("comppv_c",
               "void(f8[:, :, :], f8[:, :], f8[:, :, :], f8[:, :, :])")
    def comppv_c(vor, f, h, pv):
        """ PV at cell centers
        """
        shape = h.shape
        for k in range(shape[0]):
            for j in range(shape[1]):
                v0 = vor[k, j+1, 0]+vor[k, j, 0]+f[j, 0]+f[j+1, 0]
                for i in range(shape[2]):
                    v1 = vor[k, j+1, i+1]+vor[k, j, i+1]+f[j, i+1]+f[j+1, i+1]
                    vm = (v1+v0)*.25
                    pv[k, j, i] = vm/h[k, j, i]
                    v0 = v1

    @cc.export("curl",
               "void(f8[:, :, :], f8[:, :, :], i1[:, :], i4, i4)")
    def curl(v, vort, msk, sign, m0):
        shape = v.shape
        if sign == 1:
            for k in range(shape[0]):
                for j in range(1, shape[1]-1):
                    for i in range(1, shape[2]):
                        m = msk[j, i]+msk[j-1, i] + msk[j, i-1]+msk[j-1, i-1]
                        if m >= m0:
                            vort[k, j, i] += v[k, j, i] - v[k, j, i-1]

        elif sign == -1:
            for k in range(shape[0]):
                for j in range(1, shape[1]-1):
                    for i in range(1, shape[2]):
                        m = msk[j, i]+msk[j-1, i] + msk[j, i-1]+msk[j-1, i-1]
                        if m >= m0:
                            vort[k, j, i] -= v[k, j, i] - v[k, j, i-1]

    @cc.export("totalcurl",
               "void(f8[:, :, :], f8[:, :, :], f8[:, :, :], i1[:, :])")
    def totalcurl(u, v, vort, msk):
        """ curl computation with both u and v

        all arrays are in i convention

        this is faster than splitting the directions

        BUT: j=0 and j=ny, i=0 and i=nx are not set, this is a problem with the noslip boundary condition
        """
        shape = u.shape
        nz, ny, nx = shape
        nx -= 1
        for k in range(nz):
            for j in range(1, ny):
                for i in range(1, nx):
                    m = msk[j, i]+msk[j-1, i] + msk[j, i-1]+msk[j-1, i-1]
                    if m >= 4:
                        vort[k, j, i] = (v[k, j, i] - v[k, j, i-1]
                                         - u[k, j, i] + u[k, j-1, i])

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
                        ke[k, j, i] = (ke0+ke1)*.25
                        ke0 = ke1
        elif sign == 1:
            for k in range(shape[0]):
                for j in range(shape[1]):
                    ke0 = u[k, j, 0]*U[k, j, 0]
                    for i in range(shape[2]):
                        ke1 = u[k, j, i+1]*U[k, j, i+1]
                        ke[k, j, i] += (ke0+ke1)*.25
                        ke0 = ke1

    @cc.export("montgomery",
               "void(f8[:, :, :], f8[:, :], f8[:, :, :], f8[:, :], f8[:], f8, i1[:, :])")
    def montgomery(h, hb, p, area, rho, g, msk):
        shape = h.shape
        nz = shape[0]
        assert rho.shape[0] == nz

        for k in range(nz):
            # first pass on all levels with contribution
            # from the bottom layer and the topography
            cff = g*rho[k]
            for j in range(shape[1]):
                for i in range(shape[2]):
                    p[k, j, i] = cff*(hb[j, i]+h[0, j, i])

        # add overlaying layers contribution
        # the one layer case never enters this loop
        for l in range(1, nz):
            for k in range(nz):
                cff = g*min(rho[k], rho[l])
                for j in range(shape[1]):
                    for i in range(shape[2]):
                        p[k, j, i] += cff*h[l, j, i]

        for k in range(nz):
            for j in range(shape[1]):
                for i in range(shape[2]):
                    if msk[j, i] == 1:
                        p[k, j, i] /= area[j, i]

    @cc.export("vortex_force",
               "void(f8[:, :, :], f8[:, :], f8[:, :, :], f8[:, :, :], f8[:], i1[:, :], i4)")
    def vortex_force(U, f, vor, dv, q, order, sign):
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
        #q = np.zeros((nx,))
        #Um = np.zeros((nx-1,))
        #flux = np.zeros((nx-1,))

        assert U.shape == (nz, ny-1, nx)
        assert dv.shape == (nz, ny, nx-1)
        assert order.shape == (ny, nx)
        assert q.shape == (nx,)

        # Porder = np.zeros((nx-1,), dtype=np.int8)
        # Morder = np.zeros((nx-1,), dtype=np.int8)

        # Porder[0] = 1
        # Porder[1] = 3
        # Porder[2:-1] = 5
        # Porder[-1] = 3

        # Morder[0] = 3
        # Morder[1:-2] = 5
        # Morder[-2] = 3
        # Morder[-1] = 1

        for k in range(nz):
            for j in range(1, ny-1):
                # U0 = U[k, j-1, 0] + U[k, j, 0]
                # for i in range(nx-1):
                #     U1 = U[k, j-1, i+1] + U[k, j, i+1]
                #     Um[i] = (U0+U1)*.25
                #     U0 = U1
                for i in range(nx):
                    q[i] = vor[k, j, i]  # +f[j, i]

                # interp1d_etoc(q, Um, flux, porder, morder)
                U0 = U[k, j-1, 0] + U[k, j, 0]
                for i in range(nx-1):
                    U1 = U[k, j-1, i+1] + U[k, j, i+1]
                    Um = (U0+U1)*.25
                    U0 = U1
                    if Um > 0:
                        porder = order[j, i]
                        #porder = Porder[i]
                        if porder == 5:
                            #qi = d1*q[i-2]+d2*q[i-1]+d3*q[i]+d4*q[i+1]+d5*q[i+2]
                            qi = weno5(q[i-2], q[i-1], q[i], q[i+1], q[i+2], 1)
                        elif porder == 3:
                            #qi = c1*q[i-1]+c2*q[i]+c3*q[i+1]
                            qi = weno3(q[i-1], q[i], q[i+1], 1)
                        elif porder == 1:
                            qi = q[i]
                        else:
                            qi = 0  # q[i+1]
                    else:
                        morder = order[j, i+1]

                        #morder = Morder[i]
                        if morder == 5:
                            # qi = d5*q[i-1]+d4*q[i]+d3*q[i+1]+d2*q[i+2]+d1*q[i+3]
                            #qi = weno5(q[i-1], q[i], q[i+1], q[i+2], q[i+3], 0)
                            qi = weno5(q[i+3], q[i+2], q[i+1], q[i], q[i-1], 1)
                        elif morder == 3:
                            #qi = c3*q[i]+c2*q[i+1]+c1*q[i+2]
                            #qi = weno3(q[i], q[i+1], q[i+2], 0)
                            qi = weno3(q[i+2], q[i+1], q[i], 1)
                        elif morder == 1:
                            qi = q[i+1]
                        else:
                            qi = 0  # q[i]

                    #flux[i] = Um[i]*qi
                    ff = 0.5*(f[j, i]+f[j, i+1])
                    if sign == 1:
                        dv[k, j, i] += Um*(qi+ff)
                    else:
                        dv[k, j, i] -= Um*(qi+ff)

                # if sign == 1:
                #     for i in range(nx-1):
                #         dv[k, j, i] += flux[i]
                # elif sign == -1:
                #     for i in range(nx-1):
                #         dv[k, j, i] -= flux[i]

    @cc.export("upwindtrac",
               "void(f8[:, :, :], f8[:, :, :], f8[:, :, :], i1[:, :])")
    def upwindtrac(field, U, dfield, order):
        nz, ny, nx = field.shape

        #flux = np.zeros((nx+1,))
        assert order.shape == (ny, nx+1)
        # Porder = np.zeros((nx+1,), dtype=np.int8)
        # morder = np.zeros((nx+1,), dtype=np.int8)

        # Porder[0] = 0
        # Porder[1] = 1
        # Porder[2] = 3
        # Porder[3:-2] = 5
        # Porder[-2] = 3
        # Porder[-1] = 1

        # morder[0] = 1
        # morder[1] = 3
        # morder[2:-3] = 5
        # morder[-3] = 3
        # morder[-2] = 1
        # morder[-1] = 0

        for I in np.ndindex(nz, ny):
            k, j = I
            q = field[k, j]
            #Um = U[k, j]
            for i in range(1, nx+1):
                if U[k, j, i] > 0:
                    porder = order[j, i]
                    #porder = Porder[i]
                    if porder == 5:
                        #qi = d1*q[i-2]+d2*q[i-1]+d3*q[i]+d4*q[i+1]+d5*q[i+2]
                        qi = weno5(q[i-3], q[i-2], q[i-1], q[i], q[i+1], 1)
                    elif porder == 3:
                        #qi = c1*q[i-1]+c2*q[i]+c3*q[i+1]
                        qi = weno3(q[i-2], q[i-1], q[i], 1)
                    elif porder == 1:
                        qi = q[i-1]
                    else:
                        qi = 0.
                else:
                    if i < nx:
                        morder = order[j, i+1]
                        #morder = Porder[i+1]
                    else:
                        morder = 0
                    if morder == 5:
                        # qi = d5*q[i-1]+d4*q[i]+d3*q[i+1]+d2*q[i+2]+d1*q[i+3]
                        #qi = weno5(q[i-2], q[i-1], q[i], q[i+1], q[i+2], 0)
                        qi = weno5(q[i+2], q[i+1], q[i], q[i-1], q[i-2], 1)
                    elif morder == 3:
                        #qi = c3*q[i]+c2*q[i+1]+c1*q[i+2]
                        # qi = weno3(q[i-1], q[i], q[i+1], 0)
                        qi = weno3(q[i+1], q[i], q[i-1], 1)
                    elif morder == 1:
                        qi = q[i]
                    else:
                        qi = 0.

                #flux[i] = Um[i]*qi
                f = U[k, j, i]*qi
                dfield[k, j, i] += f
                dfield[k, j, i-1] -= f

            # interp1d_ctoe(field[k, j], U[k, j], flux)
            # for i in range(nx):
            #     dfield[k, j, i] -= flux[i+1]-flux[i]

    @cc.export("set_geostrophic_vel",
               "void(f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :], f8[:, :], f8[:, :], i1[:, :])")
    def set_geostrophic_vel(u, v, vor, p, ke, dx2, dy2, f, msk):
        nz, ny, nx = np.shape(p)
        B = np.zeros((ny+1, nx+1))
        mskv = np.zeros_like(msk)
        cff = [0., 1., 0.5, 1./3, 0.25]
        for j in range(1, ny):
            for i in range(1, nx):
                if msk[j-1, i]+msk[j-1, i-1]+msk[j, i]+msk[j, i-1] == 4:
                    mskv[j, i] = 1
        for k in range(nz):
            b = p[k]+ke[k]
            W = vor[k]+f
            # j = 0
            B[0, 0] = b[0, 0]*cff[msk[0, 0]]
            for i in range(1, nx):
                B[0, i] = (b[0, i-1]+b[0, i])*cff[msk[0, i-1]+msk[0, i]]
            B[0, nx] = b[0, nx-1]*cff[msk[0, nx-1]]

            j = ny-1
            B[j+1, 0] = b[j, 0]*cff[msk[j, 0]]
            for i in range(1, nx):
                B[j+1, i] = (b[j, i-1]+b[j, i])*cff[msk[j, i-1]+msk[j, i]]
            B[j+1, nx] = b[j, nx-1]*cff[msk[j, nx-1]]

            for j in range(1, ny):
                b0 = b[j-1, 0]+b[j, 0]
                m0 = msk[j-1, 0]+msk[j, 0]
                B[j, 0] = b0*cff[m0]
                for i in range(1, nx):
                    b1 = b[j-1, i]+b[j, i]
                    m1 = msk[j-1, i]+msk[j, i]
                    B[j, i] = (b0+b1)*cff[m0+m1]
                    b0 = b1
                    m0 = m1
                B[j, nx] = b1*cff[m1]

            for j in range(ny):
                for i in range(1, nx-1):
                    w = (W[j+1, i]+W[j, i])*0.5
                    if mskv[j+1, i]+mskv[j, i] == 2:
                        u[k, j, i] = -(B[j+1, i]-B[j, i])/w
                    u[k, j, i] *= dx2[j, i]

            for j in range(1, ny-1):
                for i in range(nx):
                    w = (W[j, i+1]+W[j, i])*0.5
                    if mskv[j, i+1]+mskv[j, i] == 2:
                        v[k, j, i] = +(B[j, i+1]-B[j, i])/w
                    v[k, j, i] *= dy2[j, i]
    cc.compile()
