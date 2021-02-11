"""
weno interpolation

Balsara, Garain, Shu, 2016, "An efficient class of WENO schemes with adaptive order", J. Comput. Phys.

"""
import numpy as np
from numba import jit
from numba.pycc import CC

# third order linear
c1 = -1./6.
c2 = 5./6.
c3 = 2./6.

# fifth order
d1 = 2./60.
d2 = -13./60.
d3 = 47./60.
d4 = 27./60.
d5 = -3./60.

# weno 5th
cff1 = 13/3.
cff2 = 1./120.
cff3 = 1./56.
cff4 = 1./12.
cff5 = 1./24.
cff6 = 123./455.
cff7 = 781./20.
cff8 = 1421461./2275.
cff9 = 1./3.

eps = 1e-8

gammaHi = 0.9
gammaLo = 0.9

# (3.5)
gamma5 = gammaHi
gamma_1 = (1-gammaHi)*(1-gammaLo)*0.5
gamma_2 = (1-gammaHi)*gammaLo
gamma_3 = (1-gammaHi)*(1-gammaLo)*0.5

# (2.1)
L1 = [1., 0.]
L2 = [1., 0, -1./12]
L3 = [1., 0, -3./20, 0.]
L4 = [1., 0., -3./14, 0, 3./560]

K1 = np.polyval(L1, 0.5)
K2 = np.polyval(L2, 0.5)
K3 = np.polyval(L3, 0.5)
K4 = np.polyval(L4, 0.5)

J1 = np.polyval(L1, -0.5)
J2 = np.polyval(L2, -0.5)
J3 = np.polyval(L3, -0.5)
J4 = np.polyval(L4, -0.5)

g1 = 1./10
g2 = 6./10.
g3 = 3./10.

h1 = 1./3.
h2 = 2./3.

cc = CC("wenos")
cc.verbose = True


@cc.export("weno5", "f8(f8, f8, f8, f8, f8, i8)")
def weno5(qmm, qm, q0, qp, qpp, sign):
    # step 1: Eqs (2.3, 2.4, 2.5)
    ux_1 = 0.5*qmm-2*qm+1.5*q0
    ux2_1 = 0.5*(qmm+q0)-qm

    ux_2 = 0.5*(qp-qm)
    ux2_2 = 0.5*(qm+qp)-q0

    ux_3 = -1.5*q0+2*qp-0.5*qpp
    ux2_3 = 0.5*(q0+qpp)-qp

    # Eq. 2.6
    beta_1 = ux_1*ux_1 + cff1*ux2_1*ux2_1
    beta_2 = ux_2*ux_2 + cff1*ux2_2*ux2_2
    beta_3 = ux_3*ux_3 + cff1*ux2_3*ux2_3

    w_1 = g1/(beta_1+eps)**2
    w_2 = g2/(beta_2+eps)**2
    w_3 = g3/(beta_3+eps)**2

    # normalization
    wnorm = 1./(w_1+w_2+w_3)

    if sign > 0:
        # (2.2)
        P1 = q0 + ux_1*K1 + ux2_1*K2
        P2 = q0 + ux_2*K1 + ux2_2*K2
        P3 = q0 + ux_3*K1 + ux2_3*K2
    else:
        # same for U<0
        P1 = q0 + ux_1*J1 + ux2_1*J2
        P2 = q0 + ux_2*J1 + ux2_2*J2
        P3 = q0 + ux_3*J1 + ux2_3*J2

    #
    return (w_1*P1 + w_2*P2 + w_3*P3)*wnorm


@cc.export("weno3", "f8(f8, f8, f8, i8)")
def weno3(qm, q0, qp, sign):

    beta_1 = (q0-qm)**2
    beta_2 = (qp-q0)**2

    w_1 = h1/(beta_1+eps)**2
    w_2 = h2/(beta_2+eps)**2

    # normalization
    wnorm = 1./(w_1+w_2)

    if sign > 0:
        P1 = 1.5*q0-0.5*qm
        P2 = 0.5*qp+0.5*q0
    else:
        P1 = 0.5*q0+0.5*qm
        P2 = 1.5*q0-0.5*qp

    #
    return (w_1*P1 + w_2*P2)*wnorm


cc.compile()


def wenoflux_edge2center(q, U, flux, porder, morder):
    """ compute flux = U * q

    staggering:

    center: U, flux
    edge: q

    U and flux have the same shape
    q has one more element
    q[i] is on the *left* of U[i]
    """
    n = U.shape[0]

    # porder = np.zeros((n,), dtype=int)
    # morder = np.zeros((n,), dtype=int)

    porder[0] = 1
    porder[1] = 3
    porder[2:-1] = 5
    porder[-1] = 3

    morder[0] = 3
    morder[1:-2] = 5
    morder[-2] = 3
    morder[-1] = 1

    #porder = 5
    #morder = 5

    for k in range(n):
        if U[k] > 0:
            if porder[k] == 5:
                #qi = d1*q[k-2]+d2*q[k-1]+d3*q[k]+d4*q[k+1]+d5*q[k+2]
                qi = weno5(q[k-2], q[k-1], q[k], q[k+1], q[k+2], 1)
            elif porder[k] == 3:
                qi = c1*q[k-1]+c2*q[k]+c3*q[k+1]
            else:
                # porder[k] == 1:
                qi = q[k]
        else:
            if morder[k] == 5:
                # qi = d5*q[k-1]+d4*q[k]+d3*q[k+1]+d2*q[k+2]+d1*q[k+3]
                qi = weno5(q[k-1], q[k], q[k+1], q[k+2], q[k+3], -1)
            elif morder[k] == 3:
                qi = c3*q[k]+c2*q[k+1]+c1*q[k+2]
            else:
                # morder[k] == 1:
                qi = q[k+1]

        flux[k] = U[k]*qi


def linearflux_edge2center(q, U, flux):
    """ compute flux = U * q

    staggering:

    center: U, flux
    edge: q

    U and flux have the same shape
    q has one more element
    q[i] is on the *left* of U[i]
    """
    n = U.shape[0]

    k = 0
    if U[k] > 0:
        # upwind 1st
        qi = q[k]
    else:
        # upwind 3rd
        qi = c3*q[k]+c2*q[k+1]+c1*q[k+2]
    flux[k] = U[k]*qi
    k += 1

    if U[k] > 0:
        # upwind 3rd
        qi = c1*q[k-1]+c2*q[k]+c3*q[k+1]
        flux[k] = U[k]*qi
        k += 1

    ok = True
    while ok:
        if U[k] > 0:
            qmm = q[k-2]
            qm = q[k-1]
            q0 = q[k]
            qp = q[k+1]
            qpp = q[k+2]

            qi = d1*qmm+d2*qm+d3*q0+d4*qp+d5*qpp
        else:
            qmm = q[k-1]
            qm = q[k]
            q0 = q[k+1]
            qp = q[k+2]
            qpp = q[k+3]
            qi = d5*qmm+d4*qm+d3*q0+d2*qp+d1*qpp

        flux[k] = U[k]*qi
        k += 1
        # stop criterion
        if (k == n-2) and (U[k] < 0):
            ok = False
        if (k == n-1):
            ok = False

    if (k == n-2):  # which implies U[k] < 0
        # upwind 3rd
        qi = c3*q[k]+c2*q[k+1]+c1*q[k+2]
        flux[k] = U[k]*qi
        k += 1

    if U[k] > 0:
        # upwind 3rd
        qi = c1*q[k-1]+c2*q[k]+c3*q[k+1]
    else:
        # upwind 1st
        qi = q[k+1]
    flux[k] = U[k]*qi


def wenoflux_center2edge(q, U, flux):
    """ compute flux = U * q

    staggering:

    edge: U, flux
    center: q

    U and flux have the same shape
    q has one less element
    q[i] is on the *right* of U[i]

    the function assumes a no incoming flux at the left and right
    but allows for a non zero outgoing flux ...
    """
    n = q.shape[0]

    k = 0
    if U[k] > 0:
        # inflow condition -> no flux!
        qi = 0.
    else:
        # upwind 1st
        qi = 0.  # q[k]
    flux[k] = U[k]*qi
    k += 1

    if U[k] > 0:
        # upwind 1st
        qi = q[k-1]
    else:
        # upwind 3rd
        qi = c3*q[k-1]+c2*q[k]+c1*q[k+1]
    flux[k] = U[k]*qi
    k += 1

    if U[k] > 0:
        # upwind 3rd
        qi = c1*q[k-2]+c2*q[k-1]+c3*q[k]
        flux[k] = U[k]*qi
        k += 1

    ok = True
    while ok:
        if U[k] > 0:
            qmm = q[k-3]
            qm = q[k-2]
            q0 = q[k-1]
            qp = q[k]
            qpp = q[k+1]
        else:
            qmm = q[k-2]
            qm = q[k-1]
            q0 = q[k]
            qp = q[k+1]
            qpp = q[k+2]

        # step 1: Eqs (2.3, 2.4, 2.5)
        ux_1 = 0.5*qmm-2*qm+1.5*q0
        ux2_1 = 0.5*(qmm+q0)-qm

        ux_2 = 0.5*(qp-qm)
        ux2_2 = 0.5*(qm+qp)-q0

        ux_3 = -1.5*q0+2*qp-0.5*qpp
        ux2_3 = 0.5*(q0+qpp)-qp

        # Eq. 2.6
        beta_1 = ux_1*ux_1 + cff1*ux2_1*ux2_1
        beta_2 = ux_2*ux_2 + cff1*ux2_2*ux2_3
        beta_3 = ux_3*ux_3 + cff1*ux2_3*ux2_3

        w_1 = g1/(beta_1+eps)**2
        w_2 = g2/(beta_2+eps)**2
        w_3 = g3/(beta_3+eps)**2

        # normalization
        wnorm = 1./(w_1+w_2+w_3)

        if U[k] > 0:
            # (2.2)
            P1 = q0 + ux_1*K1 + ux2_1*K2
            P2 = q0 + ux_2*K1 + ux2_2*K2
            P3 = q0 + ux_3*K1 + ux2_3*K2
        else:
            # same for U<0
            P1 = q0 + ux_1*J1 + ux2_1*J2
            P2 = q0 + ux_2*J1 + ux2_2*J2
            P3 = q0 + ux_3*J1 + ux2_3*J2

        #
        qi = (w_1*P1 + w_2*P2 + w_3*P3)*wnorm

        flux[k] = U[k]*qi
        k += 1
        # stop criterion
        if (k == n-2) and (U[k] < 0):
            ok = False
        if (k == n-1):
            ok = False

    if (k == n-2):  # which implies U[k] < 0
        # upwind 3rd
        qi = c3*q[k-1]+c2*q[k]+c1*q[k+1]
        flux[k] = U[k]*qi
        k += 1

    if U[k] > 0:
        # upwind 3rd
        qi = c1*q[k-2]+c2*q[k-1]+c3*q[k]
    else:
        # upwind 1st
        qi = q[k]
    flux[k] = U[k]*qi
    k += 1

    if U[k] > 0:
        # upwind 1st
        qi = 0.  # q[k-1]
    else:
        #  inflow -> no flow
        qi = 0.
    flux[k] = U[k]*qi


def linearflux_center2edge(q, U, flux):
    """ compute flux = U * q

    staggering:

    edge: U, flux
    center: q

    U and flux have the same shape
    q has one less element
    q[i] is on the *right* of U[i]

    the function assumes a no incoming flux at the left and right
    but allows for a non zero outgoing flux ...
    """
    n = q.shape[0]

    k = 0
    if U[k] > 0:
        # inflow condition -> no flux!
        qi = 0.
    else:
        # upwind 1st
        qi = 0.  # q[k]
    flux[k] = U[k]*qi
    k += 1

    if U[k] > 0:
        # upwind 1st
        qi = q[k-1]
    else:
        # upwind 3rd
        qi = q[k]
    flux[k] = U[k]*qi
    k += 1

    if U[k] > 0:
        # upwind 3rd
        qi = q[k-1]
        flux[k] = U[k]*qi
        k += 1

    ok = True
    while ok:
        if U[k] > 0:
            qi = q[k-1]
        else:
            qi = q[k]

        flux[k] = U[k]*qi
        k += 1
        # stop criterion
        if (k == n-2) and (U[k] < 0):
            ok = False
        if (k == n-1):
            ok = False

    if (k == n-2):  # which implies U[k] < 0
        # upwind 3rd
        qi = q[k]
        flux[k] = U[k]*qi
        k += 1

    if U[k] > 0:
        # upwind 3rd
        qi = q[k-1]
    else:
        # upwind 1st
        qi = q[k]
    flux[k] = U[k]*qi
    k += 1

    if U[k] > 0:
        # upwind 1st
        qi = 0.  # q[k-1]
    else:
        #  inflow -> no flow
        qi = 0.
    flux[k] = U[k]*qi


if __name__ == '__main__':
    nx = 100
    dx = 0.05
    xe = (np.arange(nx+1)-0.5)*dx
    xc = np.arange(nx)*dx
    q = np.sin(20*xe)
    flux = np.zeros((nx,))
    U = np.ones((nx,))

    wenoflux_edge2center(q, U, flux)

    import matplotlib.pyplot as plt
    plt.ion()
    plt.clf()
    plt.plot(xc, flux)
    plt.plot(xe, q)
