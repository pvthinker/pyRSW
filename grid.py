"""Provide the grid on which pyRSW is discretized.

"""

import numpy as np
from variables import Scalar
import topology as topo
import variables
try:
    import mindexing
except:
    import mask_stencils as ms
    ms.compile()
    import mindexing
from numba import jit

gridvar = {
    "hb": {
        "type": "scalar",
        "name": "total depth",
        "dimensions": ["y", "x"],
        "unit": "L",
        "constant": True,
        "prognostic": False},
    "vol": {
        "type": "scalar",
        "name": "area",
        "dimensions": ["y", "x"],
        "unit": "L^2",
        "constant": True,
        "prognostic": False},
    "volv": {
        "type": "vorticity",
        "name": "area",
        "dimensions": ["y", "x"],
        "unit": "L^2",
        "constant": True,
        "prognostic": False},
    "f": {
        "type": "vorticity",
        "name": "coriolis",
        "dimensions": ["y", "x"],
        "unit": "L^2.T^-1",
        "constant": True,
        "prognostic": False},
    "invd": {
        "type": "vector",
        "name": "inverse of squared Lame coefficient",
        "dimensions": ["y", "x"],
        "unit": "L^-2",
        "constant": True,
        "prognostic": False},
    "msk": {
        "type": "scalar",
        "dtype": "b",  # 'b' is int8
        "name": "land mask",
        "dimensions": ["y", "x"],
        "unit": "1",
        "constant": True,
        "prognostic": False}
}

nickname = "tporder"
gridvar[nickname] = {
    "type": "vector",
    "dtype": "b",  # 'b' is int8
    "name": nickname,
    "dimensions": ["y", "x"],
    "unit": "1",
    "constant": True,
    "prognostic": False}

nickname = "vporderx"
gridvar[nickname] = {
    "type": "vorticity",
    "dtype": "b",  # 'b' is int8
    "name": nickname,
    "dimensions": ["y", "x"],
    "unit": "1",
    "constant": True,
    "prognostic": False}

nickname = "vpordery"
gridvar[nickname] = {
    "type": "vorticity",
    "dtype": "b",  # 'b' is int8
    "name": nickname,
    "dimensions": ["y", "x"],
    "unit": "1",
    "constant": True,
    "prognostic": False}


def set_domain_decomposition(param):
    topo.topology = param["geometry"]
    procs = [1, param["npy"], param["npx"]]
    myrank = 0  # mpitools.get_myrank(procs)
    loc = topo.rank2loc(myrank, procs)
    neighbours = topo.get_neighbours(loc, procs)
    param["procs"] = procs
    param["myrank"] = myrank
    param["neighbours"] = neighbours
    param["loc"] = loc


class Grid(object):
    def __init__(self, param):
        set_domain_decomposition(param)

        self.arrays = variables.State(param, gridvar)
        # set msk to 1
        msk = self.arrays.msk.view("i")
        msk[:] = 1

        # Copy needed parameters
        self.nx = param["nx"]
        self.ny = param["ny"]
        self.nz = param["nz"]

        self.npx = param["npx"]
        self.npy = param["npy"]

        self.Lx = param["Lx"]
        self.Ly = param["Ly"]

        self.f0 = param["f0"]

        # grid space
        self.dx = self.Lx / (self.npx*self.nx)
        self.dy = self.Ly / (self.npy*self.ny)

        # Define useful quantities
        self.dx2 = self.dx**2
        self.dy2 = self.dy**2

        self.idx = 1 / self.dx
        self.idy = 1 / self.dy

        self.idx2 = 1 / self.dx**2
        self.idy2 = 1 / self.dy**2

        self.area = self.dx * self.dy
        self.iarea = 1/self.area

        # Define coordinates
        #  coordinates of the bottom left front corner of the subdomain
        x0 = param['loc'][1]*self.nx*self.dx
        y0 = param['loc'][0]*self.ny*self.dy

        var = {"name": "", "unit": "",
               "prognostic": False, "dimensions": ["y", "x"]}
        # at cell centers
        dummy = Scalar("dummy", var, param, stagg="")
        ny, nx = dummy.shape
        self.shape = (ny, nx)

        j0, j1, i0, i1 = dummy.domainindices
        self.xc = (0.5+np.arange(nx)-i0)*self.dx+x0
        self.yc = (0.5+np.arange(ny)-j0)*self.dy+y0
        # at cell edges
        dummy = Scalar("dummy", var, param, stagg="xy")
        j0, j1, i0, i1 = dummy.domainindices
        self.xe = (np.arange(nx+1)-i0)*self.dx+x0
        self.ye = (np.arange(ny+1)-j0)*self.dy+y0

        # self.arrays.f.view("j")
        #self.arrays.f.locked = True

    def set_coriolis(self):
        f = self.arrays.f.view("i")
        f[:] = self.f0*(self.dx*self.dy)  # self.arrays.volv.view("i")
        #f[:] = self.f0*self.arrays.volv.view("i")

        mskc = self.arrays.msk.view("i")
        ny, nx = mskc.shape
        # derive the mask at vorticity point
        mskv = np.zeros((ny+1, nx+1), dtype=mskc.dtype)
        for j in range(ny):
            for i in range(nx):
                if mskc[j, i] == 1:
                    mskv[j+1, i+1] += 1
                    mskv[j+1, i] += 1
                    mskv[j, i+1] += 1
                    mskv[j, i] += 1
        # for j in range(ny+1):
        #     for i in range(nx+1):
        #         if mskv[j, i] != 4:
        #             f[j,i]=0

    def sum(self, array3d):
        """ compute the global domain sum of array3d
        array3d should be in (k,j,i) convention """

        assert array3d.activeview == "i"

        j0, j1, i0, i1 = array3d.domainindices

        localsum = array3d[:, j0:j1, i0:i1].sum(axis=(-2, -1))

        return localsum

    def cov_to_contra(self, state):
        U = state.Ux.view("i")
        V = state.Uy.view("i")
        u = state.ux.view("i")
        v = state.uy.view("i")
        idx2 = self.arrays.invdx.view("i")
        idy2 = self.arrays.invdy.view("i")
        U[:] = u * idx2
        V[:] = v * idy2

    def finalize(self):
        """ compute the order of upwind discretizations
        depending on the proximity to solid boundaries
        """
        self.set_coriolis()
        self.arrays.f.view("j")
        self.arrays.f.locked = True
        self.arrays.msk.view("j")
        self.arrays.msk.locked = True
        msk = self.arrays.msk.view("i")
        txporder = self.arrays.tporderx.view("i")
        vyporder = self.arrays.vpordery.view("i")
        index_tracerflux(msk, txporder)
        index_vortexforce(msk, vyporder)
        # txporder = np.minimum(txporder, 0)
        # vyporder = np.minimum(vyporder, 0)
        # txmorder = self.arrays.tmorderx.view("i")
        # index_tracerflux(msk, txmorder, 0)

        msk = self.arrays.msk.view("j")
        typorder = self.arrays.tpordery.view("j")
        vxporder = self.arrays.vporderx.view("j")
        index_tracerflux(msk, typorder)
        index_vortexforce(msk, vxporder)
        # typorder = np.minimum(typorder, 0)
        # vxporder = np.minimum(vxporder, 0)
        # tymorder = self.arrays.tmordery.view("j")
        # index_tracerflux(msk, tymorder, 0)

    def msku(self):
        mskc = self.arrays.msk.view("i")
        ny, nx = mskc.shape
        msku = np.zeros((ny, nx+1), dtype=np.int8)
        for j in range(ny):
            for i in range(nx-1):
                if mskc[j, i-1]+mskc[j, i] == 2:
                    msku[j, i] = 1
        return msku

    def mskv(self):
        mskc = self.arrays.msk.view("j")
        ny, nx = mskc.shape
        msku = np.zeros((ny, nx+1), dtype=np.int8)
        for j in range(ny):
            for i in range(nx-1):
                if mskc[j, i-1]+mskc[j, i] == 2:
                    msku[j, i] = 1
        return msku


# @jit
def index_tracerflux(msk, order):
    """
    Determine the order for the biased interpolation
    of tracer along direction x at U point

    sign:
      - 1 for left-biased
      - 0 for right biased
    """
    ny, nx = msk.shape
    assert order.shape == (ny, nx+1)
    for j in range(ny):
        m = msk[j]
        for i1 in range(nx+1):
            i = i1
            if (i >= 1) and (i < nx):
                m1 = m[i-1]+m[i]
            elif (i == nx):
                m1 = m[i-1]
            else:
                m1 = 0
            if (i >= 2) and (i < nx):
                m3 = m[i-2]+m[i-1]+m[i]
            else:
                m3 = 0
            if (i >= 3) and (i < nx-1):
                m5 = m[i-3]+m[i-2]+m[i-1]+m[i]+m[i+1]
            else:
                m5 = 0

            if m5 == 5:
                order[j, i1] = 5
            elif m3 == 3:
                order[j, i1] = 3
            elif m1 >= 1:
                order[j, i1] = 1
            else:
                order[j, i1] = 0


# @jit
def index_vortexforce(mskc, order):
    """
    Determine the order for the biased interpolation
    of vorticity along direction x at V point

    mskc: [ny, nx] msk at centers
    order: [ny+1, nx+1] order at V point
    sign:
      - 1 for left-biased
      - 0 for right biased

    mskv: msk at corners
    msk: [ny+1, nx+1] msk to be used to determine the order

    """
    ny, nx = mskc.shape
    assert order.shape == (ny+1, nx+1)
    # derive the mask at vorticity point
    mskv = np.zeros((ny+1, nx+1), dtype=mskc.dtype)
    for j in range(ny):
        for i in range(nx):
            if mskc[j, i] == 1:
                mskv[j+1, i+1] += 1
                mskv[j+1, i] += 1
                mskv[j, i+1] += 1
                mskv[j, i] += 1
    # mskv[:] = 0
    # for j in range(1,ny):
    #     for i in range(1,nx):
    #         if mskc[j, i]+mskc[j-1,i]+mskc[j,i-1]+mskc[j-1,i-1] >= 1:
    #             mskv[j, i] = 1

    for j in range(ny+1):
        m = mskv[j]
        for i1 in range(nx+1):
            i = i1
            if (i >= 0) and (i < nx+1):
                m1 = m[i]
            else:
                m1 = 0
            if (i >= 1) and (i < nx) and (m1 == 4):
                m3 = m[i-1]+m[i]+m[i+1]
            else:
                m3 = 0
            if (i >= 2) and (i < nx-1) and (m3 == 12):
                m5 = m[i-2]+m[i-1]+m[i]+m[i+1]+m[i+2]
            else:
                m5 = 0

            if m5 > 0:
                order[j, i1] = 5
            elif m3 > 0:
                order[j, i1] = 3
            elif m1 > 0:
                order[j, i1] = 1
            else:
                order[j, i1] = 0
