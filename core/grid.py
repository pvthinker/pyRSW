"""Provide the grid on which pyRSW is discretized.

"""

import numpy as np
import topology as topo
import variables
import coordinates

try:
    import buffers_halo
except:
    import buffershalo as BH
    BH.compile()
    fullycompiled = False

import halo
#from numba import jit

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
        "prognostic": False},
    "mskv": {
        "type": "vorticity",
        "dtype": "b",  # 'b' is int8
        "name": "land mask at corners",
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


class Grid(object):
    def __init__(self, param):
        # this set the module "topology" global variable
        # seen outside of this module
        topo.topology = param["geometry"]

        infos = topo.get_domain_decomposition(param)
        for key, val in infos.items():
            setattr(self, key, val)

        self.param = param

        self.halo = halo.Halo(param)

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

        self.f0 = param["f0"]

        self.partialcell = param.partialcell

        # at cell centers
        shape, domainindices = topo.get_shape_and_domainindices(
            param, "yx", "")
        ny, nx = shape
        self.outershape = (ny, nx)
        self.innershape = (self.nz, self.ny, self.nx)

        j0, j1, i0, i1 = domainindices
        #print("cartesian", self.param.myrank, j0, j1, i0, i1)
        if param.coordinates == "cartesian":
            self.coord = coordinates.Cartesian(param)

            self.dx = self.coord.dx
            self.dy = self.coord.dy

        elif param.coordinates == "cylindrical":
            self.coord = coordinates.Cylindrical(param)

        elif param.coordinates == "spherical":
            self.coord = coordinates.Spherical(param)

        else:
            raise ValueError

        self.ic = (0.5+np.arange(nx)-i0)
        self.jc = (0.5+np.arange(ny)-j0)
        self.ie = (np.arange(nx+1)-i0)
        self.je = (np.arange(ny+1)-j0)

        self.xc = self.coord.x(self.jc, self.ic)
        self.yc = self.coord.y(self.jc, self.ic)
        self.xe = self.coord.x(self.je, self.ie)
        self.ye = self.coord.y(self.je, self.ie)

        area = self.arrays.vol.view("i")
        area[:] = self.coord.area(self.jc, self.ic)

        areav = self.arrays.volv.view("i")
        areav[:] = self.coord.area(self.je, self.ie)

        idx2 = self.arrays.invdx.view("i")
        idx2[:] = self.coord.idx2(self.jc, self.ie)

        idy2 = self.arrays.invdy.view("i")
        idy2[:] = self.coord.idy2(self.je, self.ic)

        self.set_coriolis()

        self.set_hmean()

    def set_hmean(self):
        if self.nz == 1:
            self.hmean = np.asarray([self.param.H])
        else:
            assert hasattr(self.param, "Hlayers")
            self.hmean = np.asarray(self.param.Hlayers)

    def set_coriolis(self):
        f = self.arrays.f.view("i")
        areav = self.arrays.volv.view("i")
        if self.param.coordinates == "spherical":
            Omega = self.param.Omega
            theta = self.coord.theta(self.je, self.ie)
            phi = self.coord.phi(self.je, self.ie)
            theta_shift = self.param.lat_pole_shift
            f[:] = 2*Omega*(np.sin(theta)*np.cos(theta_shift)
                            - np.cos(phi)*np.cos(theta)*np.sin(theta_shift))
            f[:] *= areav
        else:
            f[:] = self.f0*areav

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

    def set_mask(self, fbry, **kwargs):
        xc, yc = self.xc, self.yc
        xe, ye = self.xe, self.ye
        m = fbry(xe, ye, **kwargs)

        msk = self.arrays.msk.view("i")
        area = self.arrays.vol.view("i")
        areav = self.arrays.volv.view("i")
        idx2 = self.arrays.invdx.view("i")
        idy2 = self.arrays.invdy.view("i")
        msk[:] = 0

        ny, nx = msk.shape

        partial = np.zeros_like(msk)

        npergridcell = 20
        idx = (np.arange(npergridcell)+0.5)/npergridcell

        # define the mask
        for j in range(ny):
            for i in range(nx):
                partial[j, i] = ((m[j, i] >= 0)*1
                                 + (m[j+1, i] >= 0)*1
                                 + (m[j, i+1] >= 0)*1
                                 + (m[j+1, i+1] >= 0)*1)
                # if partial[j,i] <4:
                #     partial[j,i] = 0
                if partial[j, i] > 0:
                    msk[j, i] = 1
                    if partial[j, i] < 4:
                        xs = self.coord.x(j+idx, i+idx)
                        ys = self.coord.y(j+idx, i+idx)
                        val = fbry(xs, ys, **kwargs)
                        fraction = np.count_nonzero(val)/npergridcell**2
                        if fraction > 0.:
                            # we may declare a cell outside if
                            # fraction is too small, e.g. fraction<0.1
                            # currently all partial cells are included
                            # TODO: check how sensitive it is
                            area[j, i] *= fraction
                        else:
                            msk[j, i] = 0
                            partial[j, i] = 0

        # define area a vertices
        npergridcell = 10
        idx = (np.arange(npergridcell)+0.5)/npergridcell*0.5
        areav_backup = areav.copy()
        areav[:] = 0.
        for j in range(ny):
            for i in range(nx):
                a = 0.25*area[j, i]
                if partial[j, i] == 4:
                    areav[j, i] += a
                    areav[j, i+1] += a
                    areav[j+1, i] += a
                    areav[j+1, i+1] += a
                elif partial[j, i] > 0:
                    xs = self.coord.x(j+idx, i+idx)
                    ys = self.coord.y(j+idx, i+idx)
                    val = fbry(xs, ys, **kwargs)
                    fraction = np.count_nonzero(val)/npergridcell**2
                    if fraction > 0.:
                        areav[j, i] += a*fraction

                    xs = self.coord.x(j+idx, i+idx+0.5)
                    ys = self.coord.y(j+idx, i+idx+0.5)
                    val = fbry(xs, ys, **kwargs)
                    fraction = np.count_nonzero(val)/npergridcell**2
                    if fraction > 0.:
                        areav[j, i+1] += a*fraction

                    xs = self.coord.x(j+idx+0.5, i+idx)
                    ys = self.coord.y(j+idx+0.5, i+idx)
                    fraction = np.count_nonzero(val)/npergridcell**2
                    if fraction > 0.:
                        areav[j+1, i] += a*fraction

                    xs = self.coord.x(j+idx+0.5, i+idx+0.5)
                    ys = self.coord.y(j+idx+0.5, i+idx+0.5)
                    fraction = np.count_nonzero(val)/npergridcell**2
                    if fraction > 0.:
                        areav[j+1, i+1] += a*fraction

        # restore backup values where areav == 0
        for j in range(ny+1):
            for i in range(nx+1):
                if areav[j, i] == 0:
                    areav[j, i] = areav_backup[j, i]

        if False:
            """
            The lengths at edges might need to be changed

            TODO: determine if this has to be done
            """
            # define lengths at U points
            for j in range(ny):
                for i in range(1, nx-1):
                    s = partial[j, i-1]+partial[j, i]
                    if s == 8:
                        pass
                    elif (partial[j, i-1] > 0) and (partial[j, i] > 0):
                        #dy2 = 1./self.coord.idy2(j+0.5, i)
                        #idx2[j, i] = dy2/(area[j, i-1]*area[j, i])
                        idx2[j, i] = self.coord.idx2(j, i+0.5)

            # define lengths at V points
            for j in range(1, ny-1):
                for i in range(nx):
                    s = partial[j-1, i]+partial[j, i]
                    if s == 8:
                        pass
                    elif (partial[j-1, i] > 0) and (partial[j, i] > 0):
                        #dx2 = 1./self.coord.idx2(j, i+0.5)
                        #idy2[j, i] = dx2/(area[j-1, i]*area[j, i])
                        idy2[j, i] = self.coord.idy2(j+0.5, i)

    def set_maskv(self):
        msk = self.arrays.msk.view("i")
        mskv = self.arrays.mskv.view("i")
        mskv[:] = 0.
        ny, nx = msk.shape
        for j in range(ny):
            for i in range(nx):
                mskv[j, i] += msk[j, i]
                mskv[j+1, i] += msk[j, i]
                mskv[j, i+1] += msk[j, i]
                mskv[j+1, i+1] += msk[j, i]

    def finalize(self):
        """
        1/ determine the mask and correct the areas if partialcase

        2/ compute the order of upwind discretizations
        depending on the proximity to solid boundaries
        """
        if self.partialcell:
            msg = ["", "You forgot to set grid.boundary = ...",
                   "Look at the experiment dipole.py for an example"]
            assert hasattr(self, "boundary"), "\n".join(msg)
            fbry = self.boundary["fbry"]
            kwargs = self.boundary["kwargs"]
            self.set_mask(fbry, **kwargs)

        self.set_maskv()

        self.arrays.f.lock()
        self.arrays.msk.lock()
        self.arrays.mskv.lock()
        # self.arrays.f.view("j")
        # self.arrays.f.locked = True
        # self.arrays.msk.view("j")
        # self.arrays.msk.locked = True
        # self.arrays.mskv.view("j")
        # self.arrays.mskv.locked = True

        msk = self.arrays.msk.view("i")
        txporder = self.arrays.tporderx.view("i")
        vyporder = self.arrays.vpordery.view("i")
        index_tracerflux(msk, txporder, self.param.MF_order)
        index_vortexforce(msk, vyporder, self.param.VF_order)

        msk = self.arrays.msk.view("j")
        typorder = self.arrays.tpordery.view("j")
        vxporder = self.arrays.vporderx.view("j")
        index_tracerflux(msk, typorder, self.param.MF_order)
        index_vortexforce(msk, vxporder, self.param.VF_order)

        # self.set_coriolis()
        self.arrays.tporderx.lock()
        self.arrays.vporderx.lock()
        self.arrays.tpordery.lock()
        self.arrays.vpordery.lock()

    def msku(self):
        mskc = self.arrays.msk.view("i")
        ny, nx = mskc.shape
        msku = np.zeros((ny, nx+1), dtype=np.int8)
        for j in range(ny):
            for i in range(1, nx):
                if mskc[j, i-1]+mskc[j, i] == 2:
                    msku[j, i] = 1
        return msku

    def mskv(self):
        mskc = self.arrays.msk.view("j")
        ny, nx = mskc.shape
        msku = np.zeros((ny, nx+1), dtype=np.int8)
        for j in range(ny):
            for i in range(1, nx):
                if mskc[j, i-1]+mskc[j, i] == 2:
                    msku[j, i] = 1
        return msku


# @jit
def index_tracerflux(msk, order, maxorder):
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
        for i in range(nx+1):
            if (i >= 1) and (i < nx):
                m1 = m[i-1]+m[i]
            elif (i == 0):
                m1 = m[i]
            elif (i == nx):
                m1 = m[i-1]
            else:
                m1 = -1
            if (i >= 2) and (i < nx) and (m1 > 0):
                m3 = m[i-2]+m[i-1]+m[i]
            else:
                m3 = -1
            if (i >= 3) and (i < nx-1) and (m3 == 3):
                m5 = m[i-3]+m[i-2]+m[i-1]+m[i]+m[i+1]
            else:
                m5 = -1

            if (m5 == 5) and (maxorder >= 5):
                order[j, i] = 5
            elif (m3 == 3) and (maxorder >= 3):
                order[j, i] = 3
            elif (m1 >= 1) or (i == nx):
                order[j, i] = 1
            else:
                order[j, i] = 0


# @jit
def index_vortexforce(mskc, order, maxorder):
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

    mskv[mskv > 1] = 1

    for j in range(ny+1):
        m = mskv[j]
        for i in range(nx+1):
            if (i >= 0) and (i < nx):
                m1 = m[i]+m[i+1]
            elif i == nx:
                m1 = m[i]
            else:
                m1 = -1
            if (i >= 1) and (i < nx) and (m1 >= 0):
                m3 = m[i-1]+m[i]+m[i+1]
            else:
                m3 = -1
            if (i >= 2) and (i < nx-1) and (m3 >= 0):
                m5 = m[i-2]+m[i-1]+m[i]+m[i+1]+m[i+2]
            else:
                m5 = -1

            if (m5 >= 5) and (maxorder >= 5):
                order[j, i] = 5
            elif (m3 >= 3) and (maxorder >= 3):
                order[j, i] = 3
            elif m1 >= 0:
                order[j, i] = 1
            else:
                order[j, i] = 0
