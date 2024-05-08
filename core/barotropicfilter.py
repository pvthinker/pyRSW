import numpy as np
from scipy import sparse
from scipy.sparse import linalg

class BarotropicFilter:
    def __init__(self, param, grid):
        self.param = param
        if not param.barotropicfilter:
            return

        dt = param.dt
        Tc = dt
        g = param.g
        H = param.H
        nx = param.nx
        ny = param.ny
        nz = param.nz
        dx = param.Lx/(nx*param.npx)
        dy = dx

        self.btcst = g*Tc*dt
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Abt = set_barotropicmatrix(nx, ny, dx, dy, g, H, Tc, dt)
        self.Abt2 = set_barotropicmatrix(nx, ny, dx, dy, g, H, Tc, dt*.5)

        self.area = grid.arrays.vol.view("i")
        self.fu = np.zeros((ny, nx+1))
        self.fv = np.zeros((ny+1, nx))
        self.hstar = np.zeros((nz, ny, nx))
        self.dhstar = np.zeros((nz, ny, nx))

    def set_height_normalization(self, state):
        if not self.param.barotropicfilter:
            return

        h = state.h.view("i")
        self.cff = self.param.H/(np.sum(h, axis=0))
        #self.cff = self.param.H/(np.sum(h, axis=0))*self.area
        #H = np.sum(h, axis=0)
        #print(np.max(self.cff),np.max(H)/self.area)
        #stop
        #cff = 1./self.area
        self.cff = 1/self.area
        for k in range(self.nz):
            self.hstar[k] = h[k]*self.cff

    def compute_dstar(self, state, dstate,half=False):
        if not self.param.barotropicfilter:
            return

        h = state.h.view("i")
        u = state.u["i"].view("i")
        v = state.u["j"].view("i")

        dh = dstate.h.view("i")
        du = dstate.u["i"].view("i")
        dv = dstate.u["j"].view("i")

        self.fu[:] = 0.
        self.fv[:] = 0.

        for k in range(self.nz):
            self.dhstar[k] = dh[k]*self.cff

        for k in range(self.nz):
            self.fu[:, 1:-1] += du[k, :, 1:-1] * \
                (self.hstar[k, :, 1:]+self.hstar[k, :, :-1])
            self.fu[:, 1:-1] += u[k, :, 1:-1] * \
                (self.dhstar[k, :, 1:]+self.dhstar[k, :, :-1])

        for k in range(self.nz):
            self.fv[1:-1, :] += dv[k, 1:-1, :] * \
                (self.hstar[k, 1:, :]+self.hstar[k, :-1, :])
            self.fv[1:-1, :] += v[k, 1:-1, :] * \
                (self.dhstar[k, 1:, :]+self.dhstar[k, :-1, :])

        self.div = (self.fu[:, 1:]-self.fu[:, :-1]+self.fv[1:, :]-self.fv[:-1, :])

        if half:
            self.dstar = 0.25*self.btcst*linalg.spsolve(self.Abt2, self.div.ravel())
        else:
            self.dstar = 0.5*self.btcst*linalg.spsolve(self.Abt, self.div.ravel())

        self.dstar.shape = (self.ny, self.nx)

    def apply_dstar(self, state, dt):
        if not self.param.barotropicfilter:
            return

        u = state.u["i"].view("i")
        v = state.u["j"].view("i")
        dstar = self.dstar
        du = (dstar[:, 1:]-dstar[:, :-1])*dt
        dv = (dstar[1:, :]-dstar[:-1, :])*dt
        for k in range(self.nz):
            u[k, :, 1:-1] += du
        for k in range(self.nz):
            v[k, 1:-1, :] += dv


def set_barotropicmatrix(nx, ny, dx, dy, g, H, Tc, dt):
    N = nx*ny
    rows = np.zeros((5*N,), dtype="i")
    cols = np.zeros((5*N,), dtype="i")
    data = np.zeros((5*N,))
    count = 0
    G = np.arange(N)
    G.shape = (ny, nx)
    coef = -g*H*Tc*dt#/dx**2
    for j, i in np.ndindex((ny, nx)):
        I = G[j, i]
        diag = 0
        if i > 0:
            cols[count] = G[j, i-1]
            rows[count] = I
            data[count] = coef
            count += 1
            diag += coef
        if i < nx-1:
            cols[count] = G[j, i+1]
            rows[count] = I
            data[count] = coef
            count += 1
            diag += coef
        if j > 0:
            cols[count] = G[j-1, i]
            rows[count] = I
            data[count] = coef
            count += 1
            diag += coef
        if j < ny-1:
            cols[count] = G[j+1, i]
            rows[count] = I
            data[count] = coef
            count += 1
            diag += coef
        cols[count] = I
        rows[count] = I
        data[count] = dx**2-diag
        count += 1

    A = sparse.coo_matrix(
        (data[:count], (rows[:count], cols[:count])), shape=(N, N))

    return A.tocsr()

def set_barotropicmatrix9(nx, ny, dx, dy, g, H, Tc, dt):
    N = nx*ny
    rows = np.zeros((9*N,), dtype="i")
    cols = np.zeros((9*N,), dtype="i")
    data = np.zeros((9*N,))
    count = 0
    G = np.arange(N)
    G.shape = (ny, nx)
    coef = -0.5*g*H*Tc*dt/dx**2
    for j, i in np.ndindex((ny, nx)):
        I = G[j, i]
        diag = 0
        if i > 0:
            cols[count] = G[j, i-1]
            rows[count] = I
            data[count] = coef
            count += 1
            diag += coef
        if i < nx-1:
            cols[count] = G[j, i+1]
            rows[count] = I
            data[count] = coef
            count += 1
            diag += coef
        if j > 0:
            cols[count] = G[j-1, i]
            rows[count] = I
            data[count] = coef
            count += 1
            diag += coef
        if j < ny-1:
            cols[count] = G[j+1, i]
            rows[count] = I
            data[count] = coef
            count += 1
            diag += coef
        if i>0 and j>0:
            cols[count] = G[j-1, i-1]
            rows[count] = I
            data[count] = coef*.5
            count += 1
            diag += coef
        if i>0 and j<ny-1:
            cols[count] = G[j+1, i-1]
            rows[count] = I
            data[count] = coef*.5
            count += 1
            diag += coef
        if i<nx-1 and j>0:
            cols[count] = G[j-1, i+1]
            rows[count] = I
            data[count] = coef*.5
            count += 1
            diag += coef
        if i<nx-1 and j<ny-1:
            cols[count] = G[j+1, i+1]
            rows[count] = I
            data[count] = coef*.5
            count += 1
            diag += coef

        cols[count] = I
        rows[count] = I
        data[count] = 1-diag
        count += 1

    A = sparse.coo_matrix(
        (data[:count], (rows[:count], cols[:count])), shape=(N, N))

    return A.tocsr()
