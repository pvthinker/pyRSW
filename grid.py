"""Provide the grid on which pyRSW is discretized.

"""

import numpy as np
from variables import Scalar
import topology as topo

def set_domain_decomposition(param):
    topo.topology = param["geometry"]
    procs = [1, param["npy"], param["npx"]]
    myrank = 0#mpitools.get_myrank(procs)
    loc = topo.rank2loc(myrank, procs)
    neighbours = topo.get_neighbours(loc, procs)
    param["procs"] = procs
    param["myrank"] = myrank
    param["neighbours"] = neighbours
    param["loc"] = loc
    
class Grid(object):
    def __init__(self, param):

        set_domain_decomposition(param)

        # Copy needed parameters
        self.nx = param["nx"]
        self.ny = param["ny"]
        self.nz = param["nz"]

        self.npx = param["npx"]
        self.npy = param["npy"]

        self.Lx = param["Lx"]
        self.Ly = param["Ly"]

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

        # Define coordinates
        #  coordinates of the bottom left front corner of the subdomain
        x0 = param['loc'][1]*self.nx*self.dx
        y0 = param['loc'][0]*self.ny*self.dy

        var = {"name": "", "unit": "",
               "prognostic": False, "dimensions": ["y", "x"]}
        # at cell centers
        dummy = Scalar("dummy", var, param, stagg="")
        j0, j1, i0, i1 = dummy.domainindices
        self.xc = (0.5+np.arange(self.nx)-i0)*self.dx+x0
        self.yc = (0.5+np.arange(self.ny)-j0)*self.dy+y0
        # at cell edges
        dummy = Scalar("dummy", var, param, stagg="xy")
        j0, j1, i0, i1 = dummy.domainindices
        self.xe = (np.arange(self.nx+1)-i0)*self.dx+x0
        self.ye = (np.arange(self.ny+1)-j0)*self.dy+y0

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
        U[:] = u * self.idx2
        V[:] = v * self.idy2
        
