""" Bulk diagnostics (integrated over the domain)
"""

import numpy as np
import finitediff as fd
from timing import timeit

class Bulk(object):
    def __init__(self, param, grid):
        self.param = param
        self.grid = grid
        self.U_max = 0.
        self.H_max = param.H
        self.cellarea = grid.arrays.vol.view("i")
        self.f0 = param.f0
        self.nprocs = np.prod(grid.procs)
        if self.nprocs > 1:
            from mpi4py import MPI
            self.MPI = MPI

    @timeit
    def computebulk(self, state, diags, fulldiag=False):

        h = state.h.view("i")

        if fulldiag:

            k0, k1, j0, j1, i0, i1 = state.h.domainindices

            msk = self.grid.arrays.msk.view("i")
            areav = self.grid.arrays.volv.view("i")
            vor = state.vor.view("i")
            pv = state.pv.view("i")
            ke = state.ke.view("i")
            p = state.p.view("i")
            KE = ke
            PE = p
            # ENS = (vor+self.f0*areav)**2  # <- this ain't a 2 form ...
            PV2 = pv**2  # <- multiplication with the 2-form h

            # the above "enstrophy"
            # is not conserved by the continuous RSWE

            # What is conserved is the potential enstrophy density
            # q**2. It is conserved on parcels

            # To compute the bulk potential enstrophy (integrated over
            # the domain) we need to be multiply the density with the
            # 2-form h to make it integrable !

            # in short: q**2 is materially conserved
            # and sum(q**2 * h) is the bulk conserved quantity

            K = fd.sum_horiz(ke, h, msk, j0, j1, i0, i1)
            P = fd.sum_horiz(p, h, msk, j0, j1, i0, i1)
            Q = fd.sum2_horiz(pv, h, j0, j1, i0, i1)

            if self.nprocs > 1:
                MPI = self.MPI
                K = MPI.COMM_WORLD.allreduce(K, op=MPI.SUM)
                P = MPI.COMM_WORLD.allreduce(P, op=MPI.SUM)
                Q = MPI.COMM_WORLD.allreduce(Q, op=MPI.SUM)

            P *= 0.5
            #Q *= 0.5
            diags["ke"] = K
            diags["pe"] = P
            diags["me"] = K+P
            #diags["enstrophy"] = Z
            diags["potenstrophy"] = Q

        else:
            # these diags are used only when param.auto_dt == True
            U = state.U
            U_max = 0.
            for dim in "ij":
                U_max = max(U_max, max(np.abs(U[dim].view().flat)))
            self.U_max = U_max

            self.H_max = max(np.sum(h/self.cellarea, axis=0).flat)
