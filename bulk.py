""" Bulk diagnostics (integrated over the domain)
"""

import numpy as np


class Bulk(object):
    def __init__(self, param, grid):
        self.grid = grid
        self.U_max = 0.
        self.H_max = param.H
        self.cellarea = grid.arrays.vol.view("i")
        self.f0 = param.f0

    def compute(self, state, diags, fulldiag=False):

        h = state.h.view("i")

        if fulldiag:
            msk = self.grid.arrays.msk.view("i")
            areav = self.grid.arrays.volv.view("i")
            vor = state.vor.view("i")
            pv = state.pv.view("i")
            ke = state.ke.view("i")
            p = state.p.view("i")
            KE = ke*h
            PE = p*h
            ENS = (vor+self.f0*areav)**2  # <- this ain't a 2 form ...
            PV2 = pv**2*h  # <- multiplication with the 2-form h

            # the above "enstrophy"
            # is not conserved by the continuous RSWE

            # What is conserved is the potential enstrophy density
            # q**2. It is conserved on parcels

            # To compute the bulk potential enstrophy (integrated over
            # the domain) we need to be multiply the density with the
            # 2-form h to make it integrable !

            # in short: q**2 is materially conserved
            # and sum(q**2 * h) is the bulk conserved quantity

            K = P = Z = Q = V = 0.
            for k in range(ke.shape[0]):
                K += np.sum(KE[k]*msk)
                P += np.sum(PE[k]*msk)
                Z += np.sum(ENS[k])
                Q += np.sum(PV2[k]*msk)

            P *= 0.5
            Z *= 0.5
            Q *= 0.5
            diags["ke"] = K
            diags["pe"] = P
            diags["me"] = K+P
            diags["enstrophy"] = Z
            diags["potenstrophy"] = Q

        else:
            # these diags are used only when param.auto_dt == True
            U = state.U
            U_max = 0.
            for dim in "ij":
                U_max = max(U_max, max(np.abs(U[dim].view().flat)))
            self.U_max = U_max

            self.H_max = max(np.sum(h/self.cellarea, axis=0).flat)
