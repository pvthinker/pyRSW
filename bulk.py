""" Bulk diagnostics
"""

import numpy as np

class Bulk(object):
    def __init__(self, param, grid):
        self.U_max = 0.
        self.H_max = param.H
        self.cellarea = grid.dx*grid.dy

    def compute(self, state):        
        U = state.U
        U_max = 0.
        for dim in "ij":
            U_max = max(U_max, max(np.abs(U[dim].view().flat)))
        self.U_max = U_max

        h = state.h.view()
        self.H_max = max(np.sum(h,axis=0).flat)/self.cellarea
