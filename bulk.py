""" Bulk diagnostics
"""

import numpy as np

class Bulk(object):
    def __init__(self, param):
        self.U_max = 0.

    def compute(self, state):        
        U = state.U
        U_max = 0.
        for dim in "ij":
            U_max = max(U_max, max(np.abs(U[dim].view().flat)))
        self.U_max = U_max
