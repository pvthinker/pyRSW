import numpy as np
import timescheme as ts
import grid
from variables import State
from parameters import param
from bulk import Bulk


class RSW(object):
    def __init__(self, param):
        self.param = param
        self.grid = grid.Grid(param)
        self.state = State(param)
        self.bulk = Bulk(param)
        self.timescheme = ts.Timescheme(param, self.state)
        self.timescheme.set(self.rhs_prog, self.rhs_diag)
        self.t = 0.

    def run(self):
        ok = True
        while ok:
            self.dt = self.compute_dt()
            self.forward(self.t, self.dt)

            if self.t >= self.tend:
                ok = False
            self.bulk.compute()
            self.t += self.dt

    def rhs_prog(self, state, t, dstate, last=False):
        pass

    def rhs_diag(self, state):
        pass

    def compute_dt(self):
        if self.auto_dt:
            U_max = self.bulk.U_max
            if U_max == 0.0:
                return self.dt_max
            else:
                dt = self.cfl / U_max
                return min(dt, self.dt_max)
        else:
            return self.dt0
