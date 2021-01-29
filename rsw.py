import numpy as np
import signal
import timescheme as ts
import variables
from bulk import Bulk
import operators
import tracer
import plotting


class RSW(object):
    def __init__(self, param, grid):
        self.param = param
        self.grid = grid
        self.state = variables.State(param, variables.modelvar)
        self.state.tracers = ["h"]  # <- TODO improve for the general case
        self.set_coriolis()
        self.bulk = Bulk(param)
        self.timescheme = ts.Timescheme(param, self.state)
        self.timescheme.set(self.rhs, self.diagnose_var)
        self.t = 0.

    def set_coriolis(self):
        f = self.state.f
        f[:] = self.param.f0*self.grid.area

    def run(self):
        self.ok = True
        kite = 0
        self.diagnose_var(self.state)

        signal.signal(signal.SIGINT, self.signal_handler)

        if self.param.plot_interactive:
            fig = plotting.Figure(self.param, self.state, self.t)

        while self.ok:
            self.dt = self.compute_dt()
            self.timescheme.forward(self.state, self.t, self.dt)

            if self.t >= self.param.tend:
                self.ok = False
            else:
                self.t += self.dt
                kite += 1

            if self.param.plot_interactive and (kite % self.param.freq_plot) == 0:
                fig.update(self.t)

            self.bulk.compute(self.state)
            time_string = f"\r n={kite:3d} t={self.t:.2f} dt={self.dt:.4f}"
            print(time_string, end="")

        if self.param.plot_interactive:
            fig.finalize()

    def rhs(self, state, t, dstate, last=False):
        dstate.set_to_zero()
        # transport the tracers
        tracer.rhstrac(state, dstate, self.param, last=last)
        # vortex force
        operators.vortex_force(state, dstate, self.param)
        # bernoulli
        operators.bernoulli(state, dstate, self.param)

    def diagnose_var(self, state):
        self.applybc(state.ux)
        self.applybc(state.uy)
        self.grid.cov_to_contra(state)

        state.vor[:] = 0.
        if not self.param.linear:
            operators.vorticity(state)
            operators.kinenergy(state, self.param)

        self.applybc(state.vor)
        operators.montgomery(state, self.param)
        operators.comppv(state)

    def applybc(self, scalar):
        if self.param.geometry == "closed":
            var = scalar.view("j")
            var[..., 0, :] = 0.
            var[..., -1, :] = 0.
            var = scalar.view("i")
            var[..., 0, :] = 0.
            var[..., -1, :] = 0.

    def compute_dt(self):
        if self.param.auto_dt:
            U_max = self.bulk.U_max
            if U_max == 0.0:
                return self.param.dt_max
            else:
                dt = self.param.cfl / U_max
                return min(dt, self.param.dt_max)
        else:
            return self.param.dt

    def signal_handler(self, signal, frame):
        if self.param.myrank == 0:
            print('\n hit ctrl-C, stopping', end='')
        self.ok = False


