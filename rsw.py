import numpy as np
import sys
import signal
import variables
from bulk import Bulk

# if these modules aren't yet compiled, do it
try:
    import finitediff
except:
    import finitedifferences as FD
    FD.compile()

try:
    import optimize
except:
    import optimizers as OPT
    OPT.compile()

import timescheme as ts
import operators
import tracer
import plotting
from ncio import Ncio


class RSW(object):
    def __init__(self, param, grid):
        self.param = param
        self.grid = grid
        self.shape = grid.shape
        self.banner()
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

        self.io = Ncio(param, grid, self.state)
        if param.myrank == 0:
            print(f"Creating output file: {self.io.hist_path}")
            print(f"Backing up script to: {self.io.script_path}")
            self.io.backup_scriptfile(sys.argv[0])
            self.io.backup_paramfile()

        self.ok = True
        kite = 0
        self.diagnose_var(self.state)
        self.diagnose_supplementary(self.state)

        signal.signal(signal.SIGINT, self.signal_handler)

        self.io.create_history_file(self.state, self.grid)
        self.io.dohis(self.state, self.t)
        nexthistime = self.t + self.param.freq_his

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
                if self.param.plotvar == "pv":
                    self.diagnose_supplementary(self.state)

                fig.update(self.t)

            self.bulk.compute(self.state)

            if self.t >= nexthistime:
                self.diagnose_supplementary(self.state)
                self.io.dohis(self.state, self.t)
                nexthistime += self.param.freq_his

            time_string = f"\r n={kite:3d} t={self.t:.2f} dt={self.dt:.4f} nexthis={nexthistime:.2f}"
            print(time_string, end="")

        if self.param.plot_interactive:
            fig.finalize()
        if self.param.myrank == 0:
            print()
            if self.ok:
                print("Job is aborted")
            else:
                print("Job completed as expected")
            print(f"Output written to: {self.io.hist_path}")

    def rhs(self, state, t, dstate, last=False):
        dstate.set_to_zero()
        # transport the tracers
        tracer.rhstrac(state, dstate, self.param, last=last)
        # vortex force
        operators.vortex_force(state, dstate, self.param)
        # bernoulli
        operators.bernoulli(state, dstate, self.param)

    def diagnose_var(self, state):
        self.applybc(state.h)
        self.applybc(state.ux)
        self.applybc(state.uy)
        self.grid.cov_to_contra(state)

        state.vor[:] = 0.
        if not self.param.linear:
            operators.vorticity(state)
            operators.kinenergy(state, self.param)

        self.applybc(state.vor)
        if self.param.noslip:
            self.applynoslip(state)
        operators.montgomery(state, self.param)

    def diagnose_supplementary(self, state):
        operators.comppv(state)

    def applynoslip(self, state):
        if self.param.geometry == "closed":
            vor = state.vor.view("j")
            u = state.ux.view("j")
            vor[..., 0] = -u[..., 0]
            vor[..., -1] = -u[..., -1]

            vor = state.vor.view("i")
            v = state.uy.view("i")
            vor[..., 0] = v[..., 0]
            vor[..., -1] = -v[..., -1]

    def applybc(self, scalar):
        ny, nx = self.shape
        # if self.param.geometry == "closed":
        #     var = scalar.view("j")
        #     var[..., 0, :] = 0.
        #     var[..., -1, :] = 0.
        #     var = scalar.view("i")
        #     var[..., 0, :] = 0.
        #     var[..., -1, :] = 0.
        if self.param.geometry == "perio_x":
            nh = self.param.nh
            nh2 = nh+nh
            var = scalar.view("i")
            if scalar.shape[-1] == nx:
                var[..., :nh] = var[..., -nh2:-nh]
                var[..., -nh:] = var[..., nh:nh2]
            else:
                var[..., :nh+1] = var[..., -nh2-1:-nh]
                var[..., -nh-1:] = var[..., nh:nh2+1]

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

    def banner(self):
        logo = [
            "              _____   _______          __",
            "             |  __ \ / ____\ \        / /",
            "  _ __  _   _| |__) | (___  \ \  /\  / / ",
            " | '_ \| | | |  _  / \___ \  \ \/  \/ /  ",
            " | |_) | |_| | | \ \ ____) |  \  /\  /   ",
            " | .__/ \__, |_|  \_\_____/    \/  \/    ",
            " | |     __/ |                           ",
            " |_|    |___/                            ",
            "                                         "]
        if self.param.myrank == 0:
            print("Welcome to")
            for l in logo:
                print(" "*10+l)
