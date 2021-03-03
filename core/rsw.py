from ncio import Ncio
import plotting
import numpy as np
import sys
import signal
import variables
from bulk import Bulk
import parameters
import time
import os

fullycompiled = True
# if these modules aren't yet compiled, do it
try:
    import finitediff
except:
    import finitedifferences as FD
    FD.compile()
    fullycompiled = False

try:
    import optimize
except:
    import optimizers as OPT
    OPT.compile()
    fullycompiled = False


# operators, tracer and timescheme need to be imported AFTER the compilation
import operators
import tracer
import timescheme as ts

if not fullycompiled:
    print("compilation completed".center(80, "-"))

# can be imported ONLY IF compilation is done

# check pyRSW is properly installed
if os.path.isdir(parameters.configdir):
    if os.path.isfile(parameters.paramfile):
        pass
    else:
        raise ValueError("Please copy defaults.yaml in your ~/.pyrsw")
else:
    raise ValueError("Please install pyrsw first, with ./install.sh")


class RSW(object):
    def __init__(self, param, grid):
        self.param = param
        self.grid = grid
        self.outershape = grid.outershape
        self.banner()
        self.state = variables.State(param, variables.modelvar)
        self.state.tracers = ["h"]  # <- TODO improve for the general case
        self.fix_density()
        self.bulk = Bulk(param, grid)
        self.diags = {}
        self.timescheme = ts.Timescheme(param, self.state)
        self.timescheme.set(self.rhs, self.diagnose_var)
        self.t = 0.
        if param.myrank == 0:
            self.print_recap()

    def fix_density(self):
        """ Transform param.rho into a numpy array
        for use in the montgomery computation"""
        if self.param.nz == 1:
            if isinstance(self.param.rho, float):
                self.param.rho = np.asarray([self.param.rho])
        else:
            assert len(self.param.rho) == self.param.nz
            self.param.rho = np.asarray(self.param.rho)

    def run(self):
        nprocs = np.prod(self.param.procs)
        if nprocs > 1:
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()

        self.io = Ncio(self.param, self.grid, self.state)

        if self.param.myrank == 0:
            print(f"Creating output file: {self.io.hist_path}")
            print(f"Backing up script to: {self.io.script_path}")
            self.io.backup_scriptfile(sys.argv[0])
            self.io.backup_paramfile()

        kite = 0
        self.diagnose_var(self.state)
        self.diagnose_supplementary(self.state)

        self.bulk.compute(self.state, self.diags, fulldiag=True)

        signal.signal(signal.SIGINT, self.signal_handler)

        self.io.create_history_file(self.state)
        self.io.dohis(self.state, self.t)
        nexthistime = self.io.kt * self.param.freq_his

        self.io.create_diagnostic_file(self.diags)
        self.io.dodiags(self.diags, self.t, kite)
        nextdiagtime = self.io.ktdiag * self.param.freq_diag

        if self.param.plot_interactive:
            fig = plotting.Figure(self.param, self.grid, self.state, self.t)

        tend = self.param.tend
        self.ok = self.t < tend
        self.termination_status = "Job completed as expected"

        ngridpoints = self.param.nx*self.param.ny*self.param.nz
        walltime = time.time()
        starttime = walltime
        blowup = False

        while self.ok:
            walltime0 = walltime

            if nprocs > 1:
                MPI.COMM_WORLD.Barrier()
            self.dt = self.compute_dt()
            blowup = self.forward()
            if blowup:
                self.ok = False
                self.termination_status = "BLOW UP detected"

            if self.t >= tend:
                self.ok = False
            else:
                kite += 1
                if self.param.auto_dt:
                    self.t += self.dt
                else:
                    self.t = kite*self.dt

            if self.param.plot_interactive and (kite % self.param.freq_plot) == 0:
                if self.param.plotvar == "pv":
                    self.diagnose_supplementary(self.state)

                fig.update(self.t, self.state)

            timetohis = (self.t >= nexthistime)
            timetodiag = (self.t >= nextdiagtime)

            if timetohis or timetodiag:
                self.diagnose_supplementary(self.state)

            if timetohis:
                self.io.dohis(self.state, self.t)
                nexthistime = self.io.kt * self.param.freq_his

            if timetodiag:
                self.bulk.compute(self.state, self.diags, fulldiag=True)
                self.io.dodiags(self.diags, self.t, kite)
                nextdiagtime = self.io.ktdiag * self.param.freq_diag

            walltime = time.time()
            perf = (walltime-walltime0)/ngridpoints
            tu = self.param.timeunit

            time_string = f"\r n={kite:3d} t={self.t/tu:.2f} dt={self.dt/tu:.4f} his={nexthistime/tu:.2f}/{tend/tu:.2f} perf={perf:.2e} s"
            if self.param.myrank == 0:
                print(time_string, end="", flush=True)

        if self.param.plot_interactive:
            fig.finalize()

        if blowup:
            # save the state for debug
            self.io.dohis(self.state, self.t)

        if self.param.myrank == 0:
            print()
            print(self.termination_status, flush=True)

            wt = walltime-starttime
            if kite > 0:
                print(f"Wall time: {wt:.3} s -- {wt/kite:.2e} s/iteration")
            print(f"Output written to: {self.io.hist_path}")

        if np.prod(self.param.procs) > 1:
            MPI.COMM_WORLD.Barrier()
            if blowup:
                print()
                print(f"blowup detected by rank={self.param.myrank}")
                MPI.COMM_WORLD.Abort()

    def forward(self):
        self.timescheme.forward(self.state, self.t, self.dt)
        h = self.state.h[:]
        blowup = any(np.isnan(h.flat)) #or any(h.flat < 0)
        return blowup

    def rhs(self, state, t, dstate, last=False):
        dstate.set_to_zero()
        # transport the tracers
        tracer.rhstrac(state, dstate, self.param, self.grid, last=last)
        # vortex force
        operators.vortex_force(state, dstate, self.param, self.grid)
        # bernoulli
        operators.bernoulli(state, dstate, self.param, self.grid)
        #
        if self.param.forcing and last:
            self.forcing.add(state, dstate, t)

    def diagnose_var(self, state, full=False):
        self.applybc(state.h, full=full)
        self.applybc(state.ux, full=full)
        self.applybc(state.uy, full=full)
        self.grid.cov_to_contra(state)

        state.vor[:] = 0.
        if not self.param.linear:
            operators.vorticity(state, self.grid, self.param.noslip)

            operators.kinenergy(state, self.param)

        self.applybc(state.vor)
        operators.montgomery(state, self.grid, self.param)

    def diagnose_supplementary(self, state):
        operators.comppv_c(state, self.grid)

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

    def applybc(self, scalar, full=False):
        self.grid.halo.fill(scalar, full=full)
        # return
        # ny, nx = self.outershape
        # if ("x" in self.param.geometry):
        #     nh = self.param.nh
        #     nh2 = nh+nh
        #     var = scalar.view("i")
        #     if self.param.openbc:
        #         for i in range(var.shape[-1]):
        #             var[:, :nh, i] = var[:, nh, i]
        #             var[:, -nh:, i] = var[:, -nh-1, i]
        #     else:
        #         if scalar.shape[-1] == nx:
        #             var[..., :nh] = var[..., -nh2:-nh]
        #             var[..., -nh:] = var[..., nh:nh2]
        #         else:
        #             var[..., :nh+1] = var[..., -nh2-1:-nh]
        #             var[..., -nh-1:] = var[..., nh:nh2+1]
        # if ("y" in self.param.geometry):
        #     nh = self.param.nh
        #     nh2 = nh+nh
        #     var = scalar.view("j")
        #     if self.param.openbc:
        #         for i in range(var.shape[-1]):
        #             var[:, :nh, i] = var[:, nh, i]
        #             var[:, -nh:, i] = var[:, -nh-1, i]
        #     else:
        #         if scalar.shape[-2] == ny:
        #             var[..., :nh] = var[..., -nh2:-nh]
        #             var[..., -nh:] = var[..., nh:nh2]
        #         else:
        #             var[..., :nh+1] = var[..., -nh2-1:-nh]
        #             var[..., -nh-1:] = var[..., nh:nh2+1]

    def compute_dt(self):

        if self.param.auto_dt:

            self.bulk.compute(self.state, self.diags)

            c_max = np.sqrt(self.param.g*self.bulk.H_max)
            U_max = self.bulk.U_max

            vmax = max(c_max, U_max)
            # vmax = c_max+U_max
            if vmax == 0.0:
                return self.param.dt_max
            else:
                dt = self.param.cfl / vmax
                return min(dt, self.param.dt_max)
        else:
            return self.param.dt

    def signal_handler(self, signal, frame):
        if self.param.myrank == 0:
            print('\n hit ctrl-C, stopping', end='')
        self.ok = False
        self.termination_status = "Job manually interrupted"

    def print_recap(self):
        param = self.param
        s = "s"
        if param.nz < 4:
            nblayers = {1: "one", 2: "two", 3: "three"}[param.nz]
            if param.nz == 1:
                s = ""
        else:
            nblayers = param.nz
        ny, nx = param.ny, param.nx
        npy, npx = param.npy, param.npx
        if not param.VF_linear and not param.MF_linear and (param.VF_order == 5) and (param.MF_order == 5):
            numerics = "numerics: weno 5th on vorticity and mass flux"
        else:
            numerics = "numerics: you're not using the best combination"

        ncores = np.prod(param.procs)
        if ncores > 1:
            parallel = f"parallel computation with {ncores} cores: {param.npy} x {param.npx}"
        else:
            parallel = "single core computation, no subdomains"
        print(f"  Experiment: {param.expname}")
        print(
            f"  grid size : {nblayers} layer{s} {npy*ny} x {npx*nx} in {param.coordinates} coordinates")
        print(f"  {numerics}")
        print(f"  {parallel}")
        print("")

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
