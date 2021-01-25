import tracer
from variables import State
import operators


class TwoLayers(object):

    def __init__(self, param, grid, linear=False):
        self.param = param
        self.linear = linear
        self.grid = grid
        self.traclist = ['h']
        self.state = State(param)

        self.orderA = param["orderA"]
        self.orderVF = param["orderVF"]
        self.orderKE = param["orderKE"]

        self.diff_coef = param['diff_coef']

        self.tracer = tracer.Tracer_numerics(param,
                                             grid, self.traclist, self.orderA, self.diff_coef)

    def diagnose_var(self, state):
        # Diagnostic variables
        cov_to_contra(state, self.grid)

        if not self.linear:
            operators.vorticity(state)
            operators.kinenergy(state, self.param)
            operators.montgomery(state, self.param)

    def rhs(self, state, t, dstate, last=False):
        dstate.set_to_zero()
        # transport the tracers
        self.tracer.rhstrac(state, dstate, last=last)
        # vortex force
        if not self.linear:
            operators.vortex_force(state, dstate, self.param)
        # bernoulli
        operators.bernoulli(state, dstate, self.param)

    def forward(self, t, dt):
        self.timescheme.forward(self.state, t, dt)
        return False


def cov_to_contra(state, grid):
    U = state.U["i"].view("i")
    V = state.U["j"].view("i")
    u = state.u["i"].view("i")
    v = state.v["j"].view("i")
    U[:] = u * grid.idx2
    V[:] = v * grid.idy2
