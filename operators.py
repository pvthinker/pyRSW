import finitediff as fd


def vorticity(state):
    # + dv/dx
    v = state.u["j"].view("i")
    vor = state.vor.view("i")
    fd.curl(v, vor, 1)

    # - du/dy
    u = state.u["i"].view("j")
    vor = state.vor.view("j")
    fd.curl(u, vor, -1)


def kinenergy(state, param):
    u = state.u["i"].view("j")
    U = state.U["i"].view("j")
    ke = state.ke.view("j")

    fd.ke(u, U, ke, 0)

    v = state.u["j"].view("i")
    V = state.U["j"].view("i")
    ke = state.ke.view("i")

    fd.ke(v, V, ke, 1)


def montgomery(state, param):
    rho0, rho1 = param["rho"]
    g = param["g"]
    h = state.h.view("i")
    hb = state.hb.view("i")
    p = state.p.view("i")

    fd.montgomery(h, hb, p, rho0, rho1, g)


def vortex_force(state, dstate, param):
    du = dstate.u["i"].view("j")
    V = state.U["j"].view("j")
    vor = state.vor.view("j")
    f = state.f.view("j")
    fd.vortex_force(V, f, vor, du, -1)

    dv = dstate.u["j"].view("i")
    U = state.U["i"].view("i")
    vor = state.vor.view("i")
    f = state.f.view("i")
    fd.vortex_force(U, f, vor, dv, 1)
    

def bernoulli(state, dstate, param):
    du = dstate.u["i"].view("i")
    p = state.p.view("i")
    ke = state.ke.view("i")
    fd.bernoulli(ke, p, du, param["linear"])

    dv = dstate.u["j"].view("j")
    p = state.p.view("j")
    ke = state.ke.view("j")
    fd.bernoulli(ke, p, dv,  param["linear"])
