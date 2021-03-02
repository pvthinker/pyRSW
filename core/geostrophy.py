"""
Set the velocity field in geostrophic balance,
or a higher balance, from the mass field


"""
import numpy as np
import finitediff as FD


def set_balance(model, nite=1):
    """
    from h, compute p, the pressure

    the geostrophic velocity is

    u0 = crossdel[p] / f

    then compute ke0 the kinetic energy of u0
    and dzeta0 the vorticity of u0

    u1 = crossdel[p+ke0] / (f+dzeta0)

    iterate nite times

    u(kt) = crossdel[p+ke(kt-1)] / (f+dzeta(kt-1))
    """
    model.diagnose_var(model.state)
    for kt in range(nite):
        # model.diagnose_var(model.state)

        msk = model.grid.arrays.msk.view("i")
        u = model.state.ux.view("i")
        v = model.state.uy.view("i")
        vor = model.state.vor.view("i")
        p = model.state.p.view("i")
        ke = model.state.ke.view("i")
        f = model.grid.arrays.f.view("i")
        dx2 = 1/model.grid.arrays.invdx.view("i")
        dy2 = 1/model.grid.arrays.invdy.view("i")

        FD.set_geostrophic_vel(u, v, vor, p, ke, dx2, dy2, f, msk)

        model.diagnose_var(model.state)

        u = model.state.ux.view("i")
        v = model.state.uy.view("j")

        msku = model.grid.msku()
        mskv = model.grid.mskv()

        for k in range(model.param.nz):
            u[k] *= msku
            v[k] *= mskv
