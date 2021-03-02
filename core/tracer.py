import finitediff as fd


def rhstrac(state, rhs, param, grid, last=False):
    for tracname in state.tracers:
        trac = state.get(tracname)  # trac is a 'Field' instance
        dtrac = rhs.get(tracname)
        k0, k1, j0, j1, i0, i1 = trac.domainindices
        nz, ny, nx = trac.shape
        for direction in 'ij':
            velocity = state.U[direction].view(direction)
            order = grid.arrays.tporder[direction].view(direction)
            dfield = dtrac.view(direction)

            if param.linear:
                field = trac.view(direction)*0 + param.H
            else:
                field = trac.view(direction)

            if direction == "i":
                idx = (j0, j1, max(i0, 1), min(i1+1, nx))
            else:
                idx = (i0, i1, max(j0, 1), min(j1+1, ny))

            fd.upwindtrac(field, velocity, dfield,
                          order, param.MF_linear, *idx)
