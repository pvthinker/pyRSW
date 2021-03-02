import finitediff as fd

def rhstrac(state, rhs, param, grid, last=False):
    for tracname in state.tracers:
        trac = state.get(tracname)  # trac is a 'Field' instance
        dtrac = rhs.get(tracname)

        for direction in 'ij':
            velocity = state.U[direction].view(direction)
            order = grid.arrays.tporder[direction].view(direction)
            dfield = dtrac.view(direction)

            if param.linear:
                field = trac.view(direction)*0 + param.H
            else:
                field = trac.view(direction)

            fd.upwindtrac(field, velocity, dfield, order, param.MF_linear)
