import finitediff as fd

def rhstrac(state, rhs, param, last=False):
    for tracname in state.tracers:
        trac = state.get(tracname)  # trac is a 'Scalar' instance
        dtrac = rhs.get(tracname)

        for direction in 'ij':
            velocity = state.U[direction].view(direction)
            if param.linear:
                field = trac.view(direction)*0 + param.H
            else:
                field = trac.view(direction)
            dfield = dtrac.view(direction)
            
            fd.upwindtrac(field, velocity, dfield)
        
