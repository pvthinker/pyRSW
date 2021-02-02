import numpy as np
import pickle
from netCDF4 import Dataset
from grid import Grid
from rsw import RSW
import plotting


def domovie(expdir, varname, extraparam={}):
    """Generate a movie from a history file
    
    Parameters
    ----------
    expdir: str, the experiment path

    varname: str, the variable nickname (h, vor, pv etc)

    extraparam: dict, plotting parameters. By default, it uses the
    parameters saved in the param.pkl file

    
    Example
    -------
    >>> domovie("/home/roullet/data/pyRSW/myexp2", "vor", {"colorscheme": "imposed", "cax": [-0.005, 0.008]})
    """
    
    paramfile = f"{expdir}/param.pkl"
    with open(paramfile, "rb") as fid:
        param = pickle.load(fid)

    for key, val in extraparam.items():
        setattr(param, key, val)

    param.generate_mp4 = True
    param.plotvar = varname

    grid = Grid(param)
    model = RSW(param, grid)

    hisfile = f"{expdir}/history_00.nc"
    with Dataset(hisfile) as nc:
        nt = len(nc.dimensions["t"])
        
        data = model.state.get(varname).view("i")
        
        kt = 0
        data[:] = nc.variables[varname][kt]
        time = nc.variables["t"][kt]
        fig = plotting.Figure(param, model.state, time)
        
        for kt in range(1, nt):
            data[:] = nc.variables[varname][kt]
            time = nc.variables["t"][kt]
            print(f"\rkt={kt}  t={time:.2}", end="")
            fig.update(time)

        fig.finalize()
