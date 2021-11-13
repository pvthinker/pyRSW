from parameters import Param
from grid import Grid
#from rsw import RSW
import glob
import os
import pickle
import ncio

def saverestart(model):

    batchindex = get_batchindex(model)

    # if batchindex == 0:
    #     backup_grid(model)

    backup_state(model, batchindex)

    return batchindex

def loadrestart(expname, batchindex=-1):
    p = Param()
    datadir = os.path.expanduser(p.datadir)
    output_dir = f"{datadir}/{expname}"

    param = get_param_from_pkl(output_dir)
    grid = Grid(param)
    model = RSW(param, grid)

    update_grid(model)
    grid.finalize()

    update_state(model, batchindex)

    return model

def update_state(model, batchindex):
    stateinfos = loadpkl(model, f"restart_{batchindex:02}")
    set_stateinfos(model, stateinfos)

def update_grid(model):
    gridinfos = loadpkl(model, "grid")
    print(list(gridinfos.keys()))
    set_gridinfos(model, gridinfos)

def backup_state(model, batchindex):
    stateinfos = get_stateinfos(model)
    savepkl(model, f"restart_{batchindex:02}", stateinfos)

def backup_grid(model):
    gridinfos = get_gridinfos(model)
    savepkl(model, "grid", gridinfos)

def loadpkl(model, objectname):
    output_dir = ncio.get_expdir(model.param)#model.io.output_directory
    myrank = model.grid.myrank
    pkldir = f"{output_dir}/{objectname}"
    pklfile = f"{pkldir}/{objectname}_{myrank:04}.pkl"
    with open(pklfile, "rb") as fid:
        infos = pickle.load(fid)
    return infos

def savepkl(model, objectname, infos):
    output_dir = ncio.get_expdir(model.param)#model.io.output_directory
    myrank = model.grid.myrank
    pkldir = f"{output_dir}/{objectname}"
    pklfile = f"{pkldir}/{objectname}_{myrank:04}.pkl"

    assert not os.path.isdir(pkldir), f"{pkldir} already exists"
    os.makedirs(pkldir)

    with open(pklfile, "wb") as fid:
        pickle.dump(infos, fid)

def get_batchindex(model):
    output_dir = ncio.get_expdir(model.param)#model.io.output_directory
    restarts = glob.glob(f"{output_dir}/restart_*")
    batchindex = len(restarts)
    return batchindex

def get_gridinfos(model):
    grid = model.grid
    gridinfos = {"msk": grid.arrays.msk.view("i")}
    if hasattr(grid, "boundary"):
        gridinfos["boundary"] = copy.deepcopy(grid.boundary)
    return gridinfos

def set_gridinfos(model, gridinfos):
    grid = model.grid
    grid.arrays.msk.view("i").flat = gridinfos["msk"]
    if "boundary" in gridinfos:
        grid.boundary = gridinfos["boundary"]

def get_stateinfos(model):
    stateinfos = {}
    state = model.state
    prognostic_scalars = state.get_prognostic_scalars()
    for scalar in prognostic_scalars:
        stateinfos[scalar] = state.get(scalar).view("i")
    stateinfos["t"] = model.t
    stateinfos["kite"] = model.kite
    return stateinfos

def set_stateinfos(model, stateinfos):
    state = model.state
    prognostic_scalars = state.get_prognostic_scalars()
    for scalar in prognostic_scalars:
        state.get(scalar).view("i").flat = stateinfos[scalar]
    model.t = stateinfos["t"]
    model.kite = stateinfos["kite"]
    model.param.tend = model.t+model.param.duration


def get_param_from_pkl(output_dir):
    paramfile = f"{output_dir}/param.pkl"
    with open(paramfile, "rb") as fid:
        param = pickle.load(fid)
    return param

