import os
import shutil
import numpy as np
from netCDF4 import Dataset


class Ncio(object):
    def __init__(self, param, grid, state):
        self.param = param

        self.hist_path = 'history.nc'
        # topography is not yet defined when Ncio is called
        # self.create_history_file(state, grid)
        self.kt = 0
        self.ktdiag = 0
        # Create paths for the in- and output
        datadir = os.path.expanduser(param.datadir)
        expname = param.expname
        out_dir = os.path.join(datadir, expname)
        if param.filemode == "overwrite":
            pass
        else:
            raise ValueError(f"{param.filemode} is not yet implemented")
        hisname = f"{expname}_{param.myrank:02}_hist.nc"
        self.hist_path = os.path.join(out_dir, hisname)
        self.script_path = os.path.join(out_dir, f"{expname}.py")
        self.output_directory = out_dir
        # Create the output directory if necessary
        if not os.path.isdir(self.output_directory):
            if param.myrank == 0:
                os.makedirs(self.output_directory)

    def backup_scriptfile(self, filename):
        shutil.copyfile(filename, self.script_path)

    def create_history_file(self, state, grid):
        self.gridvar = []
        for nickname in state.variables:
            if state.variables[nickname]["constant"]:
                self.gridvar += [nickname]

        with Dataset(self.hist_path, "w", format='NETCDF4') as nc:
            # Store the experiment parameters
            attrs = {key: self.param[key] for key in self.param.toc}
            # replace bool with int and transform dict into str
            for key in attrs:
                if isinstance(attrs[key], bool):
                    attrs[key] *= 1
                if isinstance(attrs[key], dict):
                    attrs[key] = str(attrs[key])
            nc.setncatts(attrs)

            nx = len(grid.xc)
            ny = len(grid.yc)
            # Create the dimensions
            nc.createDimension("t", None)  # unlimited size
            nc.createDimension("tdiag", None)  # unlimited size
            nc.createDimension("xc", nx)
            nc.createDimension("yc", ny)
            nc.createDimension("xe", nx+1)
            nc.createDimension("ye", ny+1)
            nc.createDimension("z", self.param.nz)

            # Create the variables with one dimension
            v = nc.createVariable("n", int, ("t",))
            v.standard_name = "integration step in the model run"

            v = nc.createVariable("t", float, ("t",))
            v.standard_name = "time in the model run"
            v.units = "s"

            v = nc.createVariable("xc", float, ("xc",))
            v.standard_name = "x coordinate at cell centers"
            v.units = "m"

            v = nc.createVariable("xe", float, ("xe",))
            v.standard_name = "x coordinate at cell edges"
            v.units = "m"

            v = nc.createVariable("yc", float, ("yc",))
            v.standard_name = "y  coordinate at cell centers"
            v.units = "m"

            v = nc.createVariable("ye", float, ("ye",))
            v.standard_name = "y  coordinate at cell edges"
            v.units = "m"

        self.hisvar = []
        for nickname in self.param.var_to_save+self.gridvar:
            var = state.variables[nickname]
            # Spatial dimensions are in reversed order, because
            # the arrays are stored like this in the Scalar class
            dims0 = []
            if not var["constant"]:
                dims0 += ["t"]
            if "z" in var["dimensions"]:
                dims0 += ["z"]
            if var["type"] == "scalar":
                dims = dims0 + ["yc", "xc"]
                self.createvar(nickname, dims, var["name"], var["unit"])
            if var["type"] == "vorticity":
                dims = dims0 + ["ye", "xe"]
                self.createvar(nickname, dims, var["name"], var["unit"])
            if var["type"] == "vector":
                dims = dims0 + ["yc", "xe"]
                self.createvar(nickname+"x", dims,
                               var["name"]+" x-component", var["unit"])
                dims = dims0 + ["ye", "xc"]
                self.createvar(nickname+"y", dims,
                               var["name"]+" y-component", var["unit"])

        with Dataset(self.hist_path, "r+") as nc:
            nc.variables["xc"][:] = grid.xc
            nc.variables["yc"][:] = grid.yc
            nc.variables["xe"][:] = grid.xe
            nc.variables["ye"][:] = grid.ye

        # plot constant variables (relative to grid properties: coriolis and bathymetry)
        with Dataset(self.hist_path, "r+") as nc:
            for nickname in self.gridvar:
                nc.variables[nickname][:] = state.get(nickname).view("i")

    def createvar(self, nickname, dims, name, unit):
        with Dataset(self.hist_path, "r+", format='NETCDF4') as nc:
            v = nc.createVariable(nickname, float, tuple(dims))
            v.standard_name = name
            v.units = unit
        if nickname not in self.gridvar:
            self.hisvar += [nickname]

    def dohis(self, state, time):
        with Dataset(self.hist_path, "r+") as nc:
            nc.variables['n'][self.kt] = self.kt
            nc.variables['t'][self.kt] = time
            for nickname in self.hisvar:
                nc.variables[nickname][self.kt] = state.get(nickname).view("i")
        self.kt += 1

    def dodiag(self, diags, time):
        with Dataset(self.hist_path, "r+") as nc:
            nc.variables['tdiag'][self.ktdiag] = time
            for key, val in diags.items():
                nc.variables[key][self.ktdiag] = val
        self.ktdiag += 1
