import os
import shutil
import pickle
import numpy as np
from netCDF4 import Dataset


class Ncio(object):
    def __init__(self, param, grid, state):
        self.param = param
        self.grid = grid
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
        hisname = f"history_{param.myrank:02}.nc"
        diagname = f"diags.nc"
        self.hist_path = os.path.join(out_dir, hisname)
        self.diag_path = os.path.join(out_dir, diagname)
        self.script_path = os.path.join(out_dir, f"{expname}.py")
        self.output_directory = out_dir
        # Create the output directory if necessary
        if param.myrank == 0:
            if not os.path.isdir(self.output_directory):
                os.makedirs(self.output_directory)
        self.dtype = np.dtype(self.param.hisdtype)

        self.halo_included = param.halo_included

    def backup_paramfile(self):
        self.paramfile = f"{self.output_directory}/param.pkl"
        with open(self.paramfile, "wb") as fid:
            pickle.dump(self.param, fid)

    def backup_scriptfile(self, filename):
        if self.param.myrank == 0:
            shutil.copyfile(filename, self.script_path)

    def create_diagnostic_file(self, diags):
        if self.param.myrank == 0:
            with Dataset(self.diag_path, "w", format='NETCDF4') as nc:
                nc.createDimension("t", None)
                d = nc.createVariable("t", "f", ("t",))
                d.long_name = "model time"
                d = nc.createVariable("kt", "i", ("t",))
                d.long_name = "model iteration"

                for v in diags:
                    d = nc.createVariable(v, "f", ("t",))
                    d.long_name = v

    def create_history_file(self, state):
        self.gridvar = ["msk", "f", "hb"]
        # for nickname in state.variables:
        #     if state.variables[nickname]["constant"]:
        #         self.gridvar += [nickname]

        with Dataset(self.hist_path, "w", format='NETCDF4') as nc:
            # Store the experiment parameters
            attrs = {key: self.param[key] for key in self.param.toc}
            # replace bool with int and transform dict into str
            for key in attrs:
                if isinstance(attrs[key], bool):
                    attrs[key] *= 1
                if isinstance(attrs[key], dict):
                    attrs[key] = str(attrs[key])
                # force 32 bits integers (default is 64 in Python)
                if isinstance(attrs[key], int):
                    attrs[key] = np.int32(attrs[key])
                # force 32 bits in list of integers (namely, loc and procs)
                if isinstance(attrs[key], list):
                    for k, elem in enumerate(attrs[key]):
                        if isinstance(elem, int):
                            attrs[key][k] = np.int32(elem)

            nc.setncatts(attrs)

            self.ndim = self.grid.xc.ndim
            if self.ndim == 1:
                nx = len(self.grid.xc)
                ny = len(self.grid.yc)
            elif self.ndim == 2:
                if self.halo_included:
                    ny, nx = self.grid.xc.shape
                else:
                    ny, nx = self.grid.ny, self.grid.nx
            else:
                raise ValueError

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

            v = nc.createVariable("t", self.dtype, ("t",))
            v.standard_name = "time in the model run"
            v.units = "s"

            if self.ndim == 1:
                v = nc.createVariable("xc", self.dtype, ("xc",))
                v.standard_name = "x coordinate at cell centers"
                v.units = "m"

                v = nc.createVariable("xe", self.dtype, ("xe",))
                v.standard_name = "x coordinate at cell edges"
                v.units = "m"

                v = nc.createVariable("yc", self.dtype, ("yc",))
                v.standard_name = "y  coordinate at cell centers"
                v.units = "m"

                v = nc.createVariable("ye", self.dtype, ("ye",))
                v.standard_name = "y  coordinate at cell edges"
                v.units = "m"
            else:
                v = nc.createVariable("xc", self.dtype, ("yc", "xc"))
                v.standard_name = "x coordinate at cell centers"
                v.units = "m"

                v = nc.createVariable("xe", self.dtype, ("ye", "xe"))
                v.standard_name = "x coordinate at cell edges"
                v.units = "m"

                v = nc.createVariable("yc", self.dtype, ("yc", "xc"))
                v.standard_name = "y  coordinate at cell centers"
                v.units = "m"

                v = nc.createVariable("ye", self.dtype, ("ye", "xe"))
                v.standard_name = "y  coordinate at cell edges"
                v.units = "m"

        self.hisvar = []
        for nickname in self.param.var_to_save+self.gridvar:
            if nickname in state.variables:
                var = state.variables[nickname]
            elif nickname in self.grid.arrays.variables:
                var = self.grid.arrays.variables[nickname]
            else:
                raise ValueError
            # Spatial dimensions are in reversed order, because
            # the arrays are stored like this in the Field class
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

        if self.halo_included:
            with Dataset(self.hist_path, "r+") as nc:
                nc.variables["xc"][:] = self.grid.xc
                nc.variables["yc"][:] = self.grid.yc
                nc.variables["xe"][:] = self.grid.xe
                nc.variables["ye"][:] = self.grid.ye
        else:
            with Dataset(self.hist_path, "r+") as nc:
                j0, j1, i0, i1 = self.grid.arrays.hb.domainindices
                nc.variables["xc"][:] = self.grid.xc[j0:j1, i0:i1]
                nc.variables["yc"][:] = self.grid.yc[j0:j1, i0:i1]
                j0, j1, i0, i1 = self.grid.arrays.f.domainindices
                nc.variables["xe"][:] = self.grid.xe[j0:j1, i0:i1]
                nc.variables["ye"][:] = self.grid.ye[j0:j1, i0:i1]

        # store grid arrays (f, hb, msk)
        with Dataset(self.hist_path, "r+") as nc:
            for nickname in self.gridvar:
                var = self.grid.arrays.get(nickname)
                data = var.getproperunits(self.grid)
                if self.halo_included:
                    nc.variables[nickname][:] = data
                else:
                    j0, j1, i0, i1 = var.domainindices
                    nc.variables[nickname][self.kt] = data[j0:j1, i0:i1]

    def createvar(self, nickname, dims, name, unit):
        # print(f"create variable {nickname}")
        with Dataset(self.hist_path, "r+", format='NETCDF4') as nc:
            v = nc.createVariable(nickname, self.dtype, tuple(dims))
            v.standard_name = name
            v.units = unit
        if nickname not in self.gridvar:
            self.hisvar += [nickname]

    def dohis(self, state, time):
        with Dataset(self.hist_path, "r+") as nc:
            nc.variables['n'][self.kt] = self.kt
            nc.variables['t'][self.kt] = time
            for nickname in self.hisvar:
                data = state.get(nickname).getproperunits(self.grid)
                if self.halo_included:
                    nc.variables[nickname][self.kt] = data
                else:
                    k0, k1, j0, j1, i0, i1 = state.get(nickname).domainindices
                    nc.variables[nickname][self.kt] = data[:, j0:j1, i0:i1]
        self.kt += 1

    def dodiags(self, diags, time, kt):
        if self.param.myrank == 0:
            with Dataset(self.diag_path, "r+") as nc:
                nc.variables["t"][self.ktdiag] = time
                nc.variables["kt"][self.ktdiag] = kt
                for key, val in diags.items():
                    nc.variables[key][self.ktdiag] = val
        self.ktdiag += 1
