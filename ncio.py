import os
import numpy as np
import netCDF4 as nc
from collections import namedtuple

# Define the attributes of a netCDF variable
NetCDFVariable = namedtuple('NetCDFVariable', ['nickname', 'name', 'units'])


class Ncio(object):
    def __init__(self, param, grid, state):
        self.param = param

        full_hist_variables = [NetCDFVariable('u', 'velocity along x', 'm s-1'),
                               NetCDFVariable('v', 'velocity along y', 'm s-1'),
                               NetCDFVariable('h', 'vertical displacement', 'm'),
                               NetCDFVariable('pv', 'potential vorticity', 's-1')]

        self.hist_variables = full_hist_variables
        self.diag_variables = [NetCDFVariable('hm', 'mean height', 'm'),
                               NetCDFVariable('ke', 'mean kinetic energy', 'm2 s-2'),
                               NetCDFVariable('pe', 'mean potential energy', 'm2 s-2'),
                               NetCDFVariable('pvm', 'mean total potential vorticity', 's-1'),
                               NetCDFVariable('ens', 'mean enstrophy', 's-2')]

        self.hist_path = 'history.nc'
        self.create_history_file(state, grid)
        self.kt = 0
        self.ktdiag = 0

    def create_history_file(self, state, grid):
        with nc.Dataset(self.hist_path, "w", format='NETCDF4') as ncfile:
            # Store the experiment parameters
            attrs = {key: self.param[key] for key in self.param.toc}            
            #ncfile.setncatts(attrs)

            nx = len(grid.xc)
            ny = len(grid.yc)
            # Create the dimensions
            ncfile.createDimension("t", None)  # unlimited size
            ncfile.createDimension("tdiag", None)  # unlimited size
            ncfile.createDimension("xc", nx)
            ncfile.createDimension("yc", ny)
            ncfile.createDimension("xe", nx+1)
            ncfile.createDimension("ye", ny+1)
            ncfile.createDimension("z", self.param.nz)

            # Create the variables with one dimension
            v = ncfile.createVariable("n", int, ("t",))
            v.standard_name = "integration step in the model run"

            v = ncfile.createVariable("t", float, ("t",))
            v.standard_name = "time in the model run"
            v.units = "s"

            v = ncfile.createVariable("xc", float, ("xc",))
            v.standard_name = "x coordinate at cell centers"
            v.units = "m"

            v = ncfile.createVariable("xe", float, ("xe",))
            v.standard_name = "x coordinate at cell edges"
            v.units = "m"

            v = ncfile.createVariable("yc", float, ("yc",))
            v.standard_name = "y  coordinate at cell centers"
            v.units = "m"

            v = ncfile.createVariable("ye", float, ("ye",))
            v.standard_name = "y  coordinate at cell edges"
            v.units = "m"

            def createvar(nickname, dims, name, unit):
                v = ncfile.createVariable(nickname, float, tuple(dims))
                v.standard_name = name
                v.units = unit
                
            for nickname, var in state.variables.items():
                # Spatial dimensions are in reversed order, because
                # the arrays are stored like this in the Scalar class
                dims = ["t", "z"]
                if var["type"] == "scalar":
                    dims = ["t", "z", "yc", "xc"]
                    createvar(nickname, dims, var["name"], var["unit"])
                if var["type"] == "vorticity":
                    dims = ["t", "z", "ye", "xe"]
                    createvar(nickname, dims, var["name"], var["unit"])
                if var["type"] == "vector":
                    dims = ["t", "z", "yc", "xe"]
                    createvar(nickname+"x", dims, var["name"]+" x-component", var["unit"])
                    dims = ["t", "z", "ye", "xc"]
                    createvar(nickname+"y", dims, var["name"]+" y-component", var["unit"])
            # for variable in self.diag_variables:
            #     # Spatial dimensions are in reversed order, because
            #     # the arrays are stored like this in the Scalar class
            #     v = ncfile.createVariable(
            #         variable.nickname, float, ("tdiag"))
            #     v.standard_name = variable.name
            #     v.units = variable.units

        with nc.Dataset(self.hist_path, "r+") as ncfile:
            ncfile.variables["xc"][:] = grid.xc
            ncfile.variables["yc"][:] = grid.yc
            ncfile.variables["xe"][:] = grid.xe
            ncfile.variables["ye"][:] = grid.ye
            
    def dohis(self, state, time):
        with nc.Dataset(self.hist_path, "r+") as ncfile:
            ncfile.variables['n'][self.kt] = self.kt
            ncfile.variables['t'][self.kt] = time
            for nickname in state.toc:
                ncfile.variables[nickname][self.kt] = state.get(nickname).view("i")
        self.kt += 1

    def dodiag(self, diags, time):
        with nc.Dataset(self.hist_path, "r+") as ncfile:
            ncfile.variables['tdiag'][self.ktdiag] = time
            for key, val in diags.items():
                ncfile.variables[key][self.ktdiag] = val
        self.ktdiag += 1


if __name__ == '__main__':
    nx, ny = 40, 50
    io = Ncio({'nx': nx, 'ny': ny})
    h = np.random.uniform(size=(ny, nx))
    u = np.random.uniform(size=(ny, nx+1))
    v = np.random.uniform(size=(ny+1, nx))
    pv = np.random.uniform(size=(ny, nx))
    state = [h, u, v, pv]
    t = 0.
    for kt in range(10):
        h[:, :] = np.random.uniform(size=(ny, nx))
        u[:, :] = np.random.uniform(size=(ny, nx+1))
        v[:, :] = np.random.uniform(size=(ny+1, nx))
        pv[:, :] = np.random.uniform(size=(ny, nx))
        io.dohis(state, t)
        t += 0.1
