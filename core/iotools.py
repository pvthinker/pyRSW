from netCDF4 import Dataset
from dataclasses import dataclass, field
import os
import pickle
import sys
import shutil
import numpy as np
from variables import modelvar


@dataclass
class VariableInfo():
    nickname: str = ""
    dimensions: tuple = field(default_factory=lambda: ())
    name: str = ""
    units: str = ""
    dtype: str = "d"


class NetCDF_tools():
    """
    Basic class to create and write NetCDF files

    Parameters
    ----------
    filename : str
        The file name to be created.
    attrs : dict
        The global attributes.
    dimensions : list[(name, size), ...]
       The list of dimensions.
       size==None -> unlimited
    variables : list[VariableInfo, ...]
        The name of variable.dimensions should match one of dimensions.

    """

    def __init__(self, filename, attrs, dimensions, variables):
        self.filename = filename
        self.attrs = attrs
        self.dimensions = {dim[0]: dim[1] for dim in dimensions}
        self.variables = {var.nickname: var for var in variables}

    def create(self):
        """
        Create the empty NetCDF file with

        - attributes
        - dimensions
        - variables
        """
        with Dataset(self.filename, "w", format='NETCDF4') as nc:
            nc.setncatts(self.attrs)

            for dim, size in self.dimensions.items():
                nc.createDimension(dim, size)

            for infos in self.variables.values():
                assert isinstance(infos.dimensions, tuple)
                v = nc.createVariable(infos.nickname,
                                      infos.dtype,
                                      infos.dimensions)
                v.standard_name = infos.name
                v.units = infos.units

    def write(self, variables, nc_start={}, data_start={}):
        """
        Write variables

        Parameters
        ----------
        variables : list[(nickname, data), ...]
             where data is an ndarray
        nc_start : dict{name: (offset, size)}
             name : the dimension name
             offset : the offset of that dimension in the NetCDF file
             size : the size of data in that dimension

             If a dimension is not in nc_start it is assumed that
             the data has a size that matches the size defined in
             the NetCDF.
        data_start : dict{name: (offset, size)}
             same that nc_start but for the data in variables
        """
        with Dataset(self.filename, "r+") as nc:
            for nickname, data in variables.items():
                ncidx = self._get_idx(nickname, nc_start)
                if isinstance(data, np.ndarray):
                    dataidx = self._get_idx(nickname, data_start)
                    nc.variables[nickname][ncidx] = data[dataidx]
                else:
                    nc.variables[nickname][ncidx] = data

    def _get_idx(self, nickname, nc_start):
        """
        Return the tuple of slices

        to either slice through nc.variables or through data
        """
        infos = self.variables[nickname]
        ncidx = []
        for dim in infos.dimensions:
            if dim in nc_start:
                istart, size = nc_start[dim]
            else:
                istart, size = 0, self.dimensions[dim]
            if size is not None:
                ncidx += [slice(istart, istart+size)]
        return tuple(ncidx)


class Ncio():
    """
    Class that handles all the IO for pyRSW

    which includes
    - creating and writing model snapshots in the history.nc
    - creating and writing model bulk diagnostics in the diags.nc
    - saving the param.pkl file
    - saving the Python experiment script
    """

    def __init__(self, param, grid, batchindex=0):
        self.param = param
        self.grid = grid
        self.batchindex = batchindex

        self.nprocs = np.prod(grid.procs)
        if self.nprocs > 1:
            from mpi4py import MPI
            self.MPI = MPI

        self._create_output_directory()
        self.backup_config()

        hist_infos = get_hist_infos(param, grid)
        self.hist = NetCDF_tools(self.history_file, *hist_infos)
        if not self.singlefile or self.master:
            self.hist.create()
        self.hist_index = 0
        self.write_grid()

        diag_infos = get_diag_infos(param, grid)
        self.diag = NetCDF_tools(self.diag_file, *diag_infos)
        self.diag_index = 0
        if self.master:
            self.diag.create()

    def _create_output_directory(self):
        if self.master and not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

    @property
    def myrank(self):
        return self.grid.myrank

    @property
    def master(self):
        return self.myrank == 0

    @property
    def expname(self):
        return self.param["expname"]

    @property
    def singlefile(self):
        return self.param["singlefile"]

    @property
    def output_directory(self):
        datadir = os.path.expanduser(self.param["datadir"])
        return os.path.join(datadir, self.expname)

    @property
    def history_file(self):
        """
        Full path to the NetCDF history file
        """
        his = self._add_batchindex("history")
        basicname = f"{his}.nc"
        mpiname = f"{his}_{self.myrank:02}.nc"

        hisname = basicname if self.singlefile else mpiname
        return os.path.join(self.output_directory, hisname)

    @property
    def diag_file(self):
        """
        Full path to the NetCDF diagnostic file
        """
        diag = self._add_batchindex("diag")
        diagname = f"{diag}.nc"
        return os.path.join(self.output_directory, diagname)

    def _add_batchindex(self, filename):
        if self.param.restart:
            return filename + f"_{self.batchindex:02}"
        else:
            return filename

    def backup_config(self):
        """
        Backup the experiment configuration into the output directory

        - save param in the param.pkl
        - save the experiment Python script
        """
        if self.master and self.batchindex == 0:
            dest = f"{self.output_directory}/param.pkl"
            with open(dest, "wb") as fid:
                pickle.dump(self.param, fid)

            python_launch_script = sys.argv[0]
            dest = os.path.join(self.output_directory, f"{self.expname}.py")
            shutil.copyfile(python_launch_script, dest)

    def write_grid(self):
        """
        Write the model grid arrays into the NetCDF file (just once)
        """
        xc = self.grid.coord.x(0, self.grid.ic)[0]
        yc = self.grid.coord.y(self.grid.jc, 0)[:, 0]
        xe = self.grid.coord.x(0, self.grid.ie)[0]
        ye = self.grid.coord.y(self.grid.je, 0)[:, 0]
        layer = np.arange(self.grid.nz)
        msk = self.grid.arrays.msk.view("i")
        datagrid = {
            "x": xc,
            "y": yc,
            "xe": xe,
            "ye": ye,
            "layer": layer,
            "msk": msk
        }
        self._history_write_halo_mpi(datagrid)

    def write_hist(self, state, time, kt):
        """
        Write a model snapshot into the NetCDF file
        """
        datahist = {
            "time": time,
            "iteration": kt,
        }
        for name in self.param["var_to_save"]:
            vartype = modelvar[name]["type"]
            if vartype == "vector":
                for axis in "xy":
                    compname = name+axis
                    var = state.get(compname)
                    datahist[compname] = var.getproperunits(self.grid)
            else:
                var = state.get(name)
                datahist[name] = var.getproperunits(self.grid)

        nc_start = {"time": (self.hist_index, 1)}

        self._history_write_halo_mpi(datahist, nc_start=nc_start)

        self.hist_index += 1

    def _history_write_halo_mpi(self, data, nc_start={}):
        """
        Generic function to write data into the history NetCDF file

        handle the following special cases

        - write the arrays without the halo
        - write in a single history file, even if several MPI ranks
        """
        data_start = {}
        if not self.param.halo_included:
            j0, j1, i0, i1 = self.grid.arrays.hb.domainindices
            nx = self.param.nx
            ny = self.param.ny
            data_start["x"] = (i0, nx)
            data_start["y"] = (j0, ny)
            data_start["xe"] = (i0, nx+1)
            data_start["ye"] = (j0, ny+1)

        if self.singlefile:
            i0 = self.grid.loc[2]*self.param.nx
            j0 = self.grid.loc[1]*self.param.ny
            nc_start["x"] = (i0, nx)
            nc_start["y"] = (j0, ny)
            nc_start["xe"] = (i0, nx+1)
            nc_start["ye"] = (j0, ny+1)

        if self.singlefile and (self.nprocs > 1):
            # all MPI ranks write in the same file
            for rank in range(self.nprocs):
                if rank == self.myrank:
                    self.hist.write(data,
                                    nc_start=nc_start,
                                    data_start=data_start)
                self.MPI.COMM_WORLD.Barrier()

        else:
            # each rank writes in its own history file
            self.hist.write(data, nc_start=nc_start, data_start=data_start)

    def write_diags(self, diags, time, kt):
        """
        Write the domain integrated diagnostics into the NetCDF file
        """
        datadiag = {
            "time": time,
            "iteration": kt,
            "ke": diags["ke"],
            "pe": diags["pe"],
            "me": diags["me"],
            "enstrophy": diags["potenstrophy"],
        }
        start = {"time": (self.diag_index, 1)}

        if self.master:
            self.diag.write(datadiag, nc_start=start)

        self.diag_index += 1


def get_hist_infos(param, grid):
    attrs = {"model": "pyrsw",
             "author": "someone"}

    if param.halo_included:
        ny, nx = grid.xc.shape
    else:
        ny, nx = param.ny, param.nx
        if param.singlefile:
            nx *= param.npx
            ny *= param.npy

    nz = param.nz

    dims = [("time", None), ("layer", nz),
            ("x", nx), ("y", ny),
            ("xe", nx+1), ("ye", ny+1)]

    infos = [
        ("time", ("time",), "time", "s"),
        ("iteration", ("time",), "model iteration", "", "i4"),
        ("x", ("x",), "x coord at center", "m"),
        ("y", ("y",), "y coord at center", "m"),
        ("xe", ("xe",), "x coord at edge", "m"),
        ("ye", ("ye",), "y coord at edge", "m"),
        ("layer", ("layer",), "layer index", "", "i1"),
        ("msk", ("y", "x"), "mask at cell centers", "", "i1"),
    ]

    vardims = {
        "scalar": ("time", "layer", "y", "x"),
        "u": ("time", "layer", "y", "xe"),
        "v": ("time", "layer", "ye", "x"),
        "vorticity": ("time", "layer", "ye", "xe")
    }

    for name in param["var_to_save"]:
        longname = modelvar[name]["name"]
        units = modelvar[name]["unit"]
        vartype = modelvar[name]["type"]
        if vartype == "vector":
            infos += [(name+"x", vardims["u"], longname+" x-component", units)]
            infos += [(name+"y", vardims["v"], longname+" y-component", units)]
        else:
            infos += [(name, vardims[vartype], longname, units)]

    varinfos = [VariableInfo(*info) for info in infos]

    hist_infos = (attrs, dims, varinfos)

    return hist_infos


def get_diag_infos(param, grid):
    attrs = {"model": "pyrsw",
             "author": "someone"}

    dims = [("time", None)]

    infos = [
        ("time", ("time",), "time", "s"),
        ("iteration", ("time",), "model iteration", "", "i4"),
        ("ke", ("time",), "kinetic energy", "m^2 s^-2"),
        ("pe", ("time",), "mean available potential energy", "m^2 s^-2"),
        ("me", ("time",), "kinetic + potential energy", "m^2 s^-2"),
        ("enstrophy", ("time",), "mean enstrophy", "s^-2 m^-2"),
    ]
    varinfos = [VariableInfo(*info) for info in infos]

    diag_infos = (attrs, dims, varinfos)
    return diag_infos
