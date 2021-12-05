from netCDF4 import Dataset
from dataclasses import dataclass, field
import os
import glob
import pickle
import sys
import shutil
import variables


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

    def write(self, variables, nc_start={}):
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
        """
        with Dataset(self.filename, "r+") as nc:
            for nickname, data in variables.items():
                ncidx = self._get_ncidx(nickname, nc_start)
                nc.variables[nickname][ncidx] = data

    def _get_ncidx(self, nickname, nc_start):
        infos = self.variables[nickname]
        ncidx = []
        for dim in infos.dimensions:
            if dim in nc_start:
                istart, size = nc_start[dim]
            else:
                istart, size = 0, self.dimensions[dim]
            assert size is not None
            ncidx += [slice(istart, istart+size)]
        return tuple(ncidx)


class CurrentNcio():
    def __init__(self, param, grid, batchindex=0):
        pass

    def backup_paramfile(self):
        pass

    def backup_scriptfile(self, filename):
        pass

    def create_diagnostic_file(self, diags):
        pass

    def create_history_file(self, state):
        pass

    def createvar(self, nickname, dims, name, unit):
        pass

    def dohis(self, state, time):
        pass


class NewNcio():
    def __init__(self, param, grid, batchindex=0):
        self.param = param
        self.grid = grid
        self.batchindex = batchindex

        self._create_output_directory()
        self.backup_config()

        hist_infos = get_hist_infos(param, grid)
        self.hist = NetCDF_tools(self.history_file, *hist_infos)
        self.hist.create()
        self.hist_index = 0
        self.write_grid()

        diag_infos = get_diag_infos(param, grid)
        self.diag = NetCDF_tools(self.diag_file, *diag_infos)
        self.diag_index = 0
        self.diag.create()

    def _create_output_directory(self):
        if self.myrank == 0:
            if not os.path.isdir(self.output_directory):
                os.makedirs(self.output_directory)

    @property
    def myrank(self):
        return self.grid.myrank

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
        Return the full path to the NetCDF history file
        """
        his = self._add_batchindex("history")
        basicname = f"{his}.nc"
        mpiname = f"{his}_{self.myrank:02}.nc"

        hisname = basicname if self.singlefile else mpiname
        return os.path.join(self.output_directory, hisname)

    @property
    def diag_file(self):
        diag = self._add_batchindex("diag")
        diagname = f"{diag}.nc"
        return os.path.join(self.output_directory, diagname)

    def _add_batchindex(self, filename):
        if self.param.restart:
            return filename + f"_{self.batchindex:02}"
        else:
            return filename

    def backup_config(self):
        if self.myrank == 0:
            dest = f"{self.output_directory}/param.pkl"
            with open(dest, "wb") as fid:
                pickle.dump(self.param, fid)

            python_launch_script = sys.argv[0]
            dest = os.path.join(self.output_directory, f"{self.expname}.py")
            shutil.copyfile(python_launch_script, dest)

    def write_grid(self):
        xc = self.grid.coord.x(0, self.grid.ic)[0]
        yc = self.grid.coord.y(self.grid.jc, 0)[:, 0]
        xe = self.grid.coord.x(0, self.grid.ie)[0]
        ye = self.grid.coord.y(self.grid.je, 0)[:, 0]
        layer = list(range(self.grid.nz))
        msk = self.grid.arrays.msk.view("i")
        datagrid = {
            "x": xc,
            "y": yc,
            "xe": xe,
            "ye": ye,
            "layer": layer,
            "msk": msk
        }
        self.hist.write(datagrid)

    def write_hist(self, state, time, kt):
        datahist = {
            "time": time,
            "iteration": kt,
        }
        for name in self.param["var_to_save"]:
            vartype = variables.modelvar[name]["type"]
            if vartype == "vector":
                for axis in "xy":
                    compname = name+axis
                    var = state.get(compname)
                    datahist[compname] = var.getproperunits(self.grid)
            else:
                var = state.get(name)
                datahist[name] = var.getproperunits(self.grid)

        start = {"time": (self.hist_index, 1)}

        self.hist.write(datahist, nc_start=start)
        self.hist_index += 1

    def write_diags(self, diags, time, kt):
        datadiag = {
            "time": time,
            "iteration": kt,
            "ke": diags["ke"],
            "pe": diags["pe"],
            "me": diags["me"],
            "enstrophy": diags["potenstrophy"],
        }
        start = {"time": (self.diag_index, 1)}
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
        longname = variables.modelvar[name]["name"]
        units = variables.modelvar[name]["unit"]
        vartype = variables.modelvar[name]["type"]
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
