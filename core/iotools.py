from netCDF4 import Dataset
from dataclasses import dataclass, field


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
            ncidx += [slice(istart, istart+size)]
        return tuple(ncidx)
