import numpy as np
import iotools as io
import xarray as xr


def test_basic():
    nx = 128
    ny = 64

    x = np.arange(nx)
    y = np.arange(ny)

    ncfile = "basic.nc"

    attrs = {"model": "pyrsw",
             "author": "someone"}

    dims = [("time", None), ("i", nx), ("j", ny)]
    infos = [
        ("h", ("time", "j", "i"), "height", "m"),
        ("x", ("i",), "x coordinate", "m"),
        ("y", ("j",), "y coordinate", "m"),
        ("time", ("time",), "time", "s"),
        ("iteration", ("time",), "model iteration", "", "i4"),
        ("i", ("i",), "cell center index along i-axis", "", "i4"),
        ("j", ("j",), "cell center index along j-axis", "", "i4"),
    ]
    varinfos = [io.VariableInfo(*info) for info in infos]

    nc = io.NetCDF_tools(ncfile, attrs, dims, varinfos)
    nc.create()
    nc.write({"x": x,
              "y": y,
              "i": np.arange(nx),
              "j": np.arange(ny),
              })

    for kt in range(10):
        time = kt*1.
        h = np.random.uniform(size=(ny, nx))
        start = {"time": (kt, 1)}        
        nc.write({"time": time, "h": h, "iteration": kt}, nc_start=start)

    ds = xr.load_dataset(ncfile)
    print(ds)


def test_staggered():
    nx = 128
    ny = 64

    x = np.arange(nx)*0.1
    y = np.arange(ny)*0.1

    ncfile = "staggered.nc"

    attrs = {"model": "pyrsw",
             "author": "someone"}

    dims = [("time", None), ("i", nx), ("j", ny), ("ie", nx+1), ("je", ny+1)]
    infos = [
        ("time", ("time",), "time", "s"),
        ("i", ("i",), "cell center index along i-axis", "", "i4"),
        ("j", ("j",), "cell center index along j-axis", "", "i4"),
        ("ie", ("ie",), "cell vertex index along i-axis", "", "i4"),
        ("je", ("je",), "cell vertex index along i-axis", "", "i4"),
        #
        ("iteration", ("time",), "model iteration", "", "i4"),
        ("x", ("i",), "x coordinate", "m"),
        ("y", ("j",), "y coordinate", "m"),
        ("h", ("time", "j", "i"), "height", "m"),
        ("vor", ("time", "je", "ie"), "vorticity", "s^-1"),
        ("u", ("time", "j", "ie"), "velocity along i-axis", "m s^-1"),
        ("v", ("time", "je", "i"), "velocity along j-axis", "m s^-1"),
    ]
    varinfos = [io.VariableInfo(*info) for info in infos]

    nc = io.NetCDF_tools(ncfile, attrs, dims, varinfos)
    nc.create()
    nc.write({"x": x,
              "y": y,
              "i": np.arange(nx),
              "j": np.arange(ny),
              "ie": np.arange(nx+1),
              "je": np.arange(ny+1),
              })

    for kt in range(10):
        time = kt*1.
        h = np.random.uniform(size=(ny, nx))
        vor = np.random.uniform(size=(ny+1, nx+1))
        u = np.random.uniform(size=(ny, nx+1))
        v = np.random.uniform(size=(ny+1, nx))
        start = {"time": (kt, 1)}        
        nc.write({
            "time": time,
            "iteration": kt,
            "h": h,
            "u": u,
            "v": v,
            "vor": vor,
        }, nc_start=start)

    ds = xr.load_dataset(ncfile)
    print(ds)
    return ds



def test_subdomain():
    npx = 4
    npy = 4
    nx = 128
    ny = 64

    Lx = 2.
    Ly = 1.

    dx = Lx/(npx*nx)
    dy = Ly/(npy*ny)

    ncfile = "subdomain.nc"

    attrs = {"model": "pyrsw",
             "author": "someone"}

    dims = [("time", None), ("i", nx*npx), ("j", ny*npy)]
    infos = [
        ("h", ("time", "j", "i"), "height", "m"),
        ("x", ("i",), "x coordinate", "m"),
        ("y", ("j",), "y coordinate", "m"),
        ("time", ("time",), "time", "s"),
        ("iteration", ("time",), "model iteration", "", "i4"),
        ("i", ("i",), "cell center index along i-axis", "", "i4"),
        ("j", ("j",), "cell center index along j-axis", "", "i4"),
    ]
    varinfos = [io.VariableInfo(*info) for info in infos]

    nc = io.NetCDF_tools(ncfile, attrs, dims, varinfos)
    nc.create()

    def i(loc):
        return np.arange(nx)+loc*nx

    def j(loc):
        return np.arange(ny)+loc*ny

    def x(iloc):
        return i(iloc)*dx

    def y(jloc):
        return j(jloc)*dy

    def func(x, y,t):
        u = 0.05
        v = -0.05
        return np.cos((x-u*t)*np.pi*2)*np.sin((yy-v*t)*np.pi*2)

    for iloc in range(npx):
        i0 = iloc*nx
        start = {"i": (i0, nx)}
        nc.write({"x": x(iloc), "i": i(iloc)},nc_start=start)
    for jloc in range(npy):
        j0 = jloc*ny
        start = {"j": (j0, ny)}
        nc.write({"y": y(jloc), "j": j(jloc)},nc_start=start)

    for kt in range(10):
        time = kt*1.
        nc.write({"time": time, "iteration": kt}, nc_start={"time": (kt, 1)})
        for jloc in range(npy):
            j0 = jloc*ny
            for iloc in range(npx):
                i0 = iloc*nx
                xx, yy = np.meshgrid(x(iloc), y(jloc))
                h = func(xx, yy, time)
                start = {"i": (i0, nx), "j": (j0,ny), "time": (kt, 1)}
                nc.write({"h": h}, nc_start=start)

    ds = xr.load_dataset(ncfile)
    print(ds)

def test_diagfile():
    ncfile = "diagfile.nc"

    attrs = {"model": "pyrsw",
             "author": "someone"}

    dims = [("time", None)]
    infos = [
        ("time", ("time",), "time", "s"),
        ("iteration", ("time",), "model iteration", "", "i4"),        
        ("ke", ("time",), "mean kinetic energy", "m^2 s^-2"),
        ("enstrophy", ("time",), "mean enstrophy", "s^-2 m^-2"),
        ("pe", ("time",), "mean available potential energy", "m^2 s^-2"),
    ]
    varinfos = [io.VariableInfo(*info) for info in infos]

    nc = io.NetCDF_tools(ncfile, attrs, dims, varinfos)
    nc.create()

    for kt in range(10):
        time = kt*1.
        ke = np.cos(time*0.5)**2
        pe = 1-ke
        enstrophy = np.sin(time*0.5)
        start = {"time": (kt, 1)}
        nc.write({
            "time": time,
            "iteration": kt,
            "ke": ke,
            "pe": pe,
            "enstrophy": enstrophy,
            }, nc_start=start)
    
    ds = xr.load_dataset(ncfile)
    print(ds)
    
if __name__ == "__main__":
    test_basic()
    ds = test_staggered()

    ds = test_subdomain()

    test_diagfile()
