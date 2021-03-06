import numpy as np
import matplotlib.pyplot as plt
import pickle
from netCDF4 import Dataset
plt.ion()

plt.close("all")

with open("perfs.pkl", "rb") as fid:
    perfs = pickle.load(fid)

res  = []
mt = []
for key, stats in perfs.items():
    nxglo, npx = key
    print(nxglo)
    res.append(nxglo**2)
    mt.append(stats["meantime"]/nxglo**2)

plt.figure()
plt.loglog(res, mt)
plt.xlabel(r"$n_x\,n_y$")
plt.ylabel(r"time / $(n_t\,n_x\,n_y)$ [in s]")
plt.grid()
plt.tight_layout()

# ----------------------------------------

datadir = "/home/roullet/data/pyRSW"
fig, ax = plt.subplots(1,2)
for key, stats in perfs.items():
    nxglo, npx = key
    expname = f"merging_{nxglo}_{npx}"
    ncfile = f"{datadir}/{expname}/diags.nc"
    with Dataset(ncfile) as nc:
        t = nc.variables["t"][:]
        ke = nc.variables["ke"][:]
        pe = nc.variables["pe"][:]
        me = nc.variables["me"][:]
        Z = nc.variables["potenstrophy"][:]
        #plt.semilogy(ke)
        #plt.semilogy(pe)
        ax[0].semilogy(t, me[0]-me)
        ax[1].semilogy(t, Z[0]-Z)
        
