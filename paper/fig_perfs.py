import numpy as np
import matplotlib.pyplot as plt
import pickle
from netCDF4 import Dataset
from matplotlib import ticker
plt.ion()

plt.close("all")


def plotperf(filename, ax):
    with open(filename, "rb") as fid:
        perfs = pickle.load(fid)

    res = []
    mt = []
    for key, stats in perfs.items():
        nxglo, npx = key
        print(nxglo)
        res.append(nxglo**2)
        mt.append(stats["meantime"]/nxglo**2)

        ax.semilogx(res, mt)
        ax.set_xlabel(r"$n_x\,n_y$")
        ax.set_ylabel(r"$N$ $T$ / $n_x\,n_y$")
        ax.grid()


ylim = [0, 2.5e-6]
fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
plotperf("perfs_irene.pkl", ax[0])
plotperf("perfs.pkl", ax[0])
# ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
ax[0].yaxis.set_major_formatter(
    ticker.EngFormatter(unit="s", useMathText=True))
ax[0].set_ylim(ylim)
ax[0].grid()
ax[0].set_title(r"$N=1$")

# ------------- WEAK SCALING PERF ---------------------------
# plt.figure()

res = []
cores = []
mt = []
npxs = np.asarray([1, 2, 4, 8, 16, 32])
ncores = npxs**2
for npx in npxs:
    filename = f"perfs_weak_{npx}.pkl"
    with open(filename, "rb") as fid:
        perfs = pickle.load(fid)

    for key, stats in perfs.items():
        #nxglo, npx = key

        nxglo = 100*npx
        print(nxglo)
        res.append(nxglo**2)
        cores.append(npx**2)
        mt.append(stats["meantime"]*npx**2/nxglo**2)


ax[1].semilogx(cores, mt, "o-")
ax[1].set_ylim(ylim)
ax[1].set_xlabel(r"$N$")
#ax[1].set_ylabel(r"$N$ $T$ / $(n_x\,n_y)$ [in s]")
xticks = ncores
ax[1].set_xticks(ncores)
# ax[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0g}"))
# lab=[]
# for k, x in enumerate(xticks):
#     lab.append(plt.Text(x,0, f"{x}"))
# ax[1].set_xticklabels(lab)

ax[1].xaxis.set_major_formatter(ticker.LogFormatter(base=2))
ax[1].grid()
ax[1].set_title(r"$n_x\,n_y=10^4\, N$")

plt.tight_layout()
plt.savefig("timing.png", dpi=200)


# ------------- ENERGY AND ENSTROPHY DISSIPATION ---------------------------

datadir = "/home/roullet/data/pyRSW"
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
nx = 100
for npx in npxs:
    nxglo = nx*npx
    expname = f"merging_{nxglo}_{npx}"
    ncfile = f"{datadir}/{expname}/diags.nc"
    with Dataset(ncfile) as nc:
        t = nc.variables["t"][:]
        ke = nc.variables["ke"][:]
        pe = nc.variables["pe"][:] - 0.5  # substract the background PE
        me = nc.variables["me"][:] - 0.5
        Z = nc.variables["potenstrophy"][:]
        # plt.semilogy(ke)
        # plt.semilogy(pe)
        ax[0].semilogy(t, (me[0]-me)/me[0])
        ax[1].plot(t, (Z[0]-Z)/Z[0], label=rf"{nxglo}$^2$")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$(E_0-E)$ / $E_0$")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$(Z_0-Z)$ / $Z_0$")
    ax[1].legend(fontsize="xx-small")

ax[0].grid()
ax[1].grid()
plt.tight_layout()

eps = 0.03
ax[0].annotate("a)", (eps, 1-eps), ha="left",
               va="top", xycoords="figure fraction")
ax[0].annotate("b)", (0.5, 1-eps), ha="left",
               va="top", xycoords="figure fraction")

plt.savefig("EZ_dissipation.png", dpi=200)
