from functools import wraps
from time import time
import pickle
import numpy as np
import socket

# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
stats = {}

# list of hostnames where doing a plot fails
blacklist = ["irene"]


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        if "forward" in stats.keys():
            if len(stats["forward"]) >= 100:
                ok = True
            else:
                ok = False
        else:
            ok = False
        if ok:
            result = f(*args, **kw)
        else:
            ts = time()
            result = f(*args, **kw)
            te = time()
            if 'timing' in kw.keys():
                pass
            else:
                name = f.__name__
                if name in stats.keys():
                    stats[name] += [te-ts]
                else:
                    stats[name] = [te-ts]

            #print('func:%r took: %2.4e sec' % (f.__name__,  te-ts))
        return result
    return wrap


def write_timings(path, myrank=0):
    if myrank == 0:
        with open('%s/timing.pkl' % path, 'bw') as fid:
            pickle.dump(stats, fid)

        infos = stats_from_timing(stats)
        string = string_stats(infos)
        with open('%s/timing.txt' % path, 'w') as fid:
            fid.write(string)


def basic_stats(data):
    stats = {
        "sum": np.sum(data),
        "count": len(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
    }
    return stats


def stats_from_timing(timing):
    infos = {}
    for key, val in timing.items():
        infos[key] = basic_stats(val)
    return infos


def string_stats(infos):
    string = []
    keys = list(infos.keys())
    firstkey = keys[0]
    stats = list(infos[firstkey].keys())
    maxlengthstats = max([len(s) for s in stats])

    n = 10
    header = "|"
    header += "".center(n)+"|"

    for key in keys:
        header += key[:n].center(n)+"|"
    N = len(header)
    sepline = ("+"+"-"*n)*(1+len(keys))+"+"

    string += [sepline, header, sepline]
    # print(sepline)
    # print(header)
    # print(sepline)

    for stat in stats:
        line = "|"
        line += stat.center(n)+"|"
        for key in keys:
            value = infos[key][stat]
            if isinstance(value, int):
                line += f"{value}".center(n)+"|"
            else:
                line += f"{value:.2e}".center(n)+"|"
        string += [line]

    string += [sepline]
    return "\n".join(string)


def analyze_timing(path, myrank=0):
    host = socket.gethostname()
    skip = any([machine in host for machine in blacklist])

    if (myrank == 0) and not(skip):

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams['font.size'] = 14
        mpl.rcParams['lines.linewidth'] = 2

        filename = '%s/timing.pkl' % path
        pngtiming = '%s/timing.png' % path

        f = open(filename, 'br')
        timing = pickle.load(f)

        mean = []
        total = []
        keys = []
        for k, vals in timing.items():
            mean += [np.mean(vals)]
            total += [np.sum(vals)]
            keys += [k]

        idx = np.argsort(mean)

        plt.figure(figsize=(10, 5))
        for k in idx[::-1]:
            vals = timing[keys[k]]
            plt.loglog(vals, label=keys[k], alpha=0.8)

        plt.xlabel('# call')
        plt.ylabel('time [s]')
        gca = plt.gca()
        gca.set_ylim([1e-8, 1e-0])
        plt.grid()
        plt.legend(loc='upper right', fontsize=10)
        plt.savefig(pngtiming)
