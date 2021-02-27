try:
    from pyaml import yaml
except:
    try:
        from ruamel import yaml
    except:
        raise ValueError("install Python module: pyaml or ruamel")

import os

configdir = os.path.expanduser("~/.pyrsw")
paramfile = f"{configdir}/defaults.yaml"


def get_param():
    with open(paramfile, "r") as f:
        p = yaml.load(f, Loader=yaml.Loader)
    topics = p.keys()
    param = {}
    for topic in topics:
        elems = p[topic].keys()
        for e in elems:
            #print(e, p[topic][e])
            param[e] = p[topic][e]["default"]
    return param


class Param(object):
    def __init__(self):
        param = get_param()
        self.toc = list(param.keys())
        for key, val in param.items():
            setattr(self, key, val)

    def __getitem__(self, key):
        """ allows to retrieve a value as if param was a dict """
        return getattr(self, key)

    def __setitem__(self, key, val):
        if key not in self.toc:
            self.toc += [key]
        setattr(self, key, val)


if __name__ == "__main__":

    param = {"nz": 2, "ny": 5, "nx": 3, "nh": 2}
    neighbours = [(0, -1), (0, 1)]
    param["neighbours"] = neighbours
    param["rho"] = [1., 1.1]
    param["g"] = 1.
