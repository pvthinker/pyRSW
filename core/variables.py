import numpy as np
import topology as topo
from timing import timeit

modelvar = {
    "h": {
        "type": "scalar",
        "name": "layer thickness",
        "dimensions": ["z", "y", "x"],
        "unit": "L",
        "constant": False,
        "prognostic": True},
    "u": {
        "type": "vector",
        "name": "covariant velocity",
        "dimensions": ["z", "y", "x"],
        "unit": "L^2.T^-1",
        "constant": False,
        "prognostic": True},
    "U": {
        "type": "vector",
        "name": "contravariant velocity",
        "dimensions": ["z", "y", "x"],
        "unit": "T^-1",
        "constant": False,
        "prognostic": False},
    "w": {
        "type": "scalar",
        "name": "vertical velocity",
        "dimensions": ["z", "y", "x"],
        "unit": "L.T^-1",
        "constant": False,
        "prognostic": False},
    "vor": {
        "type": "vorticity",
        "name": "vorticity",
        "dimensions": ["z", "y", "x"],
        "unit": "L^2.T^-1",
        "constant": False,
        "prognostic": False},
    "pv": {
        "type": "scalar",
        "name": "potential vorticity",
        "dimensions": ["z", "y", "x"],
        "unit": "L.T^-1",
        "constant": False,
        "prognostic": False},
    "ke": {
        "type": "scalar",
        "name": "kinetic energy",
        "dimensions": ["z", "y", "x"],
        "unit": "L^2.T^-2",
        "constant": False,
        "prognostic": False},
    "p": {
        "type": "scalar",
        "name": "pressure",
        "dimensions": ["z", "y", "x"],
        "unit": "L^2.T^-2",
        "constant": False,
        "prognostic": False},
}


class State(object):
    def __init__(self, param, variables):
        """
        Parameters
        ----------
        param: dict, model parameters
        variables: dict, one entry per variable

        Returns
        -------
        state: an object that stores all variables in one container
        """
        self.param = param
        self.variables = variables
        toc = {}
        for nickname, var in variables.items():

            if var["type"] == 'scalar':
                obj = Field(nickname, var, param)
                toc[nickname] = var["prognostic"]

            elif var["type"] == 'vector':
                obj = Vector(nickname, var, param)

            elif var["type"] == 'vorticity':
                obj = Field(nickname, var, param, stagg="xy")
                toc[nickname] = var["prognostic"]

            setattr(self, nickname, obj)

            if var["type"] == 'vector':
                for dir, axis in zip("xy", "ij"):
                    alt_nickname = nickname+dir
                    setattr(self, alt_nickname, obj[axis])
                    toc[alt_nickname] = var["prognostic"]

        self.toc = toc

    def __repr__(self):
        string = ["State"]
        string += ["List of variables:"]
        for nickname, var in self.variables.items():
            t = var["type"]
            name = var["name"]
            string += [f"  - {nickname:4}: {t:9} : {name}"]
        return "\n".join(string)

    def __getitem__(self, scalar_name):
        assert scalar_name in self.toc
        return self.get(scalar_name).view("i")

    def get(self, nickname):
        return getattr(self, nickname)

    def get_prognostic_scalars(self):
        """Return a list of the nicknames of the prognostic scalars."""
        return [v for v, prognostic in self.toc.items() if prognostic]

    def get_arrays(self):
        return list(self.toc.keys())

    def get_prognostic_variables(self):
        return [nickname for nickname, var in self.variables.items()
                if var["prognostic"]]

    def duplicate_prognostic_variables(self):
        """Return a new state with new arrays for prognostic variables."""
        prognostic_variables = {nickname: var for nickname,
                                var in self.variables.items() if var["prognostic"]}
        return State(self.param, prognostic_variables)

    def new_state_from(self, list_variables):
        variables = {nickname: var
                     for nickname, var in self.variables.items()
                     if nickname in list_variables}
        return State(self.param, variables)

    def set_to_zero(self):
        """ set prognostic variables to zero """
        for nickname, prognostic in self.toc.items():
            if prognostic:
                obj = self.get(nickname)
                obj.view()[:] = 0.


class Vector(object):
    def __init__(self, nickname, var, param):
        self.nickname = nickname
        self.type = var["type"]
        self.name = var["name"]
        self.unit = var["unit"]
        self.prognostic = var["prognostic"]
        self.dimensions = var["dimensions"]
        self.ndim = len(self.dimensions)
        infos = topo.get_domain_decomposition(param)
        self.neighbours = infos["neighbours"]

        # each component of Vector is a Field
        # a component is accessed via a dictionary
        vector = {}
        varcopy = var.copy()
        for key, dim in zip("ji", "yx"):
            name = f"{self.name} - {dim} component"
            varcopy["name"] = name
            vector[key] = Field(nickname+dim, varcopy, param, stagg=dim)
        setattr(self, nickname, vector)

    def __repr__(self):
        string = [f"Vector: {self.nickname}"]
        string += [f" - long name: {self.name}"]
        string += [f" - unit: {self.unit}"]
        string += [f" - prognostic : {self.prognostic}"]
        string += [f" - neighbours: {self.neighbours}"]
        string += [f" - components:"]
        vector = getattr(self, self.nickname)
        for dim in "ij":
            shape = vector[dim].shape
            string += [f"   - {dim}: {shape}"]
        return "\n".join(string)

    def __getitem__(self, component):
        vector = getattr(self, self.nickname)
        return vector[component]

    def lock(self):
        vector = getattr(self, self.nickname)
        for comp in "ij":
            vector[comp].lock()

    def unlock(self):
        vector = getattr(self, self.nickname)
        for comp in "ij":
            vector[comp].unlock()


class Field(object):
    def __init__(self, nickname, var, param, stagg=""):
        self.param = param
        self.nickname = nickname
        #self.type = var["type"]
        self.name = var["name"]
        self.unit = var["unit"]
        self.prognostic = var["prognostic"]
        self.dimensions = var["dimensions"]
        self.stagg = stagg
        self.ndim = len(self.dimensions)
        if "dtype" in var:
            self.dtype = var["dtype"]
        else:
            self.dtype = "d"  # default type for arrays is float8
        self.locked = False
        self.islocked = False

        nh = param["nh"]
        infos = topo.get_domain_decomposition(param)
        self.neighbours = infos["neighbours"]

        shape, domainindices = topo.get_shape_and_domainindices(
            param, self.dimensions, stagg)

        self.domainindices = tuple(domainindices)

        if self.ndim == 2:
            alt_shape = (shape[1], shape[0])
        elif self.ndim == 3:
            alt_shape = (shape[0], shape[2], shape[1])

        self.shape = tuple(shape)

        self.data = {
            "i": np.zeros(self.shape, dtype=self.dtype),
            "j": np.zeros(alt_shape, dtype=self.dtype)}
        self.activeview = "i"

    def __repr__(self):
        string = [f"Array: {self.nickname}"]
        string += [f" - long name: {self.name}"]
        string += [f" - unit: {self.unit}"]
        string += [f" - prognostic : {self.prognostic}"]
        string += [f" - neighbours: {self.neighbours}"]
        string += [f" - shape: {self.shape}"]
        return "\n".join(string)

    def __getitem__(self, elem):
        return self.view()[elem]

    def __setitem__(self, elem, val):
        self.view()[elem] = val

    def setview(self, idx):
        if (self.activeview != idx) and (not self.islocked):
            # copy the current array into the desired one
            current = self.data[self.activeview]
            desired = self.data[idx]
            if self.ndim == 2:
                axes = [1, 0]
            elif self.ndim == 3:
                axes = [0, 2, 1]
            self.flipaxes(current, desired, axes)
        self.activeview = idx

    @timeit
    def flipaxes(self, current, desired, axes):
        desired[:] = np.transpose(current, axes)

    def lock(self):
        """ synchronize all views and prevent transpose"""
        self.islocked = False
        idx = "ij".replace(self.activeview, "")
        self.setview(idx)
        self.islocked = True

    def unlock(self):
        """ restore desynchronized views and allow transpose"""
        self.islocked = False

    def view(self, idx=None):
        if idx == self.activeview or idx is None:
            pass
        else:
            self.setview(idx)
        return self.data[self.activeview].view()

    def setviewlike(self, scalar):
        """Set view in the same convention as scalar."""
        self.setview(scalar.activeview)

    def viewlike(self, scalar):
        """Return view in the same convention as scalar."""
        return self.view(scalar.activeview)

    def getproperunits(self, grid):
        if self.param.physicalunits:
            if self.nickname in ["h", "hb"]:
                area = grid.arrays.vol.view("i")
                return self.view("i")/area

            elif self.nickname == "ux":
                idx2 = grid.arrays.invdx.view("i")
                return self.view("i")*np.sqrt(idx2)

            elif self.nickname == "uy":
                idy2 = grid.arrays.invdy.view("i")
                return self.view("i")*np.sqrt(idy2)

            elif self.nickname in ["f", "vor"]:
                area = grid.arrays.volv.view("i")
                return self.view("i")/area

            else:
                return self.view("i")

        else:
            return self.view("i")


if __name__ == "__main__":
    from parameters import Param
    param = Param()
    topo.topology = param.geometry

    state = State(param, modelvar)
    u = state.u
    print(u)

    # to access the numpy array of the x-component
    ux = u["i"].view()
    ux[:, 2, :] = 1.
    print(ux)

    # to flip i and j axes
    ux = u["i"].view("j")
    print(ux)

    # another way to assign
    ux.flat[:] = np.arange(len(ux.flat))
    print(ux)

    print(state)
