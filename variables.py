import numpy as np

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
        "type": "vorticity",
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
                obj = Scalar(nickname, var, param)
                toc[nickname] = var["prognostic"]

            elif var["type"] == 'vector':
                obj = Vector(nickname, var, param)

            elif var["type"] == 'vorticity':
                obj = Scalar(nickname, var, param, stagg="xy")
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

    def get(self, nickname):
        return getattr(self, nickname)

    def get_prognostic_scalars(self):
        """Return a list of the nicknames of the prognostic scalars."""
        return [v for v, prognostic in self.toc.items() if prognostic]

    def duplicate_prognostic_variables(self):
        """Return a new state with new arrays for prognostic variables."""
        list_of_variables = []
        prognostic_variables = {nickname: var for nickname,
                                var in modelvar.items() if var["prognostic"]}
        return State(self.param, prognostic_variables)

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
        self.neighbours = param["neighbours"]

        # each component of Vector is a Scalar
        # a component is accessed via a dictionary
        vector = {}
        varcopy = var.copy()
        for key, dim in zip("ji", "yx"):
            name = f"{self.name} - {dim} component"
            varcopy["name"] = name
            vector[key] = Scalar(nickname+dim, varcopy, param, stagg=dim)
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


class Scalar(object):
    def __init__(self, nickname, var, param, stagg=""):
        self.param = param
        self.nickname = nickname
        #self.type = var["type"]
        self.name = var["name"]
        self.unit = var["unit"]
        self.prognostic = var["prognostic"]
        self.dimensions = var["dimensions"]
        self.ndim = len(self.dimensions)
        if "dtype" in var:
            self.dtype = var["dtype"]
        else:
            self.dtype = "d"  # default type for arrays is float8
        self.locked = False

        nh = param["nh"]
        neighbours = param["neighbours"]
        self.neighbours = neighbours

        shape = []
        domainindices = []
        for dim in "zyx":
            if dim == "x":
                leftneighb = (0, 0, -1) in neighbours
                rightneighb = (0, 0, +1) in neighbours
            elif dim == "y":
                leftneighb = (0, -1, 0) in neighbours
                rightneighb = (0, +1, 0) in neighbours
            else:
                leftneighb = rightneighb = False

            if dim in self.dimensions:
                nelem = param[f"n{dim}"]
                staggered = (dim in stagg)
                if staggered:
                    nelem += 1
                if leftneighb:
                    domainindices += [nh, nh+nelem]
                    nelem += nh
                else:
                    domainindices += [0, nelem]
                if rightneighb:
                    nelem += nh
                shape += [nelem]

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
        if self.activeview != idx and not self.locked:
            # copy the current array into the desired one
            current = self.data[self.activeview]
            desired = self.data[idx]
            if self.ndim == 2:
                axes = [1, 0]
            elif self.ndim == 3:
                axes = [0, 2, 1]
            desired[:] = np.transpose(current, axes)
        self.activeview = idx

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
            if self.nickname == "h":
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
    param = {"nz": 2, "ny": 256, "nx": 256, "nh": 2}
    neighbours = [(0, -1), (0, 1)]
    param["neighbours"] = neighbours
    param["rho"] = [1., 1.1]
    param["g"] = 1.

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
