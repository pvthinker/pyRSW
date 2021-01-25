import numpy as np

modelvar = {
    "h": {
        "type": "scalar",
        "name": "layer thickness",
        "dimensions": ["z", "y", "x"],
        "unit": "L",
        "prognostic": True},
    "hb": {
        "type": "scalar",
        "name": "total depth",
        "dimensions": ["y", "x"],
        "unit": "L",
        "prognostic": False},
    "f": {
        "type": "vorticity",
        "name": "coriolis",
        "dimensions": ["y", "x"],
        "unit": "L",
        "prognostic": False},
    "u": {
        "type": "vector",
        "name": "covariant velocity",
        "dimensions": ["z", "y", "x"],
        "unit": "L^2.T^-1",
        "prognostic": True},
    "U": {
        "type": "vector",
        "name": "contravariant velocity",
        "dimensions": ["z", "y", "x"],
        "unit": "T^-1",
        "prognostic": False},
    "w": {
        "type": "scalar",
        "name": "vertical velocity",
        "dimensions": ["z", "y", "x"],
        "unit": "L.T^-1",
        "prognostic": False},
    "vor": {
        "type": "vorticity",
        "name": "vorticity",
        "dimensions": ["z", "y", "x"],
        "unit": "L^2.T^-1",
        "prognostic": False},
    "ke": {
        "type": "scalar",
        "name": "kinetic energy",
        "dimensions": ["z", "y", "x"],
        "unit": "L^2.T^-2",
        "prognostic": False},
    "p": {
        "type": "scalar",
        "name": "pressure",
        "dimensions": ["z", "y", "x"],
        "unit": "L^2.T^-2",
        "prognostic": False},
}



class State(object):
    def __init__(self, param):
        for nickname, var in modelvar.items():

            if var["type"] == 'scalar':
                obj = Scalar(nickname, var, param)

            elif var["type"] == 'vector':
                obj = Vector(nickname, var, param)

            elif var["type"] == 'vorticity':
                obj = Scalar(nickname, var, param, stagg="xy")

            setattr(self, nickname, obj)

    def set_to_zero(self):
        """ set to zero prognostic variables """
        for nickname, var in modelvar.items():
            if var["prognostic"]:
                obj = getattr(self, nickname)
                if var["type"] in ["scalar", "vorticity"]:
                    obj.view().flat[:] = 0.
                else:
                    obj["i"].view().flat[:] = 0.
                    obj["j"].view().flat[:] = 0.
                    
            
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
        for key, dim in zip("ji", "yx"):
            name = f"{self.name} - {dim} component"
            var["name"] = name
            vector[key] = Scalar(nickname, var, param, stagg=dim)
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
        self.nickname = nickname
        self.type = var["type"]
        self.name = var["name"]
        self.unit = var["unit"]
        self.prognostic = var["prognostic"]
        self.dimensions = var["dimensions"]
        self.ndim = len(self.dimensions)

        nh = param["nh"]
        neighbours = param["neighbours"]
        self.neighbours = neighbours

        shape = []
        for dim in "zyx":
            if dim == "x":
                leftneighb = (0, -1) in neighbours
                rightneighb = (0, +1) in neighbours
            elif dim == "y":
                leftneighb = (-1, 0) in neighbours
                rightneighb = (+1, 0) in neighbours
            else:
                leftneighb = rightneighb = False

            if dim in self.dimensions:
                nelem = param[f"n{dim}"]
                staggered = (dim in stagg)
                if staggered:
                    nelem += 1
                if leftneighb:
                    nelem += nh
                if rightneighb:
                    nelem += nh
                shape += [nelem]

        if self.ndim == 2:
            alt_shape = (shape[1], shape[0])
        elif self.ndim == 3:
            alt_shape = (shape[0], shape[2], shape[1])

        self.shape = tuple(shape)

        self.data = {
            "i": np.zeros(self.shape),
            "j": np.zeros(alt_shape)}
        self.activeview = "i"

    def __repr__(self):
        string = [f"Array: {self.nickname}"]
        string += [f" - long name: {self.name}"]
        string += [f" - unit: {self.unit}"]
        string += [f" - prognostic : {self.prognostic}"]
        string += [f" - neighbours: {neighbours}"]
        string += [f" - shape: {self.shape}"]
        return "\n".join(string)

    def view(self, idx=None):
        if idx == self.activeview or idx is None:
            field = self.data[self.activeview].view()
        else:
            current = self.data[self.activeview].view()
            field = self.data[idx].view()
            if self.ndim == 2:
                field[:] = np.transpose(current, [1, 0])
            elif self.ndim == 3:
                field[:] = np.transpose(current, [0, 2, 1])
        return field


if __name__ == "__main__":
    param = {"nz": 2, "ny": 5, "nx": 3, "nh": 2}
    neighbours = [(0, -1), (0, 1)]
    param["neighbours"] = neighbours




class State(object):
    def __init__(self, param):
        for nickname, var in modelvar.items():

            if var["type"] == 'scalar':
                obj = Scalar(nickname, var, param)

            elif var["type"] == 'vector':
                obj = Vector(nickname, var, param)

            elif var["type"] == 'vorticity':
                obj = Scalar(nickname, var, param, stagg="xy")

            setattr(self, nickname, obj)

    def set_to_zero(self):
        """ set to zero prognostic variables """
        for nickname, var in modelvar.items():
            if var["prognostic"]:
                obj = getattr(self, nickname)
                if var["type"] in ["scalar", "vorticity"]:
                    obj.view().flat[:] = 0.
                else:
                    obj["i"].view().flat[:] = 0.
                    obj["j"].view().flat[:] = 0.
                    
            
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
        for key, dim in zip("ji", "yx"):
            name = f"{self.name} - {dim} component"
            var["name"] = name
            vector[key] = Scalar(nickname, var, param, stagg=dim)
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
        self.nickname = nickname
        self.type = var["type"]
        self.name = var["name"]
        self.unit = var["unit"]
        self.prognostic = var["prognostic"]
        self.dimensions = var["dimensions"]
        self.ndim = len(self.dimensions)

        nh = param["nh"]
        neighbours = param["neighbours"]
        self.neighbours = neighbours

        shape = []
        for dim in "zyx":
            if dim == "x":
                leftneighb = (0, -1) in neighbours
                rightneighb = (0, +1) in neighbours
            elif dim == "y":
                leftneighb = (-1, 0) in neighbours
                rightneighb = (+1, 0) in neighbours
            else:
                leftneighb = rightneighb = False

            if dim in self.dimensions:
                nelem = param[f"n{dim}"]
                staggered = (dim in stagg)
                if staggered:
                    nelem += 1
                if leftneighb:
                    nelem += nh
                if rightneighb:
                    nelem += nh
                shape += [nelem]

        if self.ndim == 2:
            alt_shape = (shape[1], shape[0])
        elif self.ndim == 3:
            alt_shape = (shape[0], shape[2], shape[1])

        self.shape = tuple(shape)

        self.data = {
            "i": np.zeros(self.shape),
            "j": np.zeros(alt_shape)}
        self.activeview = "i"

    def __repr__(self):
        string = [f"Array: {self.nickname}"]
        string += [f" - long name: {self.name}"]
        string += [f" - unit: {self.unit}"]
        string += [f" - prognostic : {self.prognostic}"]
        string += [f" - neighbours: {neighbours}"]
        string += [f" - shape: {self.shape}"]
        return "\n".join(string)

    def view(self, idx=None):
        if idx == self.activeview or idx is None:
            field = self.data[self.activeview].view()
        else:
            current = self.data[self.activeview].view()
            field = self.data[idx].view()
            if self.ndim == 2:
                field[:] = np.transpose(current, [1, 0])
            elif self.ndim == 3:
                field[:] = np.transpose(current, [0, 2, 1])
        return field


if __name__ == "__main__":
    param = {"nz": 2, "ny": 256, "nx": 256, "nh": 2}
    neighbours = [(0, -1), (0, 1)]
    param["neighbours"] = neighbours
    param["rho"] = [1., 1.1]
    param["g"] = 1.

    state = State(param)
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
