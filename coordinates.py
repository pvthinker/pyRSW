import numpy as np


class Cartesian():
    def __init__(self, param):
        self.param = param
        self.dx = param.Lx / (param.npx*param.nx)
        self.dy = param.Ly / (param.npy*param.ny)
        self.i0 = param['loc'][1]
        self.j0 = param['loc'][0]

    def x(self, j, i):
        return self.i0+index(j, i+self.i0, "i")*self.dx

    def y(self, j, i):
        return index(j+self.j0, i, "j")*self.dy

    def idx2(self, j, i):
        return ones(j, i)*(1/self.dx**2)

    def idy2(self, j, i):
        return ones(j, i)*(1/self.dy**2)

    def area(self, j, i):
        return ones(j, i)*self.dx*self.dy


class Cylindrical():
    def __init__(self, param):
        self.param = param
        self.dtheta = (param.theta[1]-param.theta[0]) / (param.npx*param.nx)
        self.dr = (param.r[1]-param.r[0]) / (param.npy*param.ny)
        self.i0 = param['loc'][1]
        self.j0 = param['loc'][0]

    def theta(self, j, i):
        """ theta """
        return index(j, i+self.i0, "i")*self.dtheta+self.param.theta[0]

    def r(self, j, i):
        """ r """
        return index(j+self.j0, i, "j")*self.dr+self.param.r[0]

    def x(self, j, i):
        return self.r(j, i)*np.cos(self.theta(j, i))

    def y(self, j, i):
        return self.r(j, i)*np.sin(self.theta(j, i))

    def idx2(self, j, i):
        """ arc length along theta """
        return 1./(self.r(j, i)*self.dtheta)**2

    def idy2(self, j, i):
        """ arc length along r """
        return ones(j, i)/(self.dr**2)

    def area(self, j, i):
        return (self.r(j, i)*self.dr*self.dtheta)


def index(j, i, direc):
    if isinstance(i, int):
        nx = 1
        i = np.asarray(i)
    else:
        nx = i.shape[-1]
    if isinstance(j, int):
        ny = 1
        j = np.asarray(j)
    else:
        ny = j.shape[0]
    if direc == "i":
        if i.ndim == 2:
            res = i
        elif i.ndim == 1:
            res = np.ones((ny, nx))*i[np.newaxis, :]
        else:
            raise ValueError
    elif direc == "j":
        if j.ndim == 2:
            res = j
        elif j.ndim == 1:
            res = np.ones((ny, nx))*j[:, np.newaxis]
        else:
            raise ValueError
    else:
        raise ValueError
    return res


def ones(j, i):

    try:
        nx = i.shape[-1]
    except:
        nx = 1
    try:
        ny = ny = j.shape[0]
    except:
        ny = 1
    # if isinstance(i, int):
    #     nx = 1
    # else:
    #     nx = i.shape[-1]
    # if isinstance(j, int):
    #     ny = 1
    # else:
    #     ny = j.shape[0]
    return np.ones((ny, nx))
