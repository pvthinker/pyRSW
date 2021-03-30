import numpy as np
import topology as topo


class Cartesian():
    def __init__(self, param):
        self.param = param
        infos = topo.get_domain_decomposition(param)
        self.dx = param.Lx / (param.npx*param.nx)
        self.dy = param.Ly / (param.npy*param.ny)
        self.i0 = infos['loc'][-1]*param.nx
        self.j0 = infos['loc'][-2]*param.ny
        #print(param.myrank, self.j0, self.i0)

    def x(self, j, i):
        return index(j+self.j0, i+self.i0, "i")*self.dx

    def y(self, j, i):
        return index(j+self.j0, i+self.i0, "j")*self.dy

    def idx2(self, j, i):
        return ones(j+self.j0, i+self.i0)*(1/self.dx**2)

    def idy2(self, j, i):
        return ones(j+self.j0, i+self.i0)*(1/self.dy**2)

    def area(self, j, i):
        return ones(j+self.j0, i+self.i0)*self.dx*self.dy


class Cylindrical():
    def __init__(self, param):
        self.param = param
        infos = topo.get_domain_decomposition(param)

        self.dtheta = (param.theta[1]-param.theta[0]) / (param.npx*param.nx)
        self.dr = (param.r[1]-param.r[0]) / (param.npy*param.ny)
        self.i0 = infos['loc'][-1]*param.nx
        self.j0 = infos['loc'][-2]*param.ny

    def theta(self, j, i):
        """ theta """
        return index(j+self.j0, i+self.i0, "i")*self.dtheta+self.param.theta[0]

    def r(self, j, i):
        """ r """
        return index(j+self.j0, i+self.i0, "j")*self.dr+self.param.r[0]

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


class Spherical():
    """
    theta : latitude
    phi : longitude
    """

    def __init__(self, param):
        self.param = param
        infos = topo.get_domain_decomposition(param)
        self.dtheta = (param.theta[1]-param.theta[0]) / (param.npy*param.ny)
        self.dphi = (param.phi[1]-param.phi[0]) / (param.npx*param.nx)
        self.a = param.sphere_radius
        self.i0 = infos['loc'][1]*param.nx
        self.j0 = infos['loc'][0]*param.ny

    def theta(self, j, i):
        """ theta """
        return index(j, i, "j")*self.dtheta+self.param.theta[0]

    def phi(self, j, i):
        """ phi """
        return index(j, i, "i")*self.dphi+self.param.phi[0]

    def x(self, j, i):
        return self.a*np.cos(self.theta(j, i))*self.phi(j, i)

    def y(self, j, i):
        return self.a*self.theta(j, i)

    def idx2(self, j, i):
        """ arc length along phi """
        return 1./(self.a*np.cos(self.theta(j, i))*self.dphi)**2

    def idy2(self, j, i):
        """ arc length along theta """
        return ones(j, i)/(self.a*self.dtheta)**2

    def area(self, j, i):
        return (self.a**2*np.cos(self.theta(j, i))*self.dtheta*self.dphi)


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
