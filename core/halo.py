"""

Tools that implement the halo filling

"""

import itertools as itert
import numpy as np
from mpi4py import MPI
import topology as topo
import buffers_halo as bh


class Halo():
    def __init__(self, param):
        """

        scalar is an instance of the Scalar object

        create a halo method once we have a scalar available
        e.g. 'b' the buoyancy

        b = state.get('b')
        halo = Halo(b)

        we can now fill any Scalar or Vector with

        halo.fill(b)
        halo.fill(u)

        where u is a vector, e.g. u = state.get('u')

        halo defines it own preallocated buffers and MPI requests

        The local topology is known from param['neighbours'] stored
        in the scalar. It is worth saying that each halo only needs
        the local information = who are my neighbours. A halo does
        not need to know the global topology.

        """
        self.param = param
        neighbours = param['neighbours']
        nh = param['nh']
        self.neighbours = neighbours
        self.myrank = param["myrank"]
        self.nh = nh
        nz, ny, nx = shape = (param.nz, param.ny, param.nx)
        self.nz, self.ny, self.nx = nz, ny, nx

        self.centers = self.define_buffers_and_requests((nz, ny, nx))
        self.edges_u = self.define_buffers_and_requests((nz, ny, nx+1))
        self.edges_v = self.define_buffers_and_requests((nz, ny+1, nx))
        self.corners = self.define_buffers_and_requests((nz, ny+1, nx+1))

    def define_buffers_and_requests(self, shape):
        comm = MPI.COMM_WORLD
        nh = self.nh

        # allocate buffers
        sbuf = {}
        rbuf = {}
        directions = [(0, -1, 0), (0, +1, 0), (0, 0, -1), (0, 0, +1)]
        for direc in directions:
            bufshape = [nh, nh, nh]
            for l in range(3):
                if abs(direc[l]) == 0:
                    bufshape[l] = shape[l]
            sbuf[direc] = np.zeros(bufshape)
            rbuf[direc] = np.zeros(bufshape)

        # define MPI requests
        reqs = []
        reqr = []
        for direc, yourrank in self.neighbours.items():
            flipdirec = tuple([-k for k in direc])
            stag = 0
            rtag = 0
            for k, d in enumerate(direc):
                stag += (3**k)*(d+1)
                rtag += (3**k)*(-d+1)
            #rs = comm.Send_init(sbuf[direc], yourrank, tag=self.myrank+tag)
            #rr = comm.Recv_init(rbuf[direc], yourrank, tag=yourrank+ftag)
            rs = comm.Send_init(sbuf[direc], yourrank, tag=stag)
            rr = comm.Recv_init(rbuf[direc], yourrank, tag=rtag)
            reqs += [rs]
            reqr += [rr]

        return (sbuf, rbuf, reqs, reqr)

    def fill(self, field):
        """
        field is a Field
        """
        #print(f"fill halo {field.name}")

        if len(field.neighbours) > 0:
            data = field.view("i")
            nh = self.nh
            k0, k1, j0, j1, i0, i1 = field.domainindices

            nz, ny, nx = field.shape

            assert (i0 == 0) or (i0 == nh)
            assert (i1 == nx) or (i1 == nx-nh)

            infos = {"": self.centers,
                     "x": self.edges_u,
                     "y": self.edges_v,
                     "xy": self.corners}[field.stagg]

            ii = jj = 0
            if "x" in field.stagg:
                ii = 1
            if "y" in field.stagg:
                jj = 1
            sbuf, rbuf, reqs, reqr = infos

            assert len(reqs) == len(field.neighbours)
            assert len(reqr) == len(field.neighbours)

            # if self.param.myrank == 0:
            #     print(field.name, j0, j1, i0, i1)
            self.fillarray(data, nh, j0, j1, i0, i1, infos, jj, ii)

    def fillarray(self, x, nh, j0, j1, i0, i1, infos, jj, ii):
        """
        this is where all the MPI instructions are

        there are basically four steps:

        - 1) copy inner values of x into buffers
        - 2) send buffers
        - 3) receive buffers
        - 4) copy buffers into x halo

        we use predefined requests and lauch them with
        Prequest.Startall(). This must be used in
        conjunction with Prequest.Waitall()
        that ensures all requests have been completed

        """

        sbuf, rbuf, reqs, reqr = infos

        MPI.Prequest.Startall(reqr)

        # 1) halo to buffer
        west = sbuf[(0, 0, -1)]
        east = sbuf[(0, 0, +1)]
        south = sbuf[(0, -1, 0)]
        north = sbuf[(0, +1, 0)]
        #print(x.shape, west.shape, east.shape, south.shape,north.shape)
        # print(j0,j1,i0,i1)
        bh.a3_to_buf(x, west, east, south, north, j0, j1, i0, i1, jj, ii, nh)
        #print(west.shape, east.shape, south.shape,north.shape)

        # # 2)
        MPI.Prequest.Startall(reqs)

        # # 3)
        MPI.Prequest.Waitall(reqr)

        # 4) buffer to halo
        west = rbuf[(0, 0, -1)]
        east = rbuf[(0, 0, +1)]
        south = rbuf[(0, -1, 0)]
        north = rbuf[(0, +1, 0)]
        #print(x.shape, west.shape, east.shape, south.shape,north.shape)
        bh.buf_to_a3(x, west, east, south, north, j0, j1, i0, i1)
        #print(west.shape, east.shape, south.shape,north.shape)

        MPI.Prequest.Waitall(reqs)

    def _print(self):
        """
        Print the shape of send buffers

        (for debug purpose)
        """
        for key, buf in self.sbuf.items():
            print(self.myrank, key, np.shape(buf))
