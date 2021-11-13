from numba.pycc import CC
import numpy as np

fillvalue = -9.


def compile(verbose=False):

    print("** Compile the buffershalo module with numba")

    cc = CC("buffers_halo")
    cc.verbose = verbose
    @cc.export("a3_to_buf", "f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], i4, i4, i4, i4, i1, i1, i1")
    def array3_to_buffer(field, ileft, iright, jleft, jright, j0, j1, i0, i1, jj, ii, nh):
        """
        jj, ii: staggering (+1 or 0)
        """
        nz, ny, nx = field.shape

        for k in range(nz):
            # Southern halo
            for j in range(j0):
                for i in range(i0, i1):
                    jleft[k, j, i-i0] = field[k, j+nh+jj, i]
            for j in range(j0, j1):
                # Western halo
                for i in range(i0):
                    ileft[k, j-j0, i] = field[k, j, i+nh+ii]
                # Eastern halo
                for i in range(i1, nx):
                    iright[k, j-j0, i-i1] = field[k, j, i-nh-ii]
            # Northern halo
            for j in range(j1, ny):
                for i in range(i0, i1):
                    jright[k, j-j1, i-i0] = field[k, j-nh-jj, i]

    @cc.export("a3_to_buf_full", "f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], i4, i4, i4, i4, i1, i1, i1")
    def array3_to_buffer_full(field, ileft, iright, jleft, jright, sw,se,nw,ne,j0, j1, i0, i1, jj, ii, nh):
        """
        jj, ii: staggering (+1 or 0)
        """
        nz, ny, nx = field.shape

        for k in range(nz):
            # Southern halo
            for j in range(j0):
                for i in range(i0):
                    sw[k,j,i] = field[k, j+nh+jj, i+nh+ii]
                for i in range(i0, i1):
                    jleft[k, j, i-i0] = field[k, j+nh+jj, i]
                for i in range(i1, nx):
                    se[k,j,i-i1] = field[k, j+nh+jj, i-nh-ii]

            for j in range(j0, j1):
                # Western halo
                for i in range(i0):
                    ileft[k, j-j0, i] = field[k, j, i+nh+ii]
                # Eastern halo
                for i in range(i1, nx):
                    iright[k, j-j0, i-i1] = field[k, j, i-nh-ii]

            # Northern halo
            for j in range(j1, ny):
                for i in range(i0):
                    nw[k,j-j1,i] = field[k, j-nh-jj, i+nh+ii]
                for i in range(i0, i1):
                    jright[k, j-j1, i-i0] = field[k, j-nh-jj, i]
                for i in range(i1, nx):
                    ne[k,j-j1,i-i1] = field[k, j-nh-jj, i-nh-ii]

    @cc.export("buf_to_a3", "f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], i4, i4, i4, i4")
    def buffer_to_array3(field, ileft, iright, jleft, jright, j0, j1, i0, i1):
        nz, ny, nx = field.shape

        for k in range(nz):
            # Southern halo
            for j in range(j0):
                for i in range(i0):
                    field[k, j, i] = fillvalue
                for i in range(i0, i1):
                    field[k, j, i] = jleft[k, j, i-i0]
                for i in range(i1, nx):
                    field[k, j, i] = fillvalue
            for j in range(j0, j1):
                # Western halo
                for i in range(i0):
                    field[k, j, i] = ileft[k, j-j0, i]
                # Eastern halo
                for i in range(i1, nx):
                    field[k, j, i] = iright[k, j-j0, i-i1]
            # Northern halo
            for j in range(j1, ny):
                for i in range(i0):
                    field[k, j, i] = fillvalue
                for i in range(i0, i1):
                    field[k, j, i] = jright[k, j-j1, i-i0]
                for i in range(i1, nx):
                    field[k, j, i] = fillvalue

    @cc.export("buf_to_a3_full", "f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :],i4, i4, i4, i4")
    def buffer_to_array3_full(field, ileft, iright, jleft, jright, sw,se,nw,ne, j0, j1, i0, i1):
        nz, ny, nx = field.shape

        for k in range(nz):
            # Southern halo
            for j in range(j0):
                for i in range(i0):
                    field[k, j, i] = sw[k,j,i]
                for i in range(i0, i1):
                    field[k, j, i] = jleft[k, j, i-i0]
                for i in range(i1, nx):
                    field[k, j, i] = se[k,j,i-i1]
            for j in range(j0, j1):
                # Western halo
                for i in range(i0):
                    field[k, j, i] = ileft[k, j-j0, i]
                # Eastern halo
                for i in range(i1, nx):
                    field[k, j, i] = iright[k, j-j0, i-i1]
            # Northern halo
            for j in range(j1, ny):
                for i in range(i0):
                    field[k, j, i] = nw[k,j-j1,i]
                for i in range(i0, i1):
                    field[k, j, i] = jright[k, j-j1, i-i0]
                for i in range(i1, nx):
                    field[k, j, i] = ne[k,j-j1,i-i1]

    cc.compile()
