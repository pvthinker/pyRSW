from numba.pycc import CC


def compile(verbose=False):
    print("Compile the optimize module with numba")

    cc = CC("optimize")
    cc.verbose = verbose

    @cc.export("add1",
               "void(f8[:, :, :], f8[:, :, :], f8)")
    def add1(s, ds0, c0):
        nz, ny, nx = s.shape
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    s[k, j, i] += ds0[k, j, i] * c0

    @cc.export("add2",
               "void(f8[:, :, :], f8[:, :, :], f8, f8[:, :, :], f8)")
    def add2(s, ds0, c0, ds1, c1):
        nz, ny, nx = s.shape
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    s[k, j, i] += ds0[k, j, i] * c0 + ds1[k, j, i] * c1

    @cc.export("add3",
               "void(f8[:, :, :], f8[:, :, :], f8, f8[:, :, :], f8, f8[:, :, :], f8)")
    def add3(s, ds0, c0, ds1, c1, ds2, c2):
        nz, ny, nx = s.shape
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    s[k, j, i] += ds0[k, j, i] * c0 + \
                        ds1[k, j, i] * c1 + ds2[k, j, i] * c2

    cc.compile()
