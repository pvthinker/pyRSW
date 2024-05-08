"""

timescheme.py provides the various time schemes

for the forward integration in time

it relies on the abstract 'rhs' function that is
defined outside of this module. The header for 'rhs'
is

    rhs(state, t, dstate)

where 'state' and 'dstate' are 'State' instances
and 't' is the model time

To allow for any time step, a forward() function is
defined. This function simply points to one of the
timeschemes. See explanations in the code below.

To push the model state one time step ahead we simply
do

    model.forward(state, t, dt)


Note that dstate is actually allocated *within* this
module during the __init__. The reason is that for multistages
timeschemes, we need to store more than just one dstate

"""
import optimize as opt
from timing import timeit
from barotropicfilter import BarotropicFilter

class Timescheme(object):
    """Catalogue and handler of timeschemes.

    The user mainly interacts with an object of this class through its
    "forward" method, which uses an "rhs" function to calculate the
    "right hand side" of the prognostic model equation.  This "rhs"
    function must be set by calling the "set" method.

    Attributes:
     - prognostic_scalars: list with the names of prognostic scalars
        in the model, which can be Field variables or the components
        of Vector variables
     - dstate: internal instance of the State class used to store
        intermediate steps in the calculation; this State object
        contains only prognostic variables
     - other attributes may exist depending on the chosen timescheme

    Methods:
     - set
     - rhs: this method is not defined within this class definition,
        but must be implemented in the model and given to Timescheme
        by calling the "set" method
     - forward: this method is defined in the constructor as an alias
        for one of the following time stepping methods
     - EulerForward
     - LFAM3

    """

    def __init__(self, param, grid, state):
        """Initialise the time stepping method specified in param."""
        self.prognostic_scalars = state.get_prognostic_scalars()

        self.apply_bt_filter = param["barotropicfilter"]
        if self.apply_bt_filter:
            self.barotropicfilter = BarotropicFilter(param, grid)

        # Activate the timescheme given in param
        timestepping = param['timestepping']
        # ... which must be in the following dictionary of *functions*
        timeschemelist = {
            'EF': self.EulerForward,
            # 'LF': self.LeapFrog,
            # 'Heun': self.Heun,
            # 'AB2': self.AB2,
            # 'AB3': self.AB3,
            'LFAM3': self.LFAM3,
            'RK3_SSP': self.RK3_SSP,
            # 'RK3': self.RK3,
            # 'RK4_LS': self.RK4_LS,
        }
        try:
            self.forward = timeschemelist[timestepping]
            # This is a HUGE trick: forward is a function (and not a
            # variable).  It points to one of the timestepping
            # functions defined below.  They all have the same header:
            # forward(state, t, dt, **kwargs)
        except KeyError:
            raise ValueError('unknown time scheme ' + repr(timestepping))

        # Create internal arrays
        self.dstate = state.duplicate_prognostic_variables()
        if timestepping == 'LFAM3':
            self.stateb = state.duplicate_prognostic_variables()
            self.state = state.duplicate_prognostic_variables()
            self.first = True
        if timestepping == 'RK3_SSP':
            self.ds0 = self.dstate
            self.ds1 = state.duplicate_prognostic_variables()
            self.ds2 = state.duplicate_prognostic_variables()
            self.s1 = state.new_state_from(state.variables)
            self.s2 = state.new_state_from(state.variables)
            self.s1.tracers = ["h"]
            self.s2.tracers = ["h"]

    def set(self, rhs, diagnose_var):
        """Assign the right hand side of the model.

        The argument 'rhs' of this method is a function with the
        signature rhs(state, t, dstate), where 'state' is the current
        state of the model and 't' is the current timestep.  The
        result is written to 'dstate'.
        """
        self.rhs = rhs
        self.diagnose_var = diagnose_var

    # ----------------------------------------
    def EulerForward(self, state, t, dt, **kwargs):
        self.rhs(state, t, self.dstate, last=True)
        for scalar_name in self.prognostic_scalars:
            scalar = state.get(scalar_name)
            # Get a view on the data without changing its orientation
            s = scalar.view()
            # Get a view on dstate in the same orientation as state
            ds = self.dstate.get(scalar_name).viewlike(scalar)
            s += dt * ds
        self.diagnose_var(state)

    # ----------------------------------------
    def LFAM3(self, state, t, dt, **kwargs):

        if self.first:
            self.rhs(state, t, self.dstate, last=True)
            # Euler Forward if very first time step
            for scalar_name in self.prognostic_scalars:
                scalar = state.get(scalar_name)
                s = scalar.view()
                ds = self.dstate.get(scalar_name).viewlike(scalar)
                sb = self.stateb.get(scalar_name).viewlike(scalar)
                sn = self.state.get(scalar_name).viewlike(scalar)
                sn[:] = s
                sb[:] = s
                s += dt * ds
            self.first = False
            self.diagnose_var(state)

        else:
            # Predictor
            self.rhs(state, t, self.dstate, last=True)
            for scalar_name in self.prognostic_scalars:
                scalar = state.get(scalar_name)
                s = scalar.view()
                ds = self.dstate.get(scalar_name).viewlike(scalar)
                sb = self.stateb.get(scalar_name).viewlike(scalar)
                sn = self.state.get(scalar_name).viewlike(scalar)
                # backup state into 'now' state
                sn[:] = s

                # LF s is at n+1
                s[:] = sb + (2.*dt) * ds
                # AM3 gives s at n+1/2
                s[:] = (1./12.)*(5.*s+8.*sn-sb)
                #
                # previous two lines combined in one => SLOWER
                # s[:] = (1./12.)*(5.*sb + (10*dt) * ds + 8.*sn-sb)

                # and backup former 'now' state into 'before' state
                sb[:] = sn

            self.diagnose_var(state)

            # Corrector step at n+1/2
            self.rhs(state, t+dt*.5, self.dstate, last=True)

            # move from n to n+1
            for scalar_name in self.prognostic_scalars:
                scalar = state.get(scalar_name)
                s = scalar.view()
                ds = self.dstate.get(scalar_name).viewlike(scalar)
                sn = self.state.get(scalar_name).viewlike(scalar)
                s[:] = sn + dt*ds

            self.diagnose_var(state)
    # ----------------------------------------
    @timeit
    def RK3_SSPold(self, state, t, dt, **kwargs):
        """ RK3 SSP

        The three stages are

        s1 = s^n + dt*L(s^n)
        s2 = s^n + (dt/4)*( L(s^n)+L(s1) )
        s^n+1 =  s^n + (dt/6)*( L(s^n)+L(s1)+4*L(s2) )

        or equivalently

        ds0 = L(s)
        s = s+dt*ds0
        ds1 = L(s)
        s = s+(dt/4)*(ds1-3*ds0)
        ds2 = L(s)
        s = s+(dt/12)*(8*ds2-ds0-ds1)

        """
        self.rhs(state, t, self.ds0, last=False)
        for scalar_name in self.prognostic_scalars:
            s = state.get(scalar_name).view("i")
            ds = self.ds0.get(scalar_name).view("i")
            #s += dt * ds
            opt.add1(s, ds, dt)
        self.diagnose_var(state)

        self.rhs(state, t+dt, self.ds1, last=False)
        c0 = -0.75*dt
        c1 = 0.25*dt
        for scalar_name in self.prognostic_scalars:
            s = state.get(scalar_name).view("i")
            ds0 = self.ds0.get(scalar_name).view("i")
            ds1 = self.ds1.get(scalar_name).view("i")
            #s += (dt/4.) * (ds1-3*ds0)
            opt.add2(s, ds0, c0, ds1, c1)
        self.diagnose_var(state)

        self.rhs(state, t+dt*0.5, self.ds2, last=True)
        c0 = -dt/12.
        c1 = c0
        c2 = 2*dt/3.
        for scalar_name in self.prognostic_scalars:
            s = state.get(scalar_name).view("i")
            ds0 = self.ds0.get(scalar_name).view("i")
            ds1 = self.ds1.get(scalar_name).view("i")
            ds2 = self.ds2.get(scalar_name).view("i")
            #s += (dt/12.) * (8*ds2-ds0-ds1)
            opt.add3(s, ds0, c0, ds1, c1, ds2, c2)
        self.diagnose_var(state)


    @timeit
    def RK3_SSP(self, state, t, dt, **kwargs):
        """ RK3 SSP

        The three stages are

        s1 = s^n + dt*L(s^n)
        s2 = s^n + (dt/4)*( L(s^n)+L(s1) )
        s^n+1 =  s^n + (dt/6)*( L(s^n)+L(s1)+4*L(s2) )

        or equivalently

        ds0 = L(s)
        s = s+dt*ds0
        ds1 = L(s)
        s = s+(dt/4)*(ds1-3*ds0)
        ds2 = L(s)
        s = s+(dt/12)*(8*ds2-ds0-ds1)

        """
        if self.apply_bt_filter:
            self.barotropicfilter.set_height_normalization(state)

        self.rhs(state, t, self.ds0, last=False)
        if self.apply_bt_filter:
            self.barotropicfilter.compute_dstar(state, self.ds0)

        for scalar_name in self.prognostic_scalars:
            s = state.get(scalar_name).view("i")
            s1 = self.s1.get(scalar_name).view("i")
            ds = self.ds0.get(scalar_name).view("i")
            s1[:] = s + dt * ds

        if self.apply_bt_filter:
            self.barotropicfilter.apply_dstar(self.s1, dt)

        self.diagnose_var(self.s1)

        self.rhs(self.s1, t+dt, self.ds1, last=False)
        add(self.ds0, 0.5, self.ds1, 0.5, self.prognostic_scalars)

        if self.apply_bt_filter:
            self.barotropicfilter.compute_dstar(state, self.ds0)

        for scalar_name in self.prognostic_scalars:
            s = state.get(scalar_name).view("i")
            s2 =  self.s2.get(scalar_name).view("i")
            ds0 = self.ds0.get(scalar_name).view("i")
            #ds1 = self.ds1.get(scalar_name).view("i")
            #s2[:] = s + (dt/4)*( ds0+ds1 )
            s2[:] = s + (0.5*dt)*( ds0 )

        if self.apply_bt_filter:
            self.barotropicfilter.apply_dstar(self.s2, dt*0.5)

        self.diagnose_var(self.s2)

        self.rhs(self.s2, t+dt*0.5, self.ds2, last=True)
        add(self.ds0, 1./3, self.ds2, 2./3, self.prognostic_scalars)

        if self.apply_bt_filter:
            self.barotropicfilter.compute_dstar(state, self.ds0)

        for scalar_name in self.prognostic_scalars:
            s = state.get(scalar_name).view("i")
            ds0 = self.ds0.get(scalar_name).view("i")
            # ds1 = self.ds1.get(scalar_name).view("i")
            # ds2 = self.ds2.get(scalar_name).view("i")
            #s += (dt/6.) * (4*ds2+ds0+ds1)
            s += dt * ds0

        if self.apply_bt_filter:
            self.barotropicfilter.apply_dstar(state, dt)

        self.diagnose_var(state)


def add(dstate0, coef0, dstate1, coef1, prognostic_scalars):
    for scalar_name in prognostic_scalars:
        ds0 = dstate0.get(scalar_name).view("i")
        ds1 = dstate1.get(scalar_name).view("i")
        ds0[:] = ds0*coef0+ds1*coef1
