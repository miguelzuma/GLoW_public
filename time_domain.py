import warnings
import numpy as np
from scipy import integrate as sc_integrate
from scipy import interpolate as sc_interpolate
from scipy import optimize as sc_optimize
from scipy import special as sc_special

from . import lenses

class TimeDomainException(Exception):
    pass

class TimeDomainWarning(UserWarning):
    pass

##==============================================================================


class ItGeneral():
    """Base class for the time-domain integral.

    Parameters
    ----------
    Lens : PsiGeneral subclass
        Lens object.
    y : float
        Impact parameter (defined to be aligned with the :math:`x_1` axis).
    p_prec : dict, optional
        Precision parameters.

    Attributes
    ----------
    lens : PsiGeneral subclass
        Lens object.
    y : float
        Impact parameter.
    p_prec : dict
        Default precision parameters updated with the input. Default keys:

        * ``eval_mode`` (*str*) -- Computation of :math:`I(\\tau)`, i.e.\
            behaviour of the method :func:`eval_It`. Options:

            * ``'interpolate'``: Precompute a grid and then evaluate an interpolation function.
            * ``'exact'``: Compute :math:`I(\\tau)` for each point requested.

        * ``Nt``   (*int*)   -- Number of points in the grid.
        * ``tmin`` (*float*) -- Minimum :math:`\\tau` in the grid.
        * ``tmax`` (*float*) -- Maximum :math:`\\tau` in the grid.

        * ``interp_fill_value`` -- Behaviour of the interpolation function\
            outside the interpolation range. Options:

            * *None*: Raise an error if extrapolation is attempted.
            * *float*: Extrapolate with a constant value.
            * ``'extrapolate'``: Linear extrapolation.

        * ``interp_kind`` (*str*) -- Interpolation method. Any kind recognized by Scipy\
            is allowed. In particular:

            * ``'linear'``
            * ``'cubic'``

        * ``sampling``  (*str*) -- Sampling options for the grid. Options:

            * ``'linear'``
            * ``'log'``
            * ``'oversampling'``: The ``oversampling_`` parameters below control\
                the properties of the oversampling grid. In this mode, the grid is\
                sampled logarithmically (as with ``'log'``) and then the region\
                between ``oversampling_tmin`` and ``oversampling_tmax`` is oversampled\
                with ``oversampling_n`` times more points. The total number of points is\
                still ``Nt``.

        * ``oversampling_n``    (*int*)
        * ``oversampling_tmin`` (*float*)
        * ``oversampling_tmax`` (*float*)

    t_grid, It_grid : array
        (*subclass defined*) Grid of points where :math:`I(\\tau)` has been computed.
    eval_It : func(float or array) -> float or array
        (*subclass defined*) Final function to evaluate :math:`I(\\tau)`.
    name : str
        (*subclass defined*) Name of the method.
    tmin : float
        (*subclass defined*) Minimum of the Fermat potential, :math:`t_\\text{min}=\phi(\\boldsymbol{x}_\\text{min})`.
    I0 : float
        (*subclass defined*) Value of :math:`I(\\tau)` at :math:`\\tau=0`.
    p_crits : list of dict
        (*subclass defined*) Dictionaries with all the images found. Keys for each image:

        * ``type`` (*str*) -- Type of image (critical point of the Fermat potential):

            * ``'min'``, ``'max'``, ``'saddle'`` -- Minimum, maximum and saddle point.
            * ``'sing/cusp max'``, ``'sing/cusp min'`` -- When the derivative of the Fermat potential\
                is discontinuous or singular, these points may behave like critical points for the\
                computation of the contours. Examples include the center of the point lens and the SIS.

        * ``t`` (*float*) -- Time delay :math:`t=\\phi(\\boldsymbol{x}_\\text{im})` (Note: the relative\
            time delay is :math:`\\tau = t - t_\\text{min}`).
        * ``x1``, ``x2`` (*float*) -- Position of the image.
        * ``mag`` (*float*) -- Magnification :math:`\\mu`.
    """
    def __init__(self, Lens, y, p_prec={}):
        self.lens = Lens
        self.y = y

        self.p_prec = self.default_general_params()
        self.p_prec_default_keys = set(self.p_prec.keys())
        self.p_prec.update(p_prec)

        self.check_general_input()

        # ***** to be overriden by the subclass *****
        self.t_grid, self.It_grid = np.array([]), np.array([])
        self.eval_It = lambda t: np.zeros_like(t)
        self.name = 'unknown'
        self.tmin = 0.
        self.I0 = 1.
        # *******************************************

    def __str__(self):
        """Create the python call needed to replicate this object."""
        class_name = type(self).__name__
        class_call = "It = time_domain." + class_name + "(Psi, y, p_prec)"

        y_message = "y = %g\n" % self.y
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        lens_message = self.lens.__str__() + "\n\n"

        return lens_message + y_message + prec_message + class_call

    def __call__(self, tau):
        """Call :func:`eval_It`."""
        return self.eval_It(tau)

    def default_general_params(self):
        """Fill the default parameters.

        Update the precision parameters common for all methods and
        then call :func:`default_params` (*subclass defined*).

        Returns
        -------
        p_prec : dict
            Default precision parameters.
        """
        p_prec = {'Nt'       : 1000, \
                  'tmin'     : 1e-2, \
                  'tmax'     : 1e6, \
                  'eval_mode' : 'interpolate', \
                  'sampling'  : 'log', \
                  'interp_fill_value' : None, \
                  'interp_kind'       : 'linear', \
                  'oversampling_n'    : 10, \
                  'oversampling_tmin' : 1e-1, \
                  'oversampling_tmax' : 1e1}

        p_prec2 = self.default_params()
        if p_prec2 is not {}:
            p_prec.update(p_prec2)

        return p_prec

    def check_general_input(self):
        """Check the input upon initialization.

        It first calls :func:`check_input` (*subclass defined*)
        to perform any checks that the user desires. It then checks that
        the input only updates existing keys in :attr:`.p_prec`,
        otherwise throws a warning.

        The idea is that we do not use a wrong name that is then ignored.

        Warns
        -----
        TimeDomainWarning

        Warnings
        --------
        If the subclass will add a new parameter without an entry in :func:`default_params`,
        it must be manually added to :attr:`self.p_prec_default_keys` in :func:`check_input`
        with ``self.p_prec_default_keys.add(new_key)``.
        """
        self.check_input()

        # check that there are no unrecognized parameters
        p_prec_new_keys = set(self.p_prec.keys())

        diff_prec = p_prec_new_keys - self.p_prec_default_keys

        if diff_prec:
            for key in diff_prec:
                message = "unrecognized key '%s' found in p_prec will be "\
                          "(most likely) ignored" % key
                warnings.warn(message, TimeDomainWarning)

    def display_info(self):
        """Print internal information in human-readable form."""
        print("\t////////////////////////////\n"\
              "\t///   I(t) information   ///\n"\
              "\t////////////////////////////")

        print("\n * Method: %s" % self.name)
        print("\n * Impact parameter y = %g" % self.y)

        if self.p_prec != {}:
            print("\n * Precision parameters:")
            for key, value in self.p_prec.items():
                print("   **", key, "=", value)

        if self.p_prec == {}:
            print('\nNo information available')

        print("\n * Lens: %s\n" % self.lens.p_phys.get('name', 'no information'))

    def display_images(self):
        """Print the critical points of the Fermat potential in human-readable form."""
        try:
            tmin = self.p_crits[0]['t']
        except AttributeError as e:
            message = 'no critical points (p_crits) found in It (%s)' % self.name
            raise TimeDomainException(message) from e

        print("\t////////////////////////////\n"\
              "\t///        Images        ///\n"\
              "\t////////////////////////////")
        print("\n * Lens: %s  (y = %g)" % (self.lens.p_phys['name'], self.y))
        for i, p in enumerate(self.p_crits):
            print("\n * Image %d  (%s):" % (i, p['type']))
            print("   **   t = %e" % p['t'])
            print("   ** tau = %e" % (p['t']-tmin))
            print("   **   x = (%e, %e)" % (p['x1'], p['x2']))
            print("   **  mu = %e" % p['mag'])
        print('')

    def interpolate(self, xs, ys):
        """Construct an interpolation function using the options in ``p_prec``.

        Parameters
        ----------
        xs, ys : float or array
            Values to interpolate.

        Returns
        -------
        interp_func : func(float) -> float
            Interpolation function.
        """
        fill_value = self.p_prec['interp_fill_value']
        kind = self.p_prec['interp_kind']

        if fill_value is None:
            if kind == 'linear':
                interp_func = lambda x: np.interp(x, xs, ys)
            else:
                interp_func = sc_interpolate.interp1d(xs, ys, kind=kind)
        else:
            interp_func = sc_interpolate.interp1d(xs, ys, \
                                                  fill_value = fill_value, \
                                                  kind = kind, \
                                                  bounds_error = False)

        return interp_func

    def eval_low_tau(self, t, tmin, Imin):
        r"""Linear interpolation to evaluate :math:`I(\tau)` at low :math:`\tau`.

        The formula used is

        .. math::
            I(\tau) = I(0) - \big(I(0)-I(\tau_\text{min})\big)\frac{\tau}{\tau_\text{min}}

        where :math:`I(0)` corresponds to the class attribute ``I0``.

        Parameters
        ----------
        t : float or array
            :math:`\tau`
        tmin : float
            :math:`\tau_\text{min}`
        Imin : float
            :math:`I(\tau_\text{min})`

        Returns
        -------
        It : float or array
            :math:`I(\tau)` at low tau.
        """
        # linear interpolation for low tau (especially useful for log sampling)
        return self.I0 - (self.I0-Imin)*t/tmin

    def compute_grid(self):
        r"""Evaluate :math:`I(\tau)` on a grid.

        The behaviour of this function is controlled by the precision
        parameter ``p_prec['sampling']``.

        Returns
        -------
        t_grid, It_grid : float or array
            Result.
        interp_It : func(float) -> float
            Interpolation function constructed with :func:`interpolate`.
        """
        Nt = self.p_prec['Nt']
        tau_min = self.p_prec['tmin']
        tau_max = self.p_prec['tmax']

        if self.p_prec['sampling'] == 'log':
            # interpolation in log space
            t_grid = np.logspace(np.log10(tau_min), np.log10(tau_max), Nt)
            It_grid = self.compute(t_grid)
            interp_It_log = self.interpolate(np.log(t_grid), It_grid)
            interp_It = lambda t: interp_It_log(np.log(t))

        elif self.p_prec['sampling'] == 'oversampling':
            # interpolate in logspace but oversample region of small tau
            # oversampling_n times more
            tmin_over = self.p_prec['oversampling_tmin']
            tmax_over = self.p_prec['oversampling_tmax']
            n_oversampling = self.p_prec['oversampling_n']

            nt = int(Nt/n_oversampling)
            n_mid = Nt-nt

            l_low = np.log(tmin_over/tau_min)
            l_high = np.log(tau_max/tmax_over)

            n_low = int(nt*l_low/(l_low + l_high))
            n_high = nt - n_low

            t_low = np.geomspace(tau_min, tmin_over, n_low, endpoint=False)
            t_mid = np.geomspace(tmin_over, tmax_over, n_mid, endpoint=False)
            t_high = np.geomspace(tmax_over, tau_max, n_high)

            t_grid = np.concatenate([t_low, t_mid, t_high])

            It_grid = self.compute(t_grid)
            interp_It_log = self.interpolate(np.log(t_grid), It_grid)
            interp_It = lambda t: interp_It_log(np.log(t))

        elif self.p_prec['sampling'] == 'linear':
            t_grid = np.linspace(tau_min, tau_max, Nt)
            It_grid = self.compute(t_grid)
            interp_It = self.interpolate(t_grid, It_grid)

        else:
            message = "interpolation mode '%s' not recognized" % self.p_prec['sampling']
            raise TimeDomainException(message)

        return t_grid, It_grid, interp_It

    def compute_all(self):
        r"""Compute the final function needed to evaluate :math:`I(\tau)`.

        Returns
        -------
        t_grid, It_grid : array
            If ``p_prec['eval_mode'] == 'interpolate'``, they contain the
            grids computed in :func:`compute_grid`. Otherwise, empty arrays.
        eval_It : func(float) -> float
            Final function to evaluate :math:`I(\tau)`.
        """
        t_grid = np.array([])
        It_grid = np.array([])

        if self.p_prec['eval_mode'] == 'interpolate':
            t_grid, It_grid, interp_It = self.compute_grid()

            tmin = t_grid[0]
            eval_low_t = lambda t: self.eval_low_tau(t, tmin, It_grid[0])

            eval_It = lambda t: np.piecewise(t, \
                                            [t<0, (t>=0)&(t<tmin),   t>=tmin], \
                                            [  0,      eval_low_t, interp_It])

        if self.p_prec['eval_mode'] == 'exact':
            eval_It = self.compute

        return t_grid, It_grid, eval_It

    def eval_Gt(self, t, dt=1e-4):
        r"""Green function defined as :math:`G(\tau) = 1/2\pi\ \text{d}I/\text{d}\tau`.

        It is computed using a (2nd order) finite difference approximation

        .. math::
            G(\tau) \simeq \frac{1}{4\pi}\left(I(\tau+\Delta\tau) - I(\tau-\Delta\tau)\right)

        Parameters
        ----------
        t : float or array
            :math:`\tau`.
        dt : float
            :math:`\Delta\tau`.

        Returns
        -------
        Gt : float or array
            :math:`G(\tau)`.
        """
        return (self.eval_It(t+dt) - self.eval_It(t-dt))/dt/4/np.pi

    def phi_Fermat(self, x1, x2):
        r"""Fermat potential with the minimum set to zero.

        .. math::
            \tau = \tilde{\phi}(\boldsymbol{x}) = \phi(\boldsymbol{x}) - t_\text{min}

        Parameters
        ----------
        x1, x2 : float or array
            Lens plane coordinates.

        Returns
        -------
        phi : float or array
            :math:`\tilde{\phi}`
        """
        return self.lens.phi_Fermat(x1, x2, self.y) - self.tmin

    ## -----------------------------------------------------------------

    def find_tmin_root(self, x1_guess, x2_guess=0):
        """Find the minimum time delay of a lens by finding the root of\
           the derivative of the Fermat potential.

        The algorithm starts with an initial guess x1_guess (x2_min=0 always
        for axisymmetric lenses).

        Caveats:
          - Derivatives of the lens required.
          - Not implemented for non-axisymmetric lenses.
          - At the moment, it can only be used when there is only one critical
            point (hence a minimum). No checks are perfomed in this function.

        Parameters
        ----------
        x1_guess, x2_guess : float
            Initial points to start the search.

        Returns
        -------
        tmin, x1_min, x2_min : (float, float, float)
            Minimum time delay and position of the minimum.
        """
        if self.lens.isAxisym:
            y = self.y

            f  = lambda x1: self.lens.dphi_Fermat_dx1(x1, x2=0, y=y)
            df = lambda x1: self.lens.ddphi_Fermat_ddx1(x1, x2=0)

            sol = sc_optimize.root_scalar(f=f, fprime=df, x0=x1_guess, method='newton')

            x1_min = sol.root
            x2_min = 0
            tmin = self.lens.phi_Fermat(x1_min, x2_min, y)
        else:
            message = "'find_tmin_root' method not implemented for non-axisymmetric lenses yet"
            raise TimeDomainException(message)

        return tmin, x1_min, x2_min

    def find_tmin_bounds(self, x1_max, x2_max=0, x1_min=None, x2_min=None, x1_guess=None, x2_guess=None):
        """Find the minimum time delay through direct minimization of the\
           Fermat potential.

        Both algorithms (axisymmetric and non-axisymmetric case) require the bounds
        x1_max, x2_max of the optimization region (if x1_min and x2_min are not
        given, the region is assumed to be symmetric). The 2d algorithm for
        non-axisymmetric lenses also needs an initial guess (x1_guess, x2_guess)
        (default: (y, 0))

        Caveats:
          - No derivatives are required, but if present they could be used to
            improve convergence
        """
        if x1_min is None:
            x1_min = -x1_max
        if x2_min is None:
            x2_min = -x2_max

        if self.lens.isAxisym:
            f = lambda x1: self.lens.phi_Fermat(x1, x2=0, y=self.y)

            sol = sc_optimize.minimize_scalar(f, \
                                              bounds=(x1_min, x1_max), \
                                              method='bounded')

            x1_min = sol.x
            x2_min = 0
            tmin = sol.fun
        else:
            if x1_guess is None:
                x1_guess = self.y
            if x2_guess is None:
                x2_guess = 0
            guess = [x1_guess, x2_guess]

            f = lambda x: self.lens.phi_Fermat(x[0], x[1], self.y)
            sol = sc_optimize.minimize(f, \
                                       guess, \
                                       bounds=((x1_min, x1_max), (x2_min, x2_max)), \
                                       method='TNC')

            x1_min = sol.x[0]
            x2_min = sol.x[1]
            tmin = sol.fun

        return tmin, x1_min, x2_min

    # ***** to be overriden by the subclass *****
    def compute(self, t):
        """(*subclass defined*) Internal function to compute :math:`I(\\tau)`.

        Parameters
        ----------
        t : float or array
            :math:`\\tau`.

        Returns
        -------
        It : float or array
            :math:`I(\\tau)`.
        """
        return np.zeros_like(t)

    def check_input(self):
        """(*subclass defined, optional*) Check the input of the implementation."""
        pass

    def default_params(self):
        """(*subclass defined*) Initialize the default parameters."""
        p_prec = {}
        return p_prec
    # *******************************************


##==============================================================================


class It_SingleContour(ItGeneral):
    """Computation for the single contour regime (only one image).

    Additional information: :ref:`theory <SingleContour_theory>`, :ref:`default parameters <pyIt_SingleContour_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.time_domain.ItGeneral`
    for the internal information of the parent class)

    Parameters
    ----------
    p_prec : dict, optional
        Precision parameters. New keys:

        * ``soft`` (*float*) -- Softening factor.
        * ``method`` (*str*) -- ODE integration method. Any kind recognized by Scipy\
            is allowed, e.g. ``'RK45'``.
        * ``rtol`` (*float*) -- Relative tolerance for the integration.
        * ``atol`` (*float*) -- Absolute tolerance for the integration.

    Warnings
    --------
    Compared to :class:`~glow.time_domain_c.It_SingleContour_C`, this class presents several drawbacks:

    - It does not check that we are indeed in the single contour regime. Proceed with caution...
    - The ``'robust'`` option (i.e. parametric integration) is not implemented.
    - It is not parallelized and way slower than the C implementation.
    """
    def __init__(self, Lens, y, p_prec={}):
        super().__init__(Lens, y, p_prec)

        self.name = 'single contour'

        self.p_crits = self.find_all_images()

        self.x1_min = self.p_crits[0]['x1']
        self.x2_min = self.p_crits[0]['x2']
        self.tmin = self.p_crits[0]['t']
        self.I0 = 2*np.pi*np.sqrt(self.p_crits[0]['mag'])

        self.compute = np.vectorize(self.compute_novec)
        self.t_grid, self.It_grid, self.eval_It = self.compute_all()

    def default_params(self):
        p_prec = {'soft'     : 1e-8, \
                  'method'   : 'RK45', \
                  'rtol'     : 1e-5, \
                  'atol'     : 1e-5, \
                  'Nt'       : 500, \
                  'tmin'     : 1e-2, \
                  'tmax'     : 1e6}

        return p_prec

    #=====  General definitions
    #===================================================================
    def x1x2(self, R, phi, xc1=0, xc2=0):
        """ Convert polar to Cartesian coordinates."""
        return xc1 + R*np.cos(phi), xc2 + R*np.sin(phi)

    def deriv_phi(self, R, phi, xc1=0, xc2=0):
        """Compute first derivatives of the Fermat potential wrt polar coordinates."""
        x1, x2 = self.x1x2(R, phi, xc1=xc1, xc2=xc2)
        d1, d2 = self.lens.dphi_Fermat_vec(x1, x2, self.y)

        Dx1 = x1-xc1
        Dx2 = x2-xc2

        dR = (Dx1*d1 + Dx2*d2)/R
        dphi = -Dx2*d1 + Dx1*d2

        return dR, dphi
    #===================================================================

    #=====  Find minimum and R(tau)
    #===================================================================
    def find_all_images(self):
        """Find the minimum of the Fermat potential assuming that it is the only critical point."""
        if self.lens.isAxisym is True:
            tmin, x1_min, x2_min = self.find_tmin_root(x1_guess=self.y)

            # alternative method with a fixed bound and direct minimization
            # ~ tmin, x1_min, x2_min = self.find_tmin_bounds(x1_max=self.y+5)
        else:
            tmin, x1_min, x2_min = self.find_tmin_bounds(x1_max=self.y+5, \
                                                         x2_max=5, \
                                                         x1_guess=self.y, \
                                                         x2_guess=0)

        sh = self.lens.shear(x1_min, x2_min)

        images = [{'type' : 'min', \
                   't'    : tmin, \
                   'x1'   : x1_min, \
                   'x2'   : x2_min, \
                   'mag'  : sh['mag']}]

        return images

    def find_R_of_tau(self, tau):
        """Find the radius that corresponds to a given :math:`\\tau`."""

        def _f_R_of_tau(R):
            x1, x2 = self.x1x2(R, 0, xc1=self.x1_min, xc2=self.x2_min)
            f = self.lens.phi_Fermat(x1, x2, self.y) - self.tmin - tau
            return f

        def _df_R_of_tau(R):
            dphi_dR, dphi_dphi = self.deriv_phi(R, 0, xc1=self.x1_min, xc2=self.x2_min)
            df = dphi_dR
            return df

        Dx1 = self.x1_min - self.y;
        R_guess = np.sqrt(2*tau + Dx1*Dx1) - Dx1

        root = sc_optimize.root_scalar(_f_R_of_tau,\
                                       x0 = R_guess,\
                                       method = 'newton',\
                                       fprime = _df_R_of_tau)
        return root.root
    #===================================================================

    #=====  Integrate the contours
    #===================================================================
    def sys(self, t, u):
        """ODE system defining the contours (and :math:`I(t)`)."""
        phi = t
        R, I = u

        dphi_dR, dphi_dphi = self.deriv_phi(R, phi, self.x1_min, self.x2_min)

        denom = dphi_dR + self.p_prec['soft']

        dR = -dphi_dphi/denom
        dI = R/np.abs(denom)

        return dR, dI

    def integrate_contour(self, tau, dense_output=False):
        """Integration of the contour corresponding to a given :math:`\\tau`.

        Parameters
        ----------
        tau : float
            Relative time delay :math:`\\tau`.
        dense_output : bool
            Include interpolation functions in the output.

        Returns
        -------
        sol : output of Scipy's ``solve_ivp``
            Solution of the differential equation.
        """
        R0 = self.find_R_of_tau(tau)

        sol = sc_integrate.solve_ivp(fun          = self.sys, \
                                     t_span       = [0, 2*np.pi], \
                                     y0           = [R0, 0], \
                                     method       = self.p_prec['method'], \
                                     dense_output = dense_output, \
                                     atol         = self.p_prec['atol'], \
                                     rtol         = self.p_prec['rtol'])
        return sol

    def compute_novec(self, tau):
        """Compute :math:`I(\\tau)`. Non-vectorized version.

        It includes the step function, :math:`I(\\tau<0)=0`.

        Parameters
        ----------
        tau : float
            Relative time delay :math:`\\tau`.

        Returns
        -------
        I : float
            :math:`I(\\tau)`
        """
        if tau > 0:
            R, It = self.integrate_contour(tau).y
            I = It[-1]
        elif tau == 0:
            I = self.I0
        else:
            I = 0.
        return I
    #===================================================================

    def get_contour(self, tau, N=1000):
        """Contour corresponding to a given :math:`\\tau` in Cartesian coordinates.

        Parameters
        ----------
        tau : float
            Relative time delay :math:`\\tau`.
        N : int
            Number of points in the contour.

        Returns
        -------
        x1s, x2s : array
            Contour in Cartesian coordinates.
        """
        ths = np.linspace(0, 2*np.pi, N)
        sol = self.integrate_contour(tau, dense_output=True)

        Rs, Is = sol.sol(ths)

        x1s, x2s = self.x1x2(Rs, ths, self.x1_min, self.x2_min)

        return x1s, x2s


class It_SingleIntegral(ItGeneral):
    """Computation for axisymmetric lenses, solving a single radial integral.

    Additional information: :ref:`theory <SingleIntegral_theory>`, :ref:`default parameters <pyIt_SingleIntegral_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.time_domain.ItGeneral`
    for the internal information of the parent class)

    Parameters
    ----------
    p_prec : dict, optional
        Precision parameters. New keys:

        * ``method`` (*str*) -- Function used for the radial integration. Options:

            * ``'quad'``: Scipy's ``quad``, using QUADPACK.
            * ``'quadrature'``: Scipy's ``quadrature``, fixed-tolerance Gaussian quadrature.

    Warnings
    --------
    The C version :class:`~glow.time_domain_c.It_SingleIntegral_C` should always be preferred.
    It uses the same algorithm, but the implementation is faster and more robust.
    """
    def __init__(self, Lens, y, p_prec={}):
        super().__init__(Lens, y, p_prec)

        self.name = 'single integral'

        self.p_crits = self.find_all_images()
        self.tmin = self.p_crits[0]['t']
        self.I0 = 2*np.pi*np.sqrt(self.p_crits[0]['mag'])

        # sort the crit points in x order
        self.p_crits_x = sorted(self.p_crits, key=lambda point: point['x1'])

        self.t_grid, self.It_grid, self.eval_It = self.compute_all()

    def check_input(self):
        if self.lens.isAxisym is False:
            message = 'Single Integral not implemented for non-axisymmetric lenses'
            raise TimeDomainException(message)

    def default_params(self):
        p_prec = {'Nt' : 300, \
                  'method' : 'quadrature'}

        return p_prec

    #=====  Generic root solvers
    #===================================================================
    def find_image_newton_1d(self, x1_guess):
        """Find a root of :math:`\\phi'(\\boldsymbol{x})=0`, using Newton's method.

        Parameters
        ----------
        x1_guess : float
            Initial guess.

        Returns
        -------
        image : dict
            Dictionary with the format explained in :class:`~glow.time_domain.ItGeneral`.
        """
        epsabs  = 1e-8
        epsrel  = 1e-8
        maxiter = 100

        f  = lambda x: self.lens.dphi_Fermat_dx1(x, 0, self.y)
        df = lambda x: self.lens.ddphi_Fermat_ddx1(x, 0)

        sol = sc_optimize.root_scalar(f, \
                                      method = 'newton', \
                                      fprime = df, \
                                      x0 = x1_guess, \
                                      xtol = epsabs, \
                                      rtol = epsrel, \
                                      maxiter = maxiter)
        return self.classify_image(sol)

    def find_image_bracket_1d(self, x1_lo, x1_hi):
        """Find a root of :math:`\\phi'(\\boldsymbol{x})=0`, using a bracketing algorithm.

        Parameters
        ----------
        x1_lo, x1_hi : float
            Bracket containing at least one root (function must change sign).

        Returns
        -------
        image : dict
            Dictionary with the format explained in :class:`~glow.time_domain.ItGeneral`.
        """
        epsabs  = 1e-8
        epsrel  = 1e-8
        maxiter = 100

        f  = lambda x: self.lens.dphi_Fermat_dx1(x, 0, self.y)

        sol = sc_optimize.root_scalar(f, \
                                      method = 'brentq', \
                                      bracket = [x1_lo, x1_hi], \
                                      xtol = epsabs, \
                                      rtol = epsrel, \
                                      maxiter = maxiter)
        return self.classify_image(sol)

    def find_root_newton_1d(self, x1_guess, t):
        """Find a root of :math:`\\phi(x_1, 0)=t`, using Newton's method.

        Parameters
        ----------
        x1_guess : float
            Initial guess.
        t : float
            Time delay :math:`t` (Note: related to the relative time delay
            as :math:`t=\\tau+t_\\text{min}`).

        Returns
        -------
        sol : Scipy's output of ``root_scalar``
            Solution.
        """
        epsabs  = 1e-8
        epsrel  = 1e-8
        maxiter = 100

        f  = lambda x: self.lens.phi_Fermat(x, 0, self.y) - t
        df = lambda x: self.lens.dphi_Fermat_dx1(x, 0, self.y)

        sol = sc_optimize.root_scalar(f, \
                                      method = 'newton', \
                                      fprime = df, \
                                      x0 = x1_guess, \
                                      xtol = epsabs, \
                                      rtol = epsrel, \
                                      maxiter = maxiter)
        return sol

    def find_root_bracket_1d(self, x1_lo, x1_hi, t):
        """Find a root of :math:`\\phi(x_1, 0)=t`, using a bracketing algorithm.

        Parameters
        ----------
        x1_lo, x1_hi : float
            Bracket containing at least one root (function must change sign).
        t : float
            Time delay :math:`t` (Note: related to the relative time delay
            as :math:`t=\\tau+t_\\text{min}`).

        Returns
        -------
        sol : Scipy's output of ``root_scalar``
            Solution.
        """
        epsabs  = 1e-8
        epsrel  = 1e-8
        maxiter = 100

        f  = lambda x: self.lens.phi_Fermat(x, 0, self.y) - t

        sol = sc_optimize.root_scalar(f, \
                                      method = 'brentq', \
                                      bracket = [x1_lo, x1_hi], \
                                      xtol = epsabs, \
                                      rtol = epsrel, \
                                      maxiter = maxiter)
        return sol

    def find_moving_bracket_1d(self, xguess_lo, xguess_hi, t):
        r"""Find a root of :math:`\phi(x_1, 0)=t`, using a moving bracket method.

        The algorithm starts with an initial bracket :math:`[x_\text{lo}, x_\text{hi}]`.
        If :math:`f(x_\text{lo})f(x_\text{hi}) < 0`, there is a root in the bracket and
        a standard algorithm is used. Otherwise, the bracket is transformed to
        :math:`[x_\text{lo}, x_\text{hi}]\to [x_\text{hi}, x_\text{hi}+\Delta x]`. The
        process is repeated, increasing :math:`\Delta x` with each iteration, until the
        bracketing condition is met.

        Parameters
        ----------
        xguess_lo, xguess_hi : float
            Initial bracket.
        t : float
            Time delay :math:`t` (Note: related to the relative time delay
            as :math:`t=\tau+t_\text{min}`).

        Returns
        -------
        sol : Scipy's output of ``root_scalar`` or None
            Returns None if the bracketing condition is never met.
        """
        scale = 1.5
        maxiter = 100

        # notice that the window always move from lo to hi
        dR = xguess_hi - xguess_lo

        f  = lambda x: self.lens.phi_Fermat(x, 0, self.y) - t

        i = 0
        success = False
        while (i < maxiter) and (success is False):
            f_lo = f(xguess_lo)
            f_hi = f(xguess_hi)

            if f_lo*f_hi < 0:
                success = True
            else:
                dR *= scale
                xguess_lo = xguess_hi
                xguess_hi += dR
                i += 1

        if success is True:
            sol = self.find_root_bracket_1d(xguess_lo, xguess_hi, t)
        else:
            sol = None

        return sol
    #===================================================================

    #=====  Find images and brackets
    #===================================================================
    def classify_image(self, sol):
        """Classify an image.

        Parameters
        ----------
        sol : Scipy's output of ``root_scalar``
            Solution to :math:`\\phi'(\\boldsymbol{x})=0`.

        Returns
        -------
        image : dict
            Dictionary with the image information. The format is explained
            in :class:`~glow.time_domain.ItGeneral`. If the solution has
            not converged, it returns an empty dictionary instead.
        """
        if sol.converged:
            x1 = sol.root
            sh = self.lens.shear(x1, 0)

            image = {'x1'  : x1,\
                     'x2'  : 0,\
                     't'   : self.lens.phi_Fermat(x1, 0, self.y),\
                     'mag' : sh['mag']}

            if sh['detA'] > 0:
                if sh['trA'] > 0:
                    image['type'] = 'min'
                else:
                    image['type'] = 'max'
            else:
                image['type'] = 'saddle'
        else:
            image = {}

        return image

    def check_singcusp(self):
        """Check for the presence of singularities and cusps at the center.

        Since :math:`\\phi'(x=0)` may diverge, we check if
        :math:`\\partial_{x_1}\\phi(-\\delta x, 0)\\phi(\\delta x, 0) < 0`.

        Returns
        -------
        singcusp : dict
            Image-like dictionary, with the format explained
            in :class:`~glow.time_domain.ItGeneral`.
        """
        dx = 1e-6
        eps = 1e-14

        dphi_le = self.lens.dphi_Fermat_dx1(-dx, 0, self.y)
        dphi_ri = self.lens.dphi_Fermat_dx1(dx, 0, self.y)

        singcusp = {}
        if (dphi_le*dphi_ri) < 0:
            if dphi_ri < 0:
                singcusp['type'] = 'sing/cusp max'
            else:
                singcusp['type'] = 'sing/cusp min'

            sh = self.lens.shear(eps, 0)

            singcusp['t']   = self.lens.phi_Fermat(eps, 0, self.y)
            singcusp['x1']  = 0
            singcusp['x2']  = 0
            singcusp['mag'] = sh['mag']

        return singcusp

    def find_all_images(self):
        """Find all the critical points.

        Returns
        -------
        images : list
            Full list of images. The output is stored in ``p_crits``, detailed
            in :class:`~glow.time_domain.ItGeneral`.
        """
        Rmin = 1e-6
        Rmax = 100
        nR = 100

        Rs = np.geomspace(Rmin, Rmax, nR)

        # bracket right
        signs = np.sign(self.lens.dphi_Fermat_dx1(Rs, 0, self.y)).astype(int)
        ids   = np.where(signs[:-1] != signs[1:])[0]

        Rs_lo = Rs[ids]
        Rs_hi = Rs[ids+1]

        # bracket left
        signs = np.sign(self.lens.dphi_Fermat_dx1(-Rs, 0, self.y)).astype(int)
        ids   = np.where(signs[:-1] != signs[1:])[0]

        Rs_lo = np.concatenate([Rs_lo, -Rs[ids]])
        Rs_hi = np.concatenate([Rs_hi, -Rs[ids+1]])

        p_crits = [self.find_image_bracket_1d(R0, R1) for R0, R1 in zip(Rs_lo, Rs_hi)]

        # look for sing/cusp
        singcusp = self.check_singcusp()
        if singcusp:
           p_crits.append(singcusp)

        return sorted(p_crits, key=lambda point: point['t'])

    def find_bracket(self, t):
        """Find all the different brackets of integration (i.e. all the
        different contours).

        Parameters
        ----------
        t : float
            Time delay :math:`t`.

        Returns
        -------
        brackets : list
            List of brackets with the limits of integration :math:`[r_i, r_f]`.
        """
        eps = 1e-8

        beta_zeroes = []

        # check for left and right extremum
        sign = -1
        for p in (self.p_crits_x[0], self.p_crits_x[-1]):
            if t > p['t']:
                x0 = sign*max(np.sqrt(2*np.abs(t))+self.y, 2*np.abs(p['x1']))

                sol = self.find_root_newton_1d(x0, t)
                x_root = sol.root

                # if the left point is not to the left (or the right to the right)
                if (x_root - p['x1'])*sign < 0:
                    if sign == 1:
                        message = "Newton's method failed at the rightmost bracket, "
                    if sign == -1:
                        message = "Newton's method failed at the leftmost bracket, "
                    message += "using the moving bracket method instead"
                    warnings.warn(message, TimeDomainWarning)

                    sol = self.find_moving_bracket_1d(p['x1'], p['x1']+sign*np.abs(x_root), t)
                    if sol == None:
                        message += " -> also failed"
                        raise TimeDomainWarning(message)
                    else:
                        x_root = sol.root

                beta_zeroes.append(np.abs(x_root))
            sign = 1

        # look for the other points
        for p1, p2 in zip(self.p_crits_x[:-1], self.p_crits_x[1:]):
            dt1 = p1['t'] - t
            dt2 = p2['t'] - t

            if dt1*dt2 < 0:
                x0_1 = p1['x1']
                x0_2 = p2['x1']

                if p2['type'] == 'sing/cusp max':
                    x0_2 -= eps
                    dt2 = self.lens.phi_Fermat(x0_2, 0, self.y) - t    # need to recompute to get PointLens to work
                    if dt1*dt2 > 0:
                        continue

                if p1['type'] == 'sing/cusp max':
                    x0_1 += eps
                    dt1 = self.lens.phi_Fermat(x0_1, 0, self.y) - t
                    if dt1*dt2 > 0:
                        continue

                sol = self.find_root_bracket_1d(x0_1, x0_2, t)

                beta_zeroes.append(np.abs(sol.root))

        beta_zeroes = np.sort(beta_zeroes)
        n = beta_zeroes.size

        # construct and check brackets
        if n == 0:
            message = "no brackets found"
            raise TimeDomainException(message)

        if n%2 != 0:
            message = "odd number of brackets found (n=%d, tau=%e)" % (n, t-self.tmin)
            raise TimeDomainException(message)

        brackets = beta_zeroes.reshape((int(n/2), 2))

        for br in brackets:
            rmin, rmax = br

            if self._dbeta(rmin, t) < 0:
                message = "incorrect left bracket assignment at r=%e (tau=%e)" % (rmin, t-self.tmin)
                raise TimeDomainException(message)

            if self._dbeta(rmax, t) > 0:
                message = "incorrect right bracket assignment at r=%e (tau=%e)" % (rmax, t-self.tmin)
                raise TimeDomainException(message)

        return brackets
    #===================================================================

    #=====  Integration
    #===================================================================
    def _beta(self, r, t):
        phi0 = 0.5*(r*r + self.y*self.y) - self.lens.psi_x(r) - t
        ry = r*self.y
        return -(phi0 + ry)*(phi0 - ry)

    def _dbeta(self, r, t):
        phi0  = 0.5*(r*r + self.y*self.y) - self.lens.psi_x(r) - t
        dphi0 = r - self.lens.dpsi_dx(r)

        return -2*(phi0*dphi0 - r*self.y**2)

    def _alpha(self, r, t):
        b = self._beta(r, t)
        return np.where(b > 0, r/np.sqrt(np.abs(b)), 0)

    def _integrand(self, xi, t, brackets):
        I = 0
        xi2 = xi*xi

        for br in brackets:
            rmin, rmax = br
            rmid = 0.5*(rmin + rmax)

            I += (rmax-rmid)*self._alpha(rmax - (rmax-rmid)*xi2, t)
            I += (rmid-rmin)*self._alpha(rmin + (rmid-rmin)*xi2, t)

        return xi*I

    def integrate_quadrature(self, t):
        """Computation of :math:`I(t)` with the ``'quadrature'`` method.

        Parameters
        ----------
        t : float or array
            Time delay :math:`t`.

        Returns
        -------
        I : float or array
            :math:`I(t)`.
        """
        brackets = self.find_bracket(t)
        f = lambda xi: self._integrand(xi, t, brackets)

        epsabs  = 1e-5
        epsrel  = 1e-5
        maxiter = 50
        xi_min = 1e-6

        sol = sc_integrate.quadrature(f, xi_min, 1, \
                                      tol=epsabs,\
                                      rtol=epsrel,\
                                      maxiter=maxiter)
        return 4*sol[0]

    def integrate_quad(self, t):
        """Computation of :math:`I(t)` with the ``'quad'`` method.

        Parameters
        ----------
        t : float or array
            Time delay :math:`t`.

        Returns
        -------
        I : float or array
            :math:`I(t)`.
        """
        brackets = self.find_bracket(t)

        f = lambda xi: self._integrand(xi, t, brackets)

        epsabs  = 1e-5
        epsrel  = 1e-5
        limit = 50
        xi_min = 1e-3

        sol = sc_integrate.quad(f, xi_min, 1, \
                                epsabs=epsabs,\
                                epsrel=epsrel,\
                                limit=limit)
        return 4*sol[0]

    def compute(self, tau):
        """Computation of :math:`I(\\tau)`.

        Depending on the precision parameter ``method``, it chooses between
        :func:`integrate_quadrature` or :func:`integrate_quad`.

        Parameters
        ----------
        tau : float or array
            Relative time delay :math:`\\tau`.

        Returns
        -------
        I : float or array
            :math:`I(\\tau)`.
        """
        t = tau + self.tmin

        if self.p_prec['method'] == 'quadrature':
            f = np.vectorize(self.integrate_quadrature)
        elif self.p_prec['method'] == 'quad':
            f = np.vectorize(self.integrate_quad)
        else:
            message = "integration method '%s' not recognized" % self.p_prec['method']
            raise TimeDomainWarning(message)

        I = np.piecewise(t, [t > self.tmin, t == self.tmin, t < self.tmin],
                            [            f,        self.I0,            0.])

        return I
    #===================================================================

    def get_contour(self, tau, N=1000):
        """Compute the contours in Cartesian coordinates.

        Parameters
        ----------
        tau : float
            Relative time delay :math:`\\tau`.
        N : int
            Number of points in the contour.

        Returns
        -------
        contour : list
            The length of this list is determined by the number of contours
            contributing to the given :math:`\\tau`. Each element (i.e. each
            contour) is given in the format :math:`[x_1, x_2]`.
        """
        t = tau + self.tmin
        brackets = self.find_bracket(t)

        half_points = int(N/2)
        if N%2 != 0:
            half_points += 1

        def _cnt_from_br(br):
            rmin, rmax = br
            rs = np.linspace(rmin, rmax, half_points)

            psis = self.lens.psi_x(rs)
            cth  = (t + psis - 0.5*rs**2 - 0.5*self.y**2)/rs/self.y

            # avoid round-off errors
            cth = np.where(cth > 1,   1, cth)
            cth = np.where(cth < -1, -1, cth)

            sth = np.sqrt(1 - cth**2)

            x1 = rs*cth
            x2 = rs*sth

            x1 = np.concatenate([x1, x1[::-1]])
            x2 = np.concatenate([x2, -x2[::-1]])

            return x1, x2

        contour = np.array([_cnt_from_br(br) for br in brackets])

        return contour


class It_AnalyticSIS(ItGeneral):
    """Analytic :math:`I(\\tau)` for the singular isothermal sphere.

    Additional information: :ref:`theory <AnalyticSIS_theory>`, :ref:`default parameters <pyIt_AnalyticSIS_default>`.

    Parameters
    ----------
    psi0 : float
        Normalization of the lensing potential :math:`\\psi(x) = \\psi_0 x`.
    """
    def __init__(self, y, p_prec={}, psi0=1):
        lens = lenses.Psi_SIS({'psi0':psi0})

        super().__init__(lens, y, p_prec)

        self.name = 'analytic SIS'
        self.psi0 = psi0

        self.p_crits = self.find_images()
        self.tmin = self.p_crits[0]['t']
        self.I0 = 2*np.pi*np.sqrt(self.p_crits[0]['mag'])

        ## ------------------------------------------
        self.R = (self.psi0 - self.y)/(self.psi0 + self.y)
        self.tau12 = 0.5*(self.psi0 + self.y)**2
        self.tau23 = (1 - self.R**2)*self.tau12

        if self.R > 0:
            self._compute_region2 = self._compute_region2A
        else:
            self._compute_region2 = self._compute_region2B
        ## ------------------------------------------

        self.t_grid, self.It_grid, self.eval_It = self.compute_all()

    def __str__(self):
        class_name = type(self).__name__
        class_call = "It = time_domain." + class_name + "(y, psi0, p_prec)"

        y_message = "y = %g\n" % self.y
        psi0_message = "psi0 = %g\n" % self.psi0
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        return y_message + psi0_message + prec_message + class_call

    def default_params(self):
        p_prec = {'Nt' : 10000}
        return p_prec

    def find_images(self):
        """Compute (analytically) the properties of the images for the SIS.

        Returns
        -------
        images : list
            Full list of images. The output is stored in ``p_crits``, detailed
            in :class:`~glow.time_domain.ItGeneral`.
        """
        y = self.y
        psi0 = self.psi0

        p_crits = [{'type' : 'min', \
                    't' : -0.5*psi0*(2*y + psi0), \
                    'x1' : y + psi0, \
                    'x2' : 0, \
                    'mag' : 1 + psi0/y}]

        if psi0 > y:
            p_crits.append({'type' : 'saddle', \
                            't' : 0.5*psi0*(2*y - psi0), \
                            'x1' : y - psi0, \
                            'x2' : 0, \
                            'mag' : np.abs(1 - psi0/y)})

        return p_crits

    def ellip_K(self, k):
        """Complete elliptic integral of the first kind, :math:`K(k)`.

        We follow the notation of [1]_.

        Parameters
        ----------
        k : float or array

        Returns
        -------
        K : float or array

        References
        ----------
        .. [1] \ I. S. Gradshteyn and I. M. Ryzhik, *Table of Integrals, Series, and Products* (Academic Press, 2007).
        """
        return sc_special.ellipk(k**2)

    def ellip_Pi(self, n, k):
        """Complete elliptic integral of the third kind, :math:`\\Pi(n, k)`.

        We follow the notation of [1]_.

        Parameters
        ----------
        n, k : float or array

        Returns
        -------
        Pi : float or array

        References
        ----------
        .. [1] \ I. S. Gradshteyn and I. M. Ryzhik, *Table of Integrals, Series, and Products* (Academic Press, 2007).
        """
        R = sc_special.elliprj(0, 1-k**2, 1, 1-n)

        return self.ellip_K(k) + n/3.*R

    def I_func(self, a, b, c, d, soft=0):
        r"""Combination of complete elliptic integrals.

        .. math::
            I_\text{SIS}(a, b, c, d) =
            \frac{8(b-c)}{\sqrt{(a-c)(b-d)}}
            \left[
                \Pi\left(\frac{a-b}{a-c}, r\right) + \frac{c\,K(r)}{(b-c)}
            \right]

        where

        .. math::
            r \equiv \sqrt{\frac{(a-b)(c-d)}{(a-c)(b-d)}}

        Parameters
        ----------
        a, b, c, d : float or array
        soft : float
            Softening factor, to help avoid divergences in denominators.

        Returns
        -------
        I_SIS : float or array
        """
        n  = (a-b)/(a-c + soft)
        r2 = n*(c-d)/(b-d + soft)
        C  = 2/np.sqrt((a-c)*(b-d + soft))

        K = sc_special.ellipk(r2)
        R = n/3.*sc_special.elliprj(0, 1-r2, 1, 1-n)

        return C*((b-c)*R + b*K)

    def _compute_region1(self, u, R):
        sqr = np.sqrt(u*u + R*R - 1)
        return 4*self.I_func(1+u, R+sqr, 1-u, R-sqr)

    def _compute_region2A(self, u, R):
        sqr = np.sqrt(1 - u*u)
        return 4*self.I_func(1, R, sqr, -sqr)

    def _compute_region2B(self, u, R):
        sqr = np.sqrt(1 - u*u)
        return 4*self.I_func(1, sqr, -sqr, R)

    def _compute_region3(self, u, R):
        sqr = np.sqrt(1 - u*u)
        return 4*self.I_func(1, sqr, R, -sqr)

    def compute(self, tau):
        r"""Compute :math:`I(\tau)`.

        Define the integral piece-wise in three regions:

            * Region 1: :math:`(u > 1)\ \to\ (\tau > (\psi_0+y)^2/2)`
            * Region 2: :math:`(\sqrt{1-R^2} < u < 1)\ \to\ (2\psi_0 y < \tau < (\psi_0+y)^2/2)`

                * 2A: :math:`(R > 0)\ \to\ (\psi_0 > y)`
                * 2B: :math:`(R < 0)\ \to\ (\psi_0 < y)`

            * Region 3: :math:`(0 < u < \sqrt{1-R^2})\ \to\ (0 < \tau < 2\psi_0 y)`

        where

        .. math::
            u \equiv \frac{\sqrt{2\tau}}{\psi_0 + y}\ ,\qquad
            R \equiv \frac{\psi_0-y}{\psi_0 + y}

        Parameters
        ----------
        tau : float or array
            Relative time delay :math:`\tau`.

        Returns
        -------
        I : float or array
            :math:`I(\tau)`.
        """
        f1 = lambda tau: self._compute_region1(np.sqrt(2*tau)/(self.psi0 + self.y), self.R)
        f2 = lambda tau: self._compute_region2(np.sqrt(2*tau)/(self.psi0 + self.y), self.R)
        f3 = lambda tau: self._compute_region3(np.sqrt(2*tau)/(self.psi0 + self.y), self.R)

        I = np.piecewise(tau, \
            [tau > self.tau12, (tau<self.tau12)&(tau>self.tau23), (tau<self.tau23)&(tau>0),   tau<0, tau<0], \
            [              f1,                                f2,                       f3, self.I0,     0])

        return I


class It_NaiveAreaIntegral(ItGeneral):
    """Simple implementation of the binning/grid/area method for the computation of :math:`I(\\tau)`.

    The algorithm uses a uniform grid with polar coordinates centered at the lens position, :math:`\\rho=r^2`
    and :math:`\\theta`.
    First, it stores all the evaluations of the
    Fermat potential into an array. This array is then sorted (thus we can
    directly compute the minimum time delay) and divided into
    :math:`\\Delta t`-boxes of unequal size. The size (i.e. temporal
    spacing) is chosen with the criterium that each box is computed using
    the same number of points.

    This method is extraordinarily simple, provides the minimum time-delay and
    adaptative time sampling, but it cannot be scaled (since sorting becomes
    prohibitive). Also, with this method, some portion for large :math:`\\tau` must
    be thrown away (some of the contributions to the integral are always left
    out of the grid).

    Additional information: :ref:`theory <AreaIntegral_theory>`, :ref:`default parameters <pyIt_NaiveAreaIntegral_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.time_domain.ItGeneral`
    for the internal information of the parent class)

    Parameters
    ----------
    p_prec : dict, optional
        Precision parameters. New keys:

            * ``rho_min``, ``rho_max`` (*float*) -- Minimum and maximum radial coordinate :math:`\\rho=r^2`.
            * ``N_rho`` (*int*) -- Number of points in the :math:`\\rho` axis.
            * ``N_theta`` (*int*) -- Number of points in the :math:`\\theta` axis.
            * ``N_intervals`` (*int*) -- Number of points in the evaluation of :math:`I(\\tau)`.

    Warnings
    --------
    This method is only included due to its simplicity and lack of assumptions. It should never be
    used for applications that require high precision or speed. It is intended only for verification.
    The C version :class:`~glow.time_domain_c.It_AreaIntegral_C` provides a (slightly) better
    optimization, avoiding both the storage of all the grid evaluations and the time-sorting problem, using
    instead a fixed temporal grid.
    """
    def __init__(self, Lens, y, p_prec={}):
        super().__init__(Lens, y, p_prec)

        self.name = 'naive area integral'

        self.tmin, self.t_grid, self.It_grid = self.compute()
        self.I0 = self.It_grid[0]

        self.interp_It = self.interpolate(self.t_grid, self.It_grid)
        self.eval_It = lambda t: np.piecewise(t, [t<0, t>=0], [0, self.interp_It])

    def default_params(self):
        p_prec = {'rho_min'   : 0,
                  'rho_max'   : 20,
                  'N_rho'   : 10000,
                  'N_theta' : 500,
                  'N_intervals' : 500}

        return p_prec

    def compute(self):
        """Compute :math:`I(\\tau)`.

        Returns
        -------
        t0 : float
            Minimum time delay :math:`t_\\text{min}`.
        t_k : array
            Temporal grid :math:`\\tau_i = t_i - t_\\text{min}`.
        It_k : array
            Result grid :math:`I_i=I(t_i)`.
        """
        # spatial grid params (rho=r^2)
        rho_min   = self.p_prec['rho_min']
        rho_max   = self.p_prec['rho_max']
        theta_min = -np.pi
        theta_max = np.pi
        N_rho     = self.p_prec['N_rho']
        N_theta   = self.p_prec['N_theta']
        Drho   = (rho_max - rho_min)/(N_rho-1)

        sym_factor = 1
        if self.lens.isAxisym:
            theta_min = 0
            sym_factor = 2

        Dtheta = (theta_max - theta_min)/(N_theta-1)

        # grid in t
        N_t = N_rho*N_theta
        N_intervals = self.p_prec['N_intervals']  # number of dt boxes in the histogram
        N_dt = N_t/N_intervals      # number of points per dt interval

        # precompute grid quantities
        rho_i = np.linspace(rho_min, rho_max, N_rho)
        theta_j = np.linspace(theta_min, theta_max, N_theta)
        ones_j = np.ones_like(theta_j)
        r_i = np.sqrt(rho_i)
        ctheta_j = np.cos(theta_j)
        x1_ij = np.outer(r_i, ctheta_j)

        if self.lens.isAxisym:
            psi_i = self.lens.psi_x(r_i)
            psi_ij = np.outer(psi_i, ones_j)
        else:
            stheta_j = np.sin(theta_j)
            x2_ij = np.outer(r_i, stheta_j)
            psi_ij = self.lens.psi(x1_ij, x2_ij)

        phi_geo = 0.5*np.outer(rho_i, ones_j) - self.y*x1_ij + 0.5*self.y**2

        # full Fermat potential
        phi = phi_geo - psi_ij

        # compute the time delay, setting the minimum to 0
        t = np.sort(phi.flatten(), kind='mergesort')            # bottleneck
        t0 = t[0]
        t -= t0

        # compute the integral
        # (choose each dt such that it contains N_dt points)
        ks = np.arange(0, N_intervals-1)
        dt_k = np.array([t[int((k+1)*N_dt-1)] - t[int(k*N_dt)] for k in ks])
        t_k = np.array([t[int(k*N_dt)] for k in ks])
        dt_k_max = np.max(dt_k)

        It_k = 0.5*sym_factor*Dtheta*Drho*N_dt/dt_k

        return t0, t_k, It_k
