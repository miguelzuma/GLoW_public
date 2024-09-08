import os
import warnings
import numpy as np
import scipy.interpolate as sc_interpolate

from . import lenses
from . import wrapper

class TimeDomainException(Exception):
    pass
    
class TimeDomainWarning(UserWarning):
    pass

## *********************************************************************
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def convert_old_syntax(y):
    if isinstance(y, dict):
        message = "\n\n%s(DEPRECATED)%s\n"\
                  "The following syntax is deprecated and support will be removed in the future:\n"\
                  " >>> It(Psi, {'y':y}, p_prec)\n\n"\
                  " Use instead:\n"\
                  " >>> It(Psi, y, p_prec)\n" % (bcolors.WARNING, bcolors.ENDC)
        warnings.warn(message, TimeDomainWarning)
        
        if 'y' not in y:
            message = "impact parameter 'y' required"
            raise TimeDomainException(message)
        else:
            y_out = y['y']
    else:
        y_out = y
    return y_out
## *********************************************************************

##==============================================================================


class ItGeneral_C():
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
        
        * ``C_prec`` (*dict*) -- Optional dictionary to change the precision parameters in\
            the C code.

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
        * ``lens_file_fname`` (*str*) -- Usually, if the lens is not implemented in C an error\
            should be raised. However, if the lens is axisymmetric we will still try to proceed.\
            The lens is precomputed on a logarithmic grid with the parameters below, stored in \
            ``lens_file_fname`` and then evaluated in C using an interpolation function.
        * ``lens_file_Nx`` (*int*) -- Number of points in the lens grid.
        * ``lens_file_xmin`` (*float*) -- Lower limit for the lens grid.
        * ``lens_file_xmax`` (*float*) -- Upper limit for the lens grid.

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

        y = convert_old_syntax(y)
        self.y = y

        self.p_prec = self.default_general_params()
        self.p_prec_default_keys = set(self.p_prec.keys())        
        self.p_prec.update(p_prec)
        
        # update precision parameters in the C code
        self._update_Cprec()
        
        # lens that will be passed to C (same as Lens in most cases)
        self.lens_to_c = self.check_lens()
        
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
        class_call = "It = time_domain_c." + class_name + "(Psi, y, p_prec)"
    
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
        p_prec = {'Nt'       : 5000, \
                  'tmin'     : 1e-2, \
                  'tmax'     : 1e6, \
                  'eval_mode' : 'interpolate', \
                  'sampling' : 'log', \
                  'interp_fill_value' : None, \
                  'interp_kind'       : 'linear', \
                  'oversampling_n'    : 10, \
                  'oversampling_tmin' : 1e-1, \
                  'oversampling_tmax' : 1e1, \
                  'lens_file_xmin'  : 1e-7, \
                  'lens_file_xmax'  : 1e7, \
                  'lens_file_Nx'    : 10000, \
                  'lens_file_fname' : 'wrapper/glow_lib/external/tmp', \
                  'C_prec' : {}}

        p_prec2 = self.default_params()
        if p_prec2 is not {}:
            p_prec.update(p_prec2)
        
        return p_prec
    
    def check_lens(self):
        """Check that the lens is implemented in C.
        
        If the lens is axisymmetric, it will try to proceed (raising a warning), 
        precomputing it on a grid and then evaluating it in C using an interpolation 
        function. If the lens is not axisymmetric, an error will be raised an error. 
        """
        if wrapper.check_implemented_lens(self.lens) is True:
            lens = self.lens
        else:
            message = "lens '%s' not implemented in C, proceeding numerically" % self.lens.p_phys['name']
            warnings.warn(message, TimeDomainWarning)
            
            if self.lens.isAxisym is False:
                message = "numerical lens not yet implemented for non-symmetric lenses"
                raise TimeDomainException(message)
            else:
                root_name = os.path.dirname(os.path.abspath(__file__)) \
                                + '/' + self.p_prec['lens_file_fname']
                
                self.lens.to_file(root_name,\
                                  self.p_prec['lens_file_xmin'],\
                                  self.p_prec['lens_file_xmax'],\
                                  self.p_prec['lens_file_Nx'])                                  
                
                lens = lenses.PsiAxisym()
                lens.p_phys = {'name' : 'grid 1d',
                               'root' : root_name,
                               'n_grid' : self.p_prec['lens_file_Nx']}
        
        return lens
    
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
        # note: if the subclass will add a new parameter without an entry
        #       in default_params, it must be manually added to 
        #       self.p_phys_default_keys or self.p_prec_default_keys
        #       in check_input() with self.p_phys_default_keys.add(new_key)
        
        if self.p_prec['sampling'] == 'oversampling':
            if self.p_prec['tmin'] > self.p_prec['oversampling_tmin']:
                message = "'oversampling_tmin' smaller than 'tmin' (%g < %g)"\
                        % (self.p_prec['oversampling_tmin'], self.p_prec['tmin'])
                raise TimeDomainException(message)
            if self.p_prec['tmax'] < self.p_prec['oversampling_tmax']:
                message = "'oversampling_tmax' larger than 'tmax' (%g > %g)"\
                        % (self.p_prec['oversampling_tmax'], self.p_prec['tmax'])
                raise TimeDomainException(message)
            
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
    
    def get_Cprec(self):
        """Get the precision parameters currently used in the C code.
        
        Returns
        -------
        Cprec : dict
            Precision parameters.
        """
        return wrapper.get_Cprec()
        
    def display_Cprec(self):
        """Print the C precision parameters in human-readable form."""
        wrapper.display_Cprec()
        
    def _update_Cprec(self):
        # need to restart the values for different instances in the same session
        wrapper.set_default_Cprec()
        wrapper.update_Cprec(self.p_prec['C_prec'])
        
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
        if self.p_prec['eval_mode'] == 'interpolate':
            t_grid, It_grid, interp_It = self.compute_grid()
            
            tmin = t_grid[0]
            eval_low_t = lambda t: self.eval_low_tau(t, tmin, It_grid[0])
            
            eval_It = lambda t: np.piecewise(t, \
                                            [t<0, (t>=0)&(t<tmin),   t>=tmin], \
                                            [  0,      eval_low_t, interp_It])
                                            
        if self.p_prec['eval_mode'] == 'exact':
            t_grid = np.array([])
            It_grid = np.array([])
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


class It_SingleContour_C(ItGeneral_C):
    """Computation for the single contour regime (only one image).
    
    Additional information: :ref:`theory <SingleContour_theory>`, :ref:`default parameters <cIt_SingleContour_default>`.
    
    (Only new parameters and attributes are documented. See :class:`~glow.time_domain_c.ItGeneral_C`
    for the internal information of the parent class)

    Parameters
    ----------
    p_prec : dict, optional
        Precision parameters. New keys:
        
        * ``parallel`` (*bool*) -- Perform the evaluation over :math:`\\tau`-arrays in parallel.
        * ``method`` (*str*) -- Method for the computation. Options:
            
            * ``'standard'`` : Parametrize the contours with the angular variable :math:`\\theta`. If\
                it fails, it will automatically switch to ``'robutst'``.
            * ``'robust'`` : Parametric representation of the contours. 
    """
    def __init__(self, Lens, y, p_prec={}):
        super().__init__(Lens, y, p_prec)
        
        self.name = 'single contour (C code)'
        
        self.p_crits = self.find_all_images()
        self.tmin = self.p_crits[0]['t']
        self.I0 = 2*np.pi*np.sqrt(self.p_crits[0]['mag'])
            
        self.t_grid, self.It_grid, self.eval_It = self.compute_all()
            
    def default_params(self):
        p_prec = {'method'   : 'standard',\
                  'parallel' : True}                  
        return p_prec
    
    def find_all_images(self):
        """Find the minimum of the Fermat potential assuming that it is the only critical point.
        
        An error is raised if more than one critical point is found. 
        """
        n_points, x1_min, x2_min, tmin, mag = wrapper.pyCheck_min(self.y, \
                                                                  self.lens_to_c)
        
        if n_points == 1:
            p_crits = [{'type' : 'min',\
                         't'   : tmin,\
                         'x1'  : x1_min,\
                         'x2'  : x2_min,\
                         'mag' : mag}]
            
            return p_crits
        else:
            message = "More than one critical point found."
            raise TimeDomainException(message)
    
    def compute(self, tau):
        """Compute :math:`I(\\tau)`.
        
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
        return wrapper.pyContour(tau, \
                                 self.p_crits[0]['x1'], \
                                 self.p_crits[0]['x2'], \
                                 self.y, \
                                 self.lens_to_c, \
                                 method=self.p_prec['method'], \
                                 parallel=self.p_prec['parallel'])
                                 
    def get_contour(self, tau, n_points=0):
        """Contour corresponding to a given :math:`\\tau`.
        
        Parameters
        ----------
        tau : float or array
            Relative time delay :math:`\\tau`.
        n_points : int
            Number of points in the contour.
        
        Returns
        -------
        contour : array of dict
            The length of the output matches the length of the input ``tau``.
            For each element, the contour is stored as a dictionary with the following entries:
            
            * ``'sigma'`` : Parameter of the curve. 
            * ``'alpha'``, ``'R'`` : Polar coordinates (centered at the position of the minimum). 
            * ``'x1'``, ``'x2'`` : Cartesian coordinates. 
        """
        return wrapper.pyGetContour(tau, \
                                    self.p_crits[0]['x1'], \
                                    self.p_crits[0]['x2'], \
                                    self.y, \
                                    self.lens_to_c, \
                                    method=self.p_prec['method'], \
                                    n_points=n_points, \
                                    parallel=self.p_prec['parallel'])


class It_SingleIntegral_C(ItGeneral_C):
    """Computation for axisymmetric lenses, solving a single radial integral.
    
    Additional information: :ref:`theory <SingleIntegral_theory>`, :ref:`default parameters <cIt_SingleIntegral_default>`.
    
    (Only new parameters and attributes are documented. See :class:`~glow.time_domain_c.ItGeneral_C`
    for the internal information of the parent class)

    Parameters
    ----------
    p_prec : dict, optional
        Precision parameters. New keys:
        
        * ``parallel`` (*bool*) -- Perform the evaluation over :math:`\\tau`-arrays in parallel.
        * ``method`` (*str*) -- Method for the integral (see the GSL documentation for details). Options:
            
            * ``'qng'`` : Non-adaptive Gauss-Kronrod integration.
            * ``'qag15'`` : Adaptive 15 point Gauss-Kronrod integration.
            * ``'qag21'`` : Adaptive 21 point Gauss-Kronrod integration.
            * ``'qag31'`` : Adaptive 31 point Gauss-Kronrod integration.
            * ``'qag41'`` : Adaptive 41 point Gauss-Kronrod integration.
            * ``'qag51'`` : Adaptive 51 point Gauss-Kronrod integration.
            * ``'qag61'`` : Adaptive 61 point Gauss-Kronrod integration.
            * ``'direct'`` : Direct integration using the ``qags`` method from GSL, without performing\
                a change of variables to smooth the integrand.
    """
    def __init__(self, Lens, y, p_prec={}):
        super().__init__(Lens, y, p_prec)
        
        self.name = 'single integral (C code)'
        
        self.p_crits = self.find_all_images()
        self.tmin = self.p_crits[0]['t']
        self.I0 = 2*np.pi*np.sqrt(self.p_crits[0]['mag'])
        
        self.t_grid, self.It_grid, self.eval_It = self.compute_all()
    
    def check_input(self):
        if self.lens.isAxisym is False:
            message = 'Single Integral not implemented for non-axisymmetric lenses'
            raise TimeDomainException(message)
        
    def default_params(self):
        p_prec = {'parallel' : True,\
                  'method' : 'qag15'}                
        return p_prec
    
    def find_all_images(self):
        """Find all the critical points.
        
        Returns
        -------
        images : list
            Full list of images. The output is stored in ``p_crits``, detailed 
            in :class:`~glow.time_domain_c.ItGeneral_C`.
        """
        p_crits = wrapper.pyFind_all_CritPoints_1D(self.y, self.lens_to_c)
        return p_crits            
    
    def compute(self, tau):
        """Computation of :math:`I(\\tau)`.
        
        It includes the step function, :math:`I(\\tau<0)=0`.
        
        Parameters
        ----------
        tau : float or array
            Relative time delay :math:`\\tau`.
            
        Returns
        -------
        I : float or array
            :math:`I(\\tau)`.
        """
        return wrapper.pySingleIntegral(tau,
                                        self.y,
                                        self.lens_to_c,
                                        self.p_crits,
                                        method=self.p_prec['method'],
                                        parallel=self.p_prec['parallel'])
    
    def get_contour(self, tau, n_points=100):
        """Compute the contours.
        
        Parameters
        ----------
        tau : float or array
            Relative time delay :math:`\\tau`.
        n_points : int
            Number of points in the contours. If it is zero, the contour contains the points used
            in the integration. 
            
        Returns
        -------
        contour : array of dict
            The length of the output matches the length of the input ``tau``.
            For each element, the contours are stored as a dictionary with the following entries:
            
            * ``'x1'``, ``'x2'`` : Cartesian coordinates. ``contour['x1']`` is a list of arrays. The\
                length of this list is the number of different contours that contribute to the\
                given :math:`\\tau`.
        """
        return wrapper.pyGetContourSI(tau,
                                      self.y,
                                      self.lens_to_c,
                                      self.p_crits,
                                      n_points=n_points,
                                      parallel=self.p_prec['parallel'])
        

class It_AnalyticSIS_C(ItGeneral_C):
    """Analytic :math:`I(\\tau)` for the singular isothermal sphere.
    
    Additional information: :ref:`theory <AnalyticSIS_theory>`, :ref:`default parameters <cIt_AnalyticSIS_default>`.
    
    (Only new parameters and attributes are documented. See :class:`~glow.time_domain_c.ItGeneral_C`
    for the internal information of the parent class)
    
    Parameters
    ----------
    psi0 : float
        Normalization of the lensing potential :math:`\\psi(x) = \\psi_0 x`.
        
    p_prec : dict, optional
        Precision parameters. New keys:
        
        * ``parallel`` (*bool*) -- Perform the evaluation over :math:`\\tau`-arrays in parallel.
    """
    def __init__(self, y, p_prec={}, psi0=1):
        lens = lenses.Psi_SIS({'psi0':psi0})
            
        super().__init__(lens, y, p_prec)
        
        self.name = 'analytic SIS (C code)'
        self.psi0 = psi0
        
        self.p_crits = self.find_all_images()
        self.tmin = self.p_crits[0]['t']
        self.I0 = 2*np.pi*np.sqrt(self.p_crits[0]['mag'])
        
        self.t_grid, self.It_grid, self.eval_It = self.compute_all()
            
    def __str__(self):
        class_name = type(self).__name__
        class_call = "It = time_domain_c." + class_name + "(y, psi0, p_prec)"
    
        y_message = "y = %g\n" % self.y
        psi0_message = "psi0 = %g\n" % self.psi0
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"
        
        return y_message + psi0_message + prec_message + class_call
    
    def default_params(self):
        p_prec = {'Nt'       : 10000, \
                  'parallel' : True}            
        return p_prec
    
    def find_all_images(self):
        """Compute (analytically) the properties of the images for the SIS.
        
        Returns
        -------
        images : list
            Full list of images. The output is stored in ``p_crits``, detailed 
            in :class:`~glow.time_domain_c.ItGeneral_C`.
        """
        psi0 = self.psi0
        y = self.y
        
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

    def compute(self, tau):
        r"""Compute :math:`I(\tau)`.
        
        It includes the step function, :math:`I(\tau<0)=0`.
        
        Parameters
        ----------
        tau : float or array
            Relative time delay :math:`\tau`.
            
        Returns
        -------
        I : float or array
            :math:`I(\tau)`.
        """
        return wrapper.pyIt_SIS(tau, self.y, self.psi0, parallel=self.p_prec['parallel'])


class It_AreaIntegral_C(ItGeneral_C):
    """Simple implementation of the binning/grid/area method for the computation of the 
    time-domain integral.
    
    Additional information: :ref:`theory <AreaIntegral_theory>`, :ref:`default parameters <cIt_AreaIntegral_default>`.
      
    (Only new parameters and attributes are documented. See :class:`~glow.time_domain_c.ItGeneral_C`
    for the internal information of the parent class)

    Parameters
    ----------
    p_prec : dict, optional
        Precision parameters. New keys:
            
            * ``tmax`` (*float*) -- Maximum :math:`\\tau` to compute. The upper limit in the radial\
                coordinate :math:`\\rho` is chosen accordingly.
            * ``n_rho`` (*int*) -- Number of points in the :math:`\\rho` axis.
            * ``n_theta`` (*int*) -- Number of points in the :math:`\\theta` axis.
    
    Warnings
    --------
    This implementation is not heavily optimized, it is only intended for
    verification. It yields robust, albeit slow and noisy, results for any lens.
    """
    def __init__(self, Lens, y, p_prec={}):
        super().__init__(Lens, y, p_prec)
        
        self.name = 'area integral (C code)'
        
        self.tmin, self.t_grid, self.It_grid = self.compute()
        self.I0 = self.It_grid[0]
        self.mag = (self.I0/2/np.pi)**2
        
        self.interp_It = self.interpolate(self.t_grid, self.It_grid)
        self.eval_It = lambda t: np.piecewise(t, [t<0, t>=0], [0, self.interp_It])
        
    def default_params(self):
        p_prec = {'n_rho' : 20000,\
                  'n_theta' : 2000,\
                  'tmax' : 10,\
                  'Nt' : 500}
        
        return p_prec 
        
    def compute(self):
        """Compute :math:`I(\\tau)`.
        
        Returns
        -------
        t0 : float
            Minimum time delay :math:`t_\\text{min}`.
        t_grid : array
            Temporal grid :math:`\\tau_i`.
        It_grid : array
            Result grid :math:`I_i=I(\\tau_i)`.
        """
        tmin, tau_grid, It_grid = wrapper.pyAreaIntegral(self.y, \
                                                         self.lens_to_c, \
                                                         self.p_prec)
        
        return tmin, tau_grid, It_grid
        

class It_MultiContour_C(ItGeneral_C):
    """Computation using the contour method for a generic strong lensing scenario.
    
    Additional information: :ref:`theory <Multicontour_theory>`, :ref:`default parameters <cIt_Multicontour_default>`.
    
    (Only new parameters and attributes are documented. See :class:`~glow.time_domain_c.ItGeneral_C`
    for the internal information of the parent class)
    
    Parameters
    ----------
    p_prec : dict, optional
        Precision parameters. New keys:
        
        * ``parallel`` (*bool*) -- Perform the evaluation over :math:`\\tau`-arrays in parallel.
    
    Attributes
    ----------
    p_centers : list of dict
        Centers of the different families of contours. Keys:
        
        * ``type`` (*str*) -- Type of center:
            
            * ``'min'``, ``'max'`` -- The center of the family of contours is a minimum/maximum of the
                Fermat potential.
            * ``'saddle 8 minmin'``, ``'saddle 8 maxmax'`` -- The center of the contours is a saddle point,
                with a critical curve that looks like an 8 and either two minima or two maxima in the lobes.
            * ``'saddle O min'``, ``'saddle O max'`` -- The center of the contours is a saddle point,
                with a critical curve that looks like a folded 8 (e.g. the standard SIS) and either
                a minimum or a maximum in the innermost lobe.
                
        * ``x10``, ``x20`` (*float*) -- Location of the center.
        * ``tau0``, ``t0`` (*float*) -- Time delays :math:`\\tau` and :math:`t=\\tau + t_\\text{min}`\
            at the center.
        * ``alpha_out`` (*float*) -- Angle with respect to :math:`x_1` that is used to search for the\
            right contour (in a straight line out of the center) and start the integration. 
        * ``R_max`` (*float*) -- Distance to the last contour contributing in this family, in the\
            direction :math:`\\alpha_\\text{out}`.
        * ``tau_birth``, ``tau_death`` (*float*) -- Minimum and maximum :math:`\\tau` that the family\
            of contours will contribute to.
        * ``is_init_birthdeath`` (*int*) -- Wheter the contour has been fully initialized (1) or not (0).\
            There must be always one (and only one) center where this variable is 0, since\
            :math:`R_\\text{max}` and :math:`\\tau_\\text{death}` are not set. This is the outermost\
            family of contours that extends to infinity.
    """
    def __init__(self, Lens, y, p_prec={}):
        super().__init__(Lens, y, p_prec)
        
        self.name = 'multicontour (C code)'
        
        self.p_crits = self.find_all_images()
        self.tmin = self.p_crits[0]['t']
        self.I0 = 2*np.pi*np.sqrt(self.p_crits[0]['mag'])
        
        self.p_centers = self.init_all_centers()
        self.t_grid, self.It_grid, self.eval_It = self.compute_all()
            
    def default_params(self):
        p_prec = {'parallel' : True}                  
        return p_prec
        
    def display_centers(self):
        """Print the information about the centers of the contours in human-readable form."""
        print("\t////////////////////////////\n"\
              "\t///       Centers        ///\n"\
              "\t////////////////////////////")
        print("\n * Lens: %s  (y = %g)" % (self.lens.p_phys['name'], self.y))
        for i, p in enumerate(self.p_centers):                            
            print("\n * Center %d  (%s):" % (i, p['type']))
            print("   **        t = %g" % p['t0'])
            print("   **      tau = %g" % (p['tau0']))
            print("   **        x = (%g, %g)" % (p['x10'], p['x20']))
            print("   ** alpha/pi = %g" % (p['alpha_out']/np.pi))
            print("   **    R_max = %g" % p['R_max'])
            if p['is_init_birthdeath'] != 0:
                print("   **      b/d = (%g, %g)" % (p['tau_birth'], p['tau_death']))
            else:
                print("   ** outermost point")
        print('')
    
    def find_all_images(self):
        """Find all the critical points.
        
        Returns
        -------
        images : list
            Full list of images. The output is stored in ``p_crits``, detailed 
            in :class:`~glow.time_domain_c.ItGeneral_C`.
        """    
        p_crits = wrapper.pyFind_all_CritPoints_2D(self.y, self.lens_to_c)
        return p_crits
        
    def init_all_centers(self):
        """Find all the centers of contours.
        
        Returns
        -------
        centers : list
            Full list of centers. The output is stored in ``p_centers``, detailed above.
        """
        p_centers = wrapper.pyInit_all_Centers(self.p_crits, self.y, self.lens_to_c)
        return p_centers
    
    def compute(self, tau):
        """Computation of :math:`I(\\tau)`.
        
        It includes the step function, :math:`I(\\tau<0)=0`.
        
        Parameters
        ----------
        tau : float or array
            Relative time delay :math:`\\tau`.
            
        Returns
        -------
        I : float or array
            :math:`I(\\tau)`.
        """
        return wrapper.pyMultiContour(tau, \
                                      self.p_centers, \
                                      self.y, \
                                      self.lens_to_c, \
                                      parallel=self.p_prec['parallel'])
    
    def get_contour(self, tau, n_points=0):
        """Compute the contours.
        
        Parameters
        ----------
        tau : float or array
            Relative time delay :math:`\\tau`.
        n_points : int
            Number of points in the contours. If it is zero, the contour contains the points used
            in the integration. 
            
        Returns
        -------
        contour : array of dict
            The length of the output matches the length of the input ``tau``.
            For each element, the contours are stored as a dictionary with the following entries:
            
            * ``'x1'``, ``'x2'`` : Cartesian coordinates. ``contour['x1']`` is a list of arrays. The\
                length of this list is the number of different contours that contribute to the\
                given :math:`\\tau`.
            * ``'sigma'`` : Parameter of the curve. 
            * ``'alpha'``, ``'R'`` : Polar coordinates (with respect to the center of the contour).
        """
        return wrapper.pyGetMultiContour(tau, \
                                         self.p_centers, \
                                         self.y, \
                                         self.lens_to_c, \
                                         n_points=n_points, \
                                         parallel=self.p_prec['parallel'])
    
    def get_contour_x1x2(self, x10, x20, sigmaf=100, n_points=100):
        """Compute the contour passing by a given point in the lens plane.
        
        The integration is carried out directly in Cartesian coordinates.
        
        Parameters
        ----------
        x10, x20 : float or array
            Initial position to start the integration.
        sigmaf : float
            Maximum parameter of the curve. The contour is computed in the parametric form 
            :math:`\\boldsymbol{x}(\\sigma)` and the integration is carried out from :math:`\\sigma=0`
            to :math:`\\sigma=\\sigma_f`.
        n_points : int
            Number of points in the contour.
            
        Returns
        -------
        contour : dict
            The length of the output matches the length of the input ``x10``, ``x20``.
            For each element, the contours are stored as a dictionary with the following entries:
            
            * ``'sigma'`` : Parameter of the curve. 
            * ``'x1'``, ``'x2'`` : Cartesian coordinates.
        """
        return wrapper.pyGetContour_x1x2(x10=x10, \
                                         x20=x20, \
                                         y=self.y, \
                                         sigmaf=sigmaf, \
                                         n_points=n_points, \
                                         Psi=self.lens_to_c, \
                                         parallel=self.p_prec['parallel'])
