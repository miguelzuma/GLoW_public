import warnings
import numpy as np
from scipy import signal as sc_signal
from scipy import special as sc_special
from scipy import interpolate as sc_interpolate
from scipy import integrate as sc_integrate

# optional dependency: needed only for the point lens
try:
    import mpmath as mp
except ModuleNotFoundError:
    mp = None

from . import lenses
from . import time_domain

class FreqDomainException(Exception):
    pass

class FreqDomainWarning(UserWarning):
    pass

##==============================================================================


class FwGeneral():
    """Base class for the amplification factor.

    Parameters
    ----------
    It : ItGeneral subclass
        Time-domain integral object.
    p_prec : dict, optional
        Precision parameters.

    Attributes
    ----------
    lens : PsiGeneral subclass
        Lens object.
    It : ItGeneral subclass
        Time-domain integral object.
    p_prec : dict
        Default precision parameters updated with the input. Default keys:

        * ``interp_fill_value`` -- Behaviour of the interpolation function\
            outside the interpolation range. Options:

            * *None*: Raise an error if extrapolation is attempted.
            * *float*: Extrapolate with a constant value.
            * ``'extrapolate'``: Linear extrapolation.

        * ``interp_kind`` (*str*) -- Interpolation method. Any kind recognized by Scipy\
            is allowed. In particular:

            * ``'linear'``
            * ``'cubic'``

        * ``T_saddle`` (*float*) -- Free parameter :math:`T` used in the regularization of the saddle points.

    eval_It_reg : func(float or array) -> float or array
        Evaluate the regular part of :math:`I(\\tau)`, i.e. :math:`I_\\text{reg}=I-I_\\text{sing}`.
    name : str
        (*subclass defined*) Name of the method.
    w_grid, Fw_grid : array
        (*subclass defined*) Grid of points where :math:`F(w)` has been computed.
    eval_Fw : func(float or array) -> complex or array
        (*subclass defined*) Final function to evaluate :math:`F(w)`.
    eval_Fw_reg : func(float or array) -> complex or array
        (*subclass defined*) Evaluate the regular part of :math:`F(w)`, i.e. :math:`F_\\text{reg}=F-F_\\text{sing}`.
    """
    def __init__(self, It, p_prec={}):
        self.It = It
        self.lens = It.lens

        self.p_prec = self.default_general_params()
        self.p_prec_default_keys = set(self.p_prec.keys())
        self.p_prec.update(p_prec)

        self.check_general_input()

        try:
            self.p_crits = It.p_crits
        except AttributeError as e:
            message = 'no critical points (p_crits) found in It (%s)' % It.name
            raise FreqDomainException(message) from e

        # split It into regular and singular contributions
        self.p_crit_params = [(p['type'], p['t']-It.tmin, np.sqrt(p['mag'])) for p in self.p_crits]
        self.It_sing_funcs = {'min' : self._It_min,
                              'max' : self._It_max,
                              'saddle' : self._It_saddle,
                              'sing/cusp max' : lambda tau, tau_c, sqrt_mu: np.zeros_like(tau),
                              'sing/cusp min' : lambda tau, tau_c, sqrt_mu: np.zeros_like(tau)}
        self.Fw_sing_funcs = {'min' : self._Fw_min,
                              'max' : self._Fw_max,
                              'saddle' : self._Fw_saddle,
                              'sing/cusp max' : lambda w, tau_c, sqrt_mu: np.zeros_like(w),
                              'sing/cusp min' : lambda w, tau_c, sqrt_mu: np.zeros_like(w)}

        self.eval_It_reg = lambda t: It.eval_It(t) - self.eval_It_sing(t)

        # ***** to be overriden by the subclass *****
        self.name = 'unknown'
        self.w_grid, self.Fw_grid = np.array([]), np.array([])
        self.eval_Fw = lambda w: np.zeros_like(w, dtype=complex)
        self.eval_Fw_reg = lambda w: np.zeros_like(w, dtype=complex)
        # *******************************************

    def __str__(self):
        class_name = type(self).__name__
        class_call = "Fw = freq_domain." + class_name + "(It, p_prec)"

        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        It_message = self.It.__str__() + "\n\n"

        return It_message + prec_message + class_call

    def __call__(self, w):
        """Call :func:`eval_Fw`."""
        return self.eval_Fw(w)

    def default_general_params(self):
        """Fill the default parameters.

        Update the precision parameters common for all methods and
        then call :func:`default_params` (*subclass defined*).

        Returns
        -------
        p_prec : dict
            Default precision parameters.
        """
        p_prec = {'interp_fill_value' : None,
                  'interp_kind' : 'linear',
                  'T_saddle' : 3}

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
        # note: if the subclass will add a new parameter without an entry
        #       in default_params, it must be manually added to
        #       self.p_phys_default_keys or self.p_prec_default_keys
        #       in check_input() with self.p_phys_default_keys.add(new_key)
        self.check_input()

        # check that there are no unrecognized parameters
        p_prec_new_keys = set(self.p_prec.keys())
        diff_prec = p_prec_new_keys - self.p_prec_default_keys

        if diff_prec:
            for key in diff_prec:
                message = "unrecognized key '%s' found in p_prec will be "\
                          "(most likely) ignored" % key
                warnings.warn(message, FreqDomainWarning)

    def display_info(self):
        """Print internal information in human-readable form."""
        print("\t////////////////////////////\n"\
              "\t///   F(w) information   ///\n"\
              "\t////////////////////////////")

        print("\n * Method: %s" % self.name)

        if self.p_prec != {}:
            print("\n * Precision parameters:")
            for key, value in self.p_prec.items():
                print("   **", key, "=", value)
        else:
            print('\nNo information available')

        print("\n * Lens: %s" % self.lens.p_phys.get('name', 'no information'))
        print(" * Time domain method: %s" % self.It.name)
        print(" * Impact parameter: y = %g\n" % self.It.y)

    def interpolate(self, xs, ys):
        """Construct an interpolation function using the options in ``p_prec``.

        Parameters
        ----------
        xs, ys : float, complex or array
            Values to interpolate.

        Returns
        -------
        interp_func : func(array) -> array
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

    def _It_min(self, tau, tau_c, sqrt_mu):
        It_sing = 2*np.pi*sqrt_mu
        return np.where(tau >= tau_c, It_sing, 0)

    def _It_max(self, tau, tau_c, sqrt_mu):
        It_sing = 2*np.pi*sqrt_mu
        return np.where(tau <= tau_c, It_sing, 0)

    def _It_saddle(self, tau, tau_c, sqrt_mu):
        dtau = np.abs(tau - tau_c)
        It_sing = -2*sqrt_mu*np.log(dtau)
        It_windowed = np.exp(-dtau/self.p_prec['T_saddle'])*It_sing
        return It_windowed

    def eval_It_sing(self, tau):
        r"""Singular contribution to :math:`I(\tau)`.

        Parameters
        ----------
        t : float or array
            :math:`\tau`.

        Returns
        -------
        It : float or array
            :math:`I_\text{sing}(\tau)`.
        """
        It = 0
        for kind, tau_c, sqrt_mu in self.p_crit_params:
            It += self.It_sing_funcs[kind](tau, tau_c, sqrt_mu)
        return It

    def _Fw_min(self, w, tau_c, sqrt_mu):
        return sqrt_mu*np.exp(1j*w*tau_c)

    def _Fw_max(self, w, tau_c, sqrt_mu):
        return -sqrt_mu*np.exp(1j*w*tau_c)

    def _Fw_saddle(self, w, tau_c, sqrt_mu):
        tmp = 1./self.p_prec['T_saddle'] - 1j*w
        I_plus = -(np.euler_gamma + np.log(tmp))/tmp

        Fw = 1j*w/np.pi*sqrt_mu*np.exp(1j*w*tau_c)*2*np.real(I_plus)
        return Fw

    def eval_Fw_sing(self, w):
        r"""Singular contribution to :math:`F(w)`.

        Parameters
        ----------
        w : float or array
            :math:`w`.

        Returns
        -------
        Fw : float or array
            :math:`F_\text{sing}(w)`.
        """
        Fw = 0j
        for kind, tau_c, sqrt_mu in self.p_crit_params:
            Fw += self.Fw_sing_funcs[kind](w, tau_c, sqrt_mu)
        return Fw

    # ***** to be overriden by the subclass *****
    def check_input(self):
        """(*subclass defined, optional*) Check the input of the implementation."""
        pass

    def default_params(self):
        """(*subclass defined*) Initialize the default parameters."""
        p_prec = {}
        return p_prec
    # *******************************************


##==============================================================================


class Fw_FFT_OldReg(FwGeneral):
    """Computation of the amplification factor as the FFT of :math:`I(\\tau)`.

    This class uses the information about the images stored in ``p_crits`` to regularize the
    time-domain integral:

    .. math::
        I_\\text{reg}(\\tau) \\equiv I(\\tau) - I_\\text{sing}(\\tau)

    where :math:`I_\\text{sing}(\\tau)` is analytical and depends on the images. This regular part
    is then Fourier transformed (with a FFT) to obtain :math:`F_\\text{reg}(w)`. Finally, we add back the
    (analytical) Fourier transform of the singular part, :math:`F_\\text{sing}(w)`, to get the
    amplification factor:

    .. math::
        F(w) = F_\\text{reg}(w) + F_\\text{sing}(w)

    Additional information: :ref:`theory <Regularization_theory>`, :ref:`default parameters <pyFw_FFT_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.freq_domain.FwGeneral`
    for the internal information of the parent class)

    Attributes
    ----------
    w_grid : array
        Grid of frequencies where the FFT has been computed.
    Fw_reg_grid : array
        FFT of the regular part of the time-domain integral, i.e. :math:`F_\\text{reg}(w)`.
    Fw_grid : array
        If we are working in the inteporlation mode, ``Fw_grid`` stores the grid of values
        that will be used for interpolation. Otherwise it contains zeroes.

    Parameters
    ----------
    p_prec : dict, optional
        Precision parameters. New keys:

        * ``wmin``, ``wmax`` (*float*) -- Minimum and maximum frequencies to be computed.
        * ``window_transition`` (*float*) -- Width of the Tukey window function.
        * ``smallest_tau_max`` (*float*) -- Ensure that :math:`I(\\tau)` is always computed, at least,\
            up to this value.
        * ``eval_mode`` (*str*) -- Behaviour of :func:`eval_Fw`, to compute\
            :math:`F(w)=F_\\text{reg}(w)+F_\\text{sing}(w)`. In both cases, :math:`F_\\text{reg}` is\
            computed with a FFT on a grid and then evaluated using an interpolation function. The two\
            options control how :math:`F_\\text{sing}` is computed.

            * ``'interpolate'``: :math:`F_\\text{sing}(w)` is precomputed on the same grid as\
                :math:`F_\\text{reg}`, and then evaluated using an interpolation function.
            * ``'exact'``: :math:`F_\\text{sing}(w)` is evaluated analytically for each :math:`w`.

        * ``FFT type`` (*str*) -- Choose the FFT solver. The options are Numpy (``'numpy'``) and\
            Scipy (``'scipy'``). The differences are minimal.
        * ``FFT method`` (*str*) -- The options are:

            * ``'standard'``: Perform a single FFT. When the frequency range is very large, this method can\
                become very time-consuming and noisy.
            * ``'multigrid'``, ``'multigrid_stack'``: Perform independent FFTs, with varying time resolution,\
                and patch them together.

        * ``N_above_discard``, ``N_below_discard``, ``N_keep`` (*int*) -- Parameters used with\
            the ``'multigrid'`` option. In this mode, the frequency range is divided into\
            :math:`n = \\frac{1}{N_\\text{keep}}\\log(w_\\text{max}/w_\\text{min})` intervals of frequency,\
            logarithmically spaced. We then perform :math:`n` independent FFTs in these intervals. To reduce\
            the errors at high and low frequencies we actually perform the FFT in each interval starting from\
            higher and lower frequencies (that are later discarded). The actual size of each of the\
            FFTs is :math:`\\log_2(N_{FFT}) = N_\\text{below} + N_\\text{keep} + N_\\text{above}`.
    """
    def __init__(self, It, p_prec={}):
        super().__init__(It, p_prec)

        self.name = 'FFT (old regularization)'
        self.windows_dic = {}

        # split It into regular/singular contributions and interpolate
        # the regular part
        It_reg_grid = It.It_grid - self.eval_It_sing(It.t_grid)

        self.t_grid = np.concatenate([[0.], It.t_grid])
        self.It_reg_grid = np.concatenate([[0.], It_reg_grid])
        self.It_grid = np.concatenate([[It.I0], It.It_grid])

        # Fourier transform the regular part
        self.w_grid, self.Fw_reg_grid = self.compute(self.eval_It_reg)

        # compute analitically the Fourier transform of the singular part and add it
        self.eval_Fw_reg = self.interpolate(self.w_grid, self.Fw_reg_grid)

        if self.p_prec['eval_mode'] == 'interpolate':
            self.Fw_grid = self.Fw_reg_grid + self.eval_Fw_sing(self.w_grid)
            self.eval_Fw = self.interpolate(self.w_grid, self.Fw_grid)
        elif self.p_prec['eval_mode'] == 'exact':
            self.Fw_grid = np.zeros_like(self.Fw_reg_grid)
            self.eval_Fw = lambda w: self.eval_Fw_reg(w) + self.eval_Fw_sing(w)
        else:
            message = "evaluation mode '%s' not recognized" % self.p_prec['eval_mode']
            raise FreqDomainException(message)

    def default_params(self):
        p_prec = {'wmin' : 1e-2,
                  'wmax' : 1e2,
                  'window_transition' : 0.8,
                  'smallest_tau_max'  : 10,
                  'N_above_discard' : 8,
                  'N_below_discard' : 4,
                  'N_keep'          : 2,
                  'interp_kind' : 'linear',
                  'eval_mode'   : 'interpolate',
                  'FFT type'    : 'numpy',
                  'FFT method'  : 'multigrid stack'}

        return p_prec

    def tau_range(self, fmin, fmax):
        """Compute the time array for the FFT.

        Given a frequency range, we compute the times where we need to evaluate
        our function to perform the FFT. We include some corrections for the
        window size and the requirements set by ``smallest_tau_max``.

        Parameters
        ----------
        fmin, fmax : float
            Minimum and maximum frequencies for the FFT.

        Returns
        -------
        tau_min, tau_max : float
            Minimum and maximum times of the time array (which will be equally spaced).
        N_fft : int
            Size of the array, i.e. size of the FFT.
        """
        window_transition = self.p_prec['window_transition']
        smallest_tau_max = self.p_prec['smallest_tau_max']

        n = max(1, smallest_tau_max*fmin/(1-0.5*window_transition))

        Dtau = n/fmin
        tau_min = -window_transition/2.*Dtau
        tau_max = Dtau + tau_min

        max_tau_from_It = self.It.t_grid[-1]
        if tau_max > max_tau_from_It:
            print(' -> WARNING: wished tau_max=%g larger than available for interpolation (%g)'\
                  % (tau_max, max_tau_from_It))
            tau_max = max_tau_from_It

        #FFT faster for the right choices of N_fft
        N_fft = 2**int(np.log2(fmax/fmin)+1)

        fmax_real = (N_fft-1)/2/Dtau
        fmax_needed = fmax/2**self.p_prec['N_above_discard']
        if fmax_needed > fmax_real:
            N_fft = 2**int(np.log2(1 + 2*Dtau*fmax_needed)+1)

        return tau_min, tau_max, N_fft

    def get_window(self, N_fft):
        """Compute the Tukey window function.

        Parameter
        ---------
        N_fft : int
            Length.

        Returns
        -------
        window : array
            Tukey window function.
        """
        # the window functions are precomputed, if we need the same size we just pick it up
        try:
            window = self.windows_dic[N_fft]
        except KeyError:
            window = sc_signal.windows.tukey(N_fft, alpha=self.p_prec['window_transition'])
            self.windows_dic[N_fft] = window

        return window

    def compute_FFT(self, Is):
        """Compute the FFT of an array.

        Parameter
        ---------
        Is : array
            Input array.

        Returns
        -------
        I_fft : array
            Real FFT of the input.
        """
        if self.p_prec['FFT type'] == 'numpy':
            I_fft = np.fft.rfft(Is)
        elif self.p_prec['FFT type'] == 'scipy':
            I_fft = sc_fft.rfft(Is)
        else:
            message = "FFT type '%s' not recognized" % self.p_prec['FFT type']
            raise FreqDomainWarning(message)

        return I_fft

    def compute_FFT_freq(self, N_fft, dtau, imin=0, imax=None):
        """
        Compute the FFT of a quantity
        """
        # equivalent to numpy's
        # f_fft = np.fft.rfftfreq(N_fft, dtau)[imin:imax]

        if imax is None:
            imax = int(N_fft/2+1)

        df = 1/dtau/N_fft
        f_fft = np.arange(imin, imax)*df

        return f_fft

    def compute_It_FFT(self, fmin, fmax, eval_It):
        """Compute the (windowed) FFT of a function in a frequency range.

        Parameters
        ----------
        fmin, fmax : float
            Frequency range.
        eval_It : func(float or array) -> float or array
            Function to be transformed.

        Returns
        -------
        tau_min : float
            Minimum time sampled.
        dtau : float
            Spacing of the time array, i.e. temporal resolution.
        f_fft : array
            Frequencies.
        It_fft : array
            FFT of the input function in the frequency range.
        """
        tau_min, tau_max, N_fft = self.tau_range(fmin, fmax)
        tau_uni = np.linspace(tau_min, tau_max, N_fft)
        dtau    = (tau_max-tau_min)/(N_fft-1)

        window  = self.get_window(N_fft)
        It_wind = window*eval_It(tau_uni)

        f_fft  = self.compute_FFT_freq(N_fft, dtau)
        It_fft = self.compute_FFT(It_wind)

        return tau_min, dtau, f_fft, It_fft

    def transform_Fw(self, tau_min, dtau, f_fft, It_fft):
        """Final transformations of the FFT of :math:`I(\\tau)` to get :math:`F(w)`.

        Parameters
        ----------
        tau_min : float
            Minimum time in the grid.
        dtau : float
            Time spacing.
        f_fft, It_fft : array
            Frequency grid and FFT.

        Returns
        -------
        w : array
            Angular frequency, :math:`w=2\\pi f`.
        Fw : array
            Amplification factor.
        """
        w = 2*np.pi*f_fft

        # phase-shift to account for tau_min != 0
        time_shift = np.exp(-1j*w*tau_min)
        Fw = -1j*dtau*f_fft*np.conj(It_fft*time_shift)

        return w, Fw

    def compute_standard(self, wmin, wmax, eval_It):
        """Computation of :math:`F(w)` with a single FFT, between a maximum and minimum frequency.

        Parameters
        ----------
        wmin, wmax: float
            Minimum and maximum frequencies :math:`w`.
        eval_It : func(float or array) -> float or array
            :math:`I(\\tau)`.

        Returns
        -------
        w_grid, Fw_grid : array
            Grids with the frequencies and :math:`F(w)`.
        """
        tau_min, dtau, f_fft, It_fft   = self.compute_It_FFT(wmin/2/np.pi, wmax/2/np.pi, eval_It)
        w_grid, Fw_grid = self.transform_Fw(tau_min, dtau, f_fft, It_fft)

        return w_grid, Fw_grid

    def get_freq_multigrid(self, wmin, wmax):
        """Compute the frequency ranges for the FFTs with the multigrid method.

        The length of the output corresponds to the number of different FFTs that will be computed.

        Parameters
        ----------
        wmin, wmax : float
            Minimum and maximum angular frequencies, :math:`w=2\\pi f`.

        Returns
        -------
        fmin_batch, fmax_batch : array
            Minimum and maximum frequencies that will be kept for each FFT.
        fmin_real, fmax_real : array
            Minimum and maximum frequencies that will be actually computed for each FFT.
        """
        # number of frequency decades to be computed
        N_above_discard = self.p_prec['N_above_discard']
        N_below_discard = self.p_prec['N_below_discard']
        N_keep = self.p_prec['N_keep']

        log2_wmin = np.log2(wmin)
        log2_wmax = np.log2(wmax)

        N_batches = int((log2_wmax-log2_wmin)/N_keep + 1)
        dlog2_w = (log2_wmax-log2_wmin)/N_batches

        log2_wmin_batch = np.arange(N_batches)*dlog2_w + log2_wmin
        log2_wmax_batch = log2_wmin_batch + dlog2_w

        fmin_batch = 2**log2_wmin_batch/2./np.pi
        fmax_batch = 2**log2_wmax_batch/2./np.pi

        fmin_real = 2**(log2_wmin_batch - N_below_discard)/2./np.pi
        fmax_real = 2**(log2_wmax_batch + N_above_discard)/2./np.pi

        return fmin_batch, fmax_batch, fmin_real, fmax_real

    def compute_multigrid(self, wmin, wmax, eval_It):
        """Computation of :math:`F(w)` patching together several FFTs.

        Parameters
        ----------
        wmin, wmax: float
            Minimum and maximum frequencies :math:`w`.
        eval_It : func(float or array) -> float or array
            :math:`I(\\tau)`.

        Returns
        -------
        w_grid, Fw_grid : array
            Grids with the frequencies and :math:`F(w)`.
        """
        fmin_batch, fmax_batch, fmin_real, fmax_real = self.get_freq_multigrid(wmin, wmax)

        # not great but the number of batches is not very large
        def _compute_batch(i):
            tau_min, dtau, f_fft,  It_fft = self.compute_It_FFT(fmin_real[i], fmax_real[i], eval_It)

            # keep only the solution inside N_keep decades
            df = f_fft[1]
            imin = int(fmin_batch[i]/df + 1)
            imax = int(fmax_batch[i]/df + 1)

            w, Fw = self.transform_Fw(tau_min, dtau, f_fft[imin:imax], It_fft[imin:imax])

            return w, Fw

        data = [_compute_batch(i) for i, fs in enumerate(fmin_batch)]

        w_grid  = np.concatenate([d[0] for d in data])
        Fw_grid = np.concatenate([d[1] for d in data])

        return w_grid, Fw_grid

    def compute_multigrid_stack(self, wmin, wmax, eval_It):
        """Computation of :math:`F(w)` patching together several FFTs.

        Parameters
        ----------
        wmin, wmax: float
            Minimum and maximum frequencies :math:`w`.
        eval_It : func(float or array) -> float or array
            :math:`I(\\tau)`.

        Returns
        -------
        w_grid, Fw_grid : array
            Grids with the frequencies and :math:`F(w)`.
        """
        fmin_batch, fmax_batch, fmin_real, fmax_real = self.get_freq_multigrid(wmin, wmax)

        tau_range = [self.tau_range(fmin, fmax) for fmin, fmax in zip(fmin_real, fmax_real)]

        batches = {}
        for i, (tau_min, tau_max, n) in enumerate(tau_range):
            try:
                batches[n][0].append(tau_min)
                batches[n][1].append(tau_max)
                batches[n][2].append(i)
            except KeyError:
                batches[n] = [[tau_min], [tau_max], [i]]

        ## ------------------------------------------

        # convenience function to be used in the loop below
        def _compute_subbatch(i):
            i1 = imin[i]
            i2 = imax[i]

            fs = self.compute_FFT_freq(n_fft, dtau[i], i1, i2)
            Is = It_fft[i][i1:i2]

            w, Fw = self.transform_Fw(tau_min[i], dtau[i], fs, Is)
            return w, Fw

        # loop over batches of FTs with same length
        ws  = []
        Fws = []
        for n_fft, (tau_min, tau_max, ids) in batches.items():
            tau_min = np.array(tau_min)
            tau_max = np.array(tau_max)
            ids     = np.array(ids, dtype=int)

            # number of FTs of the same length n_fft
            n_batch = len(ids)

            # 2d time grid (n_batch, n_fft)
            taus = np.linspace(tau_min, tau_max, n_fft).transpose()
            dtau = (tau_max-tau_min)/(n_fft-1)

            # compute It in the grid and FT it
            window  = np.tile(self.get_window(n_fft), (n_batch, 1))
            It_wind = window*eval_It(taus)
            It_fft  = self.compute_FFT(It_wind)

            # minimum and maximum indices to keep in the FFT (N_keep)
            df = 1/dtau/n_fft
            imin = (fmin_batch[ids]/df + 1).astype(int)
            imax = (fmax_batch[ids]/df + 1).astype(int)

            # transform the FFT, get freqs and transpose
            dat = [_compute_subbatch(i) for i, d in enumerate(ids)]
            w, Fw = [list(d) for d in zip(*dat)]

            ws.append(np.concatenate(w))
            Fws.append(np.concatenate(Fw))

        w_grid  = np.concatenate(ws)
        Fw_grid = np.concatenate(Fws)

        return w_grid, Fw_grid

    def compute(self, eval_It):
        """Computation of :math:`F(w)`.

        Parameters
        ----------
        eval_It : func(float or array) -> float or array
            :math:`I(\\tau)`.

        Returns
        -------
        w_grid, Fw_grid : array
            Grids with the frequencies and :math:`F(w)`.
        """
        wmin = self.p_prec['wmin']
        wmax = self.p_prec['wmax']
        method = self.p_prec['FFT method']

        if method == 'standard':
            w, Fw = self.compute_standard(wmin, wmax, eval_It)
        elif method == 'multigrid':
            w, Fw = self.compute_multigrid(wmin, wmax, eval_It)
        elif method == 'multigrid stack':
            w, Fw = self.compute_multigrid_stack(wmin, wmax, eval_It)
        else:
            message = "FFT method '%s' not recognized" % self.p_prec['FFT method']
            raise FreqDomainException(message)

        if method == 'standard':
            w = w[1:]
            Fw = Fw[1:]
        else:
            # avoid extrapolation errors adding first/last point
            Fw_min = (Fw[1]*(wmin-w[0]) - Fw[0]*(wmin-w[1]))/(w[1]-w[0])
            Fw_max = (Fw[-1]*(wmax-w[-2]) - Fw[-2]*(wmax-w[-1]))/(w[-1]-w[-2])

            w  = np.concatenate([[wmin], w, [wmax]])
            Fw = np.concatenate([[Fw_min], Fw, [Fw_max]])

        return w, Fw


class Fw_SemiAnalyticSIS(FwGeneral):
    """Semi-analytic computation of the amplification factor for the SIS.

    To arrive at this result, the radial integral is first solved analitically
    and then the angular integral is performed numerically.

    Additional information: :ref:`theory <SemiAnalyticSIS_theory>`, :ref:`default parameters <pyFw_SemiAnalyticSIS_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.freq_domain.FwGeneral`
    for the internal information of the parent class)

    Parameters
    ----------
    y : float
        Impact parameter.
    psi0 : float
        Normalization of the SIS lensing potential, :math:`\\psi(x)=\\psi_0 x`.

    Warnings
    --------
    This implementation only allows for not very large frequencies (around :math:`w` < 100)
    and it is pretty slow. Try :class:`~glow.freq_domain_c.Fw_SemiAnalyticSIS_C` for a faster version.
    """
    def __init__(self, y, p_prec={}, psi0=1):
        It = time_domain.It_AnalyticSIS(y, {'eval_mode':'exact'}, psi0)
        super().__init__(It, p_prec)

        self.y = self.It.y
        self.psi0 = psi0
        self.name = '(pseudo) analytic SIS'

        self.compute_I_vec = np.vectorize(self._compute_I_novec)

        self.eval_Fw = self.compute
        self.eval_Fw_reg = lambda w: self.eval_Fw(w) - self.eval_Fw_sing(w)

    def __str__(self):
        class_name = type(self).__name__
        class_call = "Fw = freq_domain." + class_name + "(y, psi0, p_prec)"

        y_message = "y = %g\n" % self.y
        psi0_message = "psi0 = %g\n" % self.psi0
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        return y_message + psi0_message + prec_message + class_call

    def _fg_tilde(self, x):
        S, C = sc_special.fresnel(x)

        c = np.cos(0.5*np.pi*x**2)
        s = np.sin(0.5*np.pi*x**2)

        f = (0.5 - S)*c - (0.5 - C)*s
        g = (0.5 - C)*c + (0.5 - S)*s

        return np.array([f*x, g*x])

    def _compute_I_novec(self, alpha0, r):
        integrand = lambda theta: self._fg_tilde(-alpha0*(1+r*np.cos(theta)))
        If, Ig = sc_integrate.quad_vec(integrand, 0, np.pi)[0]

        return If, Ig

    def compute(self, w):
        """Computation of :math:`F(w)`.

        Parameters
        ----------
        w : float or array
            Dimensionless frequency :math:`w`.

        Returns
        -------
        Fw : complex or array
            :math:`F(w)`.
        """
        if np.max(w) > 1e3:
            message = "maximum value for w in PseudoAnalyticSIS too large "\
                      "to yield reliable results (w_max=1e3)"
            raise FreqDomainException(message)

        y = self.y
        psi0 = self.psi0

        r = y/psi0
        alpha0 = psi0*np.sqrt(w/np.pi)
        C = np.exp(1j*w*0.5*(psi0+y)**2)

        If, Ig = self.compute_I_vec(alpha0, r)

        # different sign convention from C code
        Fw = 1 - If + 1j*Ig

        return Fw*C


class Fw_AnalyticPointLens(FwGeneral):
    """Analytic computation of the amplification factor for the point lens.

    Additional information: :ref:`theory <AnalyticPointLens_theory>`, :ref:`default parameters <pyFw_AnalyticPointLens_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.freq_domain.FwGeneral`
    for the internal information of the parent class)

    Parameters
    ----------
    y : float
        Impact parameter.

    Warnings
    --------
    Very slow. It requires the `mpmath` module since the hypergeometric function in Scipy does
    not allow for complex parameters (only argument). The alternative implementation
    :class:`~glow.freq_domain_c.Fw_AnalyticPointLens_C` is several orders of magnitude faster. It uses
    several analytical approximations calibrated to achieve approximately single precision.
    """
    def __init__(self, y, p_prec={}):
        if mp is None:
            message = "It seems that the python module mpmath is not installed. Without it, the python "\
                      "version of the analytic Fw for the point lens cannot be used."
            raise ModuleNotFoundError(message)

        Psi = lenses.Psi_PointLens({'psi0' : 1})
        It = time_domain.ItGeneral(Psi, y)

        x1 = y*0.5*(1 + np.sqrt(1 + 4/y**2))
        pmin = {'type' : 'min',
                'x1'   : x1,
                'x2'   : 0.,
                't'    : Psi.phi_Fermat(x1, 0., y),
                'mag'  : Psi.shear(x1, 0.)['mag']}

        x1 = y*0.5*(1 - np.sqrt(1 + 4/y**2))
        psad = {'type' : 'saddle',
                'x1'   : x1,
                'x2'   : 0.,
                't'    : Psi.phi_Fermat(x1, 0., y),
                'mag'  : Psi.shear(x1, 0.)['mag']}

        pmax = {'type' : 'sing/cusp max',
                'x1'   : 0.,
                'x2'   : 0.,
                't'    : Psi.phi_Fermat(1e-15, 0., y),
                'mag'  : Psi.shear(1e-15, 0.)['mag']}

        It.p_crits = [pmin, psad, pmax]
        It.tmin = pmin['t']

        super().__init__(It, p_prec)

        self.y = self.It.y
        self.name = 'analytic point lens'

        self.xm = 0.5*(y + np.sqrt(y*y + 4))
        self.phim = 0.5*(self.xm - y)**2 - np.log(self.xm)

        self.F11_vec = np.vectorize( lambda w: complex(mp.hyp1f1(0.5j*w, 1, 0.5j*w*self.y**2)) )

        self.eval_Fw = self.compute
        self.eval_Fw_reg = lambda w: self.eval_Fw(w) - self.eval_Fw_sing(w)

    def __str__(self):
        class_name = type(self).__name__
        class_call = "Fw = freq_domain." + class_name + "(y, p_prec)"

        y_message = "y = %g\n" % self.y
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        return y_message + prec_message + class_call

    def compute(self, w):
        """Computation of :math:`F(w)`.

        Parameters
        ----------
        w : float or array
            Dimensionless frequency :math:`w`.

        Returns
        -------
        Fw : complex or array
            :math:`F(w)`.
        """
        exp = np.exp(0.5j*w*(np.log(0.5*w) - 2*self.phim))
        gamma = np.exp(sc_special.loggamma(1 - 0.5j*w) + 0.25*np.pi*w)
        F11 = self.F11_vec(w)

        return exp*gamma*F11
