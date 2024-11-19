#
# GLoW - freq_domain_c.py
#
# Copyright (C) 2023, Hector Villarrubia-Rojo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import warnings
import numpy as np
from scipy import signal as sc_signal
from scipy import fft as sc_fft
from scipy import interpolate as sc_interpolate

from . import wrapper
from . import lenses
from . import time_domain_c

class FreqDomainException(Exception):
    pass

class FreqDomainWarning(UserWarning):
    pass

##==============================================================================


class FwGeneral_C():
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

        * ``C_prec`` (*dict*) -- Optional dictionary to change the precision parameters in\
            the C code.

        * ``interp_fill_value`` -- Behaviour of the interpolation function\
            outside the interpolation range. Options:

            * *None*: Raise an error if extrapolation is attempted.
            * *float*: Extrapolate with a constant value.
            * ``'extrapolate'``: Linear extrapolation.

        * ``interp_kind`` (*str*) -- Interpolation method. Any kind recognized by Scipy\
            is allowed. In particular:

            * ``'linear'``
            * ``'cubic'``

        * ``reg_stage`` (*int*) -- Regularization of :math:`I(\\tau)`. Possible values:

            * ``0`` : No regularization performed.
            * ``1`` : Initial step function at :math:`\\tau=0`, saddle points and local minima/maxima regularized.
            * ``2`` : All of the above, plus regularization of the long-tails, assuming\
                :math:`I(\\tau\\to\\infty)\\sim A_2/\\tau^{\\sigma_2}`.\
                It can be computed analytically if we have enough information about the lens, otherwise\
                it is obtained with a numerical fit.
            * ``3`` : All of the above, plus regularization of the initial slope at :math:`\\tau=0` and\
                the second order tails, assuming :math:`I(\\tau\\to\\infty)\\sim A_2/\\tau^{\\sigma_2} + A_3/\\tau^{\\sigma_3}`.\
                Both are obtained by fitting :math:`I(\\tau)`.

    reg_sch : dict
        Regularization scheme. Keys:

        * ``stage`` (*int*) -- Regularization stage computed. The regularization stage that will be\
            actually used is the value set in ``p_prec['reg_stage']``.
        * ``p_crits`` (*dict*) -- Critical points. Content of ``It.p_crits``.
        * ``has_shear`` (*bool*) -- True if the lens contains an external shear field.
        * ``I_shear_asymp`` (*float*) -- Asymptotic value of `I(\\tau)/2\\pi`.
        * ``tau_shear_scale`` (*float*) -- Scale for step regularization of the shear field.
        * ``slope`` (*float*) -- Slope at :math:`\\tau=0`.
        * ``amp`` (*list*) -- Asymptotic amplitudes :math:`A_2` and :math:`A_3`.
        * ``index`` (*list*) -- Asymptotic indices :math:`\\sigma_2` and :math:`\\sigma_3`.

    name : str
        (*subclass defined*) Name of the method.
    w_grid, Fw_grid : array
        (*subclass defined*) Grid of points where :math:`F(w)` has been computed.
    eval_Fw : func(float or array) -> complex or array
        (*subclass defined*) Final function to evaluate :math:`F(w)`.
    eval_Fw_reg : func(float or array) -> complex or array
        (*subclass defined*) Evaluate the regular part of :math:`F(w)`, i.e. :math:`F_\\text{reg}=F-F_\\text{sing}`.
    eval_It_reg : func(float or array) -> float or array
        (*subclass defined*) Evaluate the regular part of :math:`I(\\tau)`, i.e. :math:`I_\\text{reg}=I-I_\\text{sing}`.
    """
    def __init__(self, It, p_prec={}):
        self.It = It
        self.lens = It.lens

        self.p_prec = self.default_general_params()
        self.p_prec_default_keys = set(self.p_prec.keys())
        self.p_prec.update(p_prec)

        self._update_Cprec()
        self.check_general_input()

        try:
            self.p_crits = It.p_crits
        except AttributeError as e:
            message = 'no critical points (p_crits) found in It (%s)' % It.name
            raise FreqDomainException(message) from e

        self.reg_sch = self._fill_RegScheme()

        # ***** to be overriden by the subclass *****
        self.name = 'unknown'
        self.w_grid, self.Fw_grid = np.array([]), np.array([])
        self.eval_Fw = lambda w: np.zeros_like(w, dtype=complex)
        self.eval_Fw_reg = lambda w: np.zeros_like(w, dtype=complex)
        self.eval_It_reg = lambda tau: self.It(tau) - self.eval_It_sing(tau)
        # *******************************************

    def __str__(self):
        class_name = type(self).__name__
        class_call = "Fw = freq_domain_c." + class_name + "(It, p_prec)"

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
                  'interp_kind' : 'cubic',
                  'reg_stage' : 2,
                  'C_prec' : {}}

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

    def _fill_RegScheme(self):
        reg_sch = {}
        reg_sch['stage']     = 1
        reg_sch['p_crits']   = self.p_crits
        reg_sch['slope']     = None
        reg_sch['amp']       = [None, None]
        reg_sch['index']     = [None, None]
        reg_sch['has_shear'] = False
        reg_sch['I_shear_asymp']   = 1
        reg_sch['tau_shear_scale'] = 1

        A = self.It.lens.asymp_amplitude
        index = self.It.lens.asymp_index

        if (A is not None) and (index is not None):
            reg_sch['amp'][0]   = A
            reg_sch['index'][0] = index
            reg_sch['stage'] = 2

        if self.lens.p_phys['name'] == 'combined lens':
            if self.lens.has_shear:
                kp = self.lens.kappa
                g1 = self.lens.gamma1
                g2 = self.lens.gamma2

                reg_sch['has_shear'] = True
                reg_sch['I_shear_asymp'] = 1/np.sqrt((1-kp)**2 - g1**2 - g2**2)

        return reg_sch

    def _update_RegScheme(self, stage, t_grid, It_grid, parallel=False):
        It_reg_grid = wrapper.pyUpdate_RegScheme(stage, t_grid, It_grid, self.reg_sch, parallel)
        return It_reg_grid

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

    def eval_It_sing(self, tau, stage=None, parallel=False):
        r"""Singular contribution to :math:`I(\tau)`.

        Parameters
        ----------
        tau : float or array
            :math:`\tau`.
        stage : int or None
        parallel : bool

        Returns
        -------
        It : float or array
            :math:`I_\text{sing}(\tau)`.
        """
        if stage is None:
            stage = self.p_prec['reg_stage']

        return wrapper.pyIt_sing(tau, self.reg_sch, stage, parallel)

    def eval_Fw_sing(self, w, stage=None, parallel=False):
        r"""Singular contribution to :math:`F(w)`.

        Parameters
        ----------
        w : float or array
            :math:`w`.
        stage : int or None
        parallel : bool

        Returns
        -------
        Fw : float or array
            :math:`F_\text{sing}(w)`.
        """
        if stage is None:
            stage = self.p_prec['reg_stage']

        return wrapper.pyFw_sing(w, self.reg_sch, stage, parallel)

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


class Fw_FFT_C(FwGeneral_C):
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

    Additional information: :ref:`theory <Regularization_theory>`, :ref:`default parameters <cFw_FFT_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.freq_domain_c.FwGeneral_C`
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

        * ``FFT method`` (*str*) -- The options are:

            * ``'standard'``: Perform a single FFT. When the frequency range is very large, this method can\
                become very time-consuming and noisy.
            * ``'multigrid'``: Perform independent FFTs, with varying time resolution, and patch them together.

        * ``N_above_discard``, ``N_below_discard``, ``N_keep`` (*int*) -- Parameters used with\
            the ``'multigrid'`` option. In this mode, the frequency range is divided into\
            :math:`n = \\frac{1}{N_\\text{keep}}\\log(w_\\text{max}/w_\\text{min})` intervals of frequency,\
            logarithmically spaced. We then perform :math:`n` independent FFTs in these intervals. To reduce\
            the errors at high and low frequencies we actually perform the FFT in each interval starting from\
            higher and lower frequencies (that are later discarded). The actual size of each of the\
            FFTs is :math:`\\log_2(N_{FFT}) = N_\\text{below} + N_\\text{keep} + N_\\text{above}`.

        * ``parallel`` (*bool*) -- Perform the computation in parallel (one FFT per thread).
    """
    def __init__(self, It, p_prec={}):
        super().__init__(It, p_prec)

        self.name = 'FFT (new regularization, full C version)'

        It_reg_grid = self._update_RegScheme(self.p_prec['reg_stage'], \
                                             It.t_grid, It.It_grid, \
                                             self.p_prec['parallel'])

        self.t_grid = np.concatenate([[0.], It.t_grid])
        self.It_reg_grid = np.concatenate([[0.], It_reg_grid])
        self.It_grid = np.concatenate([[self.It.I0], It.It_grid])

        self.eval_It_reg = lambda t: np.interp(t, self.t_grid, self.It_reg_grid, left=0)

        self.w_grid, self.Fw_grid, self.Fw_reg_grid = self.compute()
        self.eval_Fw_reg = self.interpolate(self.w_grid, self.Fw_reg_grid)

        if self.p_prec['eval_mode'] == 'interpolate':
            self.eval_Fw = self.interpolate(self.w_grid, self.Fw_grid)
        elif self.p_prec['eval_mode'] == 'exact':
            self.eval_Fw = lambda w: self.eval_Fw_reg(w) + self.eval_Fw_sing(w)
        else:
            message = "evaluation mode '%s' not recognized" % self.p_prec['eval_mode']
            raise FreqDomainException(message)

    def default_params(self):
        p_prec = {'wmin' : 1e-2,
                  'wmax' : 1e2,
                  'window_transition' : 0.2,
                  'smallest_tau_max'  : 10,
                  'N_above_discard' : 7,
                  'N_below_discard' : 4,
                  'N_keep'          : 5,
                  'interp_kind' : 'cubic',
                  'eval_mode'   : 'interpolate',
                  'FFT method'  : 'multigrid',
                  'parallel' : True}
        return p_prec

    def get_FreqTable(self):
        """Get internal information about the multigrid method.

        Returns
        -------
        freq_table : dict
            Each entry in this dictionary contains a list, whose length corresponds to the
            number of FFTs that have been computed in the multigrid method. The available information is:

            * ``n_fft`` (*list of int*) -- Size of each FFT performed.
            * ``n_fft_keep`` (*list of int*) -- Number of points actually used in the final result.
            * ``df`` (*list of float*) -- Frequency resolution.
            * ``dtau`` (*list of float*) -- Time resolution.
            * ``wmin_real``, ``wmax_real`` (*list of float*) -- Frequency range of the FFT.
            * ``wmin_batch``, ``wmax_batch`` (*list of float*) -- Range of frequencies kept for the final result.
            * ``tau_max`` (*list of float*) -- Maximum :math:`\\tau` where :math:`I_\\text{reg}(\\tau)`\
                has been evaluated.
        """
        args = (self.t_grid,
                self.It_reg_grid,
                self.p_prec,
                self.reg_sch)

        return wrapper.pyFreqTable_to_dic(*args)

    def compute(self):
        """Computation of :math:`F(w)`.

        Returns
        -------
        w_grid, Fw_grid, Fw_reg_grid : array
            Grids with the frequencies, :math:`F(w)` and its regular part.
        """
        method = self.p_prec['FFT method']
        self.reg_sch['stage'] = self.p_prec['reg_stage']

        args = (self.t_grid,
                self.It_reg_grid,
                self.p_prec,
                self.reg_sch)

        if method == 'standard':
            w_grid, Fw_grid, Fw_reg_grid = wrapper.pyCompute_Fw_std(*args)
        elif (method == 'multigrid') or (method == 'multigrid stack'):
            w_grid, Fw_grid, Fw_reg_grid = wrapper.pyCompute_Fw(*args)
        else:
            message = "FFT method '%s' not recognized" % method
            raise FreqDomainException(message)

        return w_grid, Fw_grid, Fw_reg_grid


class Fw_SemiAnalyticSIS_C(FwGeneral_C):
    """Semi-analytic computation of the amplification factor for the SIS.

    To arrive at this result, the radial integral is first solved analitically
    and then the angular integral is performed numerically.

    Additional information: :ref:`theory <SemiAnalyticSIS_theory>`, :ref:`default parameters <cFw_SemiAnalyticSIS_default>`.

    Parameters
    ----------
    y : float
        Impact parameter.
    psi0 : float
        Normalization of the SIS lensing potential, :math:`\\psi(x)=\\psi_0 x`.
    p_prec : dict, optional
        Precision parameters. New keys:

        * ``parallel`` (*bool*) -- Perform the evaluation in parallel.
        * ``method`` (*str*) -- Method used for the integration. Options:

            * ``'direct'`` : It can be faster in some regions of the parameter space, but it cannot\
                be used for high frequencies.
            * ``'osc'`` : Special method adapted for oscillatory integrals.
    """
    def __init__(self, y, p_prec={}, psi0=1):
        It = time_domain_c.It_AnalyticSIS_C(y, {'eval_mode':'exact', 'parallel':False}, psi0)
        super().__init__(It, p_prec)

        self.name = '(pseudo) analytic SIS (C code)'
        self.psi0 = psi0
        self.y = self.It.y

        self.eval_Fw = self.compute

        # not really necessary but useful for testing
        self.eval_It_reg = lambda tau: self.It.eval_It(tau) - self.eval_It_sing(tau)
        self.eval_Fw_reg = lambda w: self.eval_Fw(w) - self.eval_Fw_sing(w)

    def __str__(self):
        class_name = type(self).__name__
        class_call = "Fw = freq_domain." + class_name + "(y, psi0, p_prec)"

        y_message = "y = %g\n" % self.y
        psi0_message = "psi0 = %g\n" % self.psi0
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        return y_message + psi0_message + prec_message + class_call

    def default_params(self):
        p_prec = {'method' : 'osc',
                  'parallel' : True}
        return p_prec

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
        Fw = wrapper.pyFw_SIS(w,
                              y = self.y,
                              psi0 = self.psi0,
                              method = self.p_prec['method'],
                              parallel = self.p_prec['parallel'])
        return Fw


class Fw_AnalyticPointLens_C(FwGeneral_C):
    """Analytic computation of the amplification factor for the point lens.

    Additional information: :ref:`theory <AnalyticPointLens_theory>`, :ref:`default parameters <cFw_AnalyticPointLens_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.freq_domain_c.FwGeneral_C`
    for the internal information of the parent class)

    Parameters
    ----------
    y : float
        Impact parameter.
    p_prec : dict, optional
        Precision parameters. New keys:

        * ``parallel`` (*bool*) -- Perform the evaluation in parallel.
    """
    def __init__(self, y, p_prec={}):
        Psi = lenses.Psi_PointLens({'psi0' : 1})
        It = time_domain_c.ItGeneral_C(Psi, y)

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
        self.name = 'analytic point lens (C code)'

        self.eval_Fw = self.compute
        self.eval_Fw_reg = lambda w: self.eval_Fw(w) - self.eval_Fw_sing(w)

    def __str__(self):
        class_name = type(self).__name__
        class_call = "Fw = freq_domain_c." + class_name + "(y, p_prec)"

        y_message = "y = %g\n" % self.y
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        return y_message + prec_message + class_call

    def default_params(self):
        p_prec = {'parallel' : True}
        return p_prec

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
        return wrapper.pyFw_PointLens(self.y, w, self.p_prec['parallel'])


class Fw_DirectFT_C(FwGeneral_C):
    """Direct Fourier transform of :math:`I(\\tau)`.

    If we compute :math:`I(\\tau)` on a grid and approximate it as a linear interpolation function, its
    Fourier transform can be easily computed analitically. This method will be generally slower
    than :class:`~glow.freq_domain_c.Fw_FFT_C`, but its accuracy can be easily controlled by improving
    the sampling of :math:`I(\\tau)`.

    In this case no grid of frequencies is precomputed, so the evaluation is performed exactly at each
    point :math:`w`.

    Additional information: :ref:`theory <DirectFT_theory>`, :ref:`default parameters <cFw_DirectFT_default>`.

    (Only new parameters and attributes are documented. See :class:`~glow.freq_domain_c.FwGeneral_C`
    for the internal information of the parent class)

    p_prec : dict, optional
        Precision parameters. New keys:

        * ``parallel`` (*bool*) -- Perform the evaluation in parallel.
    """
    def __init__(self, It, p_prec={}):
        super().__init__(It, p_prec)

        self.name = 'direct integration of FT (C version)'

        self.It_reg_grid = self._update_RegScheme(self.p_prec['reg_stage'],
                                                  It.t_grid, It.It_grid,
                                                  self.p_prec['parallel'])

        t_grid = np.concatenate([[0.], It.t_grid])
        It_reg_grid = np.concatenate([[0.], self.It_reg_grid])

        self.eval_It_reg = lambda t: np.interp(t, t_grid, It_reg_grid, left=0)

        self.eval_Fw     = lambda w: self.compute(w)[0]
        self.eval_Fw_reg = lambda w: self.compute(w)[1]

    def default_params(self):
        p_prec = {'parallel' : True}
        return p_prec

    def compute(self, w):
        """Computation of :math:`F(w)`.

        Parameters
        ----------
        w : float or array
            Frequency :math:`w`.

        Returns
        -------
        Fw, Fw_reg : float or array
            :math:`F(w)` and its regular part.
        """
        self.reg_sch['stage'] = self.p_prec['reg_stage']

        Fw, Fw_reg = wrapper.pyCompute_Fw_directFT(w, \
                                                   self.It.t_grid, \
                                                   self.It_reg_grid, \
                                                   self.reg_sch, \
                                                   stage=None, \
                                                   parallel=self.p_prec['parallel'])
        return Fw, Fw_reg
