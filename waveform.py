#
# GLoW - waveform.py
#
# Copyright (C) 2023, Stefano Savastano
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

import os

import pycbc as pycbc
from pycbc.types import TimeSeries, FrequencySeries

from astropy import units as u
from astropy import constants as c

GMsun8pi=(8*c.G*c.M_sun*np.pi/c.c**3).decompose().value

class WaveformWarning(UserWarning):
    pass

##==============================================================================
## Detectors sensitivities


def f_bounds_detector(detector):

    if detector == 'LIGO':
            fmin= 10
            fmax= 5e3
    elif detector == 'LISA':
            fmin= 1e-5
            fmax= 0.5
    elif detector == 'ET':
            fmin= 1
            fmax= 1e4
    else:
        raise ValueError("detector '%s' not recognized" % detector)

    return fmin, fmax



def get_psd_from_file(detector, sky_pol_avg=True, inc_avg=True,
                      external_path=None, psd_pow=1):

    """
    Load the Power Spectral Density (PSD) for a specific gravitational wave detector, either from an internal
    sensitivity file or an external file provided by the user.

    Parameters
    ----------
    detector : str
        The name of the detector for which the PSD is to be generated.
        Supported detectors include 'LIGO', 'LISA', and 'ET' (Einstein Telescope).

    sky_pol_avg : bool, optional
        If True, apply averaging over sky location and polarization angle.
        Defaults to True.

    inc_avg : bool, optional
        If True, apply averaging over the inclination angle.
        Defaults to True.

    external_path : str, optional
        Path to an external file containing the PSD data. If provided, the PSD is loaded from this file.
        Defaults to None.

    pow_ext : float, optional
        The power exponent applied to the PSD data in the external file.
        This is useful if the data in the file is saved as Sn(f)^(psd_pow). Defaults to 1 (no modification).

    Returns
    -------
    psd_fun : function
        A function that returns the PSD as a function of frequency, based on the loaded data and the applied averaging factors.

    Notes
    -----
    - If `external_path` is not provided, the function loads the PSD from predefined internal sensitivity files.
    - Averaging factors for different detectors are applied according to the choices of `sky_pol_avg` and `inc_avg`.
    """

    from scipy.interpolate import interp1d

    # If an external path is provided, print a message indicating that the external PSD is being loaded.
    if external_path is not None:
        print('Loading external PSD.')
    else:
        # If no external path is provided, define the path to the internal sensitivity files.
        sens_folder = os.path.dirname(os.path.realpath(__file__)) + "/sensitivities/"

    # Calculate the inclination angle averaging factor.
    if inc_avg:
        inc_avg_factor = 4/5
    else:
        inc_avg_factor = 1

    # Determine the sky and polarization averaging factor based on the detector.
    if sky_pol_avg:
        if detector == 'LIGO':
            sky_pol_avg_factor = 1/5
        elif detector == 'LISA':
            # LISA's PSD already includes sky and polarization averaging.
            sky_pol_avg_factor = 1
        elif detector == 'ET':
            sky_pol_avg_factor = 3/10
        else:
            sky_pol_avg_factor = 1
    else:
        sky_pol_avg_factor = 1

    # The overall averaging factor is the product of the inclination and sky/polarization factors.
    avg_factor = inc_avg_factor * sky_pol_avg_factor

    # If an external PSD file path is provided, attempt to load the PSD from the file.
    if external_path is not None:
        try:
            psd_ext_arr = np.loadtxt(external_path).T
            psd_fun = interp1d(psd_ext_arr[0], psd_ext_arr[1]**(1/psd_pow), kind='linear', bounds_error=False, fill_value=np.inf)
        except:
            # Raise an error if the PSD file cannot be found or read.
            raise ValueError("PSD file not found")
    else:
        # If no external path is provided, load the PSD from internal files based on the detector.
        if detector == 'LIGO':
            psd_ligo_arr = np.loadtxt(sens_folder + "aplus.txt").T
            psd_fun = interp1d(psd_ligo_arr[0], psd_ligo_arr[1]**2 / avg_factor, kind='linear', bounds_error=False, fill_value=np.inf)
        elif detector == 'LISA':
            psd_lisa_arr = np.loadtxt(sens_folder + "lisa.txt").T
            psd_lisa_interp = interp1d(np.log10(psd_lisa_arr[0]), np.log10(psd_lisa_arr[1]**2 / avg_factor), kind='linear', bounds_error=False, fill_value=np.inf)
            psd_fun = lambda x: 10**psd_lisa_interp(np.log10(x))
        elif detector == 'ET':
            psd_et_arr = np.loadtxt(sens_folder + "et.txt").T
            psd_fun = interp1d(psd_et_arr[0], psd_et_arr[1]**2 / avg_factor, kind='linear', bounds_error=False, fill_value=np.inf)
        else:
            raise ValueError("PSD file corresponding to detector not found")

    psd_fun = np.vectorize(psd_fun)

    return psd_fun



def get_psd_from_pycbc(detector, sky_pol_avg=True, inc_avg=True):
    """
    Generate a Power Spectral Density (PSD) function for a specified gravitational wave detector.

    Parameters
    ----------
    detector : str
        The name of the detector for which the PSD is to be generated.
        Supported detectors are 'LIGO', 'LISA', and 'ET' (Einstein Telescope).

    sky_pol_avg : bool, optional
        If True, the PSD is averaged over sky location and polarization angle.
        Defaults to True.

    inc_avg : bool, optional
        If True, the PSD is averaged over the inclination angle.
        Defaults to True.

    Returns
    -------
    psd_fun : function
        A function that provides the PSD as a function of frequency, based on
        the input detector and averaging choices.

    Notes
    -----
    - Compared to the get_psd_from_file this function deals with sqrt(Sn(f)),
      so averaging factors are introduced accordingly.
    - Averaging factors are derived from arXiv:1803.01944, which accounts for
      the effects of averaging over sky location, polarization, and inclination angles.
    - Different detectors have specific averaging factors:
      - LIGO: Standard sky and polarization averaging.
      - LISA: PSD is already averaged over sky and polarization angles.
      - ET: Triangular shape detector, uses specific averaging factors.
    """
    import pycbc.psd
    from scipy.interpolate import interp1d
    # Calculate the inclination angle averaging factor.
    if inc_avg:
        inc_avg_factor = 2 / np.sqrt(5)
    else:
        inc_avg_factor = 1

    # Determine the sky and polarization averaging factor based on the detector.
    if sky_pol_avg:
        if detector == 'LIGO':
            sky_pol_avg_factor = np.sqrt(1/5)
        elif detector == 'LISA':
            # LISA's waveform is pre-averaged over sky and polarization.
            sky_pol_avg_factor = 1
        elif detector == 'ET':
            sky_pol_avg_factor = np.sqrt(3/10)
    else:
        sky_pol_avg_factor = 1

    # The overall averaging factor is the product of the inclination and sky/polarization factors.
    avg_factor = inc_avg_factor * sky_pol_avg_factor

    # Define the frequency bounds and resolution for the PSD based on the detector.
    flow, fmax = f_bounds_detector(detector)
    delta_f = flow / 10                       # Set the frequency resolution (delta_f) as one-tenth of the lower frequency bound.
    flen = int((fmax - flow) // delta_f)

    # Generate the PSD based on the specified detector.
    if detector == 'LIGO':
        psd_LIGO = pycbc.psd.analytical.AdVDesignSensitivityP1200087(flen, delta_f, flow)
        psd_fun = interp1d(np.arange(0, psd_LIGO.__len__() * psd_LIGO.delta_f, psd_LIGO.delta_f),
                           1 / avg_factor * psd_LIGO.numpy(), kind='linear', fill_value=np.inf)
    elif detector == 'LISA':
        psd_LISA = pycbc.psd.analytical_space.sensitivity_curve_lisa_semi_analytical(flen, delta_f, flow)
        psd_fun = interp1d(np.arange(0, psd_LISA.__len__() * psd_LISA.delta_f, psd_LISA.delta_f),
                           1 / avg_factor * psd_LISA.numpy(), kind='linear', fill_value=np.inf)
    elif detector == 'ET':
        psd_ET = pycbc.psd.analytical.EinsteinTelescopeP1600143(flen, delta_f, flow)
        psd_fun = interp1d(np.arange(0, (psd_ET.__len__() - 1) * psd_ET.delta_f, psd_ET.delta_f),
                           1 / avg_factor * psd_ET.numpy()[0:-1], kind='linear', fill_value=np.inf)
    else:
        raise ValueError("detector '%s' not recognized" % detector)

    return psd_fun


##==============================================================================
## Waveform quantites

def f_isco(M_tot):
    '''
    Calculate the gravitational wave frequency corresponding to the innermost stable circular orbit (ISCO) for a binary system.
    This function computes the ISCO frequency in Hertz for a given total mass of the binary black hole system.

    Args:
    M_tot (float): The total mass of the binary system in solar masses (M_sun).

    Note:
    The ISCO frequency is derived from general relativity, specifically from the condition of stable circular orbits
    around a Schwarzschild black hole. The ISCO frequency is related to the orbital frequency at the innermost
    stable circular orbit. The formula used here is derived from the Schwarzschild metric, which assumes a non-rotating
    black hole and does not account for spin effects.
    '''
    return (1 / (6 * np.sqrt(6) * np.pi * c.G / c.c**3) / u.Msun).decompose().value / M_tot

def f0_obs(M, t_obs, eta=0.25):
    '''
    This function calculates the initial gravitational wave frequency at the start of an observation
    period for a binary system, based on the total mass and observation time.

    Args:
        M (float): The total mass of the binary system in solar masses.
        t_obs (float): The observation time (duration over which the system is observed).
        eta (float, optional): The symmetric mass ratio of the binary system (default is 0.25).

    Returns:
        float: The initial gravitational wave frequency in Hz.

    Note:
        The calculation is based on solving for the initial frequency from
        Eq. 16.26 in Maggiore's book (Gravitational Waves Vol. 1).
        This does not include any redshift factor.
    '''

    # Calculate the initial frequency using the equation derived from Maggiore's book
    return (150.98 / (eta * t_obs * M ** (5/3.)) ** (3/8.))



##==============================================================================
## Waveform tools

def get_fd_waveform_unlensed(**kwargs):
    """
    Generate an unlensed frequency-domain waveform.

    Args:
        **kwargs: Keyword arguments (as in pycbc) passed to the waveform generation function.

    Returns:
        tuple: Two frequency-domain polarizations (hp, hx) representing the plus and cross polarizations.
    """
    from pycbc.waveform import get_fd_waveform
    hp, hx = get_fd_waveform(**kwargs)
    return hp, hx


def get_p_prec_f_opt(h_unlensed, units):
    """
    Calculate the optimal frequency range for the given unlensed waveform.

    Args:
        h_unlensed (WaveformFD): The unlensed frequency-domain waveform.
        units (glow.new_physical_units units objets): Contains the necessary units for the calculation.

    Returns:
        dict: Dictionary containing the minimum and maximum angular frequencies (wmin, wmax).
    """
    fmin = h_unlensed.low_frequency_cutoff
    fmax = h_unlensed.high_frequency_cutoff
    wmin = GMsun8pi * units.Mlz.to('Msun').value * fmin
    wmax = GMsun8pi * units.Mlz.to('Msun').value * fmax
    p_opt_f = {'wmin': wmin, 'wmax': wmax}
    return p_opt_f


def get_lensed_fd_from_Fw(h_unlensed, Fw, units, w_opt=False):
    """
    Generate a lensed frequency-domain waveform from a given amplification factor.

    Args:
        h_unlensed (WaveformFD): The unlensed frequency-domain waveform.
        Fw (glow.freq_domain_c Fw object): The frequency-domain amplification factor.
        units (glow.new_physical_units units objets): Contains the necessary units for the calculation.
        w_opt (bool, optional): If True, the Fw will be re-initialized to match the optimal frequency range.

    Returns:
        WaveformFD: The lensed frequency-domain waveform.
    """

    if h_unlensed.islensed:
        message = 'The input waveform is already lensed'
        warnings.warn(message, WaveformWarning)

    l_id, h_id = h_unlensed.low_frequency_cutoff_id, h_unlensed.high_frequency_cutoff_id
    fs = h_unlensed.sample_frequencies[h_unlensed.nonzeros]

    # Get the optimal frequency range
    p_prec_f_opt = get_p_prec_f_opt(h_unlensed, units)
    wmin_opt, wmax_opt = p_prec_f_opt['wmin'], p_prec_f_opt['wmax']

    try:
        wmin, wmax = Fw.w_grid[[0, -1]]  # Get the min and max frequencies from Fw
        # Check if Fw's frequency range matches the optimal range
        if wmin < wmin_opt and wmax > wmax_opt:
            pass  # Fw's range includes the optimal
        else:
            message = ('Input Fw is computed for wmin, wmax = ({:.1f}, {:.1f}), while the optimal range is '
                    '({:.1f}, {:.1f})').format(wmin, wmax, wmin_opt, wmax_opt)
            warnings.warn(message, WaveformWarning)
            if w_opt:
                message ="Recomputing Fw..."
                warnings.warn(message, WaveformWarning)
                p_prec_tmp = Fw.p_prec.copy()
                p_prec_tmp.update({'wmin': wmin_opt, 'wmax': wmax_opt})
                # ! Using __init__ is deprecated. Change this by introducing a reset method in Fw instead.
                Fw.__init__(Fw.It, p_prec=p_prec_tmp)  # Re-initialize Fw with the new parameters
    except:
        pass

    Fws = Fw(units.f_to_w(fs, un=u.Hz))  # Convert frequencies to the angular domain using Fw

    # Adjust the phase of Fws to match the Fourier transform convention
    Fws = np.abs(Fws) * np.exp(-1j * np.angle(Fws))

    h_lensed = []
    for i, pol in enumerate(h_unlensed.polarizations):
        pol_lensed = FrequencySeries(pol, copy=True)
        pol_lensed[l_id:h_id + 1] = Fws * pol_lensed[l_id:h_id + 1]  # Apply the lensing effect
        h_lensed.append(pol_lensed)

    params_source_lensed = h_unlensed.params_source.copy()
    h_lensed = WaveformFD(params_source_lensed, polarizations=h_lensed)

    h_lensed.islensed = True
    h_lensed.unlensed = h_unlensed
    h_lensed.units = units
    h_lensed.Fw = Fw

    if h_unlensed.strain:
        h_lensed.project_sky()

    if len(h_unlensed.psd_grid) > 0:
        h_lensed.load_psd(h_unlensed.psd_grid)  # Load the PSD grid if available

    return h_lensed  # Return the lensed waveform


def get_lensed_fd_from_Fw_phase(h_unlensed, Fw, units, w_opt=False):
    """
    Generate a lensed frequency-domain waveform including focusing on the phase shift only.

    Args:
        h_unlensed (WaveformFD): The unlensed frequency-domain waveform.
        Fw (glow.freq_domain_c Fw object): The frequency-domain amplification factor.
        units (glow.new_physical_units units objets): Contains the necessary units for the calculation.
        w_opt (bool, optional): If True, the Fw will be re-initialized to match the optimal frequency range.

    Returns:
        WaveformFD: The lensed frequency-domain waveform with phase shift applied.
    """
    if h_unlensed.islensed:
        message = 'The input waveform is already lensed'
        warnings.warn(message, WaveformWarning)

    l_id, h_id = h_unlensed.low_frequency_cutoff_id, h_unlensed.high_frequency_cutoff_id
    fs = h_unlensed.sample_frequencies[h_unlensed.nonzeros]

    # Get the optimal frequency range
    p_prec_f_opt = get_p_prec_f_opt(h_unlensed, units)
    wmin_opt, wmax_opt = p_prec_f_opt['wmin'], p_prec_f_opt['wmax']

    try:
        wmin, wmax = Fw.w_grid[[0, -1]]  # Get the min and max frequencies from Fw
        # Check if Fw's frequency range matches the optimal range
        if wmin < wmin_opt and wmax > wmax_opt:
            pass  # Fw's range includes the optimal
        else:
            message = ('Input Fw is computed for wmin, wmax = ({:.1f}, {:.1f}), while the optimal range is '
                    '({:.1f}, {:.1f})').format(wmin, wmax, wmin_opt, wmax_opt)
            warnings.warn(message, WaveformWarning)
            if w_opt:
                message ="Recomputing Fw..."
                warnings.warn(message, WaveformWarning)
                p_prec_tmp = Fw.p_prec.copy()
                p_prec_tmp.update({'wmin': wmin_opt, 'wmax': wmax_opt})
                # ! Using __init__ is deprecated. Change this by introducing a reset method in Fw instead.
                Fw.__init__(Fw.It, p_prec=p_prec_tmp)  # Re-initialize Fw with the new parameters
    except:
        pass
    Fws = Fw(units.f_to_w(fs, un=u.Hz))  # Convert frequencies to the angular domain using Fw
    Fws = np.exp(1j * np.angle(Fws))  # Apply phase shift

    # Adjust the phase of Fws to match the Fourier transform convention
    Fws = np.abs(Fws) * np.exp(-1j * np.angle(Fws))

    h_lensed = []
    for i, pol in enumerate(h_unlensed.polarizations):
        pol_lensed = FrequencySeries(pol, copy=True)
        pol_lensed[l_id:h_id + 1] = Fws * pol_lensed[l_id:h_id + 1]  # Apply the lensing effect
        h_lensed.append(pol_lensed)

    params_source_lensed = h_unlensed.params_source.copy()
    h_lensed = WaveformFD(params_source_lensed, polarizations=h_lensed)

    h_lensed.islensed = True
    h_lensed.unlensed = h_unlensed
    h_lensed.units = units
    h_lensed.Fw = Fw

    if h_unlensed.strain:
        h_lensed.project_sky()


    if len(h_unlensed.psd_grid) > 0:
        h_lensed.load_psd(h_unlensed.psd_grid)

    return h_lensed


def get_lensed_fd_from_Fw_amp(h_unlensed, Fw, units, w_opt=False):
    """
    Generate a lensed frequency-domain waveform focusing on the amplitude modification only.

    Args:
        h_unlensed (WaveformFD): The unlensed frequency-domain waveform.
        Fw (glow.freq_domain_c Fw object): The frequency-domain amplification factor.
        units (glow.new_physical_units units objets): Contains the necessary units for the calculation.
        w_opt (bool, optional): If True, the Fw will be re-initialized to match the optimal frequency range.

    Returns:
        WaveformFD: The lensed frequency-domain waveform with amplitude modification.
    """
    if h_unlensed.islensed:
        message = 'The input waveform is already lensed'
        warnings.warn(message, WaveformWarning)

    l_id, h_id = h_unlensed.low_frequency_cutoff_id, h_unlensed.high_frequency_cutoff_id
    fs = h_unlensed.sample_frequencies[h_unlensed.nonzeros]

    # Get the optimal frequency range
    p_prec_f_opt = get_p_prec_f_opt(h_unlensed, units)
    wmin_opt, wmax_opt = p_prec_f_opt['wmin'], p_prec_f_opt['wmax']

    try:
        wmin, wmax = Fw.w_grid[[0, -1]]  # Get the min and max frequencies from Fw
        # Check if Fw's frequency range matches the optimal range
        if wmin < wmin_opt and wmax > wmax_opt:
            pass  # Fw's range includes the optimal
        else:
            message = ('Input Fw is computed for wmin, wmax = ({:.1f}, {:.1f}), while the optimal range is '
                    '({:.1f}, {:.1f})').format(wmin, wmax, wmin_opt, wmax_opt)
            warnings.warn(message, WaveformWarning)
            if w_opt:
                message ="Recomputing Fw..."
                warnings.warn(message, WaveformWarning)
                p_prec_tmp = Fw.p_prec.copy()
                p_prec_tmp.update({'wmin': wmin_opt, 'wmax': wmax_opt})
                # ! Using __init__ is deprecated. Change this by introducing a reset method in Fw instead.
                Fw.__init__(Fw.It, p_prec=p_prec_tmp)  # Re-initialize Fw with the new parameters
    except:
        pass

    Fws = Fw(units.f_to_w(fs, un=u.Hz))  # Convert frequencies to the angular domain using Fw
    Fws = np.abs(Fws)  # Apply amplitude modification

    h_lensed = []
    for i, pol in enumerate(h_unlensed.polarizations):
        pol_lensed = FrequencySeries(pol, copy=True)
        pol_lensed[l_id:h_id + 1] = Fws * pol_lensed[l_id:h_id + 1]
        h_lensed.append(pol_lensed)

    params_source_lensed = h_unlensed.params_source.copy()
    h_lensed = WaveformFD(params_source_lensed, polarizations=h_lensed)

    h_lensed.islensed = True
    h_lensed.unlensed = h_unlensed
    h_lensed.units = units
    h_lensed.Fw = Fw

    if h_unlensed.strain:
        h_lensed.project_sky()

    if len(h_unlensed.psd_grid) > 0:
        h_lensed.load_psd(h_unlensed.psd_grid)

    return h_lensed

##==============================================================================
## Waveform class

class PolarizationFD(FrequencySeries):
    """
    A subclass of FrequencySeries designed for handling frequency domain waveform polarization data.
    """

    def __init__(self, h):
        # Initialize the FrequencySeries superclass with the input data
        super().__init__(h.numpy(), delta_f=h.delta_f, epoch=h.epoch, dtype=h.dtype, copy=True)

        self.frequency_series = FrequencySeries(h, copy=True)

        # Identify where the frequency series has non-zero values
        self.nonzeros = np.nonzero(self.numpy())[0]
        self.low_frequency_cutoff_id = self.nonzeros[0]
        self.low_frequency_cutoff = self.low_frequency_cutoff_id * self.delta_f
        self.high_frequency_cutoff_id = self.nonzeros[-1]
        self.high_frequency_cutoff = self.high_frequency_cutoff_id * self.delta_f

    @property
    def sample_frequencies(self):
        '''
        Generate an array of sample frequencies.
        This method overrides the default implementation for efficiency.
        '''
        return np.arange(0, self.__len__() * self.delta_f, self.delta_f)

    def get_trimmed(self):
        '''
        Extract the portion of the frequency series that contains non-zero values.
        '''
        # If no non-zero values exist, return the full frequency series
        if len(self.nonzeros) == 0:
            return self.sample_frequencies, self.numpy()
        # If non-zero values exist, return only the relevant portion of the series
        elif len(self.nonzeros) > 1:
            freqs_trim = self.sample_frequencies[self.low_frequency_cutoff_id:self.high_frequency_cutoff_id]
            h_arr_trim = self.numpy()[self.low_frequency_cutoff_id:self.high_frequency_cutoff_id]
            return freqs_trim, h_arr_trim

    def to_timedomain(self, dt=None, alpha=0.015, cyclic_time_shift=0):
        '''
        Convert the frequency domain polarization to the time domain.
        Apply a Tukey window to reduce spectral leakage.
        '''
        # Import necessary libraries for windowing and Fourier transforms
        from scipy import signal
        from pycbc.fft import ifft
        from pycbc.types import real_same_precision_as

        l_id, h_id = self.low_frequency_cutoff_id, self.high_frequency_cutoff_id

        # Create a Tukey window to smooth the frequency data
        window = signal.windows.tukey(len(self.nonzeros) - 1, alpha=alpha, sym=False)

        h_w = FrequencySeries(np.zeros(len(self)), dtype=self.dtype, delta_f=self.delta_f, epoch=self.epoch)

        # Apply the window to the relevant portion of the frequency series
        h_w[l_id: h_id] = window * self.frequency_series[l_id: h_id]

        # Convert the windowed frequency series to a time series and apply a cyclic time shift
        return PolarizationTD(h_w.to_timeseries(dt).cyclic_time_shift(cyclic_time_shift))

    def __len__(self):
        # Return the length of the frequency series
        return super().__len__()


class WaveformFD():
    """
    Handles frequency domain waveforms for gravitational wave signals.

    Attributes:
        params_source (dict): Dictionary containing the source parameters
                              (e.g., mass, spin, distance), as required by
                               the waveform generator in pycbc.
        polarizations (list): Optional list of two FrequencySeries objects
                              representing the plus (hp) and cross (hx)
                              polarizations. If not provided, they will be
                              generated internally using pycbc.
        p (PolarizationFD): Plus polarization.
        x (PolarizationFD): Cross polarization, if available.
        strain (PolarizationFD): Waveform strain.
        sample_frequencies (numpy.ndarray): Array of frequency values
                                            corresponding to the waveform.
        delta_f (float): Frequency resolution of the waveform.
        epoch (float): Epoch time reference of the waveform.

    Further:
        nonzeros (numpy.ndarray): Indices of non-zero frequency components.
        low_frequency_cutoff_id (int): Index of the lowest frequency with
                                       non-zero value.
        low_frequency_cutoff (float): Lowest frequency with a non-zero value.
        high_frequency_cutoff_id (int): Index of the highest frequency with
                                        non-zero value.
        high_frequency_cutoff (float): Highest frequency with a non-zero value.
        islensed (bool): Flag indicating whether the waveform is lensed.
    """

    def __init__(self, params_source, polarizations=[]):

        self.params_source = params_source.copy()
        self.islensed = False

        if not polarizations:
            polarizations = get_fd_waveform_unlensed(**self.params_source)


        self.polarizations = [PolarizationFD(h) for h in polarizations]

        self.p = self.polarizations[0]
        self.x = self.polarizations[1]

        self.sample_frequencies = self.p.sample_frequencies
        self.delta_f = self.p.delta_f
        self.epoch = self.p.epoch

        # Identify non-zero frequency range
        self.nonzeros = self.p.nonzeros
        self.low_frequency_cutoff_id = self.p.low_frequency_cutoff_id
        self.low_frequency_cutoff = self.p.low_frequency_cutoff
        self.high_frequency_cutoff_id = self.p.high_frequency_cutoff_id
        self.high_frequency_cutoff = self.p.high_frequency_cutoff

        self.strain = None
        self.psd = None
        self.psd_grid = []
        self.snr = None

        self.project_sky()


    def project_sky(self):
        """
        Initializes the additional attributes that are assigned by methods.
        """
        source_keys = self.params_source.keys()


        # If sky localization parameters are present, project strain
        sky_keys = ['declination', 'right_ascension', 'polarization']
        if any(k in source_keys for k in sky_keys):
            self.get_projected_strain({k: self.params_source[k] for k in sky_keys})


    def load_psd(self, psd):
        """
        Load a Power Spectral Density (PSD) for the waveform.

        Args:
            psd (function or array-like): PSD function or grid (evaluated
                                        at waveform points) to be applied.
        """
        if callable(psd):
            # If psd is a function, calculate the PSD grid using the function
            freqs = self.p.get_trimmed()[0]
            self.psd = psd
            psd_grid = psd(freqs)
            self.psd_grid = pycbc.types.frequencyseries.FrequencySeries(
                np.zeros(self.p.__len__()), delta_f=self.delta_f)
            self.psd_grid[:self.p.nonzeros[0]] += np.inf
            self.psd_grid[self.p.nonzeros[0]:self.p.nonzeros[-1]] += psd_grid
            self.psd_grid[self.p.nonzeros[-1]:] += np.inf
        else:
            # If psd is not callable, treat it as a precomputed grid
            self.psd = None

            # Consistency check: Ensure the grid length matches the sample frequencies length
            if len(psd) != len(self.p.sample_frequencies):
                warnings.warn(
                    "The provided PSD grid does not match the length of the sample frequencies. "
                    "Please ensure the grid is evaluated at the correct sample frequencies.",
                    WaveformWarning
                )
            self.psd_grid = psd.copy()

        # Calculate SNR based on the loaded PSD
        self.snr = self.get_snr()


    def get_snr(self, only_plus=True):
        """
        Calculate the Signal-to-Noise Ratio (SNR) of the waveform.

        Args:
            only_plus (bool, optional): If True, calculate SNR using only the
                                        plus polarization. Defaults to True.
        """
        from pycbc.filter.matchedfilter import sigmasq

        if len(self.psd_grid) == 0:
            raise ValueError('Load a psd function first')

        if only_plus:
            snr = np.sqrt(sigmasq(self.p.frequency_series,
                                  psd=self.psd_grid,
                                  low_frequency_cutoff=self.p.low_frequency_cutoff,
                                  high_frequency_cutoff=self.p.high_frequency_cutoff))
        else:
            if self.strain:
                snr = np.sqrt(sigmasq(self.strain.frequency_series,
                                      psd=self.psd_grid,
                                      low_frequency_cutoff=self.strain.low_frequency_cutoff,
                                      high_frequency_cutoff=self.strain.high_frequency_cutoff))
            else:
                raise ValueError('Produce strain first')
        return snr


    def get_projected_strain(self, sky_dict):
        """
        Project the waveform strain onto a detector using the sky localization parameters.

        Args:
            sky_dict (dict): Dictionary with sky localization parameters:
                             'declination', 'right_ascension', 'polarization'.

        """
        # Calculate antenna patterns
        sky_keys = sky_dict.keys()
        Fp, Fx = pycbc.detector.overhead_antenna_pattern(*[sky_dict[x] for x in sky_keys])

        # Combine polarizations to get the detector response
        self.strain = PolarizationFD(self.p * Fp + self.x * Fx)
        self.params_source.update(sky_dict)

        # Update unlensed strain if applicable
        if self.islensed:
            self.unlensed.strain = PolarizationFD(self.unlensed.p * Fp + self.unlensed.x * Fx)
            self.unlensed.params_source.update(sky_dict)

        return self.strain

    def to_timedomain(self, dt=None, alpha=0.015, cyclic_time_shift=0, unlensed=False):
        """
        Convert the frequency domain waveform to the time domain.

        Args:
            dt (float, optional): Time resolution for the time domain waveform.
            alpha (float, optional): Tukey window parameter. Defaults to 0.015.
            cyclic_time_shift (float, optional): Shift in time for cyclic boundary conditions.
                                                 Defaults to 0.
            unlensed (bool, optional): Whether to retain the unlensed version of the
                                            waveform if it is lensed. Defaults to False.

        Returns:
            WaveformTD: The time domain waveform.
        """
        # Convert each polarization to the time domain
        polarizations_td = [p.to_timedomain(dt, alpha, cyclic_time_shift) for p in self.polarizations]
        h_td = WaveformTD(self.params_source, polarizations=polarizations_td)
        h_td.islensed = self.islensed

        # Convert the strain to time domain if it exists
        if self.strain:
            h_td.strain = self.strain.to_timedomain(dt, alpha, cyclic_time_shift)

        # If lensed and keeping the unlensed waveform, convert that too
        if h_td.islensed and unlensed:
            polarizations_unlensed_td = [p.to_timedomain(dt, alpha, cyclic_time_shift) for p in self.unlensed.polarizations]
            h_td.unlensed = WaveformTD(self.params_source, polarizations=polarizations_unlensed_td)
            h_td.units = self.units
            h_td.Fw = self.Fw
            if self.strain:
                h_td.strain = self.strain.to_timedomain(dt, alpha, cyclic_time_shift)
                h_td.unlensed.strain = self.unlensed.strain.to_timedomain(dt, alpha, cyclic_time_shift)

        return h_td



class PolarizationTD(TimeSeries):
    """
    A subclass of TimeSeries designed for handling time domain waveform polarization data.
    """

    def __init__(self, h):
        super().__init__(h)


class WaveformTD():
    """
    Handles time domain waveforms for gravitational wave signals.

    Attributes:
        params_source (dict): Dictionary containing the source parameters
                              (e.g., mass, spin, distance), as required by
                               the waveform generator in pycbc.
        polarizations (list): Optional list of two FrequencySeries objects
                              representing the plus (hp) and cross (hx)
                              polarizations. If not provided, they will be
                              generated internally using pycbc.
        p (PolarizationTD): Plus polarization.
        x (PolarizationTD): Cross polarization, if available.
        strain (PolarizationTD): Waveform strain.
        sample_times (numpy.ndarray): Array of time values
                                            corresponding to the waveform.
    """
    def __init__(self, params_source, polarizations=[]):

        self.params_source=params_source.copy()
        self.__check_source_input()

        self.polarizations= [PolarizationTD(h) for h in polarizations]
        self.p=self.polarizations[0]
        self.x=self.polarizations[1]

        self.sample_times=self.p.sample_times
        self.strain=None

    def __check_source_input(self):

        pass
