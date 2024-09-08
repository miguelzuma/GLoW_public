import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import newton, bisect, root_scalar
from scipy.integrate import simps

from astropy import units as u
from astropy import constants as c


import glow.physical_units as pu

def initialize_cosmology(**kwargs):
    return pu.initialize_cosmology(**kwargs)
    
# Initialize cosmology passing the cosmological parameters in the function below.
cosmology={} # (dict) cosmological parameters, default is Planck18
cosmo=initialize_cosmology(**cosmology)

from . import waveform
from . import physical_units as pu

from tqdm import tqdm

import pandas as pd

from pycbc.filter.matchedfilter import match, optimized_match


##==============================================================================
## TOOLBOX

def compose_filename(detector, lens, mismatch_thr, Mtot, z_src, dir='ycr_bank/'):
    '''
    Create a filename for a given setup.

    Parameters:
    detector (str): Detector's name.
    lens (str): Lens type.
    mismatch_thr (str): Pre-factor in the snr-based mismatch threshold.
    Mtot (float): Total mass of the binary.
    z_src (float): Source redshift.
    dir (str): Data directory.

    Returns:
    -str: Filename.
    '''
    tags_read=['ycr', '_{:s}_'.format(detector), '{:s}_'.format(lens), "s{:.1f}_".format(mismatch_thr), "Mtot{:.0e}_".format(Mtot), "zsrc{:.1f}".format(z_src)]
    filename_read=dir+''.join(tags_read)
    return filename_read

##==============================================================================
## TOOLS

class ExcludeArgContext:
    def __init__(self, kwargs, keys):
        self.kwargs = kwargs
        self.keys = keys
        self.exclude_arg_values = {}

    def __enter__(self):
        # Check if the keys are present in kwargs
        for key in self.keys:
            self.exclude_arg_values[key] = self.kwargs.pop(key, None)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Put the keys back in kwargs after the block
        for key, value in self.exclude_arg_values.items():
            if value is not None:
                self.kwargs[key] = value


##==============================================================================
## MISMATCH

def mismatch(h1, h2, optimized=False, only_plus=True, **kwargs):
    '''
    Calculate the mismatch between two gravitational waveforms in the frequency domain.

    Parameters:
    - h1, h2 (WaveformFD): Gravitational waveforms.
    - optimized (bool): If True, use optimized_match for more accuracy (slower). Default is False.
    - only_plus (bool): If True, compute mismatch only for the plus polarization. Default is True.
    - **kwargs_plot: Additional keyword arguments for pycbc mismatch function.

    Returns:
    - float: Mismatch value.
    '''
    if np.all(h1.sample_frequencies!=h2.sample_frequencies):
        return print("The signals must be computed on the sample frequencies grid.")
    
    psd=h1.psd_grid

    if only_plus: # So far we only compute the mismatch of the plus
        h1_strain= h1.p
        h2_strain= h2.p
    else:
        h1_strain, h2_strain = h1.strain, h2.strain


    kwargs_internal= {'psd': psd,
                     'low_frequency_cutoff': h1.p.low_frequency_cutoff,
                     'high_frequency_cutoff': h1.p.high_frequency_cutoff}

    if not optimized:
        return 1-match(h1_strain, h2_strain, **kwargs_internal, **kwargs)[0]
    else:
        return 1-optimized_match(h1_strain, h2_strain, **kwargs_internal, **kwargs)[0]

mismatch_vec=np.vectorize(mismatch)

def mismatch_lensing(h_fd, Psi, y, MLz, amp_only=False, phase_only=False, kwargs_lensing={}, kwargs_mm={}):
    '''
    Computes the mismatch between the unlensed waveform and a weakly lensed model
    specified through its potential and parameters.

    Parameters:
    - h_fd (WaveformFD): Unlensed gravitational waveform in the frequency domain.
    - Psi (Potential): Lensing potential specifying the lensed model.
    - y (float): Lensing impact parameter.
    - MLz (float): Mass of the lens.
    - p_prec_t_update (dict): Optional dictionary of precision parameters for time domain.
    - p_prec_f_update (dict): Optional dictionary of precision parameters for frequency domain.
    - **kwargs: Additional keyword arguments for pycbc mismatch function.

    Returns:
    - float: Mismatch value between the lensed and unlensed waveforms.
    '''
    # Get the (weakly) lensed waveform in the frequency domain
    h_fd_lensed_s = waveform.get_lensed_fd_from_Psi (h_fd, Psi, y, MLz, 
                                                     **kwargs_lensing,
                                                     amp_only=amp_only, phase_only=phase_only
                                                     )
    
    # Calculate the mismatch between lensed and unlensed waveforms
    if not amp_only and not phase_only:
        m_s = mismatch(h_fd_lensed_s, h_fd, only_plus=True, **kwargs_mm) 
    else:
        m_s = [mismatch(h_fd_lensed, h_fd, only_plus=True, **kwargs_mm) for h_fd_lensed in h_fd_lensed_s]
        
    return m_s


def mismatch_ys(h_fd, Psi, ys, MLz, 
                adaptive_prec_t=False, adaptive_prec_t_pow_min=1.5, adaptive_prec_t_pow_max=1.2,
                amp_only=False, phase_only=False,  kwargs_lensing={}, kwargs_mm={}):
    '''
    Computes the mismatch between the unlensed waveform and a (weakly) lensed model
    specified through its potential and parameters on an array of impact parameters.

    Parameters:
    - h_fd (WaveformFD): Unlensed gravitational waveform in the frequency domain.
    - Psi (Potential): Lensing potential specifying the lensed model.
    - ys (array-like): 1D array of impact parameter values for lensing.
    - MLz (float): Mass of the lens.
    - p_prec_t_update (dict): Optional dictionary of precision parameters for time domain update.
    - p_prec_f_update (dict): Optional dictionary of precision parameters for frequency domain update.
    - amp_only (bool): If True, compute the amplitude-only mismatch. Default is False.
    - phase_only (bool): If True, compute the phase-only mismatch. Default is False.
    - **kwargs: Additional keyword arguments for mismatch function.

    Returns:
    - array-like: 2D array of mismatch values. If neither amp_only nor phase_only is True, the array has shape (1, len(ys)).
                  If either amp_only or phase_only is True, the array has shape (>1,len(ys)).
    '''
    mismatch_1d = []

    for y in ys:
        # Get the lensed waveform in the frequency domain for each impact parameter value
        kwargs_lensing_buff=kwargs_lensing.copy()

        if adaptive_prec_t:
            p_prec_t_buff=kwargs_lensing['p_prec_t'].copy()
            p_prec_t_buff.update({'tmin':np.amin([1e-5, p_prec_t_buff['tmin']*y**adaptive_prec_t_pow_min]),'tmax':p_prec_t_buff['tmax']*y**adaptive_prec_t_pow_max})
            kwargs_lensing_buff.update({'p_prec_t':p_prec_t_buff.copy()})
            p_prec_t_buff.clear()

        mm_s = mismatch_lensing(h_fd, Psi, y, MLz, 
                            amp_only=amp_only, phase_only=phase_only, 
                            kwargs_lensing=kwargs_lensing_buff, kwargs_mm=kwargs_mm)
        p_prec_t_buff.clear()

        mismatch_1d.append(mm_s)

    mismatch_1d_s = np.array(mismatch_1d).transpose()
    return mismatch_1d_s

def get_mismatch_grid(h_fd, h_fd_1, **kwargs):
    '''
    Compute the mismatch on 1 or 2 grids of waveforms.

    Parameters:
    - h_fd_lensed (WaveformFD): Weakly lensed gravitational waveform in the frequency domain.
    - h_fd (WaveformFD): Unlensed gravitational waveform in the frequency domain.
    - only_plus (bool): If True, compute mismatch only for the plus polarization. Default is True.
    - **kwargs: Additional keyword arguments for mismatch function.

    Returns:
    - array-like: Array of mismatch values.
    '''

    mismatch_grid = mismatch_vec(h_fd, h_fd_1, **kwargs)
    return mismatch_grid


def interpolate_mismatch_vec(grid_basis, mismatch_grid_vec, scale, scale_grid, method='linear'):
    '''
    Return an interpolating function of the mismatch evaluated on an n-dimensional grid.

    Parameters:
    - grid_basis (list): List of arrays defining the grid basis for lensing parameters.
    - mismatch_grid_vec (array-like): mismatch values on the grid defined by grid_basis.
    - scale (list): List of strings indicating the scaling of each parameter.
    - scale_grid (str): Scaling of the grid. 'log' or 'lin'.
    - method (str): Interpolation method. Default is 'linear'.

    Returns:
    - function: Interpolating function of the mismatch.
    '''

    from scipy.interpolate import RegularGridInterpolator

    lin_fun = lambda x: x

    scale_fun = [np.log10 if s == 'log' else lin_fun for s in scale]

    scale_grid_fun = np.log10 if scale_grid == 'log' else lin_fun
    scale_grid_fun_inv = lambda x: 10**x if scale_grid == 'log' else x

    # Cut-off for numerical errors
    mismatch_grid_vec[mismatch_grid_vec <= 1e-15] = 1e-15

    # Map the mismatch into the logarithmic interpolation scale
    log_mismatch_grid_vec = scale_grid_fun(mismatch_grid_vec)

    # Map each variable into the chosen interpolation scale
    mapping = lambda x: [f(var) for f, var in zip(scale_fun, x)]

    # Regular grid interpolation
    mismatch_fun_log = RegularGridInterpolator(mapping(grid_basis),
                                               np.transpose(log_mismatch_grid_vec),
                                               bounds_error=True,
                                               fill_value=None,
                                               method=method)

    # Inverse logarithmic map applied
    mismatch_fun = lambda x: scale_grid_fun_inv(mismatch_fun_log(mapping(x))[0])

    return mismatch_fun


def get_mismatch_fun(grid_basis, h_fd_lensed_grid, h_fd, scale=[], scale_grid='log', method='linear', **kwargs):
    '''
    Compute an interpolating function of the mismatch between a weakly lensed waveform and an unlensed waveform.

    Parameters:
    - grid_basis (list): List of arrays defining the grid basis for lensing parameters.
    - h_fd_lensed_grid (array-like): Grid of weakly lensed waveforms.
    - h_fd (WaveformFD): Unlensed gravitational waveform in the frequency domain.
    - only_plus (bool): If True, compute mismatch only for the plus polarization. Default is True.
    - scale (list): List of strings indicating the scaling of each parameter.
    - scale_grid (str): Scaling of the grid. 'log' or 'lin'. Default is 'log'.
    - method (str): Interpolation method. Default is 'linear'.
    - **kwargs: Additional keyword arguments for mismatch function.

    Returns:
    - function: Interpolating function of the mismatch.
    - array-like: Mismatch values on the grid.
    '''
    # If scale is not provided, default to 'log' scaling for all parameters
    if not scale:
        scale = ['log'] * len(grid_basis)

    # Compute mismatch grid vector
    mismatch_grid = get_mismatch_grid(h_fd_lensed_grid, h_fd, **kwargs)
    
    # Interpolate the mismatch vector to obtain an interpolating function
    fun = interpolate_mismatch_vec(grid_basis, mismatch_grid, scale, scale_grid, method=method)

    return fun, mismatch_grid


def get_mismatch_2d_fun_from_file(filename):
    '''
    Load a precomputed mismatch grid from a file and return the corresponding interpolating function.

    Parameters:
    - filename (str): Name of the file containing the precomputed mismatch grid.

    Returns:
    - function: Interpolating function of the mismatch.
    - array-like:  Mismatch values on the grid.
    '''
    # Read the precomputed mismatch grid from a file
    read_mm = pd.read_pickle(filename)
    
    # Extract mismatch grid and grid basis
    mismatch_grid = read_mm['mm_grid'][0]
    grid_basis = [read_mm['MLz'][0], read_mm['y'][0]]
    
    # Return the interpolating function
    return interpolate_mismatch_vec(grid_basis, mismatch_grid, scale_grid='log', scale=read_mm['scale'][0]), mismatch_grid


##==============================================================================
## Critical curves


def find_ysl_axi(Psi, xmin=1e-6, xmax=1e2, N=1000):    
    x1s = np.geomspace(xmin, xmax, N)
    try:            
        d1_neg = Psi.dphi_Fermat_dx1(-x1s, 0, y=0)
        ysl=d1_neg.max()
        if ysl<0:
            ysl=0
    except:
        ysl=0
    return ysl


def find_y_crit_Mlz(mismatch_fun, Mlz, snr, y_limits=[2, 200], guess=10, **kwargs):
    '''
    Find the critical impact parameter (y_crit) for a given lens mass (Mlz) based on a mismatch function.

    Parameters:
    - mismatch_fun (callable): A 2D interpolation function representing the mismatch on the (Mlz, y) grid.
    - Mlz (float): Lens mass for which to find y_crit.
    - snr (float): Signal-to-noise ratio.
    - y_limits (list): Range of possible impact parameter values [ymin, ymax]. Default is [2, 200].
    - guess (float): Initial guess for y_crit. Default is 10.
    - **kwargs: Additional keyword arguments for the root finder.

    Returns:
    - float: The critical impact parameter (y_crit) for the given lens mass.
    '''

    ymin, ymax = y_limits

    # Define the function to find the root of
    mismatch_equation = lambda x: np.log10(mismatch_fun(Mlz, x)) + np.log10(snr**2)

    try:
        # Use Newton's method to find the root
        y_crit_root = newton(mismatch_equation, guess, **kwargs)
    except:
        try:
            # If Newton's method fails, use bisection method
            y_crit_root = bisect(mismatch_equation, ymin, ymax, **kwargs)
        except:
            # If both methods fail, raise an exception
            raise ValueError("Failed to find the critical impact parameter.")

    return np.amax(y_crit_root)

def get_y_crit_curve_from_fun(h_fd, Mvirs, z_lens, mismatch_fun, y_limits=[2,100], guess=10, to_MLz=pu.to_MLz_SIS):
    '''
    Find the critical impact parameter curve for an list of lens virial masses, given the mismatch function.

    h_fd (WafeformFD): Unlensed gravitational wave in the frequncy domain.
    Mvirs (list): Lens virial masses for which to find y_crit.
    z_lens: Lens redshift.
    mismatch_fun: Mismatch function of MLz and y.

    Returns:
    - list: The critical impact parameter curve at Mvirs.
    '''

    z_src=h_fd.params_source['redshift']
    snr=h_fd.snr

    deff = 1/(1+z_lens)*(cosmo.angular_diameter_distance_z1z2(z_lens,z_src)*cosmo.angular_diameter_distance(z_lens)/cosmo.angular_diameter_distance(z_src)).decompose()
    MLzs = [to_MLz(Mvir,deff,z_lens) for Mvir in Mvirs]

    y_crit=[find_y_crit_Mlz(mismatch_fun, MLz, snr, y_limits, guess) for MLz in MLzs]

    return y_crit

def get_y_crit_curve_opt(h_fd, Psis, MLzs, y_min, y_max, s=1, 
                         rtol=0.05, n_iter=3, robust=False,
                         return_mm=False, include_sl=False,
                         adaptive_prec_t=False, adaptive_prec_t_pow_min=1.5, adaptive_prec_t_pow_max=1.2,
                         kwargs_mm={},
                         kwargs_lensing={}
                         ):

    '''
    Fast and accurate method to get the critical curve.
    
    Parameters:
    - h_fd (WaveformFD): Gravitational waveform strain.
    - Psi (Potential): Lensing potential specifying the lensed model.
    - MLzs (array-like): Lens mass values.
    - y_min (float): Minimum impact parameter value.
    - y_max (float or array-like): Maximum impact parameter value(s).
    - s (float): Ratio between the SNR and the mismatch threshold. Default is 1.
    - optimized (bool): If True, use optimized mismatch calculation for more accuracy (slower). Default is False.
    - robust (bool): If True, no optimization of the bracket is performed. Default is False.
    - include_sl (bool): If True, include strong-lensing threshold points in the curve. Default is False.

    Returns:
    - list: Critical impact parameter values corresponding to each lens mass in MLzs.
    '''
    if isinstance(Psis, (list)):
        Psis=Psis 
    elif isinstance(MLzs, (float)):
        Psis=[Psis]
        MLzs=[MLzs]
    else:
        Psis=[Psis]*len(MLzs)

    
    def root_fun(h_fd, Psi, y, MLz):
        
        kwargs_lensing_buff=kwargs_lensing.copy()
        if adaptive_prec_t:
            p_prec_t_buff=kwargs_lensing['p_prec_t'].copy()
            p_prec_t_buff.update({'tmin':np.amin([1e-5, p_prec_t_buff['tmin']*y**adaptive_prec_t_pow_min]),'tmax':p_prec_t_buff['tmax']*y**adaptive_prec_t_pow_max})
            kwargs_lensing_buff.update({'p_prec_t':p_prec_t_buff.copy()})
            p_prec_t_buff.clear()
        
        h_fd_lensed=waveform.get_lensed_fd_from_Psi(h_fd, Psi, y, MLz, **kwargs_lensing_buff) 
        mis_val=mismatch(h_fd_lensed, h_fd, **kwargs_mm)

        kwargs_lensing_buff.clear()
        if mis_val<0:
            mis_val=1e-15
        return np.log(mis_val)+2*np.log(s*h_fd.snr)

    
    increasing=True
    
    if isinstance(y_max, (float, int)):
        y_max=np.ones_like(MLzs)*y_max

    if isinstance(y_min, (float, int)):
        y_min=np.ones_like(MLzs)*y_min

    # Avoid array of zeros as y_min
    y_min=np.array(y_min)
    y_min_nonzero_id= y_min!=0
    y_min_zero_id= y_min_nonzero_id==False
    # y_min cannot be just zeros
    if not np.any(y_min_nonzero_id):
        raise  ValueError('y_min cannot be array of 0s.')
    y_min_min=np.min(y_min[y_min_nonzero_id])
    y_min[y_min_zero_id]=np.ones_like(y_min[y_min_zero_id])*y_min_min


    ycrs=[]

    y_cr_curve_start=False

    for i, MLz in enumerate(tqdm(MLzs)):
        # Check if the mismatch at the minimum point is larger than the threshold
        Psi=Psis[i]

        if root_fun(h_fd, Psi, y_min[i], MLz)<0: 
            if not include_sl:
                ycr=0
        else:
            if increasing==True and not y_cr_curve_start:
                min, max= y_min[i], y_max[i]
            elif increasing==True and y_cr_curve_start:
                min, max= np.max([y_min[i],ycrs[-1]]), y_max[i]
            elif robust:
                min, max=y_min[i], y_max[i]  
            else:     
                min, max=y_min[i], ycrs[-1] if ycrs[-1]!=0 else y_max[i] # the conditional definition prevents using ycrs[-1]=0 (for instance when the previous point had root_fun<0)
            lim=[min, max]

            # this pushes the y_max if the root_function has not opposite sign
            if root_fun(h_fd, Psi, max, MLz)>0 and not y_cr_curve_start:
                raise_error=True
                for j in range(1,n_iter+1):
                    boost=10**j
                    if root_fun(h_fd, Psi, y_min[i], MLz)*root_fun(h_fd, Psi, boost*y_max[i], MLz)<0:
                        y_max[i:]=boost*y_max[i:]  
                        raise_error=False
                        lim=[min, y_max[i]]
                        break
                if raise_error:
                        raise ValueError("ERROR: No starting value found for MLz = {:.1e}. Try increasing n_iter!".format(MLz))
                        
            try: 
                sol=root_scalar(lambda x: root_fun(h_fd, Psi, x, MLz), bracket=lim, rtol=rtol)
            except Exception as e:
                # At this point ycrs[-1]<y<max is excluded. 
                # Try turning point condition y in [y_min, ycrs[-1]]
                try:
                    sol=root_scalar(lambda x: root_fun(h_fd, Psi, x, MLz), bracket=[y_min[i],ycrs[-1]], rtol=rtol)
                    increasing = False
                except Exception as e:
                    # Try if there is a counter turning point.
                    try:
                        sol=root_scalar(lambda x: root_fun(h_fd, Psi, x, MLz), bracket=[ycrs[-1],y_max[i]], rtol=rtol)
                        increasing = True
                    # Try to extend the upper max bound.
                    except Exception as e: 
                        for j in range(1,n_iter+1):
                            boost=10**j
                            try: 
                                sol=root_scalar(lambda x: root_fun(h_fd, Psi, x, MLz), bracket=[y_min[i], boost*y_max[i]], rtol=rtol)
                                if sol.converged: 
                                    y_max[i:]=boost*y_max[i:]
                                    break
                            except:
                                pass
                            if j==n_iter: # If the loop reached this point, no solution was found.
                                raise ValueError("ERROR: No peak value found for MLz = {:.1e}. Try increasing n_iter!".format(MLz))
        
            ycr=sol.root

            if not y_cr_curve_start: # Sets the starting point of the curve (nonzero y_crit)
                y_cr_curve_start=True 
      
        ycrs.append(ycr)

    ycrs=np.array(ycrs)


    if not return_mm:
        return ycrs
    else:
        mmSNR2_value=np.array([np.exp(root_fun(h_fd, Psi, y, MLz)) if not y==0 else 0 for y, MLz in zip(ycrs[1:], MLzs)])
        return ycrs, mmSNR2_value

def get_y_crit_curve_Fw_opt(h_fd, Fw_analytic_s, MLzs, y_min, y_max, s=1, 
                         rtol=0.05, n_iter=3, robust=False,
                         return_mm=False, include_sl=False,
                         kwargs_mm={}, kwargs_lensing_buff={}
                         ):

    '''
    Fast and accurate method to get the critical curve.
    
    Parameters:
    - h_fd (WaveformFD): Gravitational waveform strain.
    - Fws (array-like): Lensing amplification factor.
    - MLzs (array-like): Lens mass values.
    - y_min (float): Minimum impact parameter value.
    - y_max (float or array-like): Maximum impact parameter value(s).
    - s (float): Ratio between the SNR and the mismatch threshold. Default is 1.
    - optimized (bool): If True, use optimized mismatch calculation for more accuracy (slower). Default is False.
    - robust (bool): If True, no optimization of the bracket is performed. Default is False.
    - include_sl (bool): If True, include strong-lensing threshold points in the curve. Default is False.

    Returns:
    - list: Critical impact parameter values corresponding to each lens mass in MLzs.
    '''
    if isinstance(Fw_analytic_s, (list)):
        Fw_analytic_s=Fw_analytic_s 
    elif isinstance(MLzs, (float)):
        Fw_analytic_s=[Fw_analytic_s]
        MLzs=[MLzs]
    else:
        Fw_analytic_s=[Fw_analytic_s]*len(MLzs)

    def root_fun(h_fd, Fw_analytic, y, MLz):
        
        Fw_eval=Fw_analytic(y)
        h_fd_lensed=waveform.get_lensed_fd_from_Fw(h_fd, Fw_eval, MLz, **kwargs_lensing_buff) 
        mis_val=mismatch(h_fd_lensed, h_fd, **kwargs_mm)
        kwargs_lensing_buff.clear()
        if mis_val<0:
            mis_val=1e-15
        return np.log(mis_val)+2*np.log(s*h_fd.snr)

    
    increasing=True
    
    if isinstance(y_max, (float, int)):
        y_max=np.ones_like(MLzs)*y_max

    if isinstance(y_min, (float, int)):
        y_min=np.ones_like(MLzs)*y_min

    ycrs=[]

    y_cr_curve_start=False

    for i, MLz in enumerate(tqdm(MLzs)):
        # Check if the mismatch at the minimum point is larger than the threshold
        Fw_analytic=Fw_analytic_s[i]

        if root_fun(h_fd, Fw_analytic, y_min[i], MLz)<0: 
            if not include_sl:
                ycr=0
        else:
            
            if increasing==True and not y_cr_curve_start:
                min, max= y_min[i], y_max[i]
            elif increasing==True and y_cr_curve_start:
                min, max= np.max([y_min[i],ycrs[-1]]), y_max[i]
            elif robust:
                min, max=y_min[i], y_max[i]  
            else:     
                min, max=y_min[i], ycrs[-1] if ycrs[-1]!=0 else y_max[i] # the conditional definition prevents using ycrs[-1]=0 (for instance when the previous point had root_fun<0)
            lim=[min, max]

            # this pushes the y_max if the root_function has not opposite sign
            if root_fun(h_fd, Fw_analytic, max, MLz)>0 and not y_cr_curve_start:
                raise_error=True
                print('ECCO', root_fun(h_fd, Fw_analytic, y_max[i], MLz)>0)
                for j in range(1,n_iter+1):
                    boost=10**j
                    if root_fun(h_fd, Fw_analytic, y_min[i], MLz)*root_fun(h_fd, Fw_analytic, boost*y_max[i], MLz)<0:
                        y_max[i:]=boost*y_max[i:]  
                        raise_error=False
                        lim=[min, y_max[i]]
                        break
                if raise_error:
                        raise ValueError("ERROR: No starting value found for MLz = {:.1e}. Try increasing n_iter!".format(MLz))
                        
            try: 
                sol=root_scalar(lambda x: root_fun(h_fd, Fw_analytic, x, MLz), bracket=lim, rtol=rtol)
            except Exception as error:
                # handle the exception
                # print("An exception occurred:", error) # An exception occurred: division by zero

                # Catch turning point (lower y_cr than previous iter) by trying bracket [y_min, ycrs[-1]]
                try:
                    sol=root_scalar(lambda x: root_fun(h_fd, Fw_analytic, x, MLz), bracket=[y_min[i],ycrs[-1]], rtol=rtol)
                    increasing = False 
                # Solution is not in  ycrs[-1]<y<y_max[i] nor y<ycrs[-1]. Try pushing upper bounds.
                except:
                    print('ECCO qui', root_fun(h_fd, Fw_analytic, y_max[i], MLz)>0)
                    for j in range(1,n_iter+1):
                        boost=10**j
                        try: 
                            sol=root_scalar(lambda x: root_fun(h_fd, Fw_analytic, x, MLz), bracket=[y_min[i], boost*y_max[i]], rtol=rtol)
                            if sol.converged: 
                                y_max[i:]=boost*y_max[i:]
                                break
                        except:
                            pass
                        if j==n_iter: # If the loop reached this point, no solution was found.
                            raise ValueError("ERROR: No peak value found for MLz = {:.1e}. Try increasing n_iter!".format(MLz))
        
            ycr=sol.root

            if not y_cr_curve_start: # Sets the starting point of the curve (nonzero y_crit)
                y_cr_curve_start=True 
                
        ycrs.append(ycr)

    if not return_mm:
        return ycrs
    else:
        mmSNR2_value=[np.exp(root_fun(h_fd, Fw_analytic, y, MLz)) if not y==0 else 0 for y, MLz in zip(ycrs[1:], MLzs)]
        return ycrs, mmSNR2_value
    

def get_y_crit_curve_snr(Mvirs, mismatch_fun, snr, z_src, z_lens, y_limits=[2,200], guess=10, to_MLz=pu.to_MLz_SIS):
    '''
    Compute the critical impact parameter (y_crit) vs. lens mass (MLz) curve given SNR and source redshift.

    Parameters:
    - Mvirs (array-like): Halo virial masses.
    - mismatch_fun: Mismatch function on the 2D (MLz, y) grid.
    - snr (float): Signal-to-noise ratio.
    - z_src (float): Source redshift.
    - z_lens (float): Lens redshift.
    - y_limits (list): Limits for impact parameter y. Default is [2, 200].
    - guess (float): Initial guess for the critical impact parameter. Default is 10.
    - to_MLz: Function to convert halo virial mass to lens mass. Default is pu.to_MLz_SIS.

    Returns:
    - array-like: Critical impact parameters (y_crit) corresponding to each lens mass (MLz).
    '''
    
    # Convert halo virial masses to lens masses
    deff = 1/(1+z_lens)*(cosmo.angular_diameter_distance_z1z2(z_lens,z_src)*cosmo.angular_diameter_distance(z_lens)/cosmo.angular_diameter_distance(z_src)).decompose()
    MLzs = [to_MLz(Mvir,deff,z_lens).value for Mvir in Mvirs]

    y_crit=[find_y_crit_Mlz(mismatch_fun, MLz, snr, y_limits, guess) for MLz in MLzs]

    return y_crit

def get_interp_y_crit(x, y, log=False, kwargs_interp={'fill_value':'extrapolate'}):
    # Ensure x is sorted
    order = np.argsort(x)
    x_sorted = np.array(x)[order]
    y_sorted = np.array(y)[order]

    # Avoid null values in log
    y_sorted[y_sorted==0]=1e-15
    
    # Take logarithms of x and y
    log_x = np.log(x_sorted)
    if log:
        y_sorted = np.log(y_sorted)

    # Create and return the interpolating function in log space
    interpolating_function_log = interp1d(log_x, y_sorted, kind='linear', **kwargs_interp)

    # Create a function that returns the antilog of the interpolated result
    def interpolating_function(x_new):
        result = interpolating_function_log(np.log(x_new))
        if log:
            result = np.exp(result)
            
        return result

    return interpolating_function



def store_y_crit(parameters):
    """
    Store critical impact parameters (y_crit) along with relevant information in a JSON file.

    Parameters:
    - detector (str): Name of the gravitational wave detector.
    - lens (str): Identifier for the lensing scenario or model.
    - MLzs (array-like): Array of lens masses corresponding to the critical impact parameters.
    - y_crits (array-like): Array of critical impact parameters corresponding to each lens mass.
    - mismatch_thr (float): Mismatch threshold used in the analysis.
    - Mtot (float): Total mass of the lensed system.
    - z_src (float): Redshift of the gravitational wave source.
    - params_source (dict): Dictionary containing additional parameters related to the source.
    - params_lens (dict): Dictionary containing additional parameters related to the lensing scenario.
    - snr (float): Signal-to-noise ratio of the gravitational waveform.
    - dir (str): Directory path for storing the JSON file. Default is 'ycr_bank/'.

    Returns:
    - None: The function stores the data in a JSON file but does not return any values.
    """
    # Helper function to convert NumPy arrays to lists if they are NumPy arrays
    def convert_to_list(arr):
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        else:
            return arr

    # Construct a dictionary with the data to be saved
    dict_save = {
        'MLzs': pd.Series(convert_to_list(parameters['MLzs'])),
        'y_crit_s': pd.Series(convert_to_list(parameters['y_crits'])),
        's': parameters['mismatch_thr'],
        # 'Lens': parameters['lens'],
        # 'Lens_p': parameters['params_lens'],
        # 'detector': parameters['detector'],
        # 'SNR': parameters['snr'],
        'params_source': parameters['params_source']
    }

    # Convert the dictionary to a Pandas DataFrame
    to_save = pd.DataFrame.from_dict(data=dict_save)

    # Save the DataFrame as a pickle file
    filename_wrt = compose_filename(parameters['detector'], parameters['lens'], 
                                    parameters['mismatch_thr'], 
                                    parameters['params_source']['Mtot_src'], parameters['params_source']['redshift'], 
                                    dir=parameters['dir'])
    to_save.to_pickle(filename_wrt)

def store_y_crit_csv(parameters):
    """
    Store critical impact parameters (y_crit) along with relevant information in a JSON file.

    Parameters:
    - detector (str): Name of the gravitational wave detector.
    - lens (str): Identifier for the lensing scenario or model.
    - MLzs (array-like): Array of lens masses corresponding to the critical impact parameters.
    - y_crits (array-like): Array of critical impact parameters corresponding to each lens mass.
    - mismatch_thr (float): Mismatch threshold used in the analysis.
    - Mtot (float): Total mass of the lensed system.
    - z_src (float): Redshift of the gravitational wave source.
    - params_source (dict): Dictionary containing additional parameters related to the source.
    - params_lens (dict): Dictionary containing additional parameters related to the lensing scenario.
    - snr (float): Signal-to-noise ratio of the gravitational waveform.
    - dir (str): Directory path for storing the JSON file. Default is 'ycr_bank/'.

    Returns:
    - None: The function stores the data in a JSON file but does not return any values.
    """
    # Helper function to convert NumPy arrays to lists if they are NumPy arrays
    def convert_to_list(arr):
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        else:
            return arr

    # Save the DataFrame as a pickle file
    filename_wrt = compose_filename(parameters['detector'], parameters['lens'], 
                                    parameters['mismatch_thr'], 
                                    parameters['params_source']['Mtot_src'], parameters['params_source']['redshift'], 
                                    dir=parameters['dir'])
    # to_save.to_csv(filename_wrt)

    MLzs_array = np.asarray(parameters['MLzs'], dtype=float)
    y_crit_s_array = np.asarray(parameters['y_crits'], dtype=float)
    
    np.savetxt(filename_wrt, np.column_stack((MLzs_array, y_crit_s_array)), delimiter=',', header='MLzs, ycr', comments='')
    # import csv

    # with open(filename_wrt, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(MLzs_array, y_crit_s_array))

def get_interp_y_crit_from_stored(filename, log=True, **kwargs_interp):
           
    # Load stored mm grid
    MLzs=pd.read_pickle(filename)['MLzs'][0] # Change to MLzs
    y_crit=pd.read_pickle(filename)['y_crit_s'][0]
    if np.all(y_crit==[0]*len(y_crit)):
        y_crit_fun=get_interp_y_crit(MLzs, y_crit, log=False, **kwargs_interp)
    else:
        y_crit_fun=get_interp_y_crit(MLzs, y_crit, log=log, **kwargs_interp)

    return y_crit_fun
        
get_interp_y_crit_from_stored_vec= np.vectorize(get_interp_y_crit_from_stored)


##==============================================================================
## optical depth tools

from colossus.lss.mass_function import massFunction

@np.vectorize
def halo_mass_fun(Mvir, z_lens, dict_hmf):

    kwargs_hmf_default={'cutoff_low':1e4, 'cutoff_high':1e16, 
                        'mdef' : '200m', 'model' : 'tinker08', 'q_out':'dndlnM'}
    kwargs_hmf_default.update(dict_hmf)
    if Mvir>kwargs_hmf_default['cutoff_low'] and Mvir<kwargs_hmf_default['cutoff_high']:
        with ExcludeArgContext(kwargs_hmf_default, ['cutoff_low', 'cutoff_high']):
            return massFunction(Mvir/cosmo.h, z_lens, **kwargs_hmf_default)
    else:
        return 1e-40

def dlambda_dlogMvir(Mvir, y_crit_fun_z_lens, z_src, 
                    z_lens_s=[], n_z_lens=50, # Not providing z_lens_s means that the curves do not scale with lens redshift
                    include_sl=True, y_sl=1,
                    to_MLz=pu.to_MLz_SIS, 
                    dict_hmf={}):
    
    
    # Compute corresponding MLzs
    if z_lens_s==[]:
        z_evol=False
        z_lens_s = np.linspace(0.01,0.999*z_src, n_z_lens)
    else:
        z_evol=True
    

    factor =((cosmo.h)**3/u.Mpc**3*(4*np.pi*c.G*(1+z_lens_s)/(cosmo.H(z_lens_s))*(pu.get_d_A_z1z2(z_lens_s,z_src)*pu.get_d_A(z_lens_s)/pu.get_d_A(z_src))/c.c)).decompose()
    MLz_s=[to_MLz(Mvir, z_src, z_lens) for i, z_lens in enumerate(z_lens_s)] # Mvir -> MLz at different lens redshift
    
    # Halo-mass function
    dN_dz_s= np.array([halo_mass_fun(Mvir, z_lens, dict_hmf) for z_lens in z_lens_s])

    if not z_evol:
        y_crit_s=np.array([y_crit_fun_z_lens(MLz) for MLz in MLz_s])
    else:
        y_crit_s=np.array([y_crit_fun(MLz) for y_crit_fun, MLz in zip(y_crit_fun_z_lens, MLz_s)])

    # Integral
    integrand = (y_crit_s**2*dN_dz_s*MLz_s*u.solMass*factor).decompose()
    dlambda_dlogMv = simps(integrand, z_lens_s)

    
    if include_sl:
        integrand_sl = (y_sl**2*dN_dz_s*MLz_s*u.solMass*factor).decompose()
        dlambda_dlogMv_sl = simps(integrand_sl, z_lens_s)
        return dlambda_dlogMv, dlambda_dlogMv_sl
    else:
        return dlambda_dlogMv
    
def get_dlambda_dlogMvir_curve(Mvirs, y_crit_fun, z_src, **setup):

    dlambda_dlogMvir_vec=np.vectorize(lambda Mvir: dlambda_dlogMvir(Mvir, y_crit_fun, z_src, **setup))

    dlambda_dlogMvir_curve=dlambda_dlogMvir_vec(Mvirs)
    
    return dlambda_dlogMvir_curve


def integrate_dlambda_dlogMvir(Mvirs, dlambda_dlogMvir_s):
    return simps(dlambda_dlogMvir_s, x=np.log(Mvirs))


def get_lambda_at_z(z_src, y_crit_fun, Mvirs, **setup):
                  
    results = get_dlambda_dlogMvir_curve(Mvirs, y_crit_fun, z_src, **setup)

    N_WO=integrate_dlambda_dlogMvir(Mvirs, results[0])
    N_SL=integrate_dlambda_dlogMvir(Mvirs, results[1])

    return N_WO, N_SL

def get_lambda_curve(z_src_s, y_crit_fun_s, Mvirs, **setup):
    
    N_WO_s, N_SL_s = [], [] 

    for z_src,y_crit_fun in zip(z_src_s,y_crit_fun_s):
        N=get_lambda_at_z(z_src, y_crit_fun, Mvirs, **setup)
        N_WO_s.append(N[0])
        N_SL_s.append(N[1])

    return N_WO_s, N_SL_s
