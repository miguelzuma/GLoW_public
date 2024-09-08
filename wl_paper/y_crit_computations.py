import sys
#sys.path.append('../..')
from glow import waveform
import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.optimize import minimize
import scipy.special as special
import pickle 
from math import isnan
from scipy.stats import truncnorm
from glow import lenses
from glow import mismatch  as mm
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import os
import glow.physical_units as pu
from scipy.interpolate import interp1d, interp2d
pu.initialize_cosmology()
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.11 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3, Ob0=0.049)

kwargs_mm={'optimized':False}
p_prec_t={'tmax':1e9}
kwargs_lensing={'SL':False, 'p_prec_t':p_prec_t}

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
    tags_read=['ycr', '_{:s}_'.format(detector), '{:s}_'.format(lens), "s_{:.1f}_".format(mismatch_thr), "Mtot_{:e}_".format(Mtot), "zsrc_{:.2f}".format(z_src)]
    filename_read=dir+''.join(tags_read)
    return filename_read


def d_L(zL): ## In Mpc
    return cosmo.angular_diameter_distance(zL)

def d_S(zS): ## In Mpc
    return cosmo.angular_diameter_distance(zS)

def d_LS(zL, zS): ## In Mpc
    return cosmo.angular_diameter_distance_z1z2(zL, zS)

def d_eff(zL, zS): ## In Mpc
    dl = d_L(zL)
    ds = d_S(zS)
    dls = d_LS(zL, zS)
    
    return ((dl * dls / (1. + zL) / ds).decompose()).to(u.Mpc)
    

def M_lz(M200_at_zL, zL, zS): ## In Msun
    
    M200_at_zL = M200_at_zL
    
    deff = d_eff(zL, zS).to(u.Gpc)
    
    return ((2.3e6*(1+zL)**2*(deff.value)*(M200_at_zL/1e9 * cosmo.H(zL)/cosmo.H0)**(4/3)))*(u.Msun)

def ycrit_ste(Mtot_detector, Mvirs, zL, zS, snr, fdet=7e-3, w_SL = False):
    
    Mlzs = M_lz(Mvirs, zL, zS).value
    GMsun8pi=0.00012365485629514921 # 'in seconds'
    
    l=3.5
    f_ISCO=1/(6*np.sqrt(6)*GMsun8pi/8*Mtot_detector)*l
    coeffs=(snr/Mlzs)**(1/3)
    
    ycs = np.zeros(Mlzs.shape[0])
    
    if f_ISCO<fdet:
        m=2
        M_Lz_max=1/GMsun8pi*2**(4)*np.pi**3*m**3/snr**2/f_ISCO
        idxs = np.where(Mlzs>M_Lz_max)[0]
        ycs[idxs] += coeffs[idxs]/(f_ISCO*GMsun8pi)**(1/3)/2**(1/6)
    
    else:
        m=1
        M_Lz_max=16*np.pi**3*m**3/(GMsun8pi*fdet*snr**2)
        idxs = np.where(Mlzs>M_Lz_max)[0]
        ycs[idxs] += coeffs[idxs]/2**(1/6)*(GMsun8pi*fdet)**(-1/3)

    if w_SL==True:
        idxs = np.where(ycs<1)[0]
        ycs[idxs] = 1.
        return ycs
    
    else:
        return ycs,Mlzs
        
yr_to_s=u.yr.to(u.s)
Tobs= 0.13*yr_to_s
detector='LISA'
psd=waveform.get_psd_from_file(detector) 



def get_Psis_NFW(Mvirs_NFW, z_lens, z_src_waveform, norm_factor=1):

    # Calculate lensing masses (MLzs) for NFW profile
    MLzs_NFW = [pu.to_MLz_NFW(Mvir, z_lens) for Mvir in Mvirs_NFW]

    # Calculate the effective distance ratio (deff)
    deff = pu.get_deff(z_lens, z_src_waveform)

    # Calculate the Einstein radius (xi_0) for NFW profile and normalize it
    xi_0_NFW_s = [norm_factor * pu.get_R_E(MLz, deff) for MLz in MLzs_NFW]

    # Calculate the scale radius (rs) for NFW profile
    rs_s = [pu.get_rs_NFW(Mvir, z_lens) for Mvir in Mvirs_NFW]

    # Calculate the dimensionless scale parameter xs = rs / xi_0 and extract the values
    xs_s = [(rs / xi_0).decompose().value for (rs, xi_0) in zip(rs_s, xi_0_NFW_s)]

    norm_factor = [1 / norm_factor] * len(xs_s)

    # Compute potential

    Psi_NFW_s=[lenses.Psi_NFW(p_phys={'psi0':(norm_factor[i])**2,'xs':xs},
                            p_prec = {'eps_soft': 1e-15, 'eps_NFW': 0.01}) for i, xs in enumerate(xs_s)]

    return Psi_NFW_s, MLzs_NFW, xi_0_NFW_s,xs_s


def calc_ycrit_lensed_ul_SIS(Mtot_detector,q,z_src_waveform,Mlzs=np.geomspace(1e2,1e13, 20),spin=0,Tobs=Tobs,inc=0,psd=psd,detector=detector,store=True,robust=True,ymax=1e2,fdet=1e-1,w_SL=False,kwargs_mm=kwargs_mm, kwargs_lensing=kwargs_lensing):
    """
    Computes y critical for the given source parameters for SIS lens model  as a function of the effective lens mass. 

    Parameters:
    - Mtot_detector (float): Detector frame total mass of the binary source. 
    - q (float): Mass ratio of the binary (m1/m2)
    - z_src_waveform (float): Redshift of the source 
    - Mlzs (array-like): Halo virial masses. Default 1e5 to 1e12
    - spin (float) : Default 0
    - lens (str) : SIS or NFW. Default is SIS.
    - robust (bool): Root finding method. Default False
    - store (bool) : Whether to save the output in ycr_bank/. If true (default) no output is returned.
    - ymax (float) : 1e2 default. Upper limit for y_crit
    - fdet (float): 1e-1 default, Frequency cut of the detector.
    - w_SL (bool): Include Strong lensing or not. Default: False.
    - dict: passed on to the y_crit finding function.
    
    Returns:
    - array-like : effective lens masses (Mlzs).
    - array-like: Critical impact parameters (y_crit) corresponding to each lens mass (Mvir).
    - float : SNR of the signal
    
    """

    f_isco = waveform.f_isco(Mtot_detector)
    if fdet > 10*f_isco:
        f_final = 10*f_isco
    else:
        f_final = fdet
    params_source= {'approximant': "IMRPhenomXHM",
                    #'mode_array': modes,
                    'q'              : q,
                    'Mtot_src'       : Mtot_detector/(1+z_src_waveform),
                    'Mtot_obs'       : Mtot_detector,
                    'mass1'          : Mtot_detector * q/(1. + q),
                    'mass2'          : Mtot_detector * 1/(1. + q),
                    'spin1z'         : spin,
                    'spin2z'         : spin,
                    'redshift'       : z_src_waveform,
                    'inclination'    : inc,
            
                    'f_lower'        : np.amax([waveform.f0_obs(Mtot_detector, Tobs, units='s'),waveform.f_bounds_detector(detector)[0]]),
                    'delta_f'        : 1/Tobs,
                       'f_final': f_final}#,
                       
            
    # Unlensed Waveform generated once a waveform object is initialized thorugh parameters 
    h_fd=waveform.WaveformFD(params_source)
    psd=waveform.get_psd_from_file(detector)#,sky_pol_avg=False, inc_avg=False) 
    h_fd.load_psd(psd)
    snr=h_fd.snr
    Psis = lenses.Psi_SIS()
    #h_fd_lensed=waveform.get_lensed_fd_from_Psi_WL(h_fd, Psi_SIS, y, MLz)
    #mismatch=mm.mismatch(h_fd_lensed, h_fd, only_plus=True, optimized=False)
    
  
    ycrits=mm.get_y_crit_curve_opt(h_fd, Psis, Mlzs, 1, ymax, s=1,  robust= robust, rtol=1e-3,kwargs_mm=kwargs_mm, kwargs_lensing=kwargs_lensing)
    if store==True:
            os.makedirs('ycr_bank',exist_ok=True)
            if not isinstance(Psis, (list)):
                Psis=[Psis]*len(Mlzs)
        
            lens=Psis[0].p_phys['name']
            lens_p=[Psi.p_phys for Psi in Psis]
            file_lbl= 'z_lens_{:.2f}_'.format(zL)
            parameters={'detector':detector, 
                        'lens':lens, 
                        'MLzs':Mlzs, 
                        'y_crits':y_crits, 
                        'mismatch_thr':1, 
                        'params_source':h_fd.params_source, 
                        'params_lens':[lens_p],  
                        'dir':'ycr_bank/'+file_lbl,
                       'snr':snr,
                       'z_lens': zL}
            filename_wrt = compose_filename(parameters['detector'], parameters['lens'], parameters['mismatch_thr'], parameters['params_source']['Mtot_obs'], parameters['params_source']['redshift'], dir=parameters['dir']) +'.npz'
            np.savez(filename_wrt,**parameters)
    else:
        return ycrits,  Mlzs, snr
    
def calc_ycrit_lensed_ul_NFW(Mtot_detector,q,zL,z_src_waveform,Mvirs=np.geomspace(1e2,1e13, 20),spin=0,inc=0,Tobs=Tobs,psd=psd,detector=detector,store=True,robust=True,ymax=1e1,fdet=1e-1,kwargs_mm=kwargs_mm, kwargs_lensing=kwargs_lensing):
    """
    Computes y critical for the given source parameters for NFW lens model  as a function of virial mass of the lens at the  given lens redshift. 

    Parameters:
    - Mtot_detector (float): Detector frame total mass of the binary source. 
    - q (float): Mass ratio of the binary (m1/m2)
    - zL (float): Redshift of the lens should be less than source redshift.
    - z_src_waveform (float): Redshift of the source 
    - Mvirs (array-like): Halo virial masses. Default 1e5 to 1e12
    - spin (float) : Default 0
    - inc (float) : Default np.pi/4
    - lens (str) : SIS or NFW. Default is SIS.
    - robust (bool): Root finding method. Default False
    - store (bool) : Whether to save the output in ycr_bank/. If true (default) no output is returned.
    - ymax (float) : 1e2 default. Upper limit for y_crit
    - fdet (float): 1e-1 default, Frequency cut of the detector.
    - w_SL (bool): Include Strong lensing or not. Default: False.
    - dict: passed on to the y_crit finding function.
    
    Returns:
    - array-like: Effective lens mass (Mlz) corresponding to each lens mass (Mvir).
    - array-like: Critical impact parameters (y_crit) corresponding to each lens mass (Mvir).
    - float : SNR of the signal
    - array-like : physical lengthscale for lens = 'NFW'.
    
    """
    f_isco = waveform.f_isco(Mtot_detector)
    if fdet > 10*f_isco:
        f_final = 10*f_isco
    else:
        f_final = fdet
    if zL > z_src_waveform:
        print('lens redshift %f > source redshift %f, returning 0s'%(zL,z_src_waveform))
        return np.zeros(len(Mvirs)),np.nan, np.zeros(len(Mvirs))
    else:
        #modes=[(2,2)]
        params_source= {'approximant': "IMRPhenomXHM",
                    #'mode_array': modes,
                    'q'              : q,
                    'Mtot_src'       : Mtot_detector/(1+z_src_waveform),
                    'Mtot_obs'       : Mtot_detector,
                    'mass1'          : Mtot_detector * q/(1. + q),
                    'mass2'          : Mtot_detector * 1/(1. + q),
                    'spin1z'         : spin,
                    'spin2z'         : spin,
                    'redshift'       : z_src_waveform,
                    'inclination'    : inc,
            
                    'f_lower'        : np.amax([waveform.f0_obs(Mtot_detector, Tobs, units='s'),waveform.f_bounds_detector(detector)[0]]),
                    'delta_f'        : 1/Tobs,
                       'f_final': f_final}#,
                       
        
        # Unlensed Waveform generated once a waveform object is initialized thorugh parameters 
        h_fd=waveform.WaveformFD(params_source)
        psd=waveform.get_psd_from_file(detector)#,sky_pol_avg=False, inc_avg=False) 
        h_fd.load_psd(psd)
        snr=h_fd.snr

        
        Psis, Mlzs, xis,xs_s =get_Psis_NFW(Mvirs, zL, z_src_waveform,norm_factor=1)
        
        y_crits=mm.get_y_crit_curve_opt(h_fd, Psis, Mlzs, 0.001, ymax, s=1,  robust= robust, rtol=1e-3,kwargs_mm=kwargs_mm, kwargs_lensing=kwargs_lensing)

        xi=np.array([xi0.value for xi0 in xis]) #R_E physical scale
            
        
        if store==True:
            os.makedirs('ycr_bank',exist_ok=True)
            if not isinstance(Psis, (list)):
                Psis=[Psis]*len(Mlzs)
        
            lens=Psis[0].p_phys['name']
            lens_p=[Psi.p_phys for Psi in Psis]
            file_lbl= 'z_lens_{:.2f}_'.format(zL)
            parameters={'detector':detector, 
                        'lens':lens, 
                        'MLzs':Mlzs, 
                        'y_crits':y_crits, 
                        'mismatch_thr':1, 
                        'params_source':h_fd.params_source, 
                        'params_lens':[lens_p],  
                        'dir':'ycr_bank/'+file_lbl,
                       'snr':snr,
                       'Mvirs': Mvirs,
                       'z_lens': zL}
            filename_wrt = compose_filename(parameters['detector'], parameters['lens'], parameters['mismatch_thr'], parameters['params_source']['Mtot_obs'], parameters['params_source']['redshift'], dir=parameters['dir']) +'.npz'
            np.savez(filename_wrt,**parameters)
        else:
            return Mlzs,y_crits, snr,xi
