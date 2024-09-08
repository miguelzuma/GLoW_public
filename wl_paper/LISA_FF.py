#!/home1/srashti.goyal/miniconda310/envs/test_glow/bin/python
import sys
#sys.path.append('..')
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
import sys
import time
### inputs ### Lens and Source Parameters ###

y=float(sys.argv[1])#3
MLz=float(sys.argv[2])#1e8 # hoping to get lnB of O(1) from previous results
st = time.time()
Mtot=1e6 
z_src_waveform= 5 
Mtot_detector= Mtot*(1+z_src_waveform)
q=1
spin=0
N_iter = 10 # No. of initial guesses for minimizing mismatch, injection included by default.
maxiter = 100#1e5 # Max. no. of steps for minimise match function
yr_to_s=u.yr.to(u.s)
Tobs= 0.13*yr_to_s
fdet=0.5

detector='LISA'
psd=waveform.get_psd_from_file(detector) 
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
           # 'inclination'    : inc,
    
            'f_lower'        : np.amax([waveform.f0_obs(Mtot_detector, Tobs, units='s'),waveform.f_bounds_detector(detector)[0]]),
            'delta_f'        : 1/Tobs,
               'f_final': f_final}
    
# Unlensed Waveform generated once a waveform object is initialized thorugh parameters 
h_fd=waveform.WaveformFD(params_source)
psd=waveform.get_psd_from_file(detector) 
h_fd.load_psd(psd)
snr=h_fd.snr
print('SNR of the unlensed source: ', snr)

# Lensed Waveform generated once a waveform object is initialized thorugh parameters 
Psi_SIS = lenses.Psi_SIS()
h_fd_lensed=waveform.get_lensed_fd_from_Psi(h_fd, Psi_SIS, y, MLz)
mismatch=mm.mismatch(h_fd_lensed, h_fd, only_plus=True, optimized=True)
print('Log10 Mismatch lensed unlensed:', np.log10(np.abs(mismatch)) )
print('ln Bayesfactors = SNR^2 (1-M) : ', snr**2*np.abs(mismatch))

""""
plt.loglog(h_fd.sample_frequencies,h_fd.sample_frequencies*np.abs(h_fd.p))
plt.loglog(h_fd_lensed.sample_frequencies,h_fd_lensed.sample_frequencies*np.abs(h_fd_lensed.p))
plt.loglog(h_fd.sample_frequencies,np.sqrt(h_fd.sample_frequencies*np.abs(h_fd.psd_grid)))
plt.xlabel('$f$ [Hz]')
plt.ylabel('Characteristic strain')
plt.grid()
print('snr of the signal:', h_fd.snr)



psd=waveform.get_psd_from_file(detector) 
"""

def calc_mismatch(h_fd_lensed,mchirpz,eta=0.25,eff_spin=0,inc=0, psd=psd,Tobs= 0.13*yr_to_s):
    """ Given lensed W.F. calculates mismatch with the given unlensed waveform with given set of binary source parameters and PSD."""
    mtot = mchirpz * np.power(eta, -3./5)
    fac = np.sqrt(1. - 4.*eta)
    m1, m2 = (mtot * (1. + fac) / 2., mtot * (1. - fac) / 2.)
    
    params_source= {'approximant': "IMRPhenomXHM",
            'mass1'          : m1,
            'mass2'          : m2,
            'spin1z'         : eff_spin,
            'spin2z'         : eff_spin,
            'redshift'       : z_src_waveform,
            'f_lower'        : np.amax([waveform.f0_obs(Mtot_detector, Tobs, units='s'),waveform.f_bounds_detector(detector)[0]]),
            'delta_f'        : 1/Tobs,
               'f_final': f_final}
    h_fd=waveform.WaveformFD(params_source)
    h_fd.load_psd(psd)
    mismatch=mm.mismatch(h_fd_lensed, h_fd, only_plus=True, optimized=True)
    return np.log10(np.abs(mismatch))

eta= q/(1+q)**2
print('new fn Log10 mismatch lensed unlensed for same source parameters:',calc_mismatch(h_fd_lensed,Mtot_detector*(eta**(3/5)),eta))


def minimize_mismatch(fun_log_mismatch, mchirp_0, eta_0, eff_spin_0,maxiter=maxiter): 
       
   bnds = ((1e4, 1e9), (0.02, 0.25), (-0.99,0.99))

   # minimize the function  
   res = minimize(fun_log_mismatch, (mchirp_0, eta_0, eff_spin_0), method='Nelder-Mead',tol= 5e-3,bounds=bnds, options={'adaptive':True, 'disp': True,'maxiter':maxiter})    
   
   return res.fun


def calc_ff_lensed_waveforms(mchirp, eta, eff_spin, y, MLz, Psi_SIS=lenses.Psi_SIS(), psd=psd,Tobs= 0.13*yr_to_s,N_iter=1):

    """ calculate the FF of lensed waveform with an unlensed waveform family by minimizing mismatch over chirp mass, eta and spin"""
        
    mtot = mchirp * np.power(eta, -3./5)
    fac = np.sqrt(1. - 4.*eta)
    m1, m2 = (mtot * (1. + fac) / 2., mtot * (1. - fac) / 2.)
        
    # apply some boundary 
    if m1 < 2. or m2 < 2 or m1/m2 < 1./18 or m1/m2 > 18 or m1 > 1e9 or m2 > 1e9 or m1+m2 > 2e9 or        eff_spin < -0.99 or eff_spin > 0.99:
        log_mismatch = 1e6   
    else:
        
        
        params_source= {'approximant': "IMRPhenomXHM",
            'mass1'          : m1,
            'mass2'          : m2,
            'spin1z'         : eff_spin,
            'spin2z'         : eff_spin,
            'redshift'       : z_src_waveform,
            'f_lower'        : np.amax([waveform.f0_obs(Mtot_detector, Tobs, units='s'),waveform.f_bounds_detector(detector)[0]]),
            'delta_f'        : 1/Tobs, #anyscope of improvements here?
              'f_final': f_final}
        
        # generate the lensed waveform - target 
        h_fd=waveform.WaveformFD(params_source)
        h_fd.load_psd(psd)
        h_fd_lensed=waveform.get_lensed_fd_from_Psi(h_fd, Psi_SIS, y, MLz)
       
        # function to be minimized 
        fun_log_mismatch = lambda x: calc_mismatch(h_fd_lensed,x[0], x[1], x[2])

        if N_iter > 1: 
            # spread of the distribution of starting points around the true value 
            sigma_mc, sigma_eta, sigma_spin = 0.1, 0.1, 0.1

            # generate truncated Gaussian variables centered around the true value 
            mchirp_0 = truncnorm.rvs(-3, 3, size=10*N_iter)*sigma_mc+mchirp
            eta_0 = truncnorm.rvs(-3, 3, size=10*N_iter)*sigma_eta+eta
            eff_spin_0 = truncnorm.rvs(-3, 3, size=10*N_iter)*sigma_spin+eff_spin

            # make sure that the random paramers are in the allowed region; append the true values 
            idx = (mchirp_0>1e5) & (mchirp_0<1e9) & (eta_0>0.02) & (eta_0<=0.25) & (eff_spin_0>-0.99) & (eff_spin_0<0.99)

            mchirp_0 = np.append(mchirp, np.random.choice(mchirp_0[idx], N_iter-1))
            eta_0 = np.append(eta, np.random.choice(eta_0[idx], N_iter-1))
            eff_spin_0 = np.append(eff_spin, np.random.choice(eff_spin_0[idx], N_iter-1))

            log_mismatch = np.min(np.vectorize(minimize_mismatch)(fun_log_mismatch, mchirp_0, eta_0, eff_spin_0)) # kernel dies for me here therefore switched to script or use the line below with N_iter = some large no.
            #log_mismatch = np.min(np.vectorize(calc_mismatch,excluded=[0])(h_fd_lensed,mchirp_0, eta_0, eff_spin_0))
        else: 
            log_mismatch = minimize_mismatch(fun_log_mismatch, mchirp, eta, eff_spin)

    print(' FF minimum log_mismatch = {}'.format(log_mismatch))
                
    return log_mismatch


log_mismatch = calc_ff_lensed_waveforms(Mtot_detector*(eta**(3/5)),eta ,spin, y, MLz,N_iter=N_iter)


ln_Blu = 10**log_mismatch*snr**2

print('ln Bayesfactors = SNR^2 (1-FF) : ', ln_Blu)
et = time.time()

a=np.array([MLz,y, np.log10(np.abs(mismatch)),snr**2*np.abs(mismatch), log_mismatch,ln_Blu, N_iter, maxiter,Mtot_detector,q,z_src_waveform,spin, snr,et-st])
with open("test.txt", "a") as f:
    np.savetxt(f, a.reshape(1, -1), fmt="%1.3f",delimiter=",")
    f.write("\n")
