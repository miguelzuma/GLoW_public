import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18 as cosmo

from glow import freq_domain_c, waveform
from glow import physical_units as phys

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

## ---------------------------------------------------------------------

def set_plot_settings(fig, ax,
                      fig_size=(7, 4),
                      label_fontsize=16,
                      label_pad=6,
                      alpha_legend=0.5,
                      title_pad=6,
                      legend_font=1,
                      grid=True,
                      which_grid='both'):

    fig.set_size_inches(*fig_size)

    if not isinstance(ax, (list, np.ndarray)):
        ax = np.array([ax])  # Convert single axis to array

    for axis in ax:
        axis.set_xlabel(axis.get_xlabel(), labelpad=label_pad)
        axis.set_ylabel(axis.get_ylabel(), labelpad=label_pad)
        axis.set_title(axis.get_title(), pad=title_pad)

        axis.tick_params(axis='both', which='major', labelsize=label_fontsize-1)

        if grid:
            axis.grid(which=which_grid, alpha=alpha_legend)

        axis.set_xlabel(axis.get_xlabel(), fontsize=label_fontsize)
        axis.set_ylabel(axis.get_ylabel(), fontsize=label_fontsize)
        axis.set_title(axis.get_title(), fontsize=label_fontsize)

        legend = axis.get_legend()
        if legend:
            legend_frame = legend.get_frame()
            legend_frame.set_edgecolor('white')
            for text in legend.get_texts():  # Loop through legend texts and set fontsize
                text.set_fontsize(label_fontsize - legend_font)

## ---------------------------------------------------------------------

## =====================================================================
## =======   UNLENSED WAVEFORM
## =====================================================================
detector = 'LIGO'
z_src_waveform = 0.3 * cu.redshift
dL_src = z_src_waveform.to(u.Mpc, cu.with_redshift(cosmo, distance="luminosity")).value
Mtot = 100
Mtot_detector = Mtot*(1+z_src_waveform)
q = 1
spin = 0
inc = 0
Tobs = (50*u.s).value

# Lower bound of the frequecy. Maximum between expected frequency at -Tobs and the detector observable bound.
f_lower = np.amax([waveform.f0_obs(Mtot_detector, Tobs),waveform.f_bounds_detector(detector)[0]])
f_final = 5*waveform.f_isco(Mtot_detector)*(1+z_src_waveform) # If too short, multiply by k>1

# Same keys of get_fd_waveform in pycbc
params_source= {'approximant': "IMRPhenomXHM",
            'mass1'          : Mtot_detector * q/(1. + q),
            'mass2'          : Mtot_detector * 1/(1. + q),
            'spin1z'         : spin,
            'spin2z'         : spin,
            'distance'       : dL_src,
            'inclination'    : inc,
            'long_asc_nodes' : 0,
            'f_lower'        : f_lower,
            'delta_f'        : 1/Tobs,
            'f_final':  f_final
            }

sky_dict = {'declination' : 0.05, 'right_ascension' : 3.67, 'polarization': 0.34}
params_source.update(sky_dict)

# Generate Frequency Domain (FD) Waveform object
h_fd = waveform.WaveformFD(params_source)

# If needed, load a psd function either from file or pycbc (get_psd_from_file/_pycbc)
# In this case we need it to compute the SNR
psd = waveform.get_psd_from_file(detector)
h_fd.load_psd(psd)

# Time-domain waveform - alpha tunes the windowing of the signal in the inverse fft
h_td = h_fd.to_timedomain(alpha=0.04, cyclic_time_shift=-Mtot/300)


## =====================================================================
## =======   LENSED WAVEFORM - WEAK LENSING
## =====================================================================
y_wave = 1.25

# Define the units for the lens and the amplification factor
zl = z_src_waveform/2
zs = z_src_waveform
p_lens = {'name':'SIS', 'Mvir_Msun':5e6}

units_lens = phys.Units(zl, zs, p_lens)
Fw_wave= freq_domain_c.Fw_SemiAnalyticSIS_C(y_wave)

# Get the lensed waveform
h_fd_lens_wave= waveform.get_lensed_fd_from_Fw(h_fd, Fw_wave, units_lens, w_opt=True)

# Compute the time delay between images
if len(Fw_wave.It.p_crits)>1:
    Dtau_wave = Fw_wave.It.p_crits[1]['t'] - Fw_wave.It.p_crits[0]['t']
else:
    Dtau_wave = 1

time_delay_wave = units_lens.tau_to_t(Dtau_wave, un='s')

# Time domain waveforms
shift_wave = -Mtot/300
h_td_lens_wave = h_fd_lens_wave.to_timedomain(alpha=0.04, cyclic_time_shift=shift_wave, unlensed=True)
ts_wave = h_td_lens_wave.sample_times.numpy()


## =====================================================================
## =======   LENSED WAVEFORM - OVERLAPPING
## =====================================================================
y_micro = 0.1

# Define the units for the lens and the amplification factor
zl = z_src_waveform/2
zs = z_src_waveform
p_lens = {'name':'SIS', 'Mvir_Msun':1e8}

units_lens = phys.Units(zl, zs, p_lens)
Fw_micro = freq_domain_c.Fw_SemiAnalyticSIS_C(y_micro)

# Get the lensed waveform
h_fd_lens_micro = waveform.get_lensed_fd_from_Fw(h_fd, Fw_micro, units_lens, w_opt=True)

# Compute the time delay between images
if len(Fw_micro.It.p_crits)>1:
    Dtau_micro = Fw_micro.It.p_crits[1]['t'] - Fw_micro.It.p_crits[0]['t']
else:
    Dtau_micro = 1

time_delay_micro = units_lens.tau_to_t(Dtau_micro, un='s')

# Time domain waveforms
# We shift by the approx time delay of the second image + the ringdown duration
shift_micro = -time_delay_micro - Mtot/300
h_td_lens_micro = h_fd_lens_micro.to_timedomain(alpha=0.04, cyclic_time_shift=shift_micro, unlensed=True)
ts_micro = h_td_lens_micro.sample_times.numpy()


## =====================================================================
## =======   LENSED WAVEFORM - MULTIPLE
## =====================================================================
y_strong = 0.9

# Define the units for the lens and the amplification factor
zl = z_src_waveform/2
zs = z_src_waveform
p_lens = {'name':'SIS', 'Mvir_Msun':1e8}

units_lens = phys.Units(zl, zs, p_lens)
Fw_strong = freq_domain_c.Fw_SemiAnalyticSIS_C(y_strong)

# Get the lensed waveform
h_fd_lens_strong= waveform.get_lensed_fd_from_Fw(h_fd, Fw_strong, units_lens, w_opt=True)

# Compute the time delay between images
if len(Fw_strong.It.p_crits) > 1:
    Dtau_strong = Fw_strong.It.p_crits[1]['t'] - Fw_strong.It.p_crits[0]['t']
else:
    Dtau_strong=1

time_delay_strong = units_lens.tau_to_t(Dtau_strong, un='s')

# Time domain waveforms
# We shift by the approx time delay of the second image + the ringdown duration
shift_strong = -time_delay_strong - Mtot/300
h_td_lens_strong = h_fd_lens_strong.to_timedomain(alpha=0.04, cyclic_time_shift=shift_strong, unlensed=True)
ts_strong = h_td_lens_strong.sample_times.numpy()


## =====================================================================
## =======   PLOT
## =====================================================================

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].plot(ts_wave, h_td.strain/h_fd.snr, label='unlensed', c='gray', alpha=0.6)
axs[0].plot(ts_wave, h_td_lens_wave.strain/h_fd_lens_wave.snr, label='lensed', c='C0', zorder=0)
axs[0].set_xlim([-Mtot/70, Mtot/600])
axs[0].set_title('single image')
axs[0].set_ylabel('strain')
axs[0].legend()

axs[1].plot(ts_micro, h_td.strain/h_fd.snr, label='unlensed', c='gray', alpha=0.6)
axs[1].plot(ts_micro, h_td_lens_micro.strain/h_fd_lens_micro.snr, label='lensed', c='C0', zorder=0)
axs[1].set_xlim([-Mtot/70, time_delay_micro+Mtot/600])
axs[1].set_title('multiple overlapping images')

axs[2].plot(ts_strong, h_td_lens_strong.strain, label='lensed', c='C0')
axs[2].set_xlim([-Mtot/70, time_delay_strong+Mtot/600])
axs[2].set_title('multiple images')

for ax in axs:
    ax.set_xlabel('time')
    ax.tick_params(axis="both", which='both',
                   left=False, labelright=False, right=False, labelleft=False,
                   top=False, labeltop=False, bottom=False, labelbottom=False)

set_plot_settings(fig, axs,
                  fig_size=(12, 4),
                  label_fontsize=16,
                  label_pad=6,
                  alpha_legend=0.5,
                  title_pad=6,
                  legend_font=1,
                  grid=False,
                  which_grid='both'
                  )

fig.tight_layout()
fig.savefig('waveform_time_domain.pdf', dpi=700, bbox_inches='tight')

