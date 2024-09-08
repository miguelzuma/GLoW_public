import numpy as np

import glow.mismatch  as mm
import glow.tools as tools
import glow.physical_units as pu
import glow.waveform as waveform

def initialize_cosmology(**kwargs):
    return pu.initialize_cosmology(**kwargs)
    

################################################
## toolbox

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()
    
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        import matplotlib.text as  mtext
        title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title


class ExcludeArgContext:
    def __init__(self, kwargs, key):
        self.kwargs = kwargs
        self.key = key

    def __enter__(self):
        # Check if the key is present in kwargs
        self.exclude_arg_value = self.kwargs.pop(self.key, None)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Put the key back in kwargs after the block
        if self.exclude_arg_value is not None:
            self.kwargs[self.key] = self.exclude_arg_value


def set_latex_default(plt):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}

    plt.rc('font', **font)

################################################

def produce_store_ycr_curve(Psi, MLzs, detector, params_source, psd, file_lbl=''):
    """
    Produce and store critical curves.

    Parameters:
    - params_source (dict): Set of source parameters to produce the waveform.
    - pds (callable): Psd function.

    """
    # A waveform is generated once a waveform object is initialized by the source parameters 
    h_fd=waveform.WaveformFD(params_source)
    h_fd.load_psd(psd)

    s=1.
    y_crits=mm.get_y_crit_curve_opt(h_fd, Psi, MLzs, 1, 200, s)

    # Store mismatch grid

    lens=Psi.p_phys['name']
    lens_p=Psi.p_phys

    save_dict={'detector':detector, 
        'lens':lens, 
        'MLzs':[MLzs], 
        'y_crits':[y_crits], 
        'mismatch_thr':s, 
        'params_source':params_source, 
        'params_lens':lens_p,  
        'dir':'ycr_bank/'+file_lbl}

    mm.store_y_crit(save_dict)

def produce_store_ycr_curve_Psi(Psis, MLzs, detector, h_fd, file_lbl='', 
                                y_min=1, y_max=100, s=1, **kwargs):
    """
    Produce and store critical curves.

    Parameters:
    - params_source (dict): Set of source parameters to produce the waveform.
    - pds (callable): Psd function.

    """
    # A waveform is generated once a waveform object is initialized by the source parameters 

    y_crits=mm.get_y_crit_curve_opt(h_fd, Psis, MLzs, y_min, y_max, s, **kwargs)

    # Store mismatch grid

    if not isinstance(Psis, (list)):
        Psis=[Psis]*len(MLzs)
    
    lens=Psis[0].p_phys['name']
    lens_p=[Psi.p_phys for Psi in Psis]

    save_dict={'detector':detector, 
        'lens':lens, 
        'MLzs':[MLzs], 
        'y_crits':[y_crits], 
        'mismatch_thr':s, 
        'params_source':h_fd.params_source, 
        'params_lens':[lens_p],  
        'dir':'ycr_bank/'+file_lbl}

    mm.store_y_crit(save_dict)

################################################

def plot_y_crit(ax, MLzs, y_crit_fun, **kwargs_plot):
    """
    Plot the critical parameter y_cr as a function of MLz.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - MLzs (numpy.ndarray): Array of MLz values.
    - y_crit_fun (callable): Function representing the critical parameter for lensing.
    - **kwargs_plot: Additional keyword arguments for matplotlib plot function.

    Returns:
    - p (matplotlib.lines.Line2D): Plot object.
    """
    # Calculate y_crit values using the provided function
    y_crit = y_crit_fun(MLzs)

    # Plot the curve on the given axis with specified plot settings
    p = ax.plot(MLzs, y_crit, '-', **kwargs_plot)[0]

    return p


def plot_y_crit_vec(ax, MLzs, y_crit_fun_s, c_s, **kwargs_plot):
    """
    Plot multiple y_crit curves on the same axis.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - MLzs (numpy.ndarray): Array of MLz values.
    - y_crit_fun_s (list): List of functions representing critical parameters for lensing.
    - c_s (list): List of colors for the plots.
    - **kwargs_plot: Additional keyword arguments for matplotlib plot function.

    Returns:
    - plots (list): List of plot objects.
    """
    plots = []

    # Loop through each y_crit function and plot on the same axis
    for i, y_crit_fun in enumerate(y_crit_fun_s):
        with ExcludeArgContext(kwargs_plot, 'c'):
            plots.append(plot_y_crit(ax, MLzs, y_crit_fun, c=c_s[i], **kwargs_plot))

    # Set axis properties and formatting
    fontsize = 14
    ax.set_xlabel('$M_{Lz}\,[M_\odot]$', fontsize=fontsize)
    ax.set_yscale('linear')
    ax.set_xscale('log')
    lims = [MLzs[0], MLzs[-1]]
    ax.set_xlim(*lims)
    ax.set_ylim(bottom=0.5)
    ax.set_ylabel('$y_{cr}$', fontsize=fontsize)

    # Set font sizes for axis labels and ticks
    for item in ax.axes.get_xticklabels() + ax.axes.get_yticklabels():
        item.set_fontsize(fontsize)

    # Add grid lines
    ax.grid(which='both', alpha=0.5)

    return plots


def assemble_legend(Mtots, z_src, plots, detector):
    """
    Assemble legend labels and handles for a plot.

    Parameters:
    - Mtots (list): List of  binary total masses.
    - z_src (float): Source redshift.
    - plots (list): List of plot objects.
    - detector (str): Name of the detector.

    Returns:
    - labels (list): List of labels for legends.
    - handles (list): List of handles for legends.
    """
    labels = []

    # Create labels for total masses
    for Mtot in Mtots:
        labels.append('${:s}M_\odot$'.format(tools.latex_float(Mtot)))

    # Add label for source redshift
    labels = ['$(z_s={:s})$'.format(tools.latex_float(z_src)), *labels]

    # Create handles for the legend
    handles = [detector, *plots]

    return labels, handles


def add_legend_y_crit(ax, lab, hand, pos):
    """
    Add legend to the plot.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - lab (list): List of labels for legends.
    - hand (list): List of handles for legends.
    - pos (float): Position for the legend.

    Returns:
    None
    """
    ax2 = ax.twinx()

    # Customize secondary y-axis properties
    ax2.tick_params(axis="y", which='both', left=False, right=False, labelright=False)

    fontsize = 14

    # Add legend to the secondary y-axis
    ax2.legend(handles=hand, labels=lab, bbox_to_anchor=pos, handler_map={str: LegendTitle({'fontsize': 14})},
               ncols=1, fontsize=fontsize, title_fontsize=fontsize, columnspacing=0.8, handlelength=1.7,
               edgecolor='white')


def add_top_Mvirs_y_crit(ax, z_src, z_l, ax_Mvir_par, to_Mvir=pu.to_Mvir_SIS, to_MLz=pu.to_MLz_SIS):
    """
    Add a top x-axis to the plot for Mvir values.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - z_src (float): Source redshift.
    - z_l (float): Lens redshift.
    - ax_Mvir_par (tuple): Tuple containing parameters for the secondary x-axis.
    - to_Mvir (function): Conversion function for Mvir values.
    - to_MLz (function): Conversion function for MLz values.

    Returns:
    - secax_tmp (matplotlib.axes._subplots.AxesSubplot): Secondary x-axis.
    """
    ticks, col, dir, pad = ax_Mvir_par
    fontsize = 14

    # Create a secondary x-axis with specified conversion functions
    secax_tmp = ax.secondary_xaxis('top',
                                    functions=(lambda x: to_Mvir(x, z_src, z_l),
                                               lambda x: to_MLz(x, z_src, z_l)))


    # Customize secondary x-axis ticks and appearance
    secax_tmp.tick_params(axis='x', which='both', direction=dir, pad=pad, colors=col)
    secax_tmp.set_xticks(ticks)
    secax_tmp.minorticks_off()

    # Set font sizes for ticks
    for item in secax_tmp.axes.get_xticklabels():
        item.set_fontsize(fontsize)

    return secax_tmp

################################################


def plot_mismatch(ax, ys, h_fd, Psi, MLz, setup={'phase_only': True, 'amp_only': True}, **kwargs_plot):
    """
    Plot mismatch curves.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - ys (numpy.ndarray): Array of values for the parameter y.
    - h_fd (WaveformFD): Input waveform.
    - Psi (Lens potential): Description of Psi.
    - MLz (float): Mass parameter MLz.
    - setup (dict, optional): Additional setup parameters for mismatch_ys. Default is {'phase_only': True, 'amp_only': True}.
    - **kwargs_plot: Additional keyword arguments for matplotlib plot function.

    Returns:
    None
    """
    # Calculate mismatch curves using the specified parameters
    mismatch_grid = mm.mismatch_ys(h_fd, Psi, ys, MLz, **setup)

    markers = [':', '--']
    labels = ['phase', 'amplitude']

    # Plot the first curve with label and alpha
    ax.plot(ys, mismatch_grid[0], label='${:s}M_\odot$'.format(tools.latex_float(MLz)), linewidth=2, alpha=0.8, **kwargs_plot)

    # Plot the eventual phase-only, amplitude-only curves with different linestyles
    for j, mismatch_curve in enumerate(mismatch_grid[1:]):
        ax.plot(ys, mismatch_curve, linestyle=markers[j], linewidth=1.8, alpha=0.8, **kwargs_plot)
    
    fontsize=14
    # Set label, grid and formatting of the plot
    ax.set_xlabel('$y$', fontsize=fontsize)
    ax.set_ylabel(r'$\mathcal{M}$', fontsize=fontsize)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    xticks = [5, 10, 20, 50]
    ax.set_xticks(xticks, xticks)


def add_legend_mismatch(ax, ys, h_fd, phase_only=True, amp_only=True):
    """
    Add legends and formatting to a plot with mismatch curves.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - ys (numpy.ndarray): Array of values for the parameter y.
    - h_fd (WaveformFD): Input waveform.
    - phase_only (bool, optional): Whether to print phase-only label. Default is True.
    - amp_only (bool, optional): Whether to print amplitude-only label. Default is True.

    Returns:
    None
    """

    fontsize = 14
    ax2 = ax.twinx()

    # Plot a reference line on the primary y-axis
    ax.plot(ys, np.zeros(len(ys)) + 1 / h_fd.snr ** 2, '-', c='gray', alpha=0.7, linewidth=2)

    # Add snr threhold entry to legend
    ax2.plot([], [], '-', label='1/SNR$^2$', c='gray', alpha=0.7)

    markers = [':', '--']
    labels = ['phase', 'amplitude']

    # If amp_only is True, add an entry for amplitude in the legend
    if amp_only:
        ax2.plot([], [], linestyle=markers[1], c='black', label=labels[1])

    # If phase_only is True, add an entry for phase in the legend
    if phase_only:
        ax2.plot([], [], linestyle=markers[0], c='black', label=labels[0])

    # Set labels of the plot
    ax.legend(fontsize=14, title='$M_{{Lz}}$', loc='upper right', frameon=True, edgecolor='white')

    # Set font sizes for axis labels and ticks
    for item in ax.axes.get_xticklabels() + ax.axes.get_yticklabels():
        item.set_fontsize(fontsize)

    # Adjust secondary y-axis ticks and legends
    ax2.tick_params(axis="y", which='both', left=False, labelright=False, right=False, labelleft=False)
    ax.set_ylim(1e-9)
    ax.set_xlim(ys[0], ys[-1])

    # Add legend for the secondary y-axis
    legend2 = ax2.legend(loc='lower left', frameon=True, edgecolor='white', fontsize=fontsize)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(1)

################################################


def plot_dlambda_dlogMvir_curve(ax, Mvirs, y_crit_fun, z_src, plot_sl=True, setup={}, **kwargs_plot):
    """
    Plot dn/dlogMvir curve on a given axis.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - Mvirs (numpy.ndarray): Array of virial masses.
    - y_crit_fun (callable): A function representing the critical parameter for lensing.
    - z_src (float): Source redshift.
    - plot_sl (bool, optional): Whether to plot strong lensing curve. Default is True.
    - setup (dict, optional): Additional setup parameters for get_dn_dlogMvir_curve. Default is an empty dictionary.
    - **kwargs_plot: Additional keyword arguments for matplotlib plot function.

    Returns:
    None
    """
    results = mm.get_dlambda_dlogMvir_curve(Mvirs, y_crit_fun, z_src, **setup)
    # Plot the weak lensing (WL) curve
    ax.plot(Mvirs, results[0], linewidth=2, **kwargs_plot)
    # Plot the strong lensing (SL) curve if plot_sl is True
    if plot_sl:
        # Use ExcludeArgContext to temporarily remove 'c' from kwargs_plot for this plot
        with ExcludeArgContext(kwargs_plot, 'c'):
            ax.plot(Mvirs, results[1], linewidth=1.3, c='darkgreen', **kwargs_plot)

def plot_dlambda_dlogMvir_curve_vec(ax, Mvirs, y_s, z_s, c_s, ls_s):
    """
    Plot multiple dn/dlogMvir curves on a given axis.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - Mvirs (numpy.ndarray): Array of virial masses.
    - y_s (list): List of critical parameter functions for lensing.
    - z_s (list): List of source redshifts.
    - c_s (list): List of colors for curves.
    - ls_s (list): List of line styles for curves.

    Returns:
    None
    """
    for j, (z, ls) in enumerate(zip(z_s, ls_s)):
        plot_sl = True
        # Iterate over colors to plot curves with different colors
        for i, c in enumerate(c_s):
            # Plot dn/dlogMvir curve using the plot_dn_dlogMvir_curve function
            plot_dlambda_dlogMvir_curve(ax, Mvirs, y_s[j][i], z, plot_sl=plot_sl, c=c, ls=ls)
            plot_sl = False  # Set plot_sl to False for subsequent curves in the same redshift

    fontsize=14
    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$M_{\rm vir}\,[M_\odot]$', fontsize=fontsize)
    ax.set_ylabel('$\\frac{{ {{\\rm d}} \lambda}}{{ {{\\rm d}} \\log M_{\\rm vir}}}$', fontsize=fontsize)
    ax.set_xlim(1e4, 1e16)
    ax.set_ylim(bottom=1e-4, top=1e-1)
    ax.set_xscale('log')
    ax.set_yscale('log')

def add_legend_dlambda_dlogMvir(ax, Mtot_s, z_src_s, c_s, ls_s, detector):
    """
    Add legends and formatting to a plot with multiple dn/dlogMvir curves.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis for plotting.
    - Mtot_s (list): List of total masses.
    - z_src_s (list): List of source redshifts.
    - c_s (list): List of colors for legends.
    - ls_s (list): List of line styles for legends.
    - detector (str): Name of the detector.

    Returns:
    None
    """
    # Create twin axes for secondary y-axes
    secax = ax.twinx()
    secax2 = ax.twinx()

    # Plot invisible lines for creating legends for Mtot_s and z_src_s on primary and secondary y-axes
    for Mtot, c in zip(Mtot_s, c_s):
        ax.plot([], [], label='${:s} M_\odot$'.format(tools.latex_float(Mtot)), c=c)

    for z, ls in zip(z_src_s, ls_s):
        secax.plot([], [], c='black', linestyle=ls, label='${:s}$'.format(tools.latex_float(z)), alpha=0.5)

    # Plot invisible line for creating legend for SL on the secondary y-axis
    secax2.plot([], [], c='darkgreen', label='SL', linewidth=1.3)

    # Set legend, grid, labels, limits, and scales for the primary y-axis
    fontsize = 14
    ax.legend(fontsize=fontsize, ncols=1, title='$M_{{\\rm BBH}}$', bbox_to_anchor=(0.84, 1),
              handlelength=1.5, edgecolor='white')
    ax.text(2e4, 6e-2, r"{:s} ".format(detector), fontsize=16)
    for item in ax.axes.get_xticklabels() + ax.axes.get_yticklabels():
        item.set_fontsize(fontsize)

    # Set legend and hide ticks for the secondary y-axes
    secax.legend(fontsize=fontsize, title='$z_S$', bbox_to_anchor=(1, 1), handlelength=1.5, edgecolor='white')
    secax.tick_params(axis="y", which='both', left=False, right=False, labelright=False)

    secax2.legend(fontsize=fontsize, bbox_to_anchor=(1.006, 0.75), handlelength=1.5, edgecolor='white')
    secax2.tick_params(axis="y", which='both', left=False, right=False, labelright=False)

################################################


# def plot_tau_at_z(z_src_s, Mtots, Mvirs, detector, loc = 0.77, y_lims=[3e-3,2],   c_s=['#ffde00', '#ff9200', '#ff3535'] ):


def plot_lambda(ax, z_src_s, y_crit_fun_s, Mvirs, Mtot_s, c_s, setup={}, y_lims=[3e-3,2]):

    for i, Mtot in enumerate(Mtot_s):
        y_crit_fun_zs=y_crit_fun_s.transpose()[i]
        n_curve_s=mm.get_lambda_curve(z_src_s, y_crit_fun_zs, Mvirs, **setup)
        ax.plot(z_src_s, n_curve_s[0] ,'.-', linewidth=2, c=c_s[i], label='${:s} M_\odot$'.format(tools.latex_float(Mtot)))

    ax.plot(z_src_s, n_curve_s[1],'--', linewidth=1.3, c='darkgreen', label='SL')

    fontsize=14
    ax.set_yscale('log')
    ax.set_xlabel(r'$z_{\rm S}$',fontsize=fontsize)
    ax.set_ylabel('$\\lambda $',fontsize=fontsize)
    ax.grid(axis='y', which='major', alpha=0.5)
    ax.grid(axis='x', which='both', alpha=0.5)
    ax.set_ylim(*y_lims)

def add_legend_lambda(ax, z_src_s, Mtots, detector):

    fontsize=14
    ax.text(0.02, 0.85, r"{:s} ".format(detector),backgroundcolor='white', fontsize=fontsize, alpha=1, transform=ax.transAxes)
    ax.legend(fontsize=fontsize, 
                loc='upper right',
                bbox_to_anchor= (0.99,1.025), 
                handlelength=1.5, 
                ncols=len(Mtots)+1, 
                columnspacing=0.8, 
                edgecolor='white', 
                borderpad= 0.25)#, edgecolor='white')



