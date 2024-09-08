import os
import sys
from subprocess import call

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from glow import lenses
from glow import time_domain
from glow import time_domain_c
from glow import freq_domain
from glow import freq_domain_c

from scipy.interpolate import griddata
from scipy.integrate import simps

## =========================================================================

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

## =========================================================================

def create_lens(n, R=1, seed=None, use_SIS=True):
    psi0 = 1./n
    xc1s, xc2s, seed = distribute_centers(n, R, seed=seed)
    
    p_phys = {'lenses' : []}
    for xc1, xc2 in zip(xc1s, xc2s):
        
        if use_SIS is True:
            sub_p_phys = {'psi0' : psi0, 'xc1' : xc1, 'xc2' : xc2}
            Psi = lenses.Psi_offcenterSIS(sub_p_phys)
        else:
            sub_p_phys = {'psi0' : psi0, 'b' : 1, 'xc1' : xc1, 'xc2' : xc2}
            Psi = lenses.Psi_offcenterBall(sub_p_phys)
            
        p_phys['lenses'].append(Psi)

    return lenses.CombinedLens(p_phys), seed

def plot_lenses(ax, Psi, y, R=1, seed=None):
    rfig = 2*y/1.6
    shift = rfig*0.25    # shift figure to the left
    line_l = 0.05*y/1.6
    line_w = 1
    
    ax.set_xlim([-rfig + shift, rfig + shift])
    ax.set_ylim([-rfig, rfig])
    
    circle = plt.Circle((0, 0), R, color='grey', alpha=0.4, clip_on=False)
    patches = [circle]
    
    def convert_lens_to_circle(lens):
        x1 = lens.p_phys['xc1']
        x2 = lens.p_phys['xc2']
        R = lens.p_phys['psi0']
        
        subcircle = plt.Circle((x1, x2), R, color='red', alpha=0.4, clip_on=False)
        return subcircle
        
    for lens in Psi.p_phys['lenses']:
        patches.append(convert_lens_to_circle(lens))
    
    line1 = plt.Line2D([y-line_l, y+line_l], [-line_l, line_l], lw=line_w)
    line2 = plt.Line2D([y-line_l, y+line_l], [line_l, -line_l], lw=line_w)
    artists = [line1, line2]
    
    ax.set_axis_off()
    for patch in patches:
        ax.add_patch(patch)
    for art in artists:
        ax.add_artist(art)
    
    if seed is not None:
        ax.text(0.4*rfig, -0.9*rfig, "$y$ = %g\nSeed = %d" % (y, seed), alpha=0.5)

def plot_It(ax, Psi, y, SIS=False, CIS=False, rc=0.1):
    p_phys = {'y' : y}
    p_prec = {'eval_mode' : 'exact'}
    It = time_domain_c.It_SingleContour_C(Psi, p_phys, p_prec)

    taus = np.geomspace(5e-1, 50, 1000)

    ax.plot(taus, It.eval_It(taus)/2/np.pi, label='%d SIS' % len(Psi.p_phys['lenses']), c='C0')
    ax.set_xscale('log')
    
    It_SIS = None
    if SIS is True:
        It_SIS = time_domain_c.It_AnalyticSIS_C(p_phys, p_prec)
        ax.plot(taus, It_SIS.eval_It(taus)/2/np.pi, label='SIS', alpha=0.5, c='C1')
    
    It_CIS = None
    if CIS is True:
        Psi_CIS = lenses.Psi_CIS({'psi0':1, 'rc':rc})
        It_CIS = time_domain_c.It_SingleContour_C(Psi_CIS, p_phys, p_prec)
        ax.plot(taus, It_CIS.eval_It(taus)/2/np.pi, label='CIS', alpha=0.5, c='C2')
    
    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$I(\tau)/2\pi$')
    ax.set_xlim([taus[0], taus[-1]])
    ax.legend(loc='best')
    
def plot_Gt(ax, Psi, y, SIS=False, CIS=False, rc=0.1):
    p_phys = {'y' : y}
    p_prec = {'eval_mode' : 'exact'}
    It = time_domain_c.It_SingleContour_C(Psi, p_phys, p_prec)

    dt = 1e-2
    taus = np.geomspace(5e-1, 50, 1000)

    deriv = lambda I, t: 0.5*(I.eval_It(t+dt)-I.eval_It(t-dt))/dt
    
    ax.plot(taus, deriv(It, taus), label='%d SIS' % len(Psi.p_phys['lenses']), c='C0')
    ax.set_xscale('log')
    
    It_SIS = None
    if SIS is True:
        It_SIS = time_domain_c.It_AnalyticSIS_C(p_phys, p_prec)
        ax.plot(taus, deriv(It_SIS, taus), label='SIS', alpha=0.5, c='C1')
    
    It_CIS = None
    if CIS is True:
        Psi_CIS = lenses.Psi_CIS({'psi0':1, 'rc':rc})
        It_CIS = time_domain_c.It_SingleContour_C(Psi_CIS, p_phys, p_prec)
        ax.plot(taus, deriv(It_CIS, taus), label='CIS', alpha=0.5, c='C2')
    
    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$G(\tau)$')
    ax.set_xlim([taus[0], taus[-1]])
    ax.legend(loc='best')
    
def plot_Fw(ax, Psi, y, SIS=False, CIS=False, rc=0.1):
    p_phys = {'y' : y}
    p_prec = {'tmin':1e-2, 'tmax':1e5, 'Nt':5000, 'sampling':'log'}
    
    It_MultiSIS = time_domain_c.It_SingleContour_C(Psi, p_phys, p_prec)
    
    if SIS is True:
        It_SIS = time_domain_c.It_AnalyticSIS_C(p_phys, p_prec)
    
    if CIS is True:
        Psi_CIS = lenses.Psi_CIS({'psi0':1, 'rc':rc})
        It_CIS = time_domain_c.It_SingleContour_C(Psi_CIS, p_phys, p_prec)
    
    ## ============================================================================
    
    p_prec = {'fmin':1e-2, 'fmax':1e2 }
    
    Fw_MultiSIS = freq_domain_c.Fw_SL(It_MultiSIS, p_prec=p_prec)
    
    if SIS is True:
        Fw_SIS = freq_domain_c.Fw_SL(It_SIS, p_prec=p_prec)
    if CIS is True:
        Fw_CIS = freq_domain_c.Fw_SL(It_CIS, p_prec=p_prec)
    
    ## ============================================================================
    
    ws = Fw_MultiSIS.w_grid
    
    ax.plot(ws, np.abs(Fw_MultiSIS.Fw_grid), zorder=10, label='%d SIS' % len(Psi.p_phys['lenses']), c='C0')
    if SIS is True:
        ax.plot(ws, np.abs(Fw_SIS.Fw_grid), alpha=0.5, label='SIS', c='C1')
    if CIS is True:
        ax.plot(ws, np.abs(Fw_CIS.Fw_grid), alpha=0.5, label='CIS', c='C2')
    
    ax.grid(alpha=0.5)
    ax.set_xlim([ws[0], ws[-1]])
    ax.set_xscale('log')
    ax.set_ylabel('$|F(w)|$')
    ax.set_xlabel('$w$')
    ax.legend(loc='best')

def plot_all(n_sublenses, fname, y, seed=None):
        
    Psi, seed = create_lens(n=n_sublenses, seed=seed)
    
    ## ------------------------------
    fig_size = (16, 4)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, \
                                gridspec_kw={'hspace': 0, \
                                             'width_ratios': [1, 1, 1, 1]})
    
    plot_lenses(ax1, Psi, y, seed=seed)
    plot_It(ax2, Psi, y, SIS=True, CIS=True)
    plot_Gt(ax3, Psi, y, SIS=False, CIS=True)
    plot_Fw(ax4, Psi, y, SIS=True, CIS=True)
    
    fig.set_size_inches(fig_size)
    fig.tight_layout()
    fig.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close(fig)

## ============================================================================

if __name__ == '__main__':
    exit()
    
    ## Interesting seeds:
    ## * 3646978
    ## * 5396985
    ## * 7899859
    ## * 7007333
    ## * 3748079
    ## * 5809618
    ## * 9925854
    
    n = 10
    y = 1.6
    fname = 'random.pdf'
    seed = None
    
    plot_all(n, fname, y, seed)
    exit()
    
    # -----------------------------------------
    
    seeds = [3646978, 5396985, 7899859, 7007333, 3748079, 5809618, 9925854]
    for seed in seeds:
        fname = 'plots/varSeed_%d.pdf' % seed
        plot_all(n_sublenses=3, y=1.6, fname=fname, seed=seed)
    call("pdfunite plots/varSeed* plots/all_varSeed.pdf", shell=True)
    
    ys = [1.2, 1.3, 1.6, 2.0, 2.4, 3.2, 4.8]
    for y in ys:
        fname = 'plots/varY_%.1e.pdf' % y
        plot_all(n_sublenses=3, y=y, fname=fname, seed=3646978)
    call("pdfunite plots/varY* plots/all_varY.pdf", shell=True)
        
    ns = [1, 2, 4, 6, 9, 12, 20, 50, 100]
    for n in ns:
        fname = 'plots/varN_%.3d.pdf' % n
        plot_all(n_sublenses=n, y=1.6, fname=fname, seed=3646978)
    call("pdfunite plots/varN* plots/all_varN.pdf", shell=True)
