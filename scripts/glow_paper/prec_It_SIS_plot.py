import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

from glow import lenses, time_domain_c, freq_domain_c

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

ch_grey = '#36454F'

## =====================================================================

def get_Cprec_tol(tol):
    ode_dic  = {'epsabs':tol/10, 'epsrel':0}
    root_dic = {'epsabs':tol/10, 'epsrel':tol/10}

    ## ----------------------------------------------
    Cprec = {}
    Cprec['sc_intContourStd'] = ode_dic
    Cprec['sc_intContourRob'] = ode_dic
    Cprec['sc_intdRdtau']     = ode_dic
    Cprec['sc_findRtau'] = root_dic
    Cprec['sc_findRtau_bracket'] = root_dic
    Cprec['sc_intContour_tol_brack'] = tol

    Cprec['mc_intRtau']    = ode_dic
    Cprec['mc_intContour'] = ode_dic
    Cprec['mc_findRbracket'] = root_dic
    Cprec['mc_updCondODE_tol_brack'] = tol

    Cprec['as_eps_soft'] = 0
    Cprec['no_warnings'] = 1

    return Cprec

def compare_methods_It_SIS(y, ax0=None, ax1=None):
    p_prec = {'eval_mode':'exact'}

    It0 = time_domain_c.It_AnalyticSIS_C(y, p_prec)

    Psi = lenses.Psi_SIS()
    It_MC = time_domain_c.It_MultiContour_C(Psi, y, p_prec)
    It_SI = time_domain_c.It_SingleIntegral_C(Psi, y, p_prec)

    if y > 1:
        It_SC = time_domain_c.It_SingleContour_C(Psi, y, p_prec)

    ts = np.geomspace(1e-2, 2e2, 2000)
    It0s = It0(ts)

    dRel = lambda t, I: np.abs(I(t)/It0s-1)

    ## ---------------------------------------------------
    if (ax0 is None) or (ax1 is None):
        fig, (ax0, ax1) = plt.subplots(2)

    ax0.plot(ts, It0s/2/np.pi, c=ch_grey)
    ax0.set_xscale('log')
    ## ---------------------------------------------------

    ## ---------------------------------------------------
    ax1.plot(ts, dRel(ts, It_MC), label='\\texttt{MultiContour}')
    ax1.plot(ts, dRel(ts, It_SI), label='\\texttt{SingleIntegral}')

    if y > 1:
        ax1.plot(ts, dRel(ts, It_SC), label='\\texttt{SingleContour}')

    # higher precision for this one (careful, Cprec gets reset for each object)
    p_prec_hi = {'eval_mode':'exact', 'C_prec':get_Cprec_tol(1e-8)}
    It_MC_hi = time_domain_c.It_MultiContour_C(Psi, y, p_prec_hi)
    ax1.plot(ts, dRel(ts, It_MC_hi), label='\\texttt{MultiContour} (high prec.)', c='grey', ls='--', alpha=0.7)
    ## ---------------------------------------------------

    ax0.set_title('SIS ($y=%g$)' % y)
    ax0.set_ylabel('$I(\\tau)/2\\pi$')

    ax1.set_yscale('log')
    ax1.legend(loc=2, ncol=2)
    ax1.set_xlabel('$\\tau$')
    ax1.set_ylabel('$|\\Delta I/I|$')
    ax1.set_ylim([1e-13, 1e-1])

    ax1.axhline(1e-4, c='k', ls='--')
    ax1.axhline(1e-8, c='grey', ls='--')

    for ax in (ax0, ax1):
        ax.set_xscale('log')
        ax.grid(alpha=0.5)
        ax.set_xlim([ts[0], ts[-1]])

    ax0.set_xticklabels([])

## =====================================================================

if __name__ == '__main__':
    fig = plt.figure(figsize=(11, 5))

    gs = gridspec.GridSpec(2, 2,
                           hspace=0.02,
                           wspace=0.05,
                           width_ratios=[1, 1],
                           height_ratios=[1, 2])

    ax00 = fig.add_subplot(gs[0, 0])
    ax10 = fig.add_subplot(gs[1, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax11 = fig.add_subplot(gs[1, 1])

    compare_methods_It_SIS(0.3, ax00, ax10)
    compare_methods_It_SIS(1.2, ax01, ax11)

    for ax in (ax01, ax11):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

    ## *****************************************************************
    print('Saving figure...')
    fig.savefig('prec_It_SIS.pdf', bbox_inches='tight')
    print('Done')
