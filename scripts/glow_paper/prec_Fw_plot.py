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

def compare_FFT(Fw0, ws=None, ax0=None, ax1=None):
    y = Fw0.It.y
    Psi = Fw0.It.lens

    It = time_domain_c.It_MultiContour_C(Psi, y)
    Fw_MC_FFT = freq_domain_c.Fw_FFT_C(It)

    Fw_MC_DirectFT = freq_domain_c.Fw_DirectFT_C(It)

    It = time_domain_c.It_SingleIntegral_C(Psi, y)
    Fw_SI_FFT = freq_domain_c.Fw_FFT_C(It)

    Fw0s = Fw0(ws)
    dRel = lambda w, F: np.abs(F(w)-Fw0s)/np.abs(Fw0s)

    ## ---------------------------------------------------
    if (ax0 is None) or (ax1 is None):
        fig, (ax0, ax1) = plt.subplots(2)

    ws_dense = np.geomspace(ws[0], ws[-1], ws.size*10)
    ax0.plot(ws_dense, np.abs(Fw0(ws_dense)), c=ch_grey)
    ax0.set_xscale('log')

    ax1.plot(ws, dRel(ws, Fw_MC_FFT), label='\\texttt{MultiContour}+\\texttt{FFT}')
    ax1.plot(ws, dRel(ws, Fw_SI_FFT), label='\\texttt{SingleIntegral}+\\texttt{FFT}')
    ax1.plot(ws, dRel(ws, Fw_MC_DirectFT), label='\\texttt{MultiContour}+\\texttt{DirectFT}', alpha=0.7, c='grey')
    ax1.axhline(1e-3, c='k', ls='--')
    ## ---------------------------------------------------

    ## ---------------------------------------------------
    name = Psi.p_phys['name']
    name = 'Point lens' if name=='point lens' else name

    ax0.set_title('%s ($y=%g$)' % (name, y))
    ax0.set_ylabel('$|F(w)|$')

    ax1.set_yscale('log')
    ax1.legend(loc=2)
    ax1.set_xlabel('$w$')
    ax1.set_ylabel('$|\\Delta F/F|$')
    ax1.set_ylim([1e-6, 1e-1])
    ## ---------------------------------------------------

    for ax in (ax0, ax1):
        ax.set_xscale('log')
        ax.grid(alpha=0.5)
        ax.set_xlim([ws[0], ws[-1]])

    ax0.set_xticklabels([])

## =====================================================================

if __name__ == '__main__':
    fnames = ['prec_Fw_SIS', 'prec_Fw_PL']
    methods = [freq_domain_c.Fw_SemiAnalyticSIS_C, freq_domain_c.Fw_AnalyticPointLens_C]

    for fname, method in zip(fnames, methods):
        fig = plt.figure(figsize=(11, 5))

        gs = gridspec.GridSpec(2, 2,
                               hspace=0.02,
                               wspace=0.07,
                               width_ratios=[1, 1],
                               height_ratios=[1, 2])

        ax00 = fig.add_subplot(gs[0, 0])
        ax10 = fig.add_subplot(gs[1, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax11 = fig.add_subplot(gs[1, 1])

        ws = np.geomspace(1e-2, 1e2, 1000)

        Fw0 = method(y=0.3)
        compare_FFT(Fw0, ws, ax00, ax10)

        Fw0 = method(y=1.2)
        compare_FFT(Fw0, ws, ax01, ax11)

        for ax in (ax01, ax11):
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        ## *****************************************************************
        print('Saving figure...')
        fig.savefig('%s.pdf' % fname, bbox_inches='tight')
        print('Done')
