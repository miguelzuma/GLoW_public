import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glow import lenses, time_domain_c, freq_domain_c, freq_domain

fontsize = 12

plt.rc('axes',  labelsize=fontsize, titlesize=16)
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('legend', fontsize=fontsize*0.9)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=fontsize*0.9)

ch_grey = '#36454F'

# ----------------------------------------------------------------------

def plot_sing_reg():
    p_prec = {'FFT method':'standard',
              'eval_mode':'interpolate'}

    y = 0.3
    name = 'freq_std_singreg_SL'

    It = time_domain_c.It_AnalyticSIS_C(y, {'Nt':5000})
    Fw = freq_domain_c.Fw_FFT_C(It, p_prec)

    Fw2 = freq_domain.Fw_FFT_OldReg(It)

    # ======================================================================

    ws = Fw.w_grid[1:]
    ts = np.geomspace(1e-2, 1e3, 1000)

    fig, axs = plt.subplots(ncols=2, figsize=(9, 4), gridspec_kw={'wspace':0.1})

    # ----------------------------------------------------------------------
    ax = axs[0]
    ax.plot(ts, It(ts)/2./np.pi, label='full = sing. + reg.',   c='grey')
    ax.plot(ts, Fw.eval_It_sing(ts)/2./np.pi, label='singular', c='royalblue', ls='--')
    ax.plot(ts, Fw.eval_It_reg(ts)/2./np.pi, label='regular',   c='orangered')
    ax.plot(ts, Fw2.eval_It_sing(ts)/2./np.pi, label='standard GO',   c='green', alpha=0.5, ls='-.')

    ax.grid(alpha=0.5)
    ax.set_xlim([ts[0], ts[-1]])
    ax.legend(loc=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$I(\tau)/2\pi$')

    # ----------------------------------------------------------------------
    ax = axs[1]
    ax.plot(ws, np.abs(Fw(ws)),              c='grey')
    ax.plot(ws, np.abs(Fw.eval_Fw_sing(ws)), c='royalblue', ls='--')
    ax.plot(ws, np.abs(Fw.eval_Fw_reg(ws)),  c='orangered')
    ax.plot(ws, np.abs(Fw2.eval_Fw_sing(ws)),  c='green', alpha=0.5, ls='-.')

    ax.grid(alpha=0.5)
    ax.set_xlim([ws[0], ws[-1]])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xscale('log')
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$|F(w)|$')

    # ----------------------------------------------------------------------
    return fig, ax

## ========================================================================

if __name__ == '__main__':
    fig, ax = plot_sing_reg()

    # add arrow
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="#36454F")

    alen = 0.13
    x0 = 0.45
    y0 = 0.18

    arrow = patches.FancyArrowPatch((x0, y0), (x0+alen, y0), transform=fig.transFigure,
                                    connectionstyle="arc3,rad=-.2", **kw)
    fig.patches.append(arrow)

    fig.text(x0+alen-0.03, y0+0.05, 'Fourier transform', transform=fig.transFigure)

    # save
    fig.savefig('regularization_example.pdf', bbox_inches='tight')
