import numpy as np
import mpmath
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from glow import freq_domain, freq_domain_c

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

## =====================================================================

def get_contour_plot(xx, yy, zz, ax=None, cmap='magma', vmin=None, vmax=None, label=''):
    cmap = matplotlib.colormaps.get_cmap(cmap)

    if ax is None:
        fig, ax = plt.subplots()

    # create a background
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                         hatch='////', facecolor='lightgray', edgecolor='seagreen', zorder=-1)
    ax.add_patch(rect)

    im = ax.pcolormesh(xx, yy, zz, cmap=cmap, linewidth=0, rasterized=True, vmin=vmin, vmax=vmax)

    ## add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label)

    ## configure plot
    ax.set_xlabel('$w$')
    ax.set_ylabel('$y$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax

def add_approximation_regions(ax, ws, ys):
    color = 'seagreen'
    color2 = 'white'

    lw = 1.5
    lw2 = 2.5

    # text properties
    props = {'fill': True,
             'linestyle': '-',
             'facecolor': color,
             'edgecolor': color2,
             'alpha': 1.,
             'boxstyle': 'round',
             'pad': 0.3}

    # Temme region: w > 8
    ax.axvline(8, c=color2, ls='-', lw=lw2, zorder=3)
    ax.axvline(8, c=color, ls='--', lw=lw, zorder=3)
    ws = ws[ws < 8]

    ax.text(14, 1.5e-2, "\\textbf{I}", bbox=props, color=color2)

    # Large z region: y > sqrt(44/w)
    ax.plot(ws, np.sqrt(44/ws), c=color2, ls='-', lw=lw2)
    ax.plot(ws, np.sqrt(44/ws), c=color, ls='--', lw=lw)

    x0 = 4.3
    ax.text(x0, 5e2, "\\textbf{II}", bbox=props, color=color2, ha='right')

    # Series region: y < sqrt(10/w)
    ax.plot(ws, np.sqrt(10/ws), c=color2, ls='-', lw=lw2)
    ax.plot(ws, np.sqrt(10/ws), c=color, ls='--', lw=lw)

    ax.text(x0, 8.5, "\\textbf{III}", bbox=props, color=color2, ha='right')

    # recurrence region: else
    ax.text(x0, 1.5e-2, "\\textbf{IV}", bbox=props, color=color2, ha='right')

def get_Fw_C_grid(ys, ws, fname='Fw_C_grid', recompute=False):
    if recompute is False:
        try:
            Fws = np.load(fname+'.npy')
            return Fws
        except FileNotFoundError:
            pass

    Fws = [freq_domain_c.Fw_AnalyticPointLens_C(y)(ws) for y in ys]
    Fws = np.array(Fws)
    np.save(fname+'.npy', Fws)

    return Fws

def get_Fw_Py_grid(ys, ws, fname='Fw_Py_grid', recompute=False):
    if recompute is False:
        try:
            Fws = np.load(fname+'.npy')
            return Fws
        except FileNotFoundError:
            pass

    Fws = []
    for y in ys:
        Fw = freq_domain.Fw_AnalyticPointLens(y)
        Fws_tmp = []

        # if the hypergeometric function fails, evaluate w one by one
        # and delete the problematic ones
        try:
            Fws_tmp = Fw(ws)
        except mpmath.libmp.libhyper.NoConvergence:
            for w in ws:
                try:
                    result = Fw(w)
                except mpmath.libmp.libhyper.NoConvergence:
                    result = np.nan
                Fws_tmp.append(result)

        Fws.append(Fws_tmp)

    Fws = np.array(Fws)
    np.save(fname+'.npy', Fws)

    return Fws

def plot_errors(ws, ys, ax=None):
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    print('Computing C grid\n'+'-'*20, flush=True)
    Fws_C = get_Fw_C_grid(ys, ws)
    print('Done\n')

    print('Computing Python grid\n'+'-'*20, flush=True)
    Fws_Py = get_Fw_Py_grid(ys, ws)
    print('Done\n')

    ww, yy = np.meshgrid(ws, ys)
    dFs = np.abs(Fws_C/(Fws_Py+1e-14)-1)

    print('dF max = %g\n' % np.max(dFs[~np.isnan(dFs)]))

    ax = get_contour_plot(ww, yy, np.log10(dFs), ax=ax, vmin=-10, label=r'$\log_{10}|\Delta F/F|$')
    add_approximation_regions(ax, ws, ys)

    return ax

def plot_Fw(ws, ys, ax=None):
    print('Computing C grid (dense)\n'+'-'*20, flush=True)
    Fws_C = get_Fw_C_grid(ys, ws, fname='Fw_C_dense_grid.npy')
    print('Done\n')

    ww, yy = np.meshgrid(ws, ys)
    ax = get_contour_plot(ww, yy, np.log10(np.abs(Fws_C-1)), ax=ax, cmap='viridis', label=r'$\log_{10}|F(w)-1|$', vmin=-3)

    return ax

## =====================================================================

if __name__ == '__main__':
    ys = np.geomspace(1e-2, 1e3, 200)
    ws = np.geomspace(1e-4, 1e4, 400)

    fig, ax = plt.subplots()
    plot_errors(ws, ys, ax)

    ## *****************************************************************
    print('Saving figure...')
    fig.savefig('PL_approx.pdf', bbox_inches='tight')
    print('Done')
    ## *****************************************************************

    exit()

    fig, ax = plt.subplots()

    ys = np.geomspace(ys[0], ys[-1], ys.size*10)
    ws = np.geomspace(ws[0], ws[-1], ws.size*10)
    plot_Fw(ws, ys, ax)

    ## *****************************************************************
    print('Saving figure...')
    fig.savefig('PL_Fw.pdf', bbox_inches='tight')
    print('Done')
