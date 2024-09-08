import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from glow import lenses, time_domain_c, freq_domain_c

fontsize = 16
markersize = 10

plt.rc('lines', linewidth=2.5)
plt.rc('axes',  labelsize=fontsize, titlesize=16)
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('legend', fontsize=14)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

## =====================================================================

def benchmark_lens():
    # benchmark lens for 2d SL
    xs = [[0.3, 0], [-0.6, 0.3], [0.3, -0.3], [0, 0]]
    psi0 = 1./len(xs)
    rc = 0.05
    Psis = [lenses.Psi_offcenterCIS({'psi0':psi0, 'rc':rc, 'xc1':x[0], 'xc2':x[1]}) for x in xs]
    Psi = lenses.CombinedLens({'lenses':Psis})
    return Psi

def plot_phi(It, ax, has_bar=True):
    Nx = 100
    xmax = 1.6
    xmin = -xmax
    x1s = np.linspace(xmin, xmax, Nx)
    x2s = np.linspace(xmin, xmax, Nx)
    x1_grid, x2_grid = np.meshgrid(x1s, x2s)

    dat = np.log(It.phi_Fermat(x1_grid, x2_grid)+1e-14)

    # alternative if we want derivatives
    # ~ d1 = Psi.dphi_Fermat_dx1(x1_grid, x2_grid, y)
    # ~ d2 = Psi.dphi_Fermat_dx2(x1_grid, x2_grid, y)
    # ~ dat = np.log(d1**2+d2**2)

    ## -----------------------------------------------------------------
    # ~ cmap = matplotlib.colormaps.get_cmap('plasma')
    # ~ cmap = matplotlib.colormaps.get_cmap('viridis')
    # ~ cmap = matplotlib.colormaps.get_cmap('inferno')
    cmap = matplotlib.colormaps.get_cmap('magma')
    # ~ cmap = matplotlib.colormaps.get_cmap('cividis')

    vmin = -6
    vmax = -1

    im = ax.imshow(dat, \
                   interpolation='bilinear', \
                   origin='lower',
                   cmap=cmap, \
                   extent=(x1s[0], x1s[-1], x2s[0], x2s[-1]), \
                   vmin=vmin, vmax=vmax, \
                   alpha=0.8)

    ax.axhline(0, lw=0.5, c='grey', alpha=0.5)
    ax.axvline(0, lw=0.5, c='grey', alpha=0.5)

    ## add colorbar
    if has_bar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

        # these are matplotlib.patch.Patch properties
        props = dict(fill=False, linestyle='', facecolor='white')
        ax.text(0.99, 1.08, r"$\log\phi$",
                 transform=ax.transAxes, fontsize=fontsize,
                 verticalalignment='top', bbox=props)

    return ax

def plot_contours(ax, contours, color='black', alpha=0.8, lw=0.3, ls='-'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for c in contours:
        for x1, x2 in zip(c['x1'], c['x2']):
            ax.plot(x1, x2, c=color, alpha=alpha, lw=lw, ls=ls)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax

def plot_saddle_cnt(It, ax, dx=1e-3):
    saddles = [p for p in It.p_centers if p['type'][:6] == 'saddle']

    xmax = 0
    for s in saddles:
        th = s['alpha_out']
        for R in (dx, -dx):
            x1 = s['x10'] + R*np.cos(th)
            x2 = s['x20'] + R*np.sin(th)

            cnt = It.get_contour_x1x2(x1, x2, sigmaf=50, n_points=200)
            ax.plot(cnt['x1'], cnt['x2'], lw=1.5, c='darkred')
            xmax = max(np.max(np.abs(cnt['x1'])), np.max(np.abs(cnt['x2'])), xmax)

    return xmax, ax

def plot_decorate(It, ax, markersize=6, has_images=True, has_source=True, has_lenses=True, has_legend=True):
    legend_elements = []

    # plot the location of the images
    if has_images:
        c = {'max':'grey', 'min':'white', 'saddle':'red'}
        marker_kwargs = {'marker' : 'o',
                     'ms' : markersize,
                     'ls' : '',
                     'c' : 'black',
                     'markerfacecolor' : 'black'}

        for i, p in enumerate(It.p_crits):
            if p['type'] == 'sing/cusp max':
                continue
            else:
                marker_kwargs['markerfacecolor'] = c[p['type']]
                ax.plot(p['x1'], p['x2'], **marker_kwargs)

        marker_kwargs['markerfacecolor'] = c['min']
        legend_elements.append(Line2D([], [], label='Minimum', **marker_kwargs))

        marker_kwargs['markerfacecolor'] = c['max']
        legend_elements.append(Line2D([], [], label='Maximum', **marker_kwargs))

        marker_kwargs['markerfacecolor'] = c['saddle']
        legend_elements.append(Line2D([], [], label='Saddle', **marker_kwargs))

    # plot the location of the source
    if has_source is True:
        y = It.y
        marker_kwargs = {'marker':'*', 'ms':markersize*1.5, 'ls':'', 'c' : 'black'}
        ax.plot(y, 0, **marker_kwargs)
        legend_elements.append(Line2D([], [], label='Source', **marker_kwargs))

    # plot the location of the sublenses
    if has_lenses is True:
        marker_kwargs = {'marker':'x',
                         'ms':markersize,
                         'markeredgewidth':2,
                         'ls':'',
                         'c' : 'black',
                         'alpha':0.8}
        try:
            for lens in Psi.p_phys['lenses']:
                ax.plot(lens.p_phys['xc1'], lens.p_phys['xc2'], **marker_kwargs)
        except KeyError:
            ax.plot(0, 0, **marker_kwargs)

        legend_elements.append(Line2D([], [], label='Lens', **marker_kwargs))

    # include the legend
    if has_legend is True:
        ax.legend(borderaxespad=0.05, handles=legend_elements, loc=2)

    return ax

def plot_all_contours(It, ax, markersize=6, has_legend=True):
    tau_min = 1e-4
    tau_max = 5
    taus = np.geomspace(tau_min, tau_max, 100)

    contours = It.get_contour(taus, 1000)
    ax = plot_contours(ax, contours, lw=0.5, alpha=0.3)
    xmax, ax = plot_saddle_cnt(It, ax)

    plot_decorate(It, ax, markersize=markersize, has_legend=has_legend)

    return ax

def plot_It(It, ax):
    ts = np.linspace(-0.05, 0.8, 500)
    ax.plot(ts, It(ts)/2/np.pi, color='#36454F') # charcoal grey
    ax.set_xlim([ts[0], ts[-1]])
    ax.grid(alpha=0.5)
    ax.set_box_aspect(1)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

def plot_Fw(It, ax):
    p_prec = It.p_prec.copy()
    p_prec['eval_mode'] = 'interpolate'
    It2 = time_domain_c.It_MultiContour_C(It.lens, It.y, p_prec)

    Fw = freq_domain_c.Fw_DirectFT_C(It2)
    ws = np.geomspace(1e-1, 1e3, 2000)

    ax.plot(ws, np.abs(Fw(ws)), color='#2F4F4F') # slate grey
    ax.set_xlim([ws[0], ws[-1]])
    ax.set_xscale('log')
    ax.grid(alpha=0.5)
    ax.set_box_aspect(1)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

## =====================================================================

if __name__ == '__main__':
    ys = [0.35, 0.15, 0.085, 0.055, 0.015]

    Psi = benchmark_lens()

    fig = plt.figure(figsize=(6*3, 6*len(ys)))
    gs = gridspec.GridSpec(len(ys), 3, hspace=0., wspace=0.2)

    for i, y in enumerate(ys):
        print("(%d/%d) y=%g" % ((i+1), len(ys), y))

        It = time_domain_c.It_MultiContour_C(Psi, y, p_prec={'eval_mode':'exact'})
        ax  = fig.add_subplot(gs[i, 0])

        plot_phi(It, ax, has_bar=True)
        plot_all_contours(It, ax, markersize=markersize, has_legend=(True if i==0 else False))

        ax_It = fig.add_subplot(gs[i, 1])
        plot_It(It, ax_It)

        ax_Fw = fig.add_subplot(gs[i, 2])
        plot_Fw(It, ax_Fw)

        if i != (len(ys)-1):
            ax.set_xticklabels([])
            ax_It.set_xticklabels([])
            ax_Fw.set_xticklabels([])
        else:
            ax.set_xlabel('$x_1$')
            ax_It.set_xlabel('$\\tau$')
            ax_Fw.set_xlabel('$w$')

        ax.set_ylabel('$x_2$')

        ax.text(0.03, 0.035, '$y=%g$' % y, transform=ax.transAxes, fontsize=fontsize,
                bbox={'alpha':0.5, 'ec':'white', 'fc':'white'})
        ax.text(0.03, 0.12, '4 CISs', transform=ax.transAxes, fontsize=fontsize,
                bbox={'alpha':0.5, 'ec':'white', 'fc':'white'})
        ax_It.text(0.85, 0.88, '$\\frac{I(\\tau)}{2\\pi}$', transform=ax_It.transAxes, fontsize=24,
                   bbox={'alpha':0.5, 'ec':'white', 'fc':'white'})
        ax_Fw.text(0.03, 0.90, '$|F(w)|$', transform=ax_Fw.transAxes, fontsize=18,
                   bbox={'alpha':0.5, 'ec':'white', 'fc':'white'})

    print('Saving figure...')
    fig.savefig('SL_examples.pdf', bbox_inches='tight')
    print('Done')
