import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from glow import lenses, time_domain_c, freq_domain_c

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

## =====================================================================

def create_composite_lens(n, R=1):
    r = R*np.sqrt(np.random.rand(n))
    th = np.random.rand(n)*2*np.pi

    # lenses uniformly distributed in a circle
    xc1, xc2 = r*np.cos(th), r*np.sin(th)
    xc1 -= np.sum(xc1)/n
    xc2 -= np.sum(xc2)/n

    # create list of lenses
    lens_list = []
    for x1, x2 in zip(xc1, xc2):
        p_phys = {'psi0':1./n, 'xc1':x1, 'xc2':x2}
        lens_list.append(lenses.Psi_offcenterSIS(p_phys))

    return lenses.CombinedLens({'lenses' : lens_list})

def plot_decorate(It, ax, markersize=6, has_images=True, has_source=True, has_legend=True, has_ax=True):
    legend_elements = []

    y = It.y

    # plot the location of the images
    if has_images:
        c = {'max':'grey', 'min':'white', 'saddle':'red'}
        marker_kwargs = {'marker': 'o',
                         'ms': markersize,
                         'ls': '',
                         'c': 'black',
                         'markerfacecolor': 'black'}

        for i, p in enumerate(It.p_crits):
            if p['type'] == 'sing/cusp max':
                continue
            else:
                marker_kwargs['markerfacecolor'] = c[p['type']]
                ax.plot(p['x1'], p['x2'], **marker_kwargs, zorder=3)

        marker_kwargs['markerfacecolor'] = c['min']
        legend_elements.append(Line2D([], [], label='Minimum', **marker_kwargs))

        # ~ marker_kwargs['markerfacecolor'] = c['max']
        # ~ legend_elements.append(Line2D([], [], label='Maximum', **marker_kwargs))

        # ~ marker_kwargs['markerfacecolor'] = c['saddle']
        # ~ legend_elements.append(Line2D([], [], label='Saddle', **marker_kwargs))

    # plot the location of the source
    if has_source is True:
        marker_kwargs = {'marker': '*',
                         'ms': markersize*1.5,
                         'ls': '',
                         'c' : 'black',
                         'markerfacecolor': 'white'}
        ax.plot(y, 0, **marker_kwargs, zorder=3)
        legend_elements.append(Line2D([], [], label='Source', **marker_kwargs))

    # include the legend
    if has_legend is True:
        ax.legend(borderaxespad=0.05, handles=legend_elements, loc=1)

    # add the axes
    if has_ax is True:
        ax.axhline(0, c='white', lw=0.5, ls='--')
        ax.axvline(0, c='white', lw=0.5, ls='--')

    return ax

def plot_contours(It, ax):
    n_ctr = 50
    ts = np.linspace(3e-2, 5e0, n_ctr)

    ctrs = It.get_contour(ts, 200)
    for ctr in ctrs:
        ax.plot(ctr['x1'], ctr['x2'], c='gray', alpha=0.5, lw=0.5)

def plot_Sigma(It, ax):
    x1s = np.linspace(-1.5, 2.3, 200)
    x2s = np.linspace(-1.5, 1.5, 200)
    X1, X2 = np.meshgrid(x1s, x2s)

    Psi = It.lens_to_c
    lens_list = Psi.p_phys['lenses']
    Sigma  = np.zeros_like(X1)
    for l in lens_list:
        xc1 = l.p_phys['xc1']
        xc2 = l.p_phys['xc2']

        r = np.sqrt((X1-xc1)**2 + (X2-xc2)**2)
        Sigma += 1/r

    c = ax.pcolor(X1, X2, Sigma,
                  norm=colors.LogNorm(vmin=Sigma.min(), vmax=Sigma.max()/4),
                  shading='auto',
                  rasterized=True)

    return X1, X2

def plot_additional_contour(It, ax, idx, color='white', id_label=0):
    lens_list = It.lens_to_c.p_phys['lenses']
    Psi = lens_list[idx]

    xc1 = Psi.p_phys['xc1']
    xc2 = Psi.p_phys['xc2']

    tau = It.phi_Fermat(xc1, xc2)
    contour = It.get_contour(tau, n_points=1000)

    ax.plot(contour['x1'], contour['x2'], c=color)

    props = {'fill': True,
             'linestyle': '-',
             'edgecolor': 'white',
             'alpha': 0.5,
             'boxstyle': 'round',
             'pad': 0.3}

    ax.text(xc1+0.15, xc2, "\\textbf{%d}" % id_label,
            bbox=props, color=color)

    return tau

## =====================================================================

if __name__ == '__main__':
    # ~ np.random.seed(1893)
    np.random.seed(185)

    Psi = create_composite_lens(10, R=1.4)
    It = time_domain_c.It_SingleContour_C(Psi, 1.2, {'eval_mode':'exact'})

    ## *****************************************************************

    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(2, 3, hspace=0.02, wspace=0.05)

    ## plot Sigma and the contours
    ## -----------------------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])

    X1, X2 = plot_Sigma(It, ax)
    plot_decorate(It, ax, markersize=8, has_ax=False)
    plot_contours(It, ax)

    idxs = [7, 0, 1]
    colors = ['coral', 'limegreen', 'gold']
    tau_c = []
    for i, (idx, color) in enumerate(zip(idxs, colors)):
        tau = plot_additional_contour(It, ax, idx, color=color, id_label=i)
        tau_c.append(tau)

    ax.set_title('Projected mass density $\\Sigma(\\boldsymbol{x})$')
    ax.set_xlim([-1., X1.max()])
    ax.set_ylim([X2.min(), X2.max()])

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ## -----------------------------------------------------------------

    ## plot It and Gt
    ## -----------------------------------------------------------------
    ax_It = fig.add_subplot(gs[0, 1])
    ax_Gt = fig.add_subplot(gs[1, 1])

    ts = np.geomspace(1e-1, 3.5, 3000)
    Its = It(ts)/2./np.pi
    Gts = It.eval_Gt(ts, dt=1e-3)

    colors = ['orangered', 'green', 'goldenrod']
    for dat, ax in zip([Its, Gts], [ax_It, ax_Gt]):
        ax.plot(ts, dat, c='#36454F', zorder=10)
        ax.grid(alpha=0.5)

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_xlim([ts[0], ts[-1]])

        for tau, color in zip(tau_c, colors):
            ax.axvline(tau, c=color)

    ax_It.xaxis.set_ticklabels([])

    ax_Gt.set_ylim(ymax=0.29)

    ## add the text labels
    props = {'fill': True,
             'linestyle': '-',
             'facecolor': 'white',
             'alpha': 0.5,
             'boxstyle': 'round',
             'pad': 0.3}

    for i, (tau, color) in enumerate(zip(tau_c, colors)):
        ax.text(tau+0.12, 0.225, "\\textbf{%d}" % i,
                bbox=props, color=color)

    ax_Gt.set_xlabel('$\\tau$')
    ax_Gt.set_ylabel('$G(\\tau)$')
    ax_It.set_ylabel('$I(\\tau)/2\\pi$')
    ## -----------------------------------------------------------------

    ## *****************************************************************
    print('Saving figure...')
    fig.savefig('WL_example.pdf', bbox_inches='tight')
    print('Done')
