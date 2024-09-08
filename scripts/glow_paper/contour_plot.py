import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from glow import lenses, time_domain_c, freq_domain_c, wrapper

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

## =====================================================================

def plot_inner_contours(It, ax, colors=None, alpha=0.7, lw=0.9):
    Psi = It.lens_to_c
    centers = It.p_centers
    y = It.y

    inner_centers = []
    for ctr in centers:
        if ctr['is_init_birthdeath'] == 0:
            external_center = ctr
        else:
            inner_centers.append(ctr)

    if colors is None:
        colors = ['C%d' % i for i in range(len(inner_centers))]

    eps = 5e-4
    for i, (ctr, color) in enumerate(zip(inner_centers, colors)):
        tau_min = ctr['tau_birth'] + eps
        tau_max = ctr['tau_death'] - eps

        # fine tune number of contours
        if (i==0) or (i==1):
            n_ctr = 25
        elif i==5:
            n_ctr = 25
        else:
            n_ctr = 7

        taus = np.linspace(tau_min, tau_max, n_ctr)

        # must include the minimum
        tmp = [centers[0], ctr]
        contours = wrapper.pyGetMultiContour(taus, tmp, y, Psi, n_points=2000)

        for contour in contours:
            try:
                ax.plot(contour['x1'][-1], contour['x2'][-1], c=color, alpha=alpha, lw=lw)
            except IndexError:
                pass

    return inner_centers

def plot_outer_contours(It, ax, color='gray', alpha=0.7, lw=0.8):
    Psi = It.lens_to_c
    centers = It.p_centers
    y = It.y

    inner_centers = []
    for ctr in centers:
        if ctr['is_init_birthdeath'] == 0:
            external_center = ctr
        else:
            inner_centers.append(ctr)

    eps = 5e-4
    tau_min = external_center['tau_birth'] + eps
    tau_max = 10*tau_min
    n_ctr = 30

    taus = np.geomspace(tau_min, tau_max, n_ctr)

    # must include the minimum
    tmp = [centers[0], external_center]
    contours = wrapper.pyGetMultiContour(taus, tmp, y, Psi, n_points=2000)

    for contour in contours:
        try:
            ax.plot(contour['x1'][-1], contour['x2'][-1], c='grey', alpha=alpha, lw=lw)
        except IndexError:
            pass

    return external_center

def add_regions_rectangles(ax, colors, color_out, inner_centers, external_center, tau_f):
    # rectangle base properties
    rect_kw = {'width': 2,
               'height': 5,
               'linewidth': 1.2,
               'edgecolor': 'none',
               'facecolor': 'none',
               'alpha': 0.5}

    # plot region and its border
    def get_rects(tmin, tmax, color, height, rect_kw):
        rect_kw_updt = rect_kw.copy()

        rect_kw_updt['width'] = tmax-tmin
        rect_kw_updt['facecolor'] = color
        rect_kw_updt['edgecolor'] = 'none'
        rect_kw_updt['alpha'] = 0.5
        rect_kw_updt['height'] = height

        rect  = patches.Rectangle((tmin, 0), **rect_kw_updt)

        rect_kw_updt['facecolor'] = 'none'
        rect_kw_updt['edgecolor'] = color
        rect_kw_updt['alpha'] = 1
        rect2 = patches.Rectangle((tmin, 0), **rect_kw_updt, zorder=5)

        return rect, rect2

    for i, (ctr, color) in enumerate(zip(inner_centers, colors)):
        rect, rect2 = get_rects(ctr['tau_birth'], ctr['tau_death'], color, 13-i, rect_kw)
        ax.add_patch(rect)
        ax.add_patch(rect2)

    rect, rect2 = get_rects(external_center['tau_birth'], tau_f, color_out, 13-i-1, rect_kw)
    ax.add_patch(rect)
    ax.add_patch(rect2)

def add_regions_arrows(ax, colors, color_out, inner_centers, external_center, tau_f):
    dy = 1.
    y0 = 6.5
    ctr = inner_centers[0]

    dx = 0.0007

    arrowprops = {'lw': 2,
                  'arrowstyle': '|-|',
                  'shrinkA': 0,
                  'shrinkB': 0}

    ylim = ax.get_ylim()

    # plot inner regions
    for i, (ctr, color) in enumerate(zip(inner_centers, colors)):
        height = y0 - i*dy
        x0, x1 = np.sort([ctr['tau_birth'], ctr['tau_death']])

        ax.axvline(x0+dx, c=color, ls='--', lw=1)
        ax.axvline(x1-dx, c=color, ls='--', lw=1)
        ax.fill_betweenx(ylim, x0, x1, color=color, alpha=0.3, zorder=-1)

        arrowprops['edgecolor'] = color
        ax.annotate('', xy=(x0, height), xytext=(x1, height),
                    arrowprops=arrowprops)

    # plot outer region
    height = y0 - (i+1)*dy
    x0 = external_center['tau_birth']
    x1 = tau_f

    ax.fill_betweenx(ylim, x0, x1, color=color_out, alpha=0.2, zorder=-1)

    arrowprops['edgecolor'] = color_out

    arrowstyle = matplotlib.patches.ArrowStyle.BarAB(widthA=0)
    arrowprops['arrowstyle'] = arrowstyle
    ax.annotate('', xy=(x0, height), xytext=(x1, height),
                arrowprops=arrowprops)

    # reset the limits
    ax.set_ylim(ylim)

def plot_It_regions(It, ax, colors=None, color_out='gray'):
    inner_centers = []
    for ctr in It.p_centers:
        if ctr['is_init_birthdeath'] == 0:
            external_center = ctr
        else:
            inner_centers.append(ctr)

    if colors is None:
        colors = ['C%d' % i for i in range(len(inner_centers))]

    # plot I(t)
    taus = np.linspace(-0.01, inner_centers[-1]['tau_death']*1.2, 2000)
    Its = It(taus)
    ax.plot(taus, Its/2/np.pi, c='k', zorder=6, lw=2.)

    # highlight regions
    add_regions_rectangles(ax, colors, color_out, inner_centers, external_center, taus[-1])
    # ~ add_regions_arrows(ax, colors, color_out, inner_centers, external_center, taus[-1])

    return taus, Its

## =====================================================================

if __name__ == '__main__':
    Psi = benchmark_lens()
    y = 0.015
    # ~ y = 0.15
    It = time_domain_c.It_MultiContour_C(Psi, y, p_prec={'eval_mode':'exact'})

    #########################

    cmap = matplotlib.colormaps['tab10']
    colors = [cmap(i) for i in range(cmap.N)]

    colors[5] = 'darkgoldenrod' # get rid of brown
    # ~ colors[5] = 'aquamarine' # get rid of brown

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    # plot_contours
    inner_centers = plot_inner_contours(It, ax1, alpha=1, colors=colors)
    outer_center  = plot_outer_contours(It, ax1, alpha=1)

    # plot saddles and crit points
    xmax, ax = plot_saddle_cnt(It, ax1)
    plot_decorate(It, ax1)

    xmax = 1.1*xmax
    ax1.set_xlim([-xmax, xmax])
    ax1.set_ylim([-xmax, xmax])
    ax1.set_box_aspect(1)

    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")

    # plot I(t)
    taus, Its = plot_It_regions(It, ax2, colors=colors)

    ax2.grid(alpha=0.7)
    ax2.set_xlim([taus[0], taus[-1]])
    ax2.set_ylim(ymin=0)
    ax2.set_box_aspect(1)

    ax2.spines['right'].set_zorder(6)
    ax2.spines['bottom'].set_zorder(6)

    ax2.set_xlabel("$\\tau$")
    ax2.set_ylabel("$I(\\tau)/2\\pi$")

    print('Done, saving.')
    fig.savefig('contour_example.pdf', bbox_inches='tight')
