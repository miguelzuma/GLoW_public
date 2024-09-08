import numpy as np
from . import lenses

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def set_latex_default(plt):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}

    plt.rc('font', **font)
    

def latex_float(f, thrld_exp=1e-2):
    if  abs(f) < thrld_exp: # For negative powers
        float_str="{0:.0e}".format(f)
        base, exponent = float_str.split("e")
        base = float(base)
        if int(base) == 1:
            return r"10^{{{0}}}".format(int(exponent))
        elif int(base) == -1:
            return r"-10^{{{0}}}".format(int(exponent))
        else:
            return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        float_str = "{0:.2g}".format(f)
        if "e" in float_str: # For positive powers
            base, exponent = float_str.split("e")
            base = float(base)
            if int(base) == 1:
                return r"10^{{{0}}}".format(int(exponent))
            elif int(base) == -1:
                return r"-10^{{{0}}}".format(int(exponent))
            else:
                return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
        else:
            return float_str
    

def plot_potential(y, lens, x1min=-2, x1max=2, x2min=None, x2max=None, levels=25):
    if x2min is None:
        x2min = x1min
    if x2max is None:
        x2max = x1max
    
    Nx1 = 2000
    Nx2 = 2000
    x1s = np.linspace(x1min, x1max, Nx1)
    x2s = np.linspace(x2min, x2max, Nx2)
    x1_grid, x2_grid = np.meshgrid(x1s, x2s)
    
    tau = lens.phi_Fermat(x1_grid, x2_grid, y)
    
    fig, ax = plt.subplots()   
    
    cmap = matplotlib.colormaps.get_cmap('plasma')
    # ~ cmap = matplotlib.colormaps.get_cmap('viridis')
    # ~ cmap = matplotlib.colormaps.get_cmap('inferno')
    # ~ cmap = matplotlib.colormaps.get_cmap('magma')
    # ~ cmap = matplotlib.colormaps.get_cmap('cividis')
    
    im = ax.imshow(tau, \
                   interpolation='bilinear', \
                   origin='lower',
                   cmap=cmap, \
                   extent=(x1min, x1max, x2min, x2max), \
                   alpha=0.8)
    cont = ax.contour(x1_grid, x2_grid, tau, colors='k', levels=levels)
    ax.clabel(cont, inline=True, fontsize=8)
     
    ax.axvline(x = 0, lw=0.8, c='red')
    ax.axhline(y = 0, lw=0.8, c='red')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
    return fig, ax 
    
    
def plot_potential_polar(y, lens, rmin=1e-3, rmax=3, theta_min=0, theta_max=np.pi, levels=25):    
    Nr = 2000
    Ntheta = 2000
    if lens.isAxisym is True:
        rs = np.linspace(rmin, rmax, Nr)
    else:
        rs = np.linspace(-rmax, rmax, Nr)
    thetas = np.linspace(theta_max, theta_min, Ntheta)
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)
    
    r_grid, cos_grid = np.meshgrid(rs, cos_thetas)
    r_grid, sin_grid = np.meshgrid(rs, sin_thetas)
    
    x1_grid = r_grid*cos_grid
    x2_grid = r_grid*sin_grid
    
    tau = lens.phi_Fermat(x1_grid, x2_grid, y)
    
    fig, ax = plt.subplots()   
    
    cmap = matplotlib.colormaps.get_cmap('plasma')
    # ~ cmap = matplotlib.colormaps.get_cmap('viridis')
    # ~ cmap = matplotlib.colormaps.get_cmap('inferno')
    # ~ cmap = matplotlib.colormaps.get_cmap('magma')
    # ~ cmap = matplotlib.colormaps.get_cmap('cividis')
    
    im = ax.imshow(tau, interpolation='bilinear', origin='lower',
                   cmap=cmap, extent=(rs[0], rs[-1], cos_thetas[0], cos_thetas[-1]), alpha=0.8)
    cont = ax.contour(r_grid, cos_grid, tau, colors='k', levels=levels)
    ax.clabel(cont, inline=True, fontsize=8)
    
    ax.set_xlabel('$r$')
    ax.set_ylabel('$\\cos(\\theta)$')
      
    return fig, ax 
    
    
def plot_potential_polar2(y, lens, rhomin=1e-3, rhomax=9, theta_min=-np.pi, theta_max=np.pi, levels=25):    
    Nrho = 2000
    Ntheta = 2000
    rhos = np.linspace(rhomin, rhomax, Nrho)
    thetas = np.linspace(theta_max, theta_min, Ntheta)
        
    rs = np.sqrt(rhos)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    r_grid, cos_grid = np.meshgrid(rs, cos_thetas)
    r_grid, sin_grid = np.meshgrid(rs, sin_thetas)
    rho_grid, theta_grid = np.meshgrid(rhos, thetas)
    
    x1_grid = r_grid*cos_grid
    x2_grid = r_grid*sin_grid
    
    tau = lens.phi_Fermat(x1_grid, x2_grid, y)
    
    fig, ax = plt.subplots()   

    cmap = matplotlib.colormaps.get_cmap('plasma')
    # ~ cmap = matplotlib.colormaps.get_cmap('viridis')
    # ~ cmap = matplotlib.colormaps.get_cmap('inferno')
    # ~ cmap = matplotlib.colormaps.get_cmap('magma')
    # ~ cmap = matplotlib.colormaps.get_cmap('cividis')
    
    im = ax.imshow(tau, interpolation='bilinear', origin='lower', \
                   cmap=cmap, \
                   extent=(rhos[0], rhos[-1],
                           thetas[0], thetas[-1]), \
                   alpha=0.8)
    cont = ax.contour(rho_grid, theta_grid, tau, colors='k', levels=levels)
    ax.clabel(cont, inline=True, fontsize=8)
    
    ax.set_xlabel('$r^2$')
    ax.set_ylabel('$\\theta$')

    return fig, ax 


def plot_lensEq_1d(lens, y=None, x2=0, xmin=1e-5, xmax=1e5, N=1000):
    
    x1s = np.geomspace(xmin, xmax, N)
    d1_pos = lens.dphi_Fermat_dx1(x1s, x2, y=0)
    d1_neg = lens.dphi_Fermat_dx1(-x1s, x2, y=0)
    
    fig, ax = plt.subplots()
    
    ax.plot(x1s, d1_pos, c='grey')
    ax.plot(-x1s, d1_neg, c='grey')
    
    if y is not None:
        ax.axhline(y=y, ls='--', alpha=0.5, c='red', label='y=%g' % y)
        ax.legend()
        
    ax.grid(alpha=0.5)
    ax.set_xscale('symlog')
    ax.set_yscale('log')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel(r'$\partial_{x_1}\phi + y$')
    ax.set_title('%s $(x_2=%g)$' % (lens.p_phys['name'], x2))
    
    return fig, ax
    
    
def plot_lensEq_2d(lens, y, x1min=-2, x1max=2, x2min=None, x2max=None, Nx=2000, levels=None, fig=None, ax=None):
    if x2min is None:
        x2min = x1min
    if x2max is None:
        x2max = x1max
    
    Nx1 = Nx
    Nx2 = Nx
    x1s = np.linspace(x1min, x1max, Nx1)
    x2s = np.linspace(x2min, x2max, Nx2)
    x1_grid, x2_grid = np.meshgrid(x1s, x2s)
    
    d1 = lens.dphi_Fermat_dx1(x1_grid, x2_grid, y)
    d2 = lens.dphi_Fermat_dx2(x1_grid, x2_grid, y)
    
    dat = np.log(d1**2 + d2**2)
    
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()   
    
    # ~ cmap = matplotlib.colormaps.get_cmap('plasma')
    # ~ cmap = matplotlib.colormaps.get_cmap('viridis')
    # ~ cmap = matplotlib.colormaps.get_cmap('inferno')
    cmap = matplotlib.colormaps.get_cmap('magma')
    # ~ cmap = matplotlib.colormaps.get_cmap('cividis')
    
    im = ax.imshow(dat, \
                   interpolation='bilinear', \
                   origin='lower',
                   cmap=cmap, \
                   extent=(x1min, x1max, x2min, x2max), \
                   vmin=-14, vmax=0, \
                   alpha=0.8)
    
    if levels is not None:
        cont = ax.contour(x1_grid, x2_grid, dat, colors='k', levels=levels)
        ax.clabel(cont, inline=True, fontsize=8)
    
    ## add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r'log$|\nabla\phi|^2$')
     
    ax.axvline(x = 0, lw=0.8, c='red', alpha=0.5)
    ax.axhline(y = 0, lw=0.8, c='red', alpha=0.5)
    
    ax.set_title('%s (y=%g)' % (lens.p_phys['name'], y))
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    
    return fig, ax 

