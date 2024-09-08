import os
import sys
from subprocess import call

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle

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
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

## =========================================================================

class Psi_composite_SIS(lenses.PsiAxisym):
    """
    Lens object for a collection of infinitely many SISs
    
    Physical parameters (p_phys):
      - psi0 : normalization of the lens potential (default: 1)
      - x_cut: maximum distance of sublenses (default: 1)
      
    Computes a grid when initializing
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)
        
        self.hasDeriv1 = True
        self.hasDeriv2 = True
        
        self.psi0 = self.p_phys['psi0']
        self.x_cut = self.p_phys['x_cut']
        
        self.x_grid = np.geomspace(1e-5,1e6,300)
        self.psi_grid = np.array([self.psi_avg(x,nth=50,nxi=300,eps=1e-7) for x in self.x_grid])
        self.dpsi_grid = np.gradient(self.psi_grid, self.x_grid)
        self.ddpsi_grid = np.gradient(self.dpsi_grid, self.x_grid)
           

    def psi_avg(self, x,nth=50,nxi=200,eps = 1e-6):
        '''average Psi for a collection of SISs
            P(x) = 1/x H(x_cut-x)
            x_cut -> subhalo deviation cutoff wrt center
            TODO: does not recover the right lensing potential!
        '''
        x_cut=self.x_cut
        
        #integrate only to pi, but factor of 2
        #integral for Sigma was delicate, Psi is sturdier
        ths = np.linspace(0,np.pi,nth)
        xi = np.geomspace(eps,1/x,nxi)
        
        integrand2 = []
        for th in ths:
            integrand = np.sqrt(1+xi**2-2*xi*np.cos(th))
            integrand2.append(simps(integrand,xi))
        #x^2 from jacobian
        return x**2/np.pi*simps(integrand2,ths)

    
    def default_params(self):
        p_phys = {'name' : 'composite_SIS',\
                  'psi0' : 1,\
                  'x_cut' : 1}
        p_prec = {}
        return p_phys, p_prec
        
    def check_input(self):
        x_cut = self.p_phys['x_cut']
        if x_cut <= 0:
            message = "x_cut = %g <= 0 found. It must be positive" % x_cut
            raise LensException(message)
    
    def psi_x(self, x):
        psi_int = griddata(self.x_grid,self.psi_grid,x,method='cubic')
        #interpolate above and below
        #psi_int = np.where(x<np.min(self.x_grid),psi_int,psi_int[0])
        #psi_int = np.where(x>np.max(self.x_grid),psi_int,x)
        return self.psi0*psi_int
        
    def dpsi_dx(self, x):
        dpsi_int = griddata(self.x_grid,self.dpsi_grid,x,method='cubic')
        #interpolate above and below
        #dpsi_int = np.where(x<np.min(self.x_grid),dpsi_int,dpsi_int[0])
        #psi_int = np.where(x>np.max(self.x_grid),dpsi_int,1)
        return self.psi0*dpsi_int
        
    def ddpsi_ddx(self, x):
        ddpsi_int = griddata(self.x_grid,self.ddpsi_grid,x,method='cubic')
        #interpolate above and below
        #dpsi_int = np.where(x<np.min(self.x_grid),ddpsi_int,ddpsi_int[2])
        #psi_int = np.where(x>np.max(self.x_grid),ddpsi_int,ddpsi_int[-2])
        return self.psi0*ddpsi_int
        
## =========================================================================
        
        
def distribute_centers(n, R=1, N=5000, seed=None, slope = 1, R_max = 1):
    '''Distribute according to a SIS density
    '''
    if(seed is None):
        seed = np.random.randint(0, 1e7)
        print("Seed:", seed)
        
    np.random.seed(seed)
    
    
    if slope!=0:
        r = np.abs(np.random.power(slope,N)-1)
    else:
        r = np.random.rand(N)
        
    phi = 2*np.pi*np.random.rand(N)
    
    x1, x2 = r*np.cos(phi), r*np.sin(phi)
    
    arg_inside = np.argwhere(x1**2+x2**2 < R_max**2)

    N_inside = len(arg_inside)
    if(n > N_inside):
        print('More points needed (N=%d)' % N)
        return 1

    x1_ins = x1[arg_inside].flatten()[:n]
    x2_ins = x2[arg_inside].flatten()[:n]
    
    x1CM = np.sum(x1_ins)/n
    x2CM = np.sum(x2_ins)/n
    
    x1 = x1_ins - x1CM
    x2 = x2_ins - x2CM
    
    return x1, x2, seed


def create_composite_lens(xc1s, xc2s, use_SIS=True):
    Nsub = len(xc1s)    
    psi0 = 1/Nsub

    p_phys = {'lenses' : []}
    for xc1, xc2 in zip(xc1s, xc2s):

        if use_SIS is True:
            sub_p_phys = {'psi0' : psi0, 'xc1' : xc1, 'xc2' : xc2}
            Psi = lenses.Psi_offcenterSIS(sub_p_phys)
        else:
            sub_p_phys = {'psi0' : psi0, 'b' : 1, 'xc1' : xc1, 'xc2' : xc2}
            Psi = lenses.Psi_offcenterBall(sub_p_phys)
        p_phys['lenses'].append(Psi)
    Psi = lenses.CombinedLens(p_phys)
    
    return Psi


def plot_Sigma(ax, xc1s, xc2s, \
               y = np.nan, \
               Ngrid = 200, \
               eps_soften = 0.1, \
               circle_radius = 0, \
               grid_lim = 1.1, \
               y_plt = 0, \
               y_arrow = False, \
               y_circle_r = 0, \
               tauC_contour = True, \
               tauC_psi = None, \
               raster = True):
    """
    plot the projected density
    
    Ngrid      -> number of grid ponts
    eps_soften -> soften the density distribution
    y_plt      -> impact parameter to plot (0 to omit)
    """
    N_sub = len(xc1s)
    eps = eps_soften/Ngrid
    
    #include 
    center, width = (max(grid_lim,y_plt*1.1) - grid_lim)/2, (max(grid_lim,y_plt*1.1) + grid_lim)/2
    y_grid = np.linspace(-width,width,Ngrid)
    x_grid = np.linspace(-width,width,Ngrid)+center
    
    X,Y = np.meshgrid(x_grid,y_grid)
    
    Sigma = np.zeros_like(X*Y)
    for x0, y0 in zip(xc1s, xc2s):
        r_i = np.sqrt((x0-X)**2+(y0-Y)**2+eps**2)
        Sigma += 1/r_i/N_sub
    
    c = ax.pcolor(X, Y, Sigma, norm=colors.LogNorm(vmin=Sigma.min(), vmax=Sigma.max()), shading='auto', rasterized=raster)
    
    if N_sub > 0:
        ax.annotate('%i SIS'%(N_sub),(.05,0.9),xycoords='axes fraction',color='white',fontsize=12)
    if y_plt > 0:
        ax.scatter([y_plt],[0],marker='*',c='w')
    if tauC_contour and tauC_psi != None:
        TD = time_domain_c.It_SingleContour_C(tauC_psi, {'y':y})
        Phi0 = TD.tmin
        PhiC = tauC_psi.phi_Fermat(0,0,y)-Phi0
        Phi = tauC_psi.phi_Fermat(X,Y,y)-Phi0
        ax.contour(X,Y,Phi,[tauC_psi.phi_Fermat(0,0,y)-Phi0],colors='w',linestyles='dashed',alpha=0.7)
        
    phi = np.linspace(0, 2*np.pi, 100)
    if circle_radius > 0:
        ax.plot(circle_radius*np.cos(phi),circle_radius*np.sin(phi),ls=':',c='w',zorder=10)
    if y_arrow:
        ax.annotate(None, xy=(0.99*x_grid.max(), 0), xytext=(0.7*x_grid.max(),0),
            arrowprops=dict(facecolor='w', shrink=0.02))
    if y_circle_r>0:
        ax.plot(y_circle_r*(np.cos(phi))+y,y_circle_r*np.sin(phi),ls=':',c='w',zorder=10)
    
    ax.set_yticks([])
    ax.annotate(r'$y=%g$'%y, (0.95,0.9),xycoords='axes fraction',color='white',fontsize=14,horizontalalignment='right')
    ax.set_xlim(x_grid.min(),x_grid.max());ax.set_ylim(y_grid.min(),y_grid.max())


def compute_It(Psi, y, SIS=True, avg=True, CIS=False, rc=0.1):
    """
    computes the time-domain integrals
    """
    
    p_phys = {'y' : y}
    p_prec = {'tmin':1e-2, 'tmax':1e6, 'Nt':5000, 'sampling':'log'}
    
    #Store results
    It_results = {}
    
    It = time_domain_c.It_SingleContour_C(Psi, p_phys, p_prec)
    It_results['comp'] = It
    It_results['Psi'] = Psi
    It_results['y'] = y
    
    It_SIS = None
    if SIS is True:
        It_SIS = time_domain_c.It_AnalyticSIS_C(p_phys, p_prec)
        It_results['SIS'] = It_SIS
    
    It_avg = None
    if avg is True:
        Psi_avg = Psi_composite_SIS({'psi0':1, 'x_cut':1})
        It_avg = time_domain_c.It_SingleContour_C(Psi_avg, p_phys, p_prec)
        It_results['avg'] = It_avg
    
    It_CIS = None
    if CIS is True:
        Psi_CIS = lenses.Psi_CIS({'psi0':1, 'rc':rc})
        It_CIS = time_domain_c.It_SingleContour_C(Psi_CIS, p_phys, p_prec)
        It_results['CIS'] = It_CIS
        
    return It_results
    

def plot_It(ax, results_It, tau_min=0.5, tau_max=50, labels=False, yticks=False):
    
    taus = np.geomspace(tau_min, tau_max, 1000)
    
    It = results_It['comp']
    ax.plot(taus, It.eval_It(taus)/2/np.pi, label='%d SIS' % len(results_It['Psi'].p_phys['lenses']), c='C0')
    ax.set_xscale('log')
    
    It_SIS = None
    if 'SIS' in results_It.keys():
        It_SIS = results_It['SIS']
        ax.plot(taus, It_SIS.eval_It(taus)/2/np.pi, label='SIS', alpha=0.5, c='C1')
    
    It_avg = None
    if 'avg' in results_It.keys():
        It_avg = results_It['avg']
        ax.plot(taus, It_avg.eval_It(taus)/2/np.pi,label='Avg',alpha=0.8,c='C2')
    
    It_CIS = None
    if 'CIS' in results_It.keys():
        It_CIS = results_It['CIS']
        ax.plot(taus, It_CIS.eval_It(taus)/2/np.pi, label='CIS', alpha=0.5, c='C3')
    
    ax.grid(alpha=0.5)
    if labels:
        ax.set_xlabel(r'$\tau$')
        ax.set_title(r'$I\mathcal{I}(\tau)/2\pi$')
    if yticks==False:
        ax.set_yticks([])
    ax.set_xlim([taus[0], taus[-1]])
    ax.legend(loc='best')


def plot_Gt(ax, results_It, \
            tau_min = 0.5, \
            tau_max = 50, \
            labels = False, \
            limits_comp = True, \
            legend = False, \
            yticks = False, \
            zeroline = True):

    dt = 1e-2
    taus = np.geomspace(tau_min, tau_max, 1000)

    deriv = lambda I, t: 0.5*(I.eval_It(t+dt)-I.eval_It(t-dt))/dt
    
    It = results_It['comp']
    Gt = deriv(It, taus)
    ax.plot(taus, Gt, label='%d SIS' % len(results_It['Psi'].p_phys['lenses']), c='C0')
    
    It_SIS = None
    if 'SIS' in results_It.keys():
        It_SIS = results_It['SIS']
        ax.plot(taus, deriv(It_SIS, taus), label='SIS', alpha=0.5, c='C1')
    
    It_avg = None
    if 'avg' in results_It.keys():
        It_avg = results_It['avg']
        ax.plot(taus, deriv(It_avg,taus),label='Avg',alpha=0.8,c='C2')
    
    It_CIS = None
    if 'CIS' in results_It.keys():
        It_CIS = results_It['CIS']
        ax.plot(taus, deriv(It_CIS, taus), label='CIS', alpha=0.5, c='C2')
    
    ax.grid(alpha=0.5)
    if labels:
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$G(\tau)$')
    if yticks==False:
        ax.set_yticks([])
        
    if limits_comp:
        ax.set_ylim([Gt.min()*1.2,Gt.max()*1.2])
    ax.set_xlim([taus[0], taus[-1]])
    if legend:
        ax.legend(loc='best')
    if zeroline:
        ax.axhline(0,c='gray',alpha=0.5,label='0')


def plot_Fw(ax, results_It,labels=False,tick_right=True,legend=False,wmF=1e-2,wMF=1e2,print_comparison=False):
    
    It_MultiSIS = results_It['comp']
    
    ## ============================================================================
    
    p_prec = {'fmin':wmF, 'fmax':wMF }
    
    Fw_MultiSIS = freq_domain.Fw_WL(It_MultiSIS, p_prec=p_prec)        
    
    ## ============================================================================
    
    ws = Fw_MultiSIS.w_grid
    
    ax.plot(ws, np.abs(Fw_MultiSIS.Fw_grid), zorder=10, label='%d SIS' % len(results_It['Psi'].p_phys['lenses']), c='C0')
    
    It_SIS = None
    if 'SIS' in results_It.keys():
        It_SIS = results_It['SIS']
        Fw_SIS = freq_domain.Fw_WL(It_SIS, p_prec=p_prec)
        ax.plot(ws, np.abs(Fw_SIS.Fw_grid), alpha=0.5, label='SIS', c='C1')
    
    It_avg = None
    if 'avg' in results_It.keys():
        It_avg = results_It['avg']
        Fw_avg = freq_domain.Fw_WL(It_avg, p_prec=p_prec)
        ax.plot(ws, np.abs(Fw_avg.Fw_grid), alpha=0.8, label='Avg', c='C2')
    
    It_CIS = None
    if 'CIS' in results_It.keys():
        It_CIS = results_It['CIS']
        Fw_CIS = freq_domain.Fw_WL(It_CIS, p_prec=p_prec)
        ax.plot(ws, np.abs(Fw_CIS.Fw_grid), alpha=0.5, label='CIS', c='C3')

    if print_comparison:
        maxsis = np.abs(Fw_SIS.Fw_grid).max()
        maxavg = np.abs(Fw_avg.Fw_grid).max()
        maxcomp = np.abs(Fw_MultiSIS.Fw_grid).max()
        print('SIS/AVG = %g, SIS/COMP=%g, AVG/COMP=%g'%(maxsis/maxavg, maxsis/maxcomp,maxavg/maxcomp))

    ax.grid(alpha=0.5)
    ax.set_xlim([ws[0], ws[-1]])
    ax.set_xscale('log')
    if tick_right:
        ax.yaxis.tick_right()
    if labels:
        ax.set_ylabel('$|F(w)|$')
        ax.set_xlabel('$w$')
    if legend:
        ax.legend(loc='best')
        

def plot_row(ax, I, xc1s, xc2s, It_results, \
             tmI = 5, \
             tMI = 280, \
             tmG = 9, \
             tMG = 29, \
             wmF = 1e-2, \
             wMF = 1e2, \
             show_lens_lim = False, \
             show_y_cir = False, \
             show_tC_contour = False, \
             show_source = False, \
             yticks = False, \
             tick_right = True):
    
    y = It_results['y']
    
    circle_r   = 1   if show_lens_lim else 0
    y_circle_r = y   if show_y_cir    else 0
    y_plt      = y   if show_source   else 0
    grid_lim   = 1.5 if show_source   else 1.2
    
    tauC_Psi   = It_results['Psi'] if show_tC_contour else None
    
    plot_Sigma(ax[I,0], xc1s, xc2s, y=y, y_circle_r=y_circle_r, circle_radius=circle_r, grid_lim=grid_lim, y_plt=y_plt, tauC_psi=tauC_Psi)
    plot_It(ax[I,1], It_results, tau_min=tmI, tau_max=tMI, yticks=yticks)
    plot_Gt(ax[I,2], It_results, tau_min=tmG, tau_max=tMG, yticks=yticks)
    plot_Fw(ax[I,3], It_results, wmF=wmF, wMF=wMF, tick_right=tick_right)   


def plot_row_custom(ax, I, xc1s, xc2s, It_results, \
                    plot_items = ['S','I','G','F'], \
                    tmI = 5, \
                    tMI = 280, \
                    tmG = 9, \
                    tMG = 29, \
                    wmF = 1e-2, \
                    wMF = 1e2, \
                    show_lens_lim = False,
                    \
                    show_y_cir = False, \
                    show_tC_contour = False, \
                    show_source = False, \
                    yticks = False, \
                    tick_right = True):

    y = It_results['y']

    circle_r   = 1   if show_lens_lim else 0
    y_circle_r = y   if show_y_cir    else 0
    y_plt      = y   if show_source   else 0
    grid_lim   = 1.5 if show_source   else 1.2

    tauC_Psi   = It_results['Psi'] if show_tC_contour else None

    for i, n in enumerate(plot_items):
        if 'S'==n:
#             print(I,i)
            plot_Sigma(ax[I,i], xc1s, xc2s, y=y, y_circle_r=y_circle_r, circle_radius=circle_r, grid_lim=grid_lim, y_plt=y_plt, tauC_psi=tauC_Psi)
        if 'I' == n:
            plot_It(ax[I,i], It_results, tau_min=tmI, tau_max=tMI, yticks=yticks)
        if 'G' == n:
            plot_Gt(ax[I,i], It_results, tau_min=tmG, tau_max=tMG, yticks=yticks)
        if 'F' == n:
            plot_Fw(ax[I,i], It_results, wmF=wmF, wMF=wMF, tick_right=tick_right)
#             print(I,i)

def compare_lens(Psi, y1=1.2, y2=2.4, y3=4.8):
    tmax = 1e8
    
    c1 = 'C0'
    c2 = 'C1'
    c3 = 'C2'
    
    p_prec = {'tmin':1e-2, 'tmax':tmax, 'Nt':5000, 'sampling':'log'}
    It_1_wl = time_domain.It_WL(Psi, {'y' : y1}, p_prec)
    It_2_wl = time_domain.It_WL(Psi, {'y' : y2}, p_prec)
    It_3_wl = time_domain.It_WL(Psi, {'y' : y3}, p_prec)
    
    if Psi.p_phys['name'] == 'SIS':
        p_prec = {'tmax':tmax, 'Nt':100000, 'sampling':'log'}

        It_1 = time_domain_c.It_AnalyticSIS_C({'y' : y1}, p_prec)
        It_2 = time_domain_c.It_AnalyticSIS_C({'y' : y2}, p_prec)
        It_3 = time_domain_c.It_AnalyticSIS_C({'y' : y3}, p_prec)
    else:
        p_prec = {'tmin':1e-2, 'tmax':tmax, 'Nt':50000, 'sampling':'log'}
        It_1 = time_domain_c.It_SingleContour_C(Psi, {'y' : y1}, p_prec)
        It_2 = time_domain_c.It_SingleContour_C(Psi, {'y' : y2}, p_prec)
        It_3 = time_domain_c.It_SingleContour_C(Psi, {'y' : y3}, p_prec)
    
    # ---------------------------------------------------------------------
    
    ts = np.logspace(-1, 5, 4000)

    fig1, (ax11, ax12) = plt.subplots(2,\
                                      sharex=True,\
                                      gridspec_kw={'hspace': 0, 'height_ratios': [1.6, 1]})

    ax11.plot(ts, It_1.eval_It(ts)/2/np.pi, label='$y=%g$' % y1, c=c1)
    ax11.plot(ts, It_1_wl.eval_It(ts)/2/np.pi, label='$y=%g$ (WL)' % y1, c=c1, ls='--')
    ax11.plot(ts, It_2.eval_It(ts)/2/np.pi, label='$y=%g$' % y2, c=c2)
    ax11.plot(ts, It_2_wl.eval_It(ts)/2/np.pi, label='$y=%g$ (WL)' % y2, c=c2, ls='--')
    ax11.plot(ts, It_3.eval_It(ts)/2/np.pi, label='$y=%g$' % y3, c=c3)
    ax11.plot(ts, It_3_wl.eval_It(ts)/2/np.pi, label='$y=%g$ (WL)' % y3, c=c3, ls='--')

    ax11.set_ylabel("$I(\\tau)/2\\pi$")
    ax11.set_xscale('log')
    ax11.legend(loc='best')
    ax11.set_xlim([ts[0], ts[-1]])
    ax11.set_title('%s' % Psi.p_phys['name'])
    ax11.grid(True, alpha=0.6)

    ## -------------------------

    ax12.plot(ts, np.abs(It_1.eval_It(ts)-It_1_wl.eval_It(ts))/It_1.eval_It(ts), c=c1)
    ax12.plot(ts, np.abs(It_2.eval_It(ts)-It_2_wl.eval_It(ts))/It_2.eval_It(ts), c=c2)
    ax12.plot(ts, np.abs(It_3.eval_It(ts)-It_3_wl.eval_It(ts))/It_3.eval_It(ts), c=c3)
    ax12.set_yscale('log')
    ax12.set_ylim([1e-7, 1e0])
    ax12.set_xlabel("$\\tau$")
    ax12.set_ylabel("$|\\Delta I(\\tau)|/I(\\tau)$");

    ax12.grid(True, alpha=0.6)
    
    ## ==================================================================================
    ## ==================================================================================
        
    fmin = 1e-2
    fmax = 1e3
    p_prec={'fmin':fmin, \
            'fmax':fmax, \
            'FFT method':'multigrid', \
            'N_below_discard':2,\
            'N_above_discard':2,\
            'N_keep':0.5}

    Fw_1 = freq_domain_c.Fw_SL(It_1, p_prec=p_prec)
    Fw_2 = freq_domain_c.Fw_SL(It_2, p_prec=p_prec)
    Fw_3 = freq_domain_c.Fw_SL(It_3, p_prec=p_prec)
    
    Fw_1_wl = freq_domain_c.Fw_SL(It_1_wl, p_prec=p_prec)
    Fw_2_wl = freq_domain_c.Fw_SL(It_2_wl, p_prec=p_prec)
    Fw_3_wl = freq_domain_c.Fw_SL(It_3_wl, p_prec=p_prec)

    # ---------------------------------------------------------------------

    ws = np.logspace(np.log10(fmin), np.log10(fmax), 15000)

    Fw_1s    = Fw_1.eval_Fw(ws)
    Fw_1_wls = Fw_1_wl.eval_Fw(ws)
    Fw_2s    = Fw_2.eval_Fw(ws)
    Fw_2_wls = Fw_2_wl.eval_Fw(ws)
    Fw_3s    = Fw_3.eval_Fw(ws)
    Fw_3_wls = Fw_3_wl.eval_Fw(ws)

    fig2, (ax21, ax22) = plt.subplots(2,\
                                      sharex=True,\
                                      gridspec_kw={'hspace': 0, 'height_ratios': [1.6, 1]})

    ax21.plot(ws, np.abs(Fw_1s), c=c1, label='$y=%g$' % y1)
    ax21.plot(ws, np.abs(Fw_1_wls), c=c1, label='$y=%g$ (WL)' % y1, ls='--')

    ax21.plot(ws, np.abs(Fw_2s), c=c2, label='$y=%g$' % y2)
    ax21.plot(ws, np.abs(Fw_2_wls), c=c2, label='$y=%g$ (WL)' % y2, ls='--')

    ax21.plot(ws, np.abs(Fw_3s), c=c3, label='$y=%g$' % y3)
    ax21.plot(ws, np.abs(Fw_3_wls), c=c3, label='$y=%g$ (WL)' % y3, ls='--')

    ax21.set_xscale('log')
    ax21.set_xlim([ws[0], ws[-1]])
    ax21.set_ylabel('$|F(w)|$')
    ax21.set_title('%s' % Psi.p_phys['name'])
    ax21.legend()
    ax21.grid(True, alpha=0.7)

    ## -------------------------

    ax22.plot(ws, np.abs(Fw_1s-Fw_1_wls)/np.abs(Fw_1s), c=c1)
    ax22.plot(ws, np.abs(Fw_2s-Fw_2_wls)/np.abs(Fw_2s), c=c2)
    ax22.plot(ws, np.abs(Fw_3s-Fw_3_wls)/np.abs(Fw_3s), c=c3)

    ax22.set_yscale('log')
    ax22.set_xlabel('$w$')
    ax22.set_ylabel('$|\\Delta F(w)|/|F(w)|$')
    ax22.set_ylim([1e-5, 1e0])
    ax22.grid(True, alpha=0.7)
    
    return (fig1, ax11, ax12), (fig2, ax21, ax22)
    

if __name__ == '__main__':
    pass
