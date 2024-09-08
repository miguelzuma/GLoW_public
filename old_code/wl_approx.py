import warnings
import numpy as np
from scipy import integrate as sc_integrate
from scipy import interpolate as sc_interpolate
from scipy import optimize as sc_optimize
from scipy import special as sc_special
import multiprocessing

from . import lenses

class TimeDomainException(Exception):
    pass
    
class TimeDomainWarning(UserWarning):
    pass
    
##==============================================================================


class ItGeneral():
    """
    Base Class for a time-domain integral.
    
    Internal variables:
      - lens : lens object
      - t_grid and It_grid : grid of points where It has been computed
      - eval_It : interpolation function to evaluate It at any point
    
    Physical parameters (p_phys):
      - y  : impact parameter, assumed to be in the x1 axis (default: no)
    
    Precision parameters (p_prec):
      - interp_fill_value : behaviour of the interpolation function outside the
                            interpolation range (default: 'extrapolate')
                            (other: a value can be used to fill the values)
      - interp_kind : interpolation method (default: 'linear') 
                      (other: 'linear', 'slinear', 'quadratic', 'nearest', ...)            
    """
    def __init__(self, Lens, p_phys={}, p_prec={}):
        self.lens = Lens

        self.p_phys, self.p_prec = self.default_general_params()
        self.p_phys_default_keys = set(self.p_phys.keys())
        self.p_prec_default_keys = set(self.p_prec.keys())
        
        self.p_phys.update(p_phys)
        self.p_prec.update(p_prec)
        
        self.check_general_input()
        
        # ***** to be overriden by the subclass *****
        self.t_grid, self.It_grid = np.array([]), np.array([])
        self.eval_It = lambda t: 0
        # *******************************************
    
    def __str__(self):
        class_name = type(self).__name__
        class_call = "It = time_domain." % module + class_name + "(Psi, p_phys, p_prec)"
    
        phys_message = "p_phys = " + self.p_phys.__repr__() + "\n"
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"
        
        lens_message = self.lens.__str__() + "\n\n"
        
        return lens_message + phys_message + prec_message + class_call
    
    def __call__(self, tau):
        return self.eval_It(tau)
    
    def default_general_params(self):
        p_phys = {}
        p_prec = {'interp_fill_value' : 'extrapolate',\
                  'interp_kind' : 'linear'}

        p_phys2, p_prec2 = self.default_params()
        if p_phys2 is not {}:
            p_phys.update(p_phys2)
        if p_prec2 is not {}:
            p_prec.update(p_prec2)
        
        return p_phys, p_prec
    
    def check_general_input(self):
        # note: if the subclass will add a new parameter without an entry
        #       in default_params, it must be manually added to 
        #       self.p_phys_default_keys or self.p_prec_default_keys
        #       in check_input() with self.p_phys_default_keys.add(new_key)
        
        self.p_phys_default_keys.add('y')
        if 'y' not in self.p_phys:
            message = "impact parameter 'y' required"
            raise TimeDomainException(message)
            
        self.check_input()
        
        # check that there are no unrecognized parameters
        p_phys_new_keys = set(self.p_phys.keys())
        p_prec_new_keys = set(self.p_prec.keys())
        
        diff_phys = p_phys_new_keys - self.p_phys_default_keys
        diff_prec = p_prec_new_keys - self.p_prec_default_keys
        
        if diff_phys:
            for key in diff_phys:
                message = "unrecognized key '%s' found in p_phys will be "\
                          "(most likely) ignored" % key
                warnings.warn(message, TimeDomainWarning)
                
        if diff_prec:
            for key in diff_prec:
                message = "unrecognized key '%s' found in p_prec will be "\
                          "(most likely) ignored" % key
                warnings.warn(message, TimeDomainWarning)

    def default_display_info(self):
        """
        Output all the information in the parameter dictionaries.
        """
        print("\t////////////////////////////\n"\
              "\t///   I(t) information   ///\n"\
              "\t////////////////////////////")
        
        if self.p_phys != {}:        
            print("\n * Method: %s" % self.p_phys.get('name', 'no information'))
                        
            print("\n * Physical parameters:")
            for key, value in self.p_phys.items():
                if key == 'name':
                    continue
                print("   **", key, "=", value)
        
        if self.p_prec != {}:
            print("\n * Precision parameters:")
            for key, value in self.p_prec.items():
                print("   **", key, "=", value)
                
        if (self.p_phys == {}) and (self.p_prec == {}):
            print('\nNo information available')
        
        print("\n * Lens: %s" % self.lens.p_phys.get('name', 'no information'))
        
        print('')
        
    def display_images(self):
        try:
            tmin = self.p_crits[0]['t']
        except AttributeError as e:
            message = 'no critical points (p_crits) found in It (%s)' % self.p_phys['name']
            raise TimeDomainException(message) from e
        
        print("\t////////////////////////////\n"\
              "\t///        Images        ///\n"\
              "\t////////////////////////////")
        print("\n * Lens: %s  (y = %g)" % (self.lens.p_phys['name'], self.p_phys['y']))
        for i, p in enumerate(self.p_crits):
            print("\n * Image %d  (%s):" % (i, p['type']))
            print("   **   t = %e" % p['t'])
            print("   ** tau = %e" % (p['t']-tmin))
            print("   **   x = (%e, %e)" % (p['x1'], p['x2']))
            print("   **  mu = %e" % p['mag'])
        print('')
        
    def interpolate(self, x, y):
        return sc_interpolate.interp1d(x, y, \
                                       fill_value = self.p_prec['interp_fill_value'],\
                                       kind = self.p_prec['interp_kind'])
                                       
    def find_tmin_root(self, x1_guess, x2_guess=0):
        """
        Method to find the minimum time delay of a lens by finding the root of 
        the derivative of the Fermat potential.
        
        The algorithm starts with an initial guess x1_guess (x2_min=0 always
        for axisymmetric lenses).
        
        Caveats:
          - Derivatives of the lens required
          - Not implemented for non-axisymmetric lenses
          - At the moment, it can only be used when there is only one critical 
            point (hence a minimum). No checks are perfomed in this function.
        """
        if (self.lens.hasDeriv1 and self.lens.hasDeriv2) is False:
            message = "'find_tmin_root' method requires the lens derivatives"
            raise TimeDomainException(message)
        
        if self.lens.isAxisym:
            y = self.p_phys['y']
            
            f  = lambda x1: self.lens.dphi_Fermat_dx1(x1, x2=0, y=y)
            df = lambda x1: self.lens.ddphi_Fermat_ddx1(x1, x2=0, y=y)
        
            sol = sc_optimize.root_scalar(f=f, fprime=df, x0=x1_guess, method='newton')
            
            x1_min = sol.root
            x2_min = 0
            tmin = self.lens.phi_Fermat(x1_min, x2_min, y)
        else:
            message = "'find_tmin_root' method not implemented for non-axisymmetric lenses yet"
            raise TimeDomainException(message)
            
        return tmin, x1_min, x2_min
    
    def find_tmin_bounds(self, x1_max, x2_max=0, x1_min=None, x2_min=None, x1_guess=None, x2_guess=None):
        """
        Method to find the minimum time delay through direct minimization of the 
        Fermat potential.
        
        Both algorithms (axisymmetric and non-axisymmetric case) require the bounds
        x1_max, x2_max of the optimization region (if x1_min and x2_min are not
        given, the region is assumed to be symmetric). The 2d algorithm for 
        non-axisymmetric lenses also needs an initial guess (x1_guess, x2_guess)
        (default: (y, 0))
        
        Caveats:
          - No derivatives are required, but if present they could be used to 
            improve convergence
        """
        if x1_min is None:
            x1_min = -x1_max
        if x2_min is None:
            x2_min = -x2_max
            
        if self.lens.isAxisym:
            y = self.p_phys['y']
            f = lambda x1: self.lens.phi_Fermat(x1, x2=0, y=y)

            sol = sc_optimize.minimize_scalar(f, \
                                              bounds=(x1_min, x1_max), \
                                              method='bounded')
            
            x1_min = sol.x
            x2_min = 0
            tmin = sol.fun
        else:
            y = self.p_phys['y']
            
            if x1_guess is None:
                x1_guess = y
            if x2_guess is None:
                x2_guess = 0
            guess = [x1_guess, x2_guess]
            
            f = lambda x: self.lens.phi_Fermat(x[0], x[1], y)
            sol = sc_optimize.minimize(f, \
                                       guess, \
                                       bounds=((x1_min, x1_max), (x2_min, x2_max)), \
                                       method='TNC')
                                       
            x1_min = sol.x[0]
            x2_min = sol.x[1]
            tmin = sol.fun
        
        return tmin, x1_min, x2_min
                
    # ***** to be overriden by the subclass *****
    def check_input(self):
        pass
        
    def default_params(self):
        p_phys = {}
        p_prec = {}
        return p_phys, p_prec
        
    def display_info(self):
        self.default_display_info()
    # *******************************************
    

##==============================================================================


class It_WL(ItGeneral):
    """
    Weak-lensing approximation of the time-domain integral.
    
    Internal variables:
      - t_grid and It_grid : grid of points where It has been computed
      - eval_It : interpolation function to evaluate It at any point
      - tmin : minimum time delay
      - mag : magnification
      - it_grid : 'integral' of It (plus some shifting and prefactors)
      
    Precision parameters (p_prec):
      - Nt : number of points to be computed (default: 1000)
      - dt : temporal increment used to compute the numerical derivative of it
             (default: 1e-4)
      - tmin : minimum tau to be computed (tau by definition starts at 0)
               (default: 0)
      - tmax : maximum tau to be computed (default: 10)
      - sampling : type of temporal grid, either 'linear' or 'log' (default: 'linear')
      - (Base class) interp_fill_value
      - (Base class) interp_kind
                     
    Physical parameters (p_phys):
      - (Base class) y
      
    HVR -> the code could be cleaned up a little bit, especially changing the
           notation so that we need less if/else to distinguish axisymmetric and
           non-axisymmetric cases
    """
    def __init__(self, Lens, p_phys={}, p_prec={}):
        super().__init__(Lens, p_phys, p_prec)
        
        t_grid, self.it_grid, It_grid = self.compute()
        
        self.mag, self.tmin, self.x1_min, self.x2_min = self.compute_magnification_tmin()
        
        self.p_crits = [{'type' : 'min',\
                         't'    : self.tmin,\
                         'x1'   : self.x1_min,\
                         'x2'   : self.x2_min,\
                         'mag'  : self.mag}]
        
        # shift and interpolate the first point (tau=0) with the analytical 
        # result (given by the magnification)
        t0 = np.array([0])
        t_grid -= self.tmin
        self.t_grid = np.concatenate([t0, t_grid])
        
        I0 = np.array([2*np.pi*np.sqrt(self.mag)])
        self.It_grid = np.concatenate([I0, It_grid])
        
        self.interp_It = self.interpolate(self.t_grid, self.It_grid)        
        self.eval_It = lambda t: np.piecewise(t, [t<0, t>=0], [0, self.interp_It])
        
    def default_params(self):
        p_phys = {'name'     : 'weak lensing'}
        p_prec = {'Nt'       : 1000, \
                  'dt'       : 1e-4, \
                  'tmin'     : 0, \
                  'tmax'     : 10, \
                  'phi0'     : 0, \
                  'sampling' : 'linear'}
                  
        if self.lens.isAxisym:
            p_prec['phif'] = 0.5*np.pi
        else:
            p_prec['phif'] = np.pi
            
        return p_phys, p_prec
    
    def x_Axisym(self, phi, r, y):
        q = 4*r/(1+r)/(1+r)
        sphi = np.sin(phi)
        x = y*(1+r)*np.sqrt(1-q*sphi*sphi)
        return x
    
    def integrand_Axisym(self, phi, r, y):
        x = self.x_Axisym(phi, r, y)
        return self.lens.psi_x(x)
        
    def x1x2_General(self, phi, R, y):
        x1 = y + R*np.sin(2*phi)
        x2 = R*np.cos(2*phi)
        return x1, x2
    
    def integrand_General(self, phi, tau, y):
        x1, x2 = self.x1x2_General(phi, tau, y)
        return self.lens.psi(x1, x2)
        
    def compute(self):
        phi0 = self.p_prec['phi0']
        phif = self.p_prec['phif']
        y = self.p_phys['y']
        
        dt = self.p_prec['dt']
        tmin = self.p_prec['tmin']
        if tmin < dt:
            tmin += dt
        tmax = self.p_prec['tmax']
        Nt = self.p_prec['Nt']
        
        if self.p_prec['sampling'] == 'linear':
            t_grid = np.linspace(tmin, tmax, Nt)
        if self.p_prec['sampling'] == 'log':
            t_grid = np.logspace(np.log10(tmin), np.log10(tmax), Nt)
        
        # obviously twice as slow as integrating for all t and then computing
        # the derivative, but in this way we can compute a sparse logarithmic 
        # grid of t points
        
        R_plus = np.sqrt(2.*(t_grid + dt))
        if self.lens.isAxisym:
            r_plus = R_plus/y
            integrand_vec = lambda phi: self.integrand_Axisym(phi, r_plus, y)
        else:
            integrand_vec = lambda phi: self.integrand_General(phi, R_plus, y)
        it_grid_plus = sc_integrate.quad_vec(integrand_vec, phi0, phif)[0]
        
        R_minus = np.sqrt(2.*(t_grid - dt))
        if self.lens.isAxisym:
            r_minus = R_minus/y
            integrand_vec = lambda phi: self.integrand_Axisym(phi, r_minus, y)
        else:
            integrand_vec = lambda phi: self.integrand_General(phi, R_minus, y)
        it_grid_minus = sc_integrate.quad_vec(integrand_vec, phi0, phif)[0]
        
        # compute the numerical derivative
        it_grid = 0.5*(it_grid_plus + it_grid_minus)
        It1_grid = 2*(it_grid_plus - it_grid_minus)/2./dt
        
        # factor of 2 for the symmetry in the angular integration
        if self.lens.isAxisym:
            it_grid = 2*it_grid
            It1_grid = 2*It1_grid
        
        # join the first-order correction with the 0th order
        It_grid = 2*np.pi + It1_grid
        
        return t_grid, it_grid, It_grid
    
    def compute_magnification_tmin(self):        
        """
        Compute the magnification and the minimum time delay of the lens,
        assuming that we are in the WL regime and there is only one critical
        point in the Fermat potential.
        """    
        if (self.lens.hasDeriv1 and self.lens.hasDeriv2) is False:
            message = "lens derivatives needed to compute magnification"
            raise TimeDomainException(message)
            
        if self.lens.isAxisym is True:
            tmin, x1_min, x2_min = self.find_tmin_root(x1_guess=self.p_phys['y'])
            
            # alternative method with a fixed bound and direct minimization
            # ~ tmin, x1_min, x2_min = self.find_tmin_bounds(x1_max=self.p_phys['y']+5)
            
            xmin = np.sqrt(x1_min**2 + x2_min**2)
            dpsi = self.lens.dpsi_dx(xmin)
            ddpsi = self.lens.ddpsi_ddx(xmin)
            
            inv_mag = (1-dpsi/xmin)*(1-ddpsi)
            mag = 1./np.abs(inv_mag)
        else:
            tmin, x1_min, x2_min = self.find_tmin_bounds(x1_max=self.p_phys['y']+5, \
                                                         x2_max=5, \
                                                         x1_guess=self.p_phys['y'], \
                                                         x2_guess=0)
            
            psi11 = self.lens.ddpsi_ddx1(x1_min, x2_min)
            psi22 = self.lens.ddpsi_ddx2(x1_min, x2_min)
            psi12 = self.lens.ddpsi_dx1dx2(x1_min, x2_min)
            
            # notation from [Schneider, Ehlers, Falco, ch.5, p.162]
            kappa = 0.5*(psi11 + psi22)
            gamma1 = 0.5*(psi11 - psi22)
            gamma2 = psi12
            gamma_sq = gamma1**2 + gamma2**2
            
            inv_mag = (1-kappa)**2 - gamma_sq
            mag = 1./np.abs(inv_mag)
            
        return mag, tmin, x1_min, x2_min
            
            
##==============================================================================       


if __name__ == "__main__":
    pass
    
