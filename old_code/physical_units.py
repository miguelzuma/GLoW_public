
import numpy as np

from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as c
from colossus.lss import mass_function
from colossus.cosmology import cosmology
from colossus.halo import concentration
from . import lenses
from scipy.interpolate import UnivariateSpline, interp1d

# The cosmology can be changed here:
cosmology.setCosmology('planck18')

def initialize_cosmology(**kwargs):

    global cosmo

    from astropy.cosmology import Planck18, FlatLambdaCDM

    default_cosmo={'H0': Planck18.H0,
                    'Om0': Planck18.Om0,
                    'Tcmb0': Planck18.Tcmb0 ,
                    'm_nu': Planck18.m_nu,
                    'Ob0': Planck18.Ob0}
    
    default_cosmo.update(kwargs).copy() if kwargs else default_cosmo

    cosmo=FlatLambdaCDM(**default_cosmo)
    return cosmo

initialize_cosmology()
    
##==============================================================================
## Constants

GMsun8pi=(8*c.G*c.M_sun*np.pi/c.c**3).decompose().value

G4pi=4*np.pi*c.G/c.c**2

yr_to_s=u.yr.to(u.s)


##==============================================================================
## Conversions

def get_d_A(z):
    return cosmo.angular_diameter_distance(z)

def get_d_A_z1z2(z1, z2):
    return cosmo.angular_diameter_distance_z1z2(z1, z2)

def get_deff(z_lens, z_src, u_out=u.pc):
    """
    Calculate the effective distance ratio.

    Parameters:
    - z_lens (float): Redshift of the lens.
    - z_src (float): Redshift of the source.

    Returns:
    - float: Effective distance ratio.
    """
    d_ls=get_d_A_z1z2(z_lens,z_src)
    d_lens=get_d_A(z_lens)
    d_src=get_d_A(z_src)
    return (d_ls * d_lens / d_src / (1+z_lens)).to(u_out)


def get_z(d_L):
    """
    Convert luminosity distance to redshift using root_scalar.

    Parameters:
    - d_L (float): Luminosity distance in specified units.

    Returns:
    - float: Redshift.
    """
    return z_at_value(cosmo.luminosity_distance,d_L).value


def get_d_L(z, u_out='Mpc'):
    """
    Convert redshift to luminosity distance.

    Parameters:
    - z (float): Redshift.
    - u (str): Units for luminosity distance. Default is 'Mpc'.

    Returns:
    - float: Luminosity distance.
    """
    return cosmo.luminosity_distance(z).to(u_out)

def get_rho_c(z_lens):
    rho_c = cosmo.critical_density(z_lens)
    return rho_c

# Functions to convert virial mass to scale radius for an SIS profile


def get_Sigma_cr(z_lens, z_src, u_out=u.Msun/u.pc**2):

    return (1/((G4pi)*(1+z_lens)*(get_deff(z_lens, z_src)))).to(u_out)


def w_to_f(h_fd, MLz):
    '''
    Functions that return F(f) evaluated on waveform points
    '''
    p_prec_f_opt = get_p_prec_optimal(h_fd.p, MLz)[1]
    p_prec_f_tmp.update(p_prec_f_opt)

    Fw = freq_domain.Fw_WL(I, p_prec=p_prec_f_tmp)

    return Fw

def get_R_E(MLz, deff, u_out=u.Mpc):
    return np.sqrt(4*c.G/c.c**2*MLz*u.Msun*deff).decompose().to(u_out)
##==============================================================================
## SIS

def get_xi_0_SIS(Mvir, z_lens, z_src, u_out=u.Mpc):
    MLz=to_MLz_SIS(Mvir, z_src, z_lens)  
    deff=get_deff(z_lens, z_src)   
    xi_0_SIS = get_R_E(MLz, deff, u_out=u_out) 
    return xi_0_SIS

def to_phys_SIS(y_dimless, Mvir, z_lens, z_src, u_out=u.Mpc):
    xi_0_SIS = get_xi_0_SIS(Mvir, z_lens, z_src, u_out=u_out)
    return y_dimless*xi_0_SIS

def to_dimless_SIS(y_phys, Mvir, z_lens, z_src):
    xi_0_SIS = get_xi_0_SIS(Mvir, z_lens, z_src) 
    return (y_phys/xi_0_SIS).decompose()

def to_Mvir_SIS(MLz, z_src, z_lens):
    """
    Convert from MLz to virial mass for an SIS profile.

    Parameters:
    - MLz (float): Virial mass.
    - deff (float): Effective distance.
    - z_lens (float): Redshift of the lens.
    - rho_evol (bool): Whether to evolve the critical density to lens redshift. Default is True.

    Returns:
    - float: Scale radius.
    """
    deff=get_deff(z_lens, z_src)
    rho_c_z_lens=get_rho_c(z_lens)
    return (((c.c**2*c.G*(MLz*u.solMass)/(deff*(1+z_lens)**2))**(3/4)/(2*c.G**(3/2)*np.pi**2)*1/np.sqrt(200*rho_c_z_lens)).decompose().to(u.solMass)/u.Msun).value

def to_MLz_SIS(Mvir, z_src, z_lens, rho_evol=True):
    """
    Convert virial mass to MLz for an SIS profile.

    Parameters:
    - Mvir (float): Virial mass.
    - deff (float): Effective distance.
    - z_lens (float): Redshift of the lens.
    - rho_evol (bool): Whether to evolve the critical density to lens redshift. Default is True.

    Returns:
    - float: Virial mass.
    """
    deff=get_deff(z_lens, z_src)
    rho_c_z_lens = (3 * cosmo.H(z_lens)**2 / (8 * np.pi * c.G)).decompose() if rho_evol else (3 * cosmo.H(0)**2 / (8 * np.pi * c.G)).decompose()
    return ((2 * 2**(1/3) * deff * c.G * (Mvir * u.Msun)**(4/3) * np.pi**(8/3) * (1 + z_lens)**2 * (200 * rho_c_z_lens)**(2/3) / c.c**2).decompose().to(u.solMass)/u.Msun).value

def get_R_E_Mvir(Mvir, z_lens, z_src, u_out=u.pc):
    """
    Calculate the Einstein radius for strong gravitational lensing in a Singular Isothermal Sphere (SIS) lens.

    :param Mvir: Virial mass at redshift z_L in solar masses.
    :param z_lens: Redshift of the lens.
    :param z_src: Redshift of the source.
    :return: Einstein radius in parsecs.
    """
    MLz = to_MLz_SIS(Mvir, z_lens, z_src)
    deff = get_deff(z_lens, z_src)
    R_E = get_R_E(MLz, deff)
    return R_E.to(u_out)

def get_sigma_v_SIS(Mvir, z_lens):
    """
    Calculate the velocity dispersion for a Singular Isothermal Sphere (SIS) lens.

    :param Mvir: Virial mass at redshift z_L in solar masses.
    :param z_lens: Redshift of the lens.
    :return: Velocity dispersion in m/s.
    """
    return (((5. * np.sqrt(6) / 2.) * c.G * cosmo.H(z_lens) * Mvir*u.Msun).decompose())**(1./3.)

def get_rvir_SIS(Mvir, z_lens, u_out=u.Mpc):
    """
    Calculate the virial radius for a Singular Isothermal Sphere (SIS) lens.

    :param Mvir: Virial mass at redshift z_L in solar masses.
    :param z_lens: Redshift of the lens.
    :return: Virial radius in Mpc.
    """
    sigma_v = get_sigma_v_SIS(Mvir, z_lens)
    rvir = sigma_v / cosmo.H(z_lens) / np.sqrt(150)
    return rvir.to(u_out)


##==============================================================================
## NFW 

## The parameters are Mvir and Delta. 
## The virial radius is defined by (1) in  https://arxiv.org/pdf/1809.07326.pdf.
## Then plug the virial radius in (4) and get rho_s.
## The scale radius is where the logarithmic slope of the profile is −2.
## We use the definitions in https://arxiv.org/pdf/1809.07326.pdf.

def get_xi_0_NFW(Mvir, z_lens, z_src, c_nfw=None, u_out=u.Mpc):
    MLz=to_MLz_NFW(Mvir, z_lens, c_nfw=c_nfw)  
    deff=get_deff(z_lens, z_src)   
    xi_0_NFW = get_R_E(MLz, deff, u_out=u_out) 
    return xi_0_NFW

def to_phys_NFW(y_dimless, Mvir, z_lens, z_src, c_nfw=None, u_out=u.Mpc):
    xi_0_NFW = get_xi_0_NFW(Mvir, z_lens, z_src, c_nfw=c_nfw, u_out=u_out)
    return y_dimless*xi_0_NFW

def to_dimless_NFW(y_phys, Mvir, z_lens, z_src, c_nfw=None):
    xi_0_NFW = get_xi_0_NFW(Mvir, z_lens, z_src, c_nfw=c_nfw)
    return (y_phys/xi_0_NFW).decompose()

def get_c_NFW(Mvir, z_lens):
    """
    Calculate the concentration parameter for a halo with Delta=200c using the Ishiyama21 model.

    :param Mvir: Virial mass in solar masses.
    :param z_lens: Redshift.
    :return: Concentration parameter.
    """
    c_nfw = concentration.concentration(Mvir, '200c', z_lens, model='ishiyama21')
    #diemer 19
    return c_nfw

def get_c_NFW_boost(Mvir, z_lens):
    """
    Calculate the concentration parameter for a halo with Delta=200c using the Ishiyama21 model.

    :param Mvir: Virial mass in solar masses.
    :param z_lens: Redshift.
    :return: Concentration parameter.
    """
    if z_lens<4:
        alpha = 1.62774 - 0.2458*(1 + z_lens) + 0.01716*(1 + z_lens)**2
        beta = 1.66079 + 0.00359*(1 + z_lens) - 1.6901*(1 + z_lens)**(0.00417)
        gamma = -0.02049 + 0.0253*(1 + z_lens)**(- 0.1044)

        log10c_nfw = alpha + beta* np.log10(Mvir)*(1+gamma*(np.log10(Mvir))**2)
    else:
        alpha = 1.226 - 0.1009*(1 + z_lens) + 0.00378*(1 + z_lens)**2
        beta = 0.008634 - 0.08814*(1 + z_lens)**(- 0.58816)

        log10c_nfw= alpha + beta*np.log10(Mvir)

    c_nfw= 10**log10c_nfw
    
    return c_nfw

def get_c_NFW_pw(Mvir, z_lens, pw=0.5, pivot=1e9):
    c_NFW=get_c_NFW(Mvir, z_lens)
    boost=(pivot/Mvir)**pw
    return c_NFW+boost


def get_c_NFW_gui(Mvir, z_lens, factor=3, z_evo_pw=0, cut=1e9):
    
    Ms = np.logspace(1, 17, 200)
    log10Ms = np.log10(Ms)
    log10Mcut = np.log10(cut) * (1. + z_lens)**(z_evo_pw)
       
    cnfw_orig = concentration.concentration(Ms, '200c', z_lens, model='ishiyama21')
    
    exponent1 = interp1d(log10Ms, np.log10(cnfw_orig))
    
    exp_fun = np.piecewise(log10Ms, [log10Ms < log10Mcut, log10Ms >= log10Mcut], [lambda log10Ms: factor*exponent1(log10Ms) - (factor-1.)*exponent1(log10Mcut), lambda log10Ms: exponent1(log10Ms)])
    exp_fun_int = interp1d(log10Ms, exp_fun)#(log10Ms)

    c_nfw = 10**exp_fun_int(np.log10(Mvir))

    return c_nfw

def get_rvir_NFW(Mvir, z_lens, Delta=200, u_out=u.Mpc):
    """
    Calculate the virial radius rvir of a halo given its virial mass Mvir and the redshift z_L.
    At the virial radius the density is 200 times the critical density.

    :param Mvir: Virial mass in solar masses.
    :param z_lens: Redshift.
    :param u: Output units for the virial radius (default is Mpc).
    :return: Virial radius rvir.
    """

    rho_c = cosmo.critical_density(z_lens)
    rvir = ((3 / (4 * np.pi) / (Delta * rho_c) ) * Mvir * u.Msun)**(1./3.)
    return rvir.decompose().to(u_out)

def get_Mvir_NFW(rvir, z_lens):
    """
    Calculate the virial mass Mvir of a halo given its virial radius rvir and the redshift z_L.

    :param rvir: Virial radius in Mpc.
    :param z_lens: Redshift.
    :param u: Output units for the virial mass (default is solar masses).
    :return: Virial mass Mvir.
    """
    rho_c = cosmo.critical_density(z_lens)
    Mvir = (4 * np.pi) / 3 * (200 * rho_c) * rvir**3
    return Mvir.decompose().to(u_out)

def get_rho_s_NFW(Mvir, z_lens, c_nfw=None, u_out=(u.Msun/u.pc**3)):
    '''
    Constant density factor in the NFW density profile, rho_s.
    It is obtained enforcing the condition the condition M(r_vir)=Mvir.
    '''
    if not c_nfw:
        c_nfw = get_c_NFW(Mvir, z_lens)
    elif type(c_nfw)==type(get_c_NFW):
        c_nfw = c_nfw(Mvir, z_lens)

    rvir = get_rvir_NFW(Mvir, z_lens)
    return ((c_nfw**3*(1 + c_nfw)*Mvir*u.Msun)/(4.*np.pi*rvir**3*(-c_nfw + (1 + c_nfw)*np.log(1 + c_nfw)))).decompose().to(u_out)
    

def get_rs_NFW(Mvir, z_lens, c_nfw=None, u_out=u.Mpc):
    """
    Calculate the scale radius (rs) for a halo with Delta=200c using the concentration parameter.

    :param Mvir: Virial mass in solar masses.
    :param z_lens: Redshift.
    :param u: Output units for the scale radius (default is Mpc).
    :return: Scale radius rs.
    """
    if not c_nfw:
        c_nfw = get_c_NFW(Mvir, z_lens)
    elif type(c_nfw)==type(get_c_NFW):
        c_nfw = c_nfw(Mvir, z_lens)

  
    rvir = get_rvir_NFW(Mvir, z_lens)
    rs = rvir / c_nfw
    return rs.to(u_out)

def get_M_NFW(MLz, z_lens):
    return MLz/(1+z_lens)

def to_MLz_NFW(Mvir, z_lens, c_nfw=None):
    """
    Convert from the virial mass (Mvir) to MLz.

    :param Mvir: Virial mass in solar masses.
    :param z_lens: Redshift.
    :param c: concentration parameter
    :param u: Output units for the lensing mass (default is solar masses).
    :return: Lensing mass MLz.
    """
    if not c_nfw:
        c_nfw = get_c_NFW(Mvir, z_lens)
    elif type(c_nfw)==type(get_c_NFW):
        c_nfw = c_nfw(Mvir, z_lens)

    c_factor= (np.log(1+c_nfw)-c_nfw/(1+c_nfw))
    return (1+z_lens) / c_factor * Mvir

def to_Mvir_NFW(MLz, z_lens, c_nfw):
    """
    Convert from the virial mass (Mvir) to MLz.

    :param Mvir: Virial mass in solar masses.
    :param z_lens: Redshift.
    :param c: concentration parameter
    :param u: Output units for the lensing mass (default is solar masses).
    :return: Lensing mass MLz.
    """
    c_factor= (np.log(1+c_nfw)-c_nfw/(1+c_nfw))
    return c_factor / (1+z_lens) * MLz

def get_Psis_NFW(Mvirs_NFW, z_lens, z_src, c_nfw=None, norm_factor=1, p_prec_NFW={}):

    p_prec={'eps_soft': 1e-15, 'eps_NFW': 0.01}
    p_prec.update(p_prec_NFW)
    # Calculate lensing masses (MLzs) for NFW profile
    MLzs_NFW = [to_MLz_NFW(Mvir, z_lens, c_nfw=c_nfw) for Mvir in Mvirs_NFW]

    # Calculate the effective distance ratio (deff)
    deff = get_deff(z_lens, z_src)

    # Calculate the Einstein radius (xi_0) for NFW profile and normalize it
    xi_0_NFW_s = [norm_factor * get_R_E(MLz, deff) for MLz in MLzs_NFW]

    # Calculate the scale radius (rs) for NFW profile
    rs_s = [get_rs_NFW(Mvir, z_lens, c_nfw=c_nfw) for Mvir in Mvirs_NFW]

    # Calculate the dimensionless scale parameter xs = rs / xi_0 and extract the values
    xs_s = [(rs / xi_0).decompose().value for (rs, xi_0) in zip(rs_s, xi_0_NFW_s)]

    norm_factor = [1 / norm_factor] * len(xs_s)

    # Compute potential

    Psi_NFW_s=[lenses.Psi_NFW(p_phys={'psi0':(norm_factor[i])**2,'xs':xs},
                            p_prec = p_prec) for i, xs in enumerate(xs_s)]

    return Psi_NFW_s, MLzs_NFW