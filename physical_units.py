#
# GLoW - physical_units.py
#
# Copyright (C) 2024, Stefano Savastano, Hector Villarrubia-Rojo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import warnings
import numpy as np

from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck18

from . import lenses

# optional dependency: needed for NFW concentration-Mvir relation
try:
    from colossus.cosmology import cosmology as col_cosmology
    from colossus.halo import concentration as col_concentration

    col_cosmology.setCosmology('planck18')
except ModuleNotFoundError:
    col_concentration = None


class PhysException(Exception):
    pass

class PhysWarning(UserWarning):
    pass

##==============================================================================


class Units_from_Mlz():
    def __init__(self, Mlz_Msun):
        self.Mlz = Mlz_Msun*u.Msun

        self.F     = self.get_F()
        self.F_inv = (0.5/np.pi/self.F).to(u.Hz)

    def _get_unit(self, u_str):
        return eval("u.%s" % u_str)

    def _add_unit(self, x, u_str=None):
        if u_str is None:
            # if we don't specify units, check that x has them, then we don't do anything
            try:
                unit_x = x.unit
            except AttributeError:
                message = "the input quantity has no astropy units, but no unit has been manually specified."
                raise PhysException(message)
        else:
            # add the units that we specified, only if x does not have already
            try:
                unit_x = x.unit
                message = f"the input quantity has astropy units = {unit_x}, "\
                          f"but some external unit has been manually specified ({u_str}). Choose one."
                raise PhysException(message)
            except AttributeError:
                pass

            un = self._get_unit(u_str)
            x = x*un

        return x

    def _strip_unit(self, x, u_str=None):
        if u_str is not None:
            un = self._get_unit(u_str)
            x = x.to(un).value
        return x

    def get_F(self):
        return (4*c.G*self.Mlz/c.c**3).to(u.s)

    ## --------------------------------
    def w_to_f(self, w, un=None):
        f = self.F_inv*w
        return self._strip_unit(f, un)

    def f_to_w(self, f, un=None):
        f = self._add_unit(f, un)
        w = (f/self.F_inv).decompose().value
        return w

    def tau_to_t(self, tau, un=None):
        t = self.F*tau
        return self._strip_unit(t, un)

    def t_to_tau(self, t, un=None):
        t = self._add_unit(t, un)
        tau = (t/self.F).decompose().value
        return tau
    ## --------------------------------


class UnitsGeneral(Units_from_Mlz):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        # check for 'external' and units_ext (not None iff external)
        if prescription == 'external':
            if units_ext == None:
                message = "the prescription has been set to 'external', but no units_ext has been specified"
                raise PhysException(message)

        if units_ext != None:
            if prescription != 'external':
                message = "some units_ext have been specified. For them to be used, the prescription "\
                          "must be set to 'external'."
                raise PhysException(message)

        # update all the distances and cosmological info
        self.update_dist(zl, zs, units_ext=units_ext)

        # update and check the parameters of the lens
        self.p_lens = p_lens
        self.prescription = prescription
        self.units_ext = units_ext
        self.update_lens_parameters()

        # xi0_default is the one that makes psi0 = 1
        self.xi0_default = self.get_xi0_default()

        self.prescriptions = {'default'  : self.xi0_default,
                              'angular'  : self.d_l,
                              'external' : units_ext.xi0 if units_ext is not None else None}
        self.add_prescriptions()

        # check that prescription chosen makes sense
        try:
            self.xi0 = self.prescriptions[prescription]
        except KeyError:
            message = f"prescription '{prescription}' not recognized. "\
                      f"Choose among: {list(self.prescriptions.keys())}"
            raise PhysException(message)

        # update the p_phys that we need for the lens (dimensionless)
        self.p_phys = self.get_pphys(self.xi0)

        Mlz_Msun = self.get_Mlz(self.xi0, self.d_eff).value
        super().__init__(Mlz_Msun)

    def check_lens_keys(self, m_keys, opt_keys=[]):
        total_keys = set(m_keys + opt_keys)
        input_keys = set([k for k in self.p_lens.keys() if k != 'name'])

        diff_keys = set(m_keys) - input_keys
        if diff_keys:
            message = f"missing the following mandatory keys in p_lens: {diff_keys}"
            raise PhysException(message)

        diff_keys = input_keys - total_keys
        if diff_keys:
            for key in diff_keys:
                message = "unrecognized key '%s' found in p_lens will be "\
                          "(most likely) ignored" % key
                warnings.warn(message, PhysWarning)

    def get_Mlz(self, xi0, d_eff):
        return ((c.c*xi0)**2/4./c.G/self.d_eff).to(u.Msun)

    def get_d_A(self, z, cosmo=Planck18):
        return cosmo.angular_diameter_distance(z)

    def get_d_A_z1z2(self, z1, z2, cosmo=Planck18):
        return cosmo.angular_diameter_distance_z1z2(z1, z2)

    def update_dist(self, zl, zs, cosmo=Planck18, units_ext=None):
        if units_ext is None:
            if zl > zs:
                message = "redshift of the lens (zl=%s) larger than the source (zs=%g)" % (zl, zs)
                raise PhysException(message)

            self.zl = zl
            self.zs = zs

            self.cosmo = cosmo
            self.d_l   = self.get_d_A(zl, cosmo)
            self.d_s   = self.get_d_A(zs, cosmo)
            self.d_ls  = self.get_d_A_z1z2(zl, zs, cosmo)
            self.d_eff = self.d_ls*self.d_l/self.d_s/(1+zl)
            self.rho_crit   = self.cosmo.critical_density(zl).to(u.Msun/u.pc**3)
            self.Sigma_crit = (c.c**2/4/np.pi/c.G/(1+zl)/self.d_eff).to(u.Msun/u.pc**2)
        else:
            self.zl = units_ext.zl
            self.zs = units_ext.zs

            self.cosmo = units_ext.cosmo
            self.d_l   = units_ext.d_l
            self.d_s   = units_ext.d_s
            self.d_ls  = units_ext.d_ls
            self.d_eff = units_ext.d_eff
            self.rho_crit   = units_ext.rho_crit
            self.Sigma_crit = units_ext.Sigma_crit

    def get_Rvir_from_Mvir(self, Mvir, Delta=200):
        Rvir = (3*Mvir/4./np.pi/Delta/self.rho_crit)**(1./3)
        return Rvir.to(u.kpc)

    ## --------------------------------
    def x_to_xi(self, x, un=None, xi0=None):
        xi0 = self.xi0 if xi0 is None else xi0

        xi = x*xi0
        return self._strip_unit(xi, un)

    def xi_to_x(self, xi, un=None, xi0=None):
        xi0 = self.xi0 if xi0 is None else xi0

        xi = self._add_unit(xi, un)
        x = (xi/xi0).decompose().value
        return x
    ## --------------------------------

    def update_lens_parameters(self):
        pass

    def get_xi0_default(self):
        return None

    def add_prescriptions(self):
        pass

    def get_psi0(self, xi0):
        psi0 = (self.xi0_default/xi0)**2
        return psi0.decompose().value

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0)}
        return p_phys


##==============================================================================
##==============================================================================


class Units_SIS(UnitsGeneral):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['Mvir_Msun'])

        self.Mvir = self.p_lens['Mvir_Msun']*u.Msun

    def get_sigma_v(self):
        H_zl = self.cosmo.H(self.zl)
        sigmav3 = (5.*np.sqrt(6)/2.)*c.G*H_zl*self.Mvir
        sigmav = (sigmav3.decompose())**(1./3.)
        return sigmav.to(u.km/u.s)

    def get_xi0_default(self):
        self.Rvir = self.get_Rvir_from_Mvir(self.Mvir)
        self.sigmav = self.get_sigma_v()
        xi0 = (self.sigmav**2/c.G/self.Sigma_crit).decompose().to(u.pc)
        return xi0

    def get_psi0(self, xi0):
        psi0 = self.xi0_default/xi0
        return psi0.decompose().value

class Units_offcenterSIS(Units_SIS):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['Mvir_Msun', 'x1_kpc', 'x2_kpc'])

        self.Mvir = self.p_lens['Mvir_Msun']*u.Msun
        self.x1   = self.p_lens['x1_kpc']*u.kpc
        self.x2   = self.p_lens['x2_kpc']*u.kpc

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'xc1'  : self.xi_to_x(self.x1, xi0=xi0),
                  'xc2'  : self.xi_to_x(self.x2, xi0=xi0)}
        return p_phys

class Units_CIS(Units_SIS):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['Mvir_Msun', 'rc_pc'])

        self.Mvir = self.p_lens['Mvir_Msun']*u.Msun
        self.rc   = self.p_lens['rc_pc']*u.pc

    def add_prescriptions(self):
        self.prescriptions['radius'] = self.rc

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'rc'   : self.xi_to_x(self.rc, xi0=xi0)}
        return p_phys

class Units_offcenterCIS(Units_SIS):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['Mvir_Msun', 'rc_pc', 'x1_kpc', 'x2_kpc'])

        self.Mvir = self.p_lens['Mvir_Msun']*u.Msun
        self.rc   = self.p_lens['rc_pc']*u.pc
        self.x1   = self.p_lens['x1_kpc']*u.kpc
        self.x2   = self.p_lens['x2_kpc']*u.kpc

    def add_prescriptions(self):
        self.prescriptions['radius'] = self.rc

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'rc'   : self.xi_to_x(self.rc, xi0=xi0),
                  'xc1'  : self.xi_to_x(self.x1, xi0=xi0),
                  'xc2'  : self.xi_to_x(self.x2, xi0=xi0)}
        return p_phys

class Units_PointLens(UnitsGeneral):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['M_Msun'])

        self.M = self.p_lens['M_Msun']*u.Msun

    def get_RE(self):
        RE = np.sqrt(4*c.G*self.M/c.c**2*self.d_eff*(1+self.zl))
        return RE.decompose().to(u.pc)

    def get_xi0_default(self):
        self.RE = self.get_RE()
        return self.RE

class Units_offcenterPointLens(Units_PointLens):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['M_Msun', 'x1_pc', 'x2_pc'])

        self.M  = self.p_lens['M_Msun']*u.Msun
        self.x1 = self.p_lens['x1_pc']*u.pc
        self.x2 = self.p_lens['x2_pc']*u.pc

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'xc1'  : self.xi_to_x(self.x1, xi0=xi0),
                  'xc2'  : self.xi_to_x(self.x2, xi0=xi0)}
        return p_phys

class Units_Ball(Units_PointLens):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['M_Msun', 'x1_pc', 'x2_pc'])

        self.M  = self.p_lens['M_Msun']*u.Msun
        self.R  = self.p_lens['R_Rsun']*u.Rsun

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'b'    : self.xi_to_x(self.R, xi0=xi0)}
        return p_phys

class Units_offcenterBall(Units_PointLens):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['M_Msun', 'R_Rsun', 'x1_pc', 'x2_pc'])

        self.M  = self.p_lens['M_Msun']*u.Msun
        self.R  = self.p_lens['R_Rsun']*u.Rsun
        self.x1 = self.p_lens['x1_pc']*u.pc
        self.x2 = self.p_lens['x2_pc']*u.pc

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'b'    : self.xi_to_x(self.R,  xi0=xi0),
                  'xc1'  : self.xi_to_x(self.x1, xi0=xi0),
                  'xc2'  : self.xi_to_x(self.x2, xi0=xi0)}
        return p_phys

class Units_NFW(UnitsGeneral):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        # here Mvir_Msun is mandatory and c_NFW is optional
        self.check_lens_keys(['Mvir_Msun'], ['c_NFW'])

        self.update_NFW()

    def update_NFW(self):
        self.Mvir = self.p_lens['Mvir_Msun']*u.Msun
        self.Rvir = self.get_Rvir_from_Mvir(self.Mvir)

        try:
            cc = self.p_lens['c_NFW']
        except KeyError:
            cc = self.get_colossus_concentration()

        self.c_NFW = cc
        self.M_NFW = self.Mvir/(np.log(1+cc) - cc/(1+cc))

        self.Rs = self.Rvir/cc

    def get_colossus_concentration(self):
        if col_concentration is None:
            message = "It seems that the python module colossus is not installed. Without it, the NFW "\
                      "concentration parameter p['c_NFW'] must be set manually."
            raise PhysException(message)

        return col_concentration.concentration(self.Mvir.value, '200c', self.zl, model='ishiyama21')

    def add_prescriptions(self):
        self.prescriptions['radius'] = self.Rs

    def get_xi0_default(self):
        xi0 = np.sqrt(4*c.G/c.c**2*self.d_eff*(1+self.zl)*self.M_NFW).decompose()
        return xi0.to(u.pc)

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'xs'   : self.xi_to_x(self.Rs, xi0=xi0)}
        return p_phys

class Units_offcenterNFW(Units_NFW):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        # here Mvir_Msun is mandatory and c_NFW is optional
        self.check_lens_keys(['Mvir_Msun', 'x1_kpc', 'x2_kpc'], ['c_NFW'])

        self.update_NFW()
        self.x1   = self.p_lens['x1_kpc']*u.kpc
        self.x2   = self.p_lens['x2_kpc']*u.kpc

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'xs'   : self.xi_to_x(self.Rs, xi0=xi0),
                  'xc1'  : self.xi_to_x(self.x1, xi0=xi0),
                  'xc2'  : self.xi_to_x(self.x2, xi0=xi0)}
        return p_phys

class Units_tSIS(UnitsGeneral):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['M_Msun', 'R_kpc'])

        self.M = self.p_lens['M_Msun']*u.Msun
        self.R = self.p_lens['R_kpc']*u.kpc

    def add_prescriptions(self):
        self.prescriptions['radius'] = self.R

    def get_xi0_default(self):
        xi0 = self.M/np.sqrt(np.pi)/self.Sigma_crit/self.R
        return xi0.decompose().to(u.pc)

    def get_psi0(self, xi0):
        psi0 = self.xi0_default/xi0
        return psi0.decompose().value

    def get_pphys(self, xi0):
        p_phys = {'psi0' : self.get_psi0(xi0),
                  'xb' : self.xi_to_x(self.R, xi0=xi0)}
        return p_phys

class Units_eSIS(Units_SIS):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        self.check_lens_keys(['Mvir_Msun', 'q', 'alpha', 'x1_kpc', 'x2_kpc'])

        self.Mvir  = self.p_lens['Mvir_Msun']*u.Msun
        self.q     = self.p_lens['q']
        self.alpha = self.p_lens['alpha']
        self.x1    = self.p_lens['x1_kpc']*u.kpc
        self.x2    = self.p_lens['x2_kpc']*u.kpc

    def get_pphys(self, xi0):
        p_phys = {'psi0'  : self.get_psi0(xi0),
                  'q'     : self.q,
                  'alpha' : self.alpha,
                  'xc1'   : self.xi_to_x(self.x1, xi0=xi0),
                  'xc2'   : self.xi_to_x(self.x2, xi0=xi0)}
        return p_phys

class Units_Ext(UnitsGeneral):
    def __init__(self, zl, zs, p_lens, prescription='default', units_ext=None):
        super().__init__(zl, zs, p_lens, prescription, units_ext)

    def update_lens_parameters(self):
        m_keys = ['kappa', 'gamma1', 'gamma2']

        if self.prescription != 'external':
            m_keys.append('xi0_kpc')

        self.check_lens_keys(m_keys)

        self.kappa  = self.p_lens['kappa']
        self.gamma1 = self.p_lens['gamma1']
        self.gamma2 = self.p_lens['gamma2']

        if self.prescription != 'external':
            self.xi0_ext = self.p_lens['xi0_kpc']*u.kpc
        else:
            self.xi0_ext = 1*u.kpc   ## any value, just to define the variable

    def get_xi0_default(self):
        return self.xi0_ext

    def get_psi0(self, xi0):
        # not relevant for this lens
        return 1

    def get_pphys(self, xi0):
        p_phys = {'kappa'  : self.kappa,
                  'gamma1' : self.gamma1,
                  'gamma2' : self.gamma2}
        return p_phys


##==============================================================================
##==============================================================================

# init functions
inits = {}
inits['SIS']                   = (Units_SIS,                lenses.Psi_SIS)
inits['off-center SIS']        = (Units_offcenterSIS,       lenses.Psi_offcenterSIS)
inits['CIS']                   = (Units_CIS,                lenses.Psi_CIS)
inits['off-center CIS']        = (Units_offcenterCIS,       lenses.Psi_offcenterCIS)
inits['point lens']            = (Units_PointLens,          lenses.Psi_PointLens)
inits['off-center point lens'] = (Units_offcenterPointLens, lenses.Psi_offcenterPointLens)
inits['ball']                  = (Units_Ball,               lenses.Psi_Ball)
inits['off-center ball']       = (Units_offcenterBall,      lenses.Psi_offcenterBall)
inits['NFW']                   = (Units_NFW,                lenses.Psi_NFW)
inits['off-center NFW']        = (Units_offcenterNFW,       lenses.Psi_offcenterNFW)
inits['tSIS']                  = (Units_tSIS,               lenses.Psi_tSIS)
inits['eSIS']                  = (Units_eSIS,               lenses.Psi_eSIS)
inits['ext']                   = (Units_Ext,                lenses.Psi_Ext)

##==============================================================================
##==============================================================================

## WRAPPERS

def Units(zl, zs, p_lens, prescription='default', units_ext=None):
    name = p_lens['name']
    units_t, lens_t = inits[name]

    return units_t(zl, zs, p_lens, prescription, units_ext)

def _init_single_lens_units(zl, zs, p_lens, prescription='default', units_ext=None, p_prec={}):
    try:
        name = p_lens['name']
    except KeyError:
        message = "name of the lens not found in p_lens. Available lenses are p_lens['name'] = "\
                  f"{list(inits.keys())}"
        raise PhysException(message)

    try:
        units_t, lens_t = inits[name]
    except KeyError:
        message = f"name of the lens ('{name}') not recognized. Valid names are p_lens['name'] = "\
                  f"{list(inits.keys())}"
        raise PhysException(message)

    units = units_t(zl, zs, p_lens, prescription, units_ext)
    Psi   = lens_t(units.p_phys, p_prec)

    return Psi, units

def Lens_Units(zl, zs, p_lens, prescription='default', units_ext=None, p_prec=None):
    p_lens = np.array(p_lens).flatten()
    p_lens_0 = p_lens[0]

    if p_prec is not None:
        p_prec = np.array(p_prec).flatten()
        p_prec_0 = p_prec[0]

        if p_prec.size != p_lens.size:
            message = "for composite lenses, the length of the (optional) precision parameters p_prec must match the number of lenses."
            raise PhysException(message)
    else:
        p_prec = np.full_like(p_lens, {}, dtype=dict)
        p_prec_0 = {}

    Psi0, units = _init_single_lens_units(zl, zs, p_lens_0, prescription, units_ext, p_prec=p_prec_0)

    if p_lens.size > 1:
        # we have a composite lens
        Psis = [Psi0]
        for p, pp in zip(p_lens[1:], p_prec[1:]):
            l, un = _init_single_lens_units(zl, zs, p, prescription='external', units_ext=units, p_prec=pp)
            Psis.append(l)

        Psi = lenses.CombinedLens({'lenses':Psis})
    else:
        Psi = Psi0

    return Psi, units
