#
# GLoW - lenses.py
#
# Copyright (C) 2023, Hector Villarrubia-Rojo
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
import scipy.special as sc_special

class LensException(Exception):
    pass

class LensWarning(UserWarning):
    pass

##==============================================================================


class PsiGeneral():
    """Base class for a generic non-axisymmetric lens.

    Parameters
    ----------
    p_phys : dict, optional
        Physical parameters of the lens.
    p_prec : dict, optional
        Precision parameters of the lens.

    Attributes
    ----------
    p_phys : dict
        Default parameters updated with the input.
    p_prec : dict
        Default parameters updated with the input.
    isAxisym : bool, default=False
        Allows to easily differentiate a general lens from the special
        axisymmetric case (instance of :class:`PsiAxisym`).
    asymp_index : float or None, default=None
    asymp_amplitude : float or None, default=None
        If the asymptotic behaviour of the lens is

        .. math ::
            \\psi'\\sim\\frac{A_\\psi}{r^\\gamma}\\ ,\\quad r\\to\\infty

        the asymptotic index and amplitude are defined as

        .. math ::
            \\begin{align}
                \\gamma_\\text{asymp} &= (\\gamma+1)/2\\\\
                A_\\text{asymp} &= A_\\psi/2^{\\gamma_\\text{asymp}}
            \\end{align}
    """
    def __init__(self, p_phys={}, p_prec={}):
        self.isAxisym = False

        self.p_phys, self.p_prec = self.default_general_params()
        self.p_phys_default_keys = set(self.p_phys.keys())
        self.p_prec_default_keys = set(self.p_prec.keys())

        self.p_phys.update(p_phys)
        self.p_prec.update(p_prec)

        self.check_general_input()

        # to be overriden by the subclass
        self.asymp_index = None
        self.asymp_amplitude = None
        # *******************************

    def __str__(self, idx=''):
        """Create the python call needed to replicate this object."""
        class_name = type(self).__name__
        class_call = "Psi%s = lenses." % idx + class_name + "(p_phys, p_prec)"

        phys_message = "p_phys = " + self.p_phys.__repr__() + "\n"
        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        return phys_message + prec_message + class_call

    def default_general_params(self):
        """Fill the default parameters.

        Update the parameters common for all lenses (none in this case).
        Then call :func:`default_params` (*subclass defined*).

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Default precision parameters.
        """
        p_phys = {}
        p_prec = {}

        # HVR -> weird behaviour (getting None) when
        #        updating an empty dictionary twice.
        #        Come back to this and clean this mess
        p_phys2, p_prec2 = self.default_params()
        if p_phys2 is not {}:
            p_phys.update(p_phys2)
        if p_prec2 is not {}:
            p_prec.update(p_prec2)

        return p_phys, p_prec

    def check_general_input(self):
        """Check the input upon initialization.

        It first calls :func:`check_input` (*subclass defined*)
        to perform any checks that the user desires. It then checks that
        the input only updates existing keys in :attr:`p_phys` and :attr:`.p_prec`,
        otherwise throws a warning.

        The idea is that we do not use a wrong name that is then ignored.

        Warns
        -----
        LensWarning

        Warnings
        --------
        If the subclass will add a new parameter without an entry in :func:`default_params`,
        it must be manually added to :attr:`self.p_phys_default_keys` or
        :attr:`self.p_prec_default_keys` in :func:`check_input`
        with ``self.p_phys_default_keys.add(new_key)``.
        """
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
                warnings.warn(message, LensWarning)

        if diff_prec:
            for key in diff_prec:
                message = "unrecognized key '%s' found in p_prec will be "\
                          "(most likely) ignored" % key
                warnings.warn(message, LensWarning)

    def display_info(self):
        """Print internal information in human-readable form."""
        print("\t////////////////////////////\n"\
              "\t///   Lens information   ///\n"\
              "\t////////////////////////////")

        if self.p_phys != {}:
            print("\n * Name: %s" % self.p_phys.get('name', 'no information'))

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

        print('')

    def phi_Fermat(self, x1, x2, y):
        r"""Fermat potential.

        Defined *without* setting the minimum time-delay to zero

        .. math::
            \phi_F(x_1, x_2, y) = \frac{1}{2}(x_1^2+x_2^2+y^2) -x_1y -\psi(x_1, x_2)

        Parameters
        ----------
        x1 : float or array
            Coordinate x1 (x1 is defined aligned with y).
        x2 : float or array
            Coordinate x2.
        y : float or array
            Impact parameter.

        Returns
        -------
        phi : float or array
        """
        phi_geo = 0.5*(x1*x1 + x2*x2) - x1*y + 0.5*y*y
        return phi_geo - self.psi(x1, x2)

    def dphi_Fermat_vec(self, x1, x2, y):
        """Gradient of the Fermat potential.

        Parameters
        ----------
        x1 : float or array
            Coordinate x1 (x1 is defined aligned with y).
        x2 : float or array
            Coordinate x2.
        y : float or array
            Impact parameter.

        Returns
        -------
        d1 : float or array
            Derivative of the Fermat potential wrt x1.
        d2 : float or array
            Derivative of the Fermat potential wrt x2.
        """
        dpsi_1, dpsi_2 = self.dpsi_vec(x1, x2)

        d1 = x1 - y - dpsi_1
        d2 = x2 - dpsi_2

        return d1, d2

    def ddphi_Fermat_vec(self, x1, x2):
        """Hessian of the Fermat potential.

        Parameters
        ----------
        x1 : float or array
            Coordinate x1.
        x2 : float or array
            Coordinate x2.

        Returns
        -------
        d11 : float or array
            2nd derivative of the Fermat potential wrt x1 twice.
        d12 : float or array
            2nd derivative of the Fermat potential wrt x1 and x2.
        d22 : float or array
            2nd derivative of the Fermat potential wrt x2 twice.
        """
        ddpsi_11, ddpsi_12, ddpsi_22 = self.ddpsi_vec(x1, x2)

        d11 = 1 - ddpsi_11
        d12 = - ddpsi_12
        d22 = 1 - ddpsi_22

        return d11, d12, d22

    def dphi_Fermat_dx1(self, x1, x2, y):
        """Return only **d1** from :func:`dphi_Fermat_vec`."""
        d1, d2 = self.dphi_Fermat_vec(x1, x2, y)
        return d1

    def dphi_Fermat_dx2(self, x1, x2, y):
        """Return only **d2** from :func:`dphi_Fermat_vec`."""
        d1, d2 = self.dphi_Fermat_vec(x1, x2, y)
        return d2

    def ddphi_Fermat_ddx1(self, x1, x2):
        """Return only **d11** from :func:`ddphi_Fermat_vec`."""
        d11, d12, d22 = self.ddphi_Fermat_vec(x1, x2)
        return d11

    def ddphi_Fermat_ddx2(self, x1, x2):
        """Return only **d22** from :func:`ddphi_Fermat_vec`."""
        d11, d12, d22 = self.ddphi_Fermat_vec(x1, x2)
        return d22

    def ddphi_Fermat_dx1dx2(self, x1, x2):
        """Return only **d12** from :func:`ddphi_Fermat_vec`."""
        d11, d12, d22 = self.ddphi_Fermat_vec(x1, x2)
        return d12

    def dpsi_dx1(self, x1, x2):
        """Return only **d1** from :func:`dpsi_vec`."""
        d1, d2 = self.dpsi_vec(x1, x2)
        return d1

    def dpsi_dx2(self, x1, x2):
        """Return only **d2** from :func:`dpsi_vec`."""
        d1, d2 = self.dpsi_vec(x1, x2)
        return d2

    def ddpsi_ddx1(self, x1, x2):
        """Return only **d11** from :func:`ddpsi_vec`."""
        d11, d12, d22 = self.ddpsi_vec(x1, x2)
        return d11

    def ddpsi_ddx2(self, x1, x2):
        """Return only **d22** from :func:`ddpsi_vec`."""
        d11, d12, d22 = self.ddpsi_vec(x1, x2)
        return d22

    def ddpsi_dx1dx2(self, x1, x2):
        """Return only **d12** from :func:`ddpsi_vec`."""
        d11, d12, d22 = self.ddpsi_vec(x1, x2)
        return d12

    def shear(self, x1, x2):
        r"""Magnification matrix.

        The Jacobian matrix for the lens mapping is [1]_

        .. math::
            A_{ij} = \frac{\partial y_i}{\partial x_j} = \partial_i\partial_j\phi_F
                = \delta_{ij} - \partial_i\partial_j\psi

        It can be written as

        .. math::
            A = \begin{pmatrix}1-\kappa-\gamma_1 & -\gamma_2\\
                               -\gamma_2 & 1-\kappa+\gamma_1
                \end{pmatrix}

        The *convergence*, :math:`\kappa`, and *shear*, :math:`\gamma`, can be
        expressed in terms of the second derivatives of the lensing potential

        .. math::
            \begin{align}
                \kappa   &= (\psi_{11} + \psi_{22})/2\\
                \gamma_1 &= (\psi_{11} - \psi_{22})/2\\
                \gamma_2 &= \psi_{12}\\
                \gamma   &= \sqrt{\gamma_1^2 + \gamma_2^2}
            \end{align}

        The eigenvalues of :math:`A` are

        .. math::
            \begin{align}
                \lambda_1  &= 1 - \kappa - \gamma\\
                \lambda_2  &= 1 - \kappa + \gamma\\
                \text{tr}\,A &= \lambda_1+\lambda_2 = 2(1-\kappa)\\
                \det A     &= \lambda_1\lambda_2 = (1-\kappa)^2-\gamma^2
            \end{align}

        Finally, the *magnification*, :math:`\mu`, is defined as

        .. math::
            \mu = \frac{1}{|\det A|}

        Parameters
        ----------
        x1, x2 : float or array
            Coordinates.

        Returns
        -------
        d : dict
            Dictionary with keys:

            * ``gamma1`` (*float or array*)  -- :math:`\gamma_1`
            * ``gamma2`` (*float or array*)  -- :math:`\gamma_2`
            * ``kappa`` (*float or array*)   -- :math:`\kappa`
            * ``gamma`` (*float or array*)   -- :math:`\gamma`
            * ``lambda1`` (*float or array*) -- :math:`\lambda_1`
            * ``lambda2`` (*float or array*) -- :math:`\lambda_2`
            * ``detA`` (*float or array*)    -- :math:`\det A`
            * ``trA`` (*float or array*)     -- :math:`\text{tr}\,A`
            * ``mag`` (*float or array*)     -- :math:`\mu`

        References
        ----------
        .. [1] \ P. Schneider, J. Ehlers, and E. E. Falco, Gravitational Lenses, (1992).

        """
        d11, d12, d22 = self.ddpsi_vec(x1, x2)

        d = {}
        d['gamma1']  = 0.5*(d11 - d22)
        d['gamma2']  = d12
        d['kappa']   = 0.5*(d11 + d22)
        d['gamma']   = np.sqrt(d['gamma1']**2 + d['gamma2']**2)
        d['lambda1'] = 1 - d['kappa'] - d['gamma']
        d['lambda2'] = 1 - d['kappa'] + d['gamma']
        d['detA']    = d['lambda1']*d['lambda2']
        d['trA']     = d['lambda1'] + d['lambda2']
        d['mag']     = 1/np.abs(d['detA'])

        return d

    # ***** to be overriden by the subclass *****
    def default_params(self):
        """ (*subclass defined*) Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Default precision parameters.
        """
        p_phys = {'name' : 'unknown'}
        p_prec = {}
        return p_phys, p_prec

    def check_input(self):
        """(*subclass defined, optional*) Check the input of the lens implementation."""
        pass

    def psi(self, x1, x2):
        """(*subclass defined*) Lensing potential.

        Parameters
        ----------
        x1, x2 : float or array
            Coordinates.

        Returns
        -------
        psi : float or array or None, default=None
        """
        return None

    def dpsi_vec(self, x1, x2):
        """(*subclass defined*) Gradient of the lensing potential.

        Parameters
        ----------
        x1, x2 : float or array
            Coordinates.

        Returns
        -------
        d1 : float or array or None, default=None
            Derivative of the lensing potential wrt x1.
        d2 : float or array or None, default=None
            Derivative of the lensing potential wrt x2.
        """
        d1, d2 = None, None
        return d1, d2

    def ddpsi_vec(self, x1, x2):
        """(*subclass defined*) Hessian of the lensing potential.

        Parameters
        ----------
        x1, x2 : float or array
            Coordinate x1.

        Returns
        -------
        d11 : float or array or None, default=None
            2nd derivative of the lensing potential wrt x1 twice.
        d12 : float or array or None, default=None
            2nd derivative of the lensing potential wrt x1 and x2.
        d22 : float or array or None, default=None
            2nd derivative of the lensing potential wrt x2 twice.
        """
        d11, d12, d22 = None, None, None
        return d11, d12, d22
    # *******************************************

class PsiAxisym(PsiGeneral):
    """Base class for an axisymmetric lens.

    Parameters
    ----------
    p_phys : dict, optional
        Physical parameters of the lens.
    p_prec : dict, optional
        Precision parameters of the lens.

    Attributes
    ----------
    isAxisym : bool, default=True
        Allows to easily differentiate this class from :class:`PsiGeneral`.
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.isAxisym = True

    def psi(self, x1, x2):
        r"""Wrapper for :func:`psi_x`.

        This method allows to use axisymmetric lenses with algorithms designed only
        for generic lenses. It defines the general lensing potential from its
        symmetric version as

        .. math::
            \psi(x_1, x_2) = \psi\left(x=\sqrt{x_1^2+x_2^2}\right)

        Parameters
        ----------
        x1, x2 : float or array
            Coordinates.

        Returns
        -------
        psi : float or array
            Lensing potential at (x1, x2).
        """
        x = np.sqrt(x1*x1 + x2*x2)
        return self.psi_x(x)

    def dpsi_vec(self, x1, x2):
        r"""Wrapper for :func:`dpsi_dx`.

        The derivatives are evaluated as
         .. math::
            \begin{align}
                \partial_1\psi &= \psi'\frac{x_1}{x}\\
                \partial_2\psi &= \psi'\frac{x_2}{x}
            \end{align}

        Parameters
        ----------
        x1, x2 : float or array
            Coordinates.

        Returns
        -------
        d1 : float or array
            :math:`\partial_1\psi`
        d2 : float or array
            :math:`\partial_2\psi`

        """
        x = np.sqrt(x1*x1 + x2*x2)
        dpsi = self.dpsi_dx(x)

        d1 = x1*dpsi/x
        d2 = x2*dpsi/x

        return d1, d2

    def ddpsi_vec(self, x1, x2):
        r"""Wrapper for :func:`ddpsi_ddx`.

        The derivatives are evaluated as
         .. math::
            \begin{align}
                \partial_{11}\psi &= (1-r_1^2)\psi'/x + r_1^2\psi''\\
                \partial_{12}\psi &= r_1r_2(\psi'' - \psi'/x)\\
                \partial_{22}\psi &= (1-r_2^2)\psi'/x + r_2^2\psi''
            \end{align}

        with :math:`r_{1,2}=x_{1,2}/x`.

        Parameters
        ----------
        x1, x2 : float or array
            Coordinates x1.

        Returns
        -------
        d11 : float or array
            :math:`\partial_{11}\psi`
        d12 : float or array
            :math:`\partial_{12}\psi`
        d22 : float or array
            :math:`\partial_{22}\psi`
        """
        x = np.sqrt(x1*x1 + x2*x2)
        dpsi = self.dpsi_dx(x)
        ddpsi = self.ddpsi_ddx(x)

        r1 = (x1/x)
        r2 = (x2/x)

        d11 = dpsi/x*(1 - r1*r1) + ddpsi*r1*r1
        d12 = r1*r2*(ddpsi - dpsi/x)
        d22 = dpsi/x*(1 - r2*r2) + ddpsi*r2*r2

        return d11, d12, d22

    def to_file(self, fname, xmin, xmax, Nx, extension='_lens.dat'):
        """Evaluate the lensing potential and its derivatives on a grid and store it.

        This method is called when a lens that has not been implemented in C is
        used in the C part of the code.
        The lens is then evaluated from :obj:`xmin` to :obj:`xmax` in a logarithmic grid with
        :obj:`Nx` points, stored and later read and interpolated from C. The files are
        stored in:

        * :obj:`fname` + ``'_psi'`` + :obj:`extension`
        * :obj:`fname` + ``'_dpsi'`` + :obj:`extension`
        * :obj:`fname` + ``'_ddpsi'`` + :obj:`extension`

        Notes
        -----
        This method is intended for internal use, but can be used to save the lens to disk.

        Parameters
        ----------
        fname : str
            Root of the file name.
        xmin : float
            Minimum radius in the grid.
        xmax : float
            Maximum radius in the grid.
        Nx : float
            Number of grid points.
        extension : str, optional
            Extension of the file.
        """
        xs = np.geomspace(xmin, xmax, Nx)

        psis = self.psi_x(xs)
        dpsis = self.dpsi_dx(xs)
        ddpsis = self.ddpsi_ddx(xs)

        np.savetxt(fname+'_psi'+extension, np.transpose([xs, psis]))
        np.savetxt(fname+'_dpsi'+extension, np.transpose([xs, dpsis]))
        np.savetxt(fname+'_ddpsi'+extension, np.transpose([xs, ddpsis]))

    # ***** to be overriden by the subclass *****
    def psi_x(self, x):
        """(*subclass defined*) Lensing potential.

        Parameters
        ----------
        x : float or array
            Radius.

        Returns
        -------
        psi : float or array or None, default=None
        """
        return None

    def dpsi_dx(self, x):
        """(*subclass defined*) First derivative of the lensing potential.

        Parameters
        ----------
        x : float or array
            Radius.

        Returns
        -------
        dpsi : float or array or None, default=None
            Derivative of the lensing potential wrt x.
        """
        return None

    def ddpsi_ddx(self, x):
        """(*subclass defined*) Second derivative of the lensing potential.

        Parameters
        ----------
        x : float or array
            Radius.

        Returns
        -------
        ddpsi : float or array or None, default=None
            Second derivative of the lensing potential wrt x.
        """
        return None
    # *******************************************


##==============================================================================


class Psi_SIS(PsiAxisym):
    """Lens object for the singular isothermal sphere (SIS).

    Additional information: :ref:`theory <Psi_SIS_theory>`, :ref:`default parameters <Psi_SIS_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.

    Attributes
    ----------
    asymp_index : float
        0.5
    asymp_amplitude : float
        :math:`\\psi_0/\\sqrt{2}`
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']

        self.asymp_index = 0.5
        self.asymp_amplitude = self.psi0/np.sqrt(2)

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'SIS',
                  'psi0' : 1}
        p_prec = {}
        return p_phys, p_prec

    def psi_x(self, x):
        r"""Lensing potential.

        .. math::
            \psi(x) = \psi_0 x
        """
        return self.psi0*x

    def dpsi_dx(self, x):
        r"""First derivative of the lensing potential.

        .. math::
            \psi'(x) = \psi_0
        """
        return np.full_like(x, self.psi0)

    def ddpsi_ddx(self, x):
        r"""Second derivative of the lensing potential.

        .. math::
            \psi''(x) = 0
        """
        return np.zeros_like(x)


class Psi_CIS(PsiAxisym):
    """Lens object for the cored isothermal sphere (CIS).

    Additional information: :ref:`theory <Psi_CIS_theory>`, :ref:`default parameters <Psi_CIS_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``rc`` (*float*) -- Core radius.

    Attributes
    ----------
    asymp_index : float
        0.5
    asymp_amplitude : float
        :math:`\\psi_0/\\sqrt{2}`

    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.rc = self.p_phys['rc']

        self.asymp_index = 0.5
        self.asymp_amplitude = self.psi0/np.sqrt(2)

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'CIS',
                  'psi0' : 1,
                  'rc'   : 0.05}
        p_prec = {}
        return p_phys, p_prec

    def check_input(self):
        """Check that the core radius is not negative.

        Raises
        ------
        LensException
        """
        rc = self.p_phys['rc']
        if rc < 0:
            message = "rc = %g < 0 found. Core radius in the CIS must be positive" % rc
            raise LensException(message)

    def psi_x(self, x):
        r"""Lensing potential.

        .. math::
            \begin{align}
                r &\equiv \sqrt{x^2 + r_c^2}\\
                \psi(x) &= \psi_0 r + \psi_0 r_c\log\left(\frac{2r_c}{r+r_c}\right)
            \end{align}
        """
        sqr = np.sqrt(x*x + self.rc*self.rc)
        return self.psi0*(sqr + self.rc*np.log(2*self.rc/(sqr + self.rc)))

    def dpsi_dx(self, x):
        r"""First derivative of the lensing potential.

        .. math::
            \begin{align}
                r &\equiv \sqrt{x^2 + r_c^2}\\
                \psi'(x) &= \psi_0\frac{x}{r}\left(1 - \frac{r_c}{r + r_c}\right)
            \end{align}
        """
        sqr = np.sqrt(x*x + self.rc*self.rc)
        return self.psi0*x/sqr*(1 - self.rc/(sqr + self.rc))

    def ddpsi_ddx(self, x):
        r"""Second derivative of the lensing potential.

        .. math::
            \begin{align}
                r &\equiv \sqrt{x^2 + r_c^2}\\
                R &\equiv (x/r)^2\\
                \psi''(x) &= \psi_0\frac{r_cR}{(r + r_c)^2}
                    + \psi_0\left(1-\frac{r_c}{R + r_c}\right)\frac{1-R}{r}
            \end{align}
        """
        sqr = np.sqrt(x*x + self.rc*self.rc)
        R = (x/sqr)**2
        tmp1 = self.psi0*R*self.rc/(sqr + self.rc)**2
        tmp2 = self.psi0*(1 - self.rc/(sqr + self.rc))*(1-R)/sqr
        return tmp1+tmp2


class Psi_PointLens(PsiAxisym):
    """Lens object for the point lens.

    Additional information: :ref:`theory <Psi_PointLens_theory>`, :ref:`default parameters <Psi_PointLens_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.

    p_prec : dict
        Precision parameters, with keys:

        * ``xc`` (*float*) -- Point mass regularization (Plummer sphere).

    Attributes
    ----------
    asymp_index : float
        1.
    asymp_amplitude : float
        :math:`\\psi_0/2`
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.xc   = self.p_prec['xc']

        self.asymp_index = 1.
        self.asymp_amplitude = self.psi0/2.

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Default precision parameters.
        """
        p_phys = {'name' : 'point lens',
                  'psi0' : 1}
        p_prec = {'xc' : 1e-10}
        return p_phys, p_prec

    def psi_x(self, x):
        r"""Lensing potential.

        .. math::
            \psi(x) = \frac{1}{2}\psi_0\log(x^2 + x_c^2)

        """
        return 0.5*self.psi0*np.log(x*x + self.xc*self.xc)

    def dpsi_dx(self, x):
        r"""First derivative of the lensing potential.

        .. math::
            \psi'(x) = \frac{\psi_0 x}{x^2 + x_c^2}

        """
        return self.psi0*x/(x*x + self.xc*self.xc)

    def ddpsi_ddx(self, x):
        r"""Second derivative of the lensing potential.

        .. math::
            \psi''(x) = \psi_0\frac{x_c^2-x^2}{(x_c^2 + x^2)^2}

        """
        return self.psi0*(self.xc**2 - x**2)/(self.xc**2 + x**2)**2


class Psi_Ball(PsiAxisym):
    """Lens object for the uniform density sphere.

    Additional information: :ref:`theory <Psi_Ball_theory>`, :ref:`default parameters <Psi_Ball_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``b`` (*float*) -- Radius of the sphere.

    Attributes
    ----------
    asymp_index : float
        1.
    asymp_amplitude : float
        :math:`\\psi_0/2`
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.b    = self.p_phys['b']

        self.asymp_index = 1.
        self.asymp_amplitude = self.psi0/2.

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'ball',
                  'psi0' : 1.,
                  'b'    : 1.}
        p_prec = {}
        return p_phys, p_prec

    def check_input(self):
        """Check that the radius is positive.

        Raises
        ------
        LensException
        """
        b = self.p_phys['b']
        if b < 0:
            message = "b = %g < 0 found. Radius in the uniform sphere must be positive" % b
            raise LensException(message)

    def psi_small_x(self, x):
        r"""Lensing potential inside the sphere.

        .. math::
            \psi(x) = \psi_0\log(b(1+X)) - \psi_0X(1+X^2/3)\ ,\quad x<b

        with :math:`X\equiv\sqrt{1-(x/b)^2}`.
        """
        b = self.p_phys['b']
        X = np.sqrt(1-(x/b)**2)
        return self.psi0*(np.log(b*(1 + X)) - X*(1 + X**2/3.))

    def psi_x(self, x):
        r"""Lensing potential.

        Piece together the lensing potential inside the sphere, :func:`psi_small_x`, and
        outside, :func:`Psi_PointLens.psi_x`.
        """
        return np.piecewise(x, [x<=self.b, x>self.b], [self.psi_small_x, lambda x: self.psi0*np.log(x)])

    def dpsi_small_x(self, x):
        r"""First derivative of the lensing potential inside the sphere.

        .. math::
            \psi'(x) = \psi_0\frac{x}{b^2}\left(1+\frac{X^2}{1+X}\right)\ ,\quad x<b

        with :math:`X\equiv\sqrt{1-(x/b)^2}`.
        """
        b = self.p_phys['b']
        X = np.sqrt(1-(x/b)**2)
        return self.psi0*x*(1 + X**2/(X+1))/b**2

    def dpsi_dx(self, x):
        r"""First derivative of the lensing potential.

        Piece together the derivative inside the sphere, :func:`dpsi_small_x`, and
        outside, :func:`Psi_PointLens.dpsi_dx`.
        """
        return np.piecewise(x, [x<=self.b, x>self.b], [self.dpsi_small_x, lambda x: self.psi0/x])

    def ddpsi_small_x(self, x):
        r"""Second derivative of the lensing potential inside the sphere.

        .. math::
            \psi''(x) = \psi_0\frac{2X^2 + 2X - 1}{b^2(1+X)}\ ,\quad x<b

        with :math:`X\equiv\sqrt{1-(x/b)^2}`.
        """
        b = self.p_phys['b']
        X = np.sqrt(1-(x/b)**2)
        return self.psi0*(2*X**2 + 2*X - 1)/(X+1)/b**2

    def ddpsi_ddx(self, x):
        r"""Second derivative of the lensing potential.

        Piece together the derivative inside the sphere, :func:`ddpsi_small_x`, and
        outside, :func:`Psi_PointLens.ddpsi_ddx`.
        """
        return np.piecewise(x, [x<=self.b, x>self.b], [self.ddpsi_small_x, lambda x: -self.psi0/x/x])


class Psi_NFW(PsiAxisym):
    """Lens object for the Navarro-Frenk-White (NFW) profile.

    Additional information: :ref:`theory <Psi_NFW_theory>`, :ref:`default parameters <Psi_NFW_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``xs`` (*float*) -- Rescaled NFW radius.

    p_prec : dict
        Precision parameters, with keys:

        * ``eps_soft`` (*float*) -- Softening factor.
        * ``eps_NFW`` (*float*) -- Switch on Taylor expansion \
                                    when :math:`x/x_s < \\epsilon_\\text{NFW}`.
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.xs   = self.p_phys['xs']
        self.eps  = self.p_prec['eps_soft']

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Default precision parameters.
        """
        p_phys = {'name' : 'NFW',
                  'psi0' : 1.0,
                  'xs'   : 0.1}
        p_prec = {'eps_soft' : 1e-15,
                  'eps_NFW'  : 1e-2}
        return p_phys, p_prec

    def F_nfw_large_x(self, x):
        r"""Auxiliary NFW function for radius larger than 1.

        .. math::
            \mathcal{F}(x) = \frac{1}{\sqrt{x^2-1}}\arctan\left(\sqrt{x^2-1}\right), \quad x>1

        """
        sqr = np.sqrt(x**2 - 1)
        return np.arctan(sqr) / sqr

    def F_nfw_small_x(self, x):
        r"""Auxiliary NFW function for radius smaller than 1.

        .. math::
            \mathcal{F}(x) = \frac{1}{\sqrt{1-x^2}}\text{arctanh}\left(\sqrt{1-x^2}\right), \quad x<1

        """
        sqr = np.sqrt(1 - x**2)
        return np.arctanh(sqr) / sqr

    def F_nfw_v(self, x):
        r"""Auxiliary NFW function for any value of the radius.

        Piece together :func:`F_nfw_large_x` and :func:`F_nfw_xmall_x`.

        .. math::
            \mathcal{F}(x) = \begin{cases}
                                \mathcal{F}_\text{large}(x) & x > 1+\epsilon\\
                                \mathcal{F}_\text{small}(x) & x < 1-\epsilon\\
                                1 + 2|x-1| & \text{otherwise}
                             \end{cases}

        where :math:`\epsilon` is set by ``p_phys['epsilon_soft']``.
        """
        return np.piecewise(x, \
            [       x<1-self.eps,       x>1+self.eps, (x>1-self.eps)&(x<1+self.eps) ],\
            [ self.F_nfw_small_x, self.F_nfw_large_x,     lambda x: 1+2*np.abs(x-1) ])

    def psi_large_u(self, x):
        r"""Lensing potential for large values of :math:`x/x_s`.

        This is the analytical expression of the lensing potential. The actual
        potential used in the code combines this one with an approximation for
        small radius in :func:`psi_x`.

        .. math::
            \psi(x) = \frac{1}{2}\psi_0\left(\log^2(u/2) + (u^2-1)\mathcal{F}^2(u)\right)

        with :math:`u\equiv x/x_s`.
        """
        u = x/self.xs
        F = self.F_nfw_v(u)
        return self.psi0/2.*( np.log(u/2.)**2 + (u**2 - 1)*F*F )

    def dpsi_dx_large_u(self, x):
        r"""First derivative of the lensing potential for large values of :math:`x/x_s`.

        This is the analytical expression. The actual potential used in the code
        combines this one with an approximation for small radius in :func:`dpsi_dx`.

        .. math::
            \psi'(x) = \frac{\psi_0}{x_su}\left(\log(u/2) + \mathcal{F}(u)\right)

        with :math:`u\equiv x/x_s`.
        """
        u = x/self.xs
        F = self.F_nfw_v(u)
        return self.psi0*( np.log(u/2.) + F )/u/self.xs

    def ddpsi_ddx_large_u(self, x):
        r"""Second derivative of the lensing potential for large values of :math:`x/x_s`.

        This is the analytical expression. The actual potential used in the code
        combines this one with an approximation for small radius in :func:`ddpsi_ddx`.

        .. math::
            \psi''(x) = -\frac{\psi_0}{x_s^2u^2}\left(\log(u/2)
                + \frac{u^2 + \mathcal{F}(u)(1-2u^2)}{1-u^2}\right)

        with :math:`u\equiv x/x_s`.
        """
        u = x/self.xs
        u2 = u*u
        F = self.F_nfw_v(u)
        return -self.psi0*( np.log(u/2) + (u2 + F*(1 - 2*u2))/(1-u2) )/u2/self.xs**2

    def psi_small_u(self, x):
        r"""Expansion of :func:`psi_large_u` for :math:`x/x_s\ll 1`."""
        u = x/self.xs
        lg = np.log(0.5*u + self.eps)
        return -self.psi0/4.*( u**2*lg + u**4/8.*(1+3*lg) + u**6/32.*(3+20/3.*lg) + u**8/128*(107./12+35/2.*lg) )

    def dpsi_dx_small_u(self, x):
        r"""Expansion of :func:`dpsi_dx_large_u` for :math:`x/x_s\ll 1`."""
        u = x/self.xs
        lg = np.log(0.5*u + self.eps)
        return -self.psi0/2./self.xs*( u*(0.5+lg) + u**3*(7./16+3./4*lg) + u**5*(37./96+5./8*lg) + u**7*(533./1536+35./64*lg) )

    def ddpsi_ddx_small_u(self, x):
        r"""Expansion of :func:`ddpsi_ddx_large_u` for :math:`x/x_s\ll 1`."""
        u = x/self.xs
        lg = np.log(0.5*u + self.eps)
        return -self.psi0/self.xs**2/2.*( 1.5 + lg + u**2*(33./16+9./4*lg) + u**4*(245./96+25./8*lg) + u**6*(4571./1536+245./64*lg) )

    def psi_x(self, x):
        r"""Lensing potential.

        Piece together :func:`psi_large_u` and :func:`psi_small_u`.

        .. math::
            \psi(x) = \begin{cases}
                          \psi_\text{large}(x) & x > \epsilon_\text{NFW}\\
                          \psi_\text{small}(x) & x \leq \epsilon_\text{NFW}
                       \end{cases}

        where :math:`\epsilon_\text{NFW}` is set by ``p_phys['epsilon_NFW']``.
        """
        eps = self.p_prec['eps_NFW']

        return np.piecewise(x, [  x > self.xs*eps, x <= self.xs*eps ],\
                               [ self.psi_large_u, self.psi_small_u ])

    def dpsi_dx(self, x):
        r"""First derivative of the lensing potential.

        Piece together :func:`dpsi_large_u` and :func:`dpsi_small_u`.

        .. math::
            \psi'(x) = \begin{cases}
                          \psi'_\text{large}(x) & x > \epsilon_\text{NFW}\\
                          \psi'_\text{small}(x) & x \leq \epsilon_\text{NFW}
                       \end{cases}

        where :math:`\epsilon_\text{NFW}` is set by ``p_phys['epsilon_NFW']``.
        """
        eps = self.p_prec['eps_NFW']

        return np.piecewise(x, [      x > self.xs*eps,     x <= self.xs*eps ],\
                               [ self.dpsi_dx_large_u, self.dpsi_dx_small_u ])

    def ddpsi_ddx(self, x):
        r"""Second derivative of the lensing potential.

        Piece together :func:`ddpsi_large_u` and :func:`ddpsi_small_u`.

        .. math::
            \psi''(x) = \begin{cases}
                           \psi''_\text{large}(x) & x > \epsilon_\text{NFW}\\
                           \psi''_\text{small}(x) & x \leq \epsilon_\text{NFW}
                        \end{cases}

        where :math:`\epsilon_\text{NFW}` is set by ``p_phys['epsilon_NFW']``.
        """
        eps = self.p_prec['eps_NFW']

        return np.piecewise(x, [        x > self.xs*eps,       x <= self.xs*eps ],\
                               [ self.ddpsi_ddx_large_u, self.ddpsi_ddx_small_u ])


class Psi_tSIS(PsiAxisym):
    """Lens object for the truncated singular isothermal sphere (tSIS).

    Additional information: :ref:`theory <Psi_tSIS_theory>`, :ref:`default parameters <Psi_tSIS_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``xb`` (*float*) -- radius.

    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.xb = self.p_phys['xb']

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'tSIS',
                  'psi0' : 1,
                  'xb'   : 10}
        p_prec = {}
        return p_phys, p_prec

    def check_input(self):
        """Check that the radius is not negative.

        Raises
        ------
        LensException
        """
        xb = self.p_phys['xb']
        if xb < 0:
            message = "xb = %g < 0 found. Radius in the tSIS must be positive" % xb
            raise LensException(message)

    def psi_x(self, x):
        r"""Lensing potential.

        .. math::
            \begin{align}
                u &\equiv x/x_b\\
                \psi(x) &= \psi_0 x\left(\text{erfc}(u) + \frac{1-\text{e}^{-u^2}}{u\sqrt{\pi}}
                    +\frac{1}{2u\sqrt{\pi}}\left[2\log(u) + E_1(u^2) + \gamma_E\right]\right)
            \end{align}
        """
        u = x/self.xb
        u2 = u**2
        rel_exp = u*sc_special.exprel(-u2)/np.sqrt(np.pi)
        exp_int = (2*np.log(u) + sc_special.exp1(u2) + np.euler_gamma)/2./np.sqrt(np.pi)/u
        return self.psi0*x*(sc_special.erfc(u) + rel_exp + exp_int)

    def dpsi_dx(self, x):
        r"""First derivative of the lensing potential.

        .. math::
            \begin{align}
                u &\equiv x/x_b\\
                \psi'(x) &= \psi_0\left(\text{erfc}(u) + \frac{1-\text{e}^{-u^2}}{u\sqrt{\pi}}\right)
            \end{align}
        """
        u = x/self.xb
        return self.psi0*(sc_special.erfc(u) + u*sc_special.exprel(-u**2)/np.sqrt(np.pi))

    def ddpsi_ddx(self, x):
        r"""Second derivative of the lensing potential.

        .. math::
            \begin{align}
                u &\equiv x/x_b\\
                \psi''(x) &= -\frac{\psi_0}{x_b\sqrt{\pi}}\frac{1-\text{e}^{-u^2}}{u^2}
            \end{align}
        """
        u = x/self.xb
        return -self.psi0*sc_special.exprel(-u**2)/np.sqrt(np.pi)/self.xb


class Psi_offcenterSIS(PsiGeneral):
    """Lens object for the singular isothermal sphere (SIS) centered at (xc1, xc2).

    Additional information: :ref:`theory <Psi_SIS_theory>`, :ref:`default parameters <Psi_offcenterSIS_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``xc1`` (*float*) -- Location in the x1 axis.
        * ``xc2`` (*float*) -- Location in the x2 axis.

    Attributes
    ----------
    asymp_index : float
        0.5
    asymp_amplitude : float
        :math:`\\psi_0/\\sqrt{2}`
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.xc1 = self.p_phys['xc1']
        self.xc2 = self.p_phys['xc2']

        self.sym_lens = Psi_SIS({'psi0':self.psi0})

        self.asymp_index = 0.5
        self.asymp_amplitude = self.psi0/np.sqrt(2)

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'off-center SIS',
                  'psi0' : 1,
                  'xc1'  : 0,
                  'xc2'  : 0}
        p_prec = {}
        return p_phys, p_prec

    def psi(self, x1, x2):
        """Lensing potential.

        Evaluate the equivalent method of :class:`Psi_SIS` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.psi(x1-self.xc1, x2-self.xc2)

    def dpsi_vec(self, x1, x2):
        """First derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_SIS` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.dpsi_vec(x1-self.xc1, x2-self.xc2)

    def ddpsi_vec(self, x1, x2):
        """Second derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_SIS` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.ddpsi_vec(x1-self.xc1, x2-self.xc2)


class Psi_offcenterCIS(PsiGeneral):
    """Lens object for the cored isothermal sphere (CIS) centered at (xc1, xc2).

    Additional information: :ref:`theory <Psi_CIS_theory>`, :ref:`default parameters <Psi_offcenterCIS_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``rc`` (*float*) -- Core radius.
        * ``xc1`` (*float*) -- Location in the x1 axis.
        * ``xc2`` (*float*) -- Location in the x2 axis.

    Attributes
    ----------
    asymp_index : float
        0.5
    asymp_amplitude : float
        :math:`\\psi_0/\\sqrt{2}`
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.rc   = self.p_phys['rc']
        self.xc1  = self.p_phys['xc1']
        self.xc2  = self.p_phys['xc2']

        self.sym_lens = Psi_CIS({'psi0':self.psi0, 'rc':self.rc})

        self.asymp_index = 0.5
        self.asymp_amplitude = self.psi0/np.sqrt(2)

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'off-center CIS',
                  'psi0' : 1,
                  'xc1'  : 0,
                  'xc2'  : 0,
                  'rc'   : 0.05}
        p_prec = {}
        return p_phys, p_prec

    def psi(self, x1, x2):
        """Lensing potential.

        Evaluate the equivalent method of :class:`Psi_CIS` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.psi(x1-self.xc1, x2-self.xc2)

    def dpsi_vec(self, x1, x2):
        """First derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_CIS` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.dpsi_vec(x1-self.xc1, x2-self.xc2)

    def ddpsi_vec(self, x1, x2):
        """Second derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_CIS` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.ddpsi_vec(x1-self.xc1, x2-self.xc2)


class Psi_offcenterPointLens(PsiGeneral):
    """Lens object for the point lens centered at (xc1, xc2).

    Additional information: :ref:`theory <Psi_PointLens_theory>`, :ref:`default parameters <Psi_offcenterPointLens_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``xc1`` (*float*) -- Location in the x1 axis.
        * ``xc2`` (*float*) -- Location in the x2 axis.

    p_prec : dict
        Precision parameters, with keys:

        * ``xc`` (*float*) -- Point mass regularization (Plummer sphere).

    Attributes
    ----------
    asymp_index : float
        1.
    asymp_amplitude : float
        :math:`\\psi_0/2`
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.xc1  = self.p_phys['xc1']
        self.xc2  = self.p_phys['xc2']

        self.sym_lens = Psi_PointLens({'psi0':self.psi0}, self.p_prec)

        self.asymp_index = 1.
        self.asymp_amplitude = self.psi0/2.

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'off-center point lens',
                  'psi0' : 1,
                  'xc1'  : 0,
                  'xc2'  : 0}
        p_prec = {'xc' : 1e-10}
        return p_phys, p_prec

    def psi(self, x1, x2):
        """Lensing potential.

        Evaluate the equivalent method of :class:`Psi_PointLens` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.psi(x1-self.xc1, x2-self.xc2)

    def dpsi_vec(self, x1, x2):
        """First derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_PointLens` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.dpsi_vec(x1-self.xc1, x2-self.xc2)

    def ddpsi_vec(self, x1, x2):
        """Second derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_PointLens` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.ddpsi_vec(x1-self.xc1, x2-self.xc2)


class Psi_offcenterBall(PsiGeneral):
    """Lens object for the uniform density sphere centered at (xc1, xc2).

    Additional information: :ref:`theory <Psi_Ball_theory>`, :ref:`default parameters <Psi_offcenterBall_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``b`` (*float*) -- Radius of the sphere.
        * ``xc1`` (*float*) -- Location in the x1 axis.
        * ``xc2`` (*float*) -- Location in the x2 axis.

    Attributes
    ----------
    asymp_index : float
        1.
    asymp_amplitude : float
        :math:`\\psi_0/2`
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0 = self.p_phys['psi0']
        self.b    = self.p_phys['b']
        self.xc1  = self.p_phys['xc1']
        self.xc2  = self.p_phys['xc2']

        self.sym_lens = Psi_Ball({'psi0':self.psi0, 'b':self.b})

        self.asymp_index = 1.
        self.asymp_amplitude = self.psi0/2.

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'off-center ball',
                  'psi0' : 1.,
                  'b'    : 1.,
                  'xc1'  : 0.,
                  'xc2'  : 0.}
        p_prec = {}
        return p_phys, p_prec

    def check_input(self):
        """Check that the radius is positive.

        Raises
        ------
        LensException
        """
        b = self.p_phys['b']
        if b < 0:
            message = "b = %g < 0 found. Radius in the uniform sphere must be positive" % b
            raise LensException(message)

    ## -------------------------------------------

    def psi(self, x1, x2):
        """Lensing potential.

        Evaluate the equivalent method of :class:`Psi_Ball` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.psi(x1-self.xc1, x2-self.xc2)

    def dpsi_vec(self, x1, x2):
        """First derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_Ball` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.dpsi_vec(x1-self.xc1, x2-self.xc2)

    def ddpsi_vec(self, x1, x2):
        """Second derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_Ball` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.ddpsi_vec(x1-self.xc1, x2-self.xc2)


class Psi_offcenterNFW(PsiGeneral):
    """Lens object for the NFW lens centered at (xc1, xc2).

    Additional information: :ref:`theory <Psi_NFW_theory>`, :ref:`default parameters <Psi_offcenterNFW_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``xs`` (*float*) -- Rescaled NFW radius.
        * ``xc1`` (*float*) -- Location in the x1 axis.
        * ``xc2`` (*float*) -- Location in the x2 axis.

    p_prec : dict
        Precision parameters, with keys:

        * ``eps_soft`` (*float*) -- Softening factor.
        * ``eps_NFW`` (*float*) -- Switch on Taylor expansion \
                                    when :math:`x/x_s < \\epsilon_\\text{NFW}`.
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.xc1  = self.p_phys['xc1']
        self.xc2  = self.p_phys['xc2']

        p_phys = {key:val for key, val in p_phys.items() if (key != 'xc1') and (key != 'xc2')}
        self.sym_lens = Psi_NFW(p_phys, p_prec)

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Default precision parameters.
        """
        p_phys = {'name' : 'off-center NFW',
                  'psi0' : 1,
                  'xs'   : 0.1,
                  'xc1'  : 0,
                  'xc2'  : 0}
        p_prec = {'eps_soft' : 1e-15,
                  'eps_NFW'  : 1e-2}
        return p_phys, p_prec

    def psi(self, x1, x2):
        """Lensing potential.

        Evaluate the equivalent method of :class:`Psi_NFW` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.psi(x1-self.xc1, x2-self.xc2)

    def dpsi_vec(self, x1, x2):
        """First derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_NFW` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.dpsi_vec(x1-self.xc1, x2-self.xc2)

    def ddpsi_vec(self, x1, x2):
        """Second derivatives of the lensing potential.

        Evaluate the equivalent method of :class:`Psi_NFW` at (x1-xc1, x2-xc2).
        """
        return self.sym_lens.ddpsi_vec(x1-self.xc1, x2-self.xc2)


class Psi_Ext(PsiGeneral):
    """Lens object for an external field (convergence & shear).

    Additional information: :ref:`theory <Psi_Ext_theory>`, :ref:`default parameters <Psi_Ext_default>`.

    Parameters
    ----------
    p_phys : dict, optional
        Physical parameters, with keys:

        * ``kappa`` (*float*) -- Convergence.
        * ``gamma1`` (*float*) -- Shear along x1.
        * ``gamma2`` (*float*) -- Shear along x2.

    Attributes
    ----------
    kappa : float
        Convergence of the lens.
    gamma1 : float
        Shear along x1 of the lens.
    gamma2 : float
        Shear along x2 of the lens.
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.kappa = self.p_phys['kappa']
        self.gamma1 = self.p_phys['gamma1']
        self.gamma2 = self.p_phys['gamma2']

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'ext',
                  'kappa' : 0,
                  'gamma1' : 0,
                  'gamma2' : 0}
        p_prec = {}
        return p_phys, p_prec

    def psi(self, x1, x2):
        r"""Lensing potential.

        .. math::
            \begin{align}
                \psi(x_1, x_2) &= \kappa (x_1^2+x_2^2)/2 + \gamma_1(x_1^2-x_2^2)/2 + \gamma_2x_1x_2
            \end{align}
        """
        return self.kappa*(x1*x1 + x2*x2)/2. + self.gamma1*(x1*x1 - x2*x2)/2. + self.gamma2*x2*x1

    def dpsi_vec(self, x1, x2):
        r"""First derivatives of the lensing potential.

        .. math::
            \begin{align}
                \partial_{1}\psi &= \kappa x_1 + \gamma_1 x_1 + \gamma_2 x_2\\
                \partial_{2}\psi &= \kappa x_2 - \gamma_1 x_2 + \gamma_2 x_1
            \end{align}
        """
        d1 = self.kappa*x1 + self.gamma1*x1 + self.gamma2*x2
        d2 = self.kappa*x2 - self.gamma1*x2 + self.gamma2*x1
        return d1, d2

    def ddpsi_vec(self, x1, x2):
        r"""Second derivatives of the lensing potential.

        .. math::
            \begin{align}
                \partial_{11}\psi &= \kappa + \gamma_1\\
                \partial_{12}\psi &= \gamma_2\\
                \partial_{22}\psi &= \kappa - \gamma_1
            \end{align}
        """
        d11 = np.full_like(x1, self.kappa + self.gamma1)
        d12 = np.full_like(x1, self.gamma2)
        d22 = np.full_like(x1, self.kappa - self.gamma1)
        return d11, d12, d22


class CombinedLens(PsiAxisym):
    """Create a composite lens.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``lenses`` (*list*) -- List of lens objects.

    Attributes
    ----------
    has_shear : bool
        True if the composite lens contains an external shear field.
    kappa, gamma1, gamma2 : float
        Convergence and shear of the external field.
    asymp_index, asymp_amplitude : float
        If the attributes :attr:`asymp_index`, :attr:`asymp_amplitude` are
        defined for all the lenses, we assign to the composite lens the
        following values:

        * :attr:`asymp_index` = minimum of all the indices
        * :attr:`asymp_amplitude` = sum of all the amplitudes

    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.lenses = self.p_phys['lenses']

        self.has_shear = False
        self.kappa = 0
        self.gamma1 = 0
        self.gamma2 = 0

        asymp_indices = []
        asymp_amplitudes = []

        for l in self.lenses:
            asymp_indices.append(l.asymp_index)
            asymp_amplitudes.append(l.asymp_amplitude)

            # the composite lens is axisym if all lenses are
            self.isAxisym = self.isAxisym and l.isAxisym

            if l.p_phys['name'] == 'ext':
                self.has_shear = True
                self.kappa  += l.p_phys['kappa']
                self.gamma1 += l.p_phys['gamma1']
                self.gamma2 += l.p_phys['gamma2']

        asymp_indices = np.array(asymp_indices)
        asymp_amplitudes = np.array(asymp_amplitudes)

        try:
            # only strictly true if all the lenses are equal, still
            # should improve the computation in most cases
            self.asymp_amplitude = np.sum(asymp_amplitudes)
            self.asymp_index = np.min(asymp_indices)
        except:
            pass

    def __str__(self):
        class_name = type(self).__name__
        class_call = "Psi = lenses." + class_name + "(p_phys, p_prec)"

        prev_lenses = ""
        lenses_string = "["
        for i, lens in enumerate(self.p_phys['lenses']):
            idx = str(i+1)
            prev_lenses += lens.__str__(idx) + "\n\n"
            lenses_string += "Psi%s, " % idx
        lenses_string = lenses_string[:-2] +"]"

        phys_message = "p_phys = {"
        if self.p_phys == {}:
            phys_message += "}\n"
        else:
            for key, p in self.p_phys.items():
                if key == 'lenses_p_phys':
                    continue
                elif key == 'lenses':
                    param = key.__repr__() + ': ' + lenses_string + ', '
                else:
                    param = key.__repr__() + ': ' + p.__repr__() + ', '
                phys_message += param
            phys_message = phys_message[:-2] + "}\n"

        prec_message = "p_prec = " + self.p_prec.__repr__() + "\n"

        return prev_lenses + phys_message + prec_message + class_call

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Default precision parameters.
        """
        p_phys = {'name' : 'combined lens',
                  'lenses' : []}
        p_prec = {}
        return p_phys, p_prec

    def check_input(self):
        """If the composite lens contains a combination of composite lenses, we flatten them to create
        a single composite lens."""
        lens_list = self.p_phys['lenses']
        for i, l in enumerate(lens_list):
            if l.p_phys['name'] == 'combined lens':
                lens_list[i] = l.p_phys['lenses']

        self.p_phys['lenses'] = np.array(lens_list).flatten().tolist()

    def display_info(self):
        # HVR -> perhaps an option could be added to display all the info
        #        of the individual lenses
        print("\t////////////////////////////\n"\
              "\t///   Lens information   ///\n"\
              "\t////////////////////////////\n")

        print(" * Name: %s" % self.p_phys['name'])

        print("\n * Physical parameters:")
        for key, value in self.p_phys.items():
            if key == 'name':
                continue
            if key == 'lenses':
                print("   ** %s:" % key)
                for l in self.p_phys['lenses']:
                    print("      *** %s" % l.p_phys['name'])
                continue
            print("   **", key, "=", value)

        if self.p_prec != {}:
            print("\n * Precision parameters:")
            for key, value in self.p_prec.items():
                print("   **", key, "=", value)
        print('')

    def psi_x(self, x):
        """Lensing potential for axisymmetric lenses.

        Sum over all the lenses.
        """
        psis = [l.psi_x(x) for l in self.lenses]
        return np.sum(psis, axis=0)

    def dpsi_dx(self, x):
        """First derivatives of the lensing potential for axisymmetric lenses.

        Sum over all the lenses.
        """
        dpsis = [l.dpsi_dx(x) for l in self.lenses]
        return np.sum(dpsis, axis=0)

    def ddpsi_ddx(self, x):
        """Second derivatives of the lensing potential for axisymmetric lenses.

        Sum over all the lenses.
        """
        ddpsis = [l.ddpsi_ddx(x) for l in self.lenses]
        return np.sum(ddpsis, axis=0)

    def psi(self, x1, x2):
        """Lensing potential.

        Sum over all the lenses.
        """
        psis = [l.psi(x1, x2) for l in self.lenses]
        return np.sum(psis, axis=0)

    def dpsi_vec(self, x1, x2):
        """First derivatives of the lensing potential.

        Sum over all the lenses.
        """
        dpsis = [l.dpsi_vec(x1, x2) for l in self.lenses]
        return np.sum(dpsis, axis=0)

    def ddpsi_vec(self, x1, x2):
        """Second derivatives of the lensing potential.

        Sum over all the lenses.
        """
        ddpsis = [l.ddpsi_vec(x1, x2) for l in self.lenses]
        return np.sum(ddpsis, axis=0)


class Psi_eSIS(PsiGeneral):
    """Lens object for the elliptical SIS (eSIS).

    Additional information: :ref:`theory <Psi_eSIS_theory>`, :ref:`default parameters <Psi_eSIS_default>`.

    Parameters
    ----------
    p_phys : dict
        Physical parameters, with keys:

        * ``psi0`` (*float*) -- Normalization of the lens.
        * ``q`` (*float*) -- Ellipticity parameter.
        * ``alpha`` (*float*) -- Orientation angle.
        * ``xc1`` (*float*) -- Location in the x1 axis.
        * ``xc2`` (*float*) -- Location in the x2 axis.

    Attributes
    ----------
    asymp_index : float
        0.5
    asymp_amplitude : float
        :math:`\\psi_0/\\sqrt{2}`
    """
    def __init__(self, p_phys={}, p_prec={}):
        super().__init__(p_phys, p_prec)

        self.psi0  = self.p_phys['psi0']
        self.q     = self.p_phys['q']
        self.alpha = self.p_phys['alpha']

        self.eps = 1e-14
        self.ca = np.cos(self.alpha)
        self.sa = np.sin(self.alpha)

        self.asymp_index = 0.5
        self.asymp_amplitude = self.psi0/np.sqrt(2)

    def default_params(self):
        """Initialize the default parameters.

        Returns
        -------
        p_phys : dict
            Default physical parameters.
        p_prec : dict
            Empty dictionary.
        """
        p_phys = {'name' : 'eSIS',
                  'psi0'  : 1.,
                  'q'     : 1.,
                  'alpha' : 0,
                  'xc1'   : 0.,
                  'xc2'   : 0.}
        p_prec = {}
        return p_phys, p_prec

    # lens potential and its derivatives
    def _psi_x(self, x):
        """SIS-like lensing potential."""
        return self.psi0*x

    def _dpsi_dx(self, x):
        """First derivative of the SIS-like lensing potential."""
        return np.full_like(x, self.psi0)

    def _ddpsi_ddx(self, x):
        """Second derivative of the SIS-like lensing potential."""
        return np.zeros_like(x)

    # coordinate transformation
    def rotate_vector(self, x1, x2):
        r"""Rotation of a vector.

        Linear transformation defined as

        .. math::
            x'_1 &= \cos\alpha\,x_1 - \sin\alpha\,x_2\\
            x'_2 &= \sin\alpha\,x_1 + \cos\alpha\,x_2

        Parameters
        ----------
        x1, x2 : float or array
            Input vector.

        Returns
        -------
        xx1 : float or array
            :math:`x'_1`
        xx2 : float or array
            :math:`x'_2`
        """
        xx1 = x1*self.ca - x2*self.sa
        xx2 = x1*self.sa + x2*self.ca
        return xx1, xx2

    def rotate_grad(self, f1, f2):
        r"""Rotation of a gradient.

        Linear transformation defined as

        .. math::
            f'_1 &= \cos\alpha\,f_1 + \sin\alpha\,f_2\\
            f'_2 &= -\sin\alpha\,f_1 + \cos\alpha\,f_2

        Parameters
        ----------
        f1, f2 : float or array
            Input vector.

        Returns
        -------
        ff1 : float or array
            :math:`f'_1`
        ff2 : float or array
            :math:`f'_2`
        """
        ff1 =  f1*self.ca + f2*self.sa
        ff2 = -f1*self.sa + f2*self.ca
        return ff1, ff2

    def rotate_hessian(self, h11, h12, h22):
        r"""Rotation of a Hessian.

        Linear transformation defined as

        .. math::
            A &\equiv 2\cos\alpha\sin\alpha\,h_{12}\\
            B &\equiv \sin\alpha (h_{11} - h_{22})\\
            h'_{11} &= h_{11} + A - \sin\alpha\,B\\
            h'_{12} &= h_{12}(1-2\sin^2\alpha) - \cos\alpha\,B\\
            h'_{22} &= h_{22} - A + \sin\alpha\,B

        Parameters
        ----------
        h11, h12, h22 : float or array
            Input vector.

        Returns
        -------
        hh11 : float or array
            :math:`h'_{11}`
        hh12 : float or array
            :math:`h'_{12}`
        hh22 : float or array
            :math:`h'_{22}`
        """
        A = 2*self.ca*self.sa*h12
        B = self.sa*(h11-h22)

        hh11 = h11 + A - self.sa*B
        hh22 = h22 - A + self.sa*B
        hh12 = h12*(1-2*self.sa**2) - self.ca*B

        return hh11, hh12, hh22

    # include the coordinate transformation
    def psi_a0(self, x1, x2):
        r"""Lensing potential (zero angle).

        .. math::
            \psi_{\alpha=0}=\sqrt{x_1^2 + x_2^2/q^2}

        """
        x = np.sqrt(x1*x1 + x2*x2/self.q**2)
        return self._psi_x(x)

    def dpsi_vec_a0(self, x1, x2):
        r"""First derivatives of the lensing potential (zero angle).

        .. math::
            \begin{align}
                \partial_{1}\psi_{\alpha=0} &= \psi_0 x_1/X\\
                \partial_{2}\psi_{\alpha=0} &= \psi_0 x_2/X/q^2
            \end{align}

        with :math:`X \equiv \sqrt{x_1^2 + x^2_2/q^2}`.
        """
        q2 = self.q*self.q
        x = np.sqrt(x1*x1 + x2*x2/q2)

        dpsi = self._dpsi_dx(x)
        R1 = x1/(x+self.eps)
        R2 = x2/(x+self.eps)/q2

        return dpsi*R1, dpsi*R2

    def ddpsi_vec_a0(self, x1, x2):
        r"""Second derivatives of the lensing potential (zero angle).

        .. math::
            \begin{align}
                \partial_{11}\psi_{\alpha=0} &= \psi_0(1-r_1^2)/X\\
                \partial_{12}\psi_{\alpha=0} &= -\psi_0r_1r_2/X\\
                \partial_{22}\psi_{\alpha=0} &= \psi_0(1/q^2-r_2^2)/X
            \end{align}

        with :math:`X \equiv \sqrt{x_1^2 + x^2_2/q^2}`, :math:`r_{1}\equiv x_{1}/X`
        and :math:`r_{2}\equiv x_{2}/X/q^2`.
        """
        q2 = self.q*self.q
        x = np.sqrt(x1*x1 + x2*x2/q2)

        dpsi  = self._dpsi_dx(x)
        ddpsi = self._ddpsi_ddx(x)
        R1 = x1/(x+self.eps)
        R2 = x2/(x+self.eps)/q2

        d11 = ddpsi*R1**2 + dpsi*(1 - R1**2)/(x+self.eps)
        d12 = (ddpsi - dpsi/(x+self.eps))*R1*R2
        d22 = ddpsi*R2**2 + dpsi*(1/q2 - R2**2)/(x+self.eps)

        return d11, d12, d22

    # rotate the lens with an angle alpha
    def psi(self, x1, x2):
        r"""Lensing potential.

        Defining the transformed coordinates
        :math:`\pmb{\tilde{x}}\equiv R(\alpha)(\pmb{x}-\pmb{x_c})`
        where :math:`R(\alpha)` is the rotation defined in :func:`rotate_vector`,
        we can write the lensing potential from :func:`psi_a0` as

        .. math::
            \psi(x_1, x_2) = \psi_{\alpha=0}(\tilde{x}_1, \tilde{x}_2)
        """
        x1 = x1 - self.p_phys['xc1']
        x2 = x2 - self.p_phys['xc2']

        if self.alpha == 0:
            psi = self.psi_a0(x1, x2)
        else:
            x1, x2 = self.rotate_vector(x1, x2)
            psi = self.psi_a0(x1, x2)
        return psi

    def dpsi_vec(self, x1, x2):
        r"""Gradient of the lensing potential.

        Defining the transformed coordinates
        :math:`\pmb{\tilde{x}}\equiv R(\alpha)(\pmb{x}-\pmb{x_c})`
        where :math:`R(\alpha)` is the rotation defined in :func:`rotate_vector`,
        we can write the first derivatives of the potential from :func:`dpsi_a0` as

        .. math::
            \begin{align}
                \pmb{g} &\equiv \pmb{\nabla}\psi(\pmb{x})\\
                \pmb{\tilde{g}} &\equiv \pmb{\nabla}\psi_{\alpha=0}(\tilde{\pmb{x}})\\
                \pmb{g} &= R_g(\alpha)\pmb{\tilde{g}}
            \end{align}

        where the linear transformation :math:`R_g` is defined in :func:`rotate_grad`.
        """
        x1 = x1 - self.p_phys['xc1']
        x2 = x2 - self.p_phys['xc2']

        if self.alpha == 0:
            dpsi = self.dpsi_vec_a0(x1, x2)
        else:
            x1, x2 = self.rotate_vector(x1, x2)
            dpsi = self.dpsi_vec_a0(x1, x2)
            dpsi = self.rotate_grad(*dpsi)
        return dpsi

    def ddpsi_vec(self, x1, x2):
        r"""Hessian of the lensing potential.

        Defining the transformed coordinates
        :math:`\pmb{\tilde{x}}\equiv R(\alpha)(\pmb{x}-\pmb{x_c})`
        where :math:`R(\alpha)` is the rotation defined in :func:`rotate_vector`,
        we can write the second derivatives of the potential from :func:`ddpsi_a0` as

        .. math::
            \begin{align}
                \pmb{h} &\equiv (\partial_{11}\psi,\,\partial_{12}\psi,\,\partial_{22}\psi)(\pmb{x})\\
                \pmb{\tilde{h}} &\equiv (\partial_{11}\psi_{\alpha=0},\,
                    \partial_{12}\psi_{\alpha=0},\,\partial_{22}\psi_{\alpha=0})(\pmb{\tilde{x}})\\
                \pmb{h} &= R_h(\alpha)\pmb{\tilde{h}}
            \end{align}

        where the linear transformation :math:`R_h` is defined in :func:`rotate_hessian`.
        """
        x1 = x1 - self.p_phys['xc1']
        x2 = x2 - self.p_phys['xc2']

        if self.alpha == 0:
            ddpsi = self.ddpsi_vec_a0(x1, x2)
        else:
            x1, x2 = self.rotate_vector(x1, x2)
            ddpsi = self.ddpsi_vec_a0(x1, x2)
            ddpsi = self.rotate_hessian(*ddpsi)
        return ddpsi
