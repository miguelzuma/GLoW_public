#
# GLoW - carea.pyx
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

#cython: language_level=3

import numpy as np

cimport cython
from cython.parallel import prange

from clenses cimport pNamedLens, free_pLens, convert_pphys_to_pLens
from carea cimport pAreaIntegral, integrate_AreaIntegral

## ----------  Precision ------------- ##
handle_GSL_errors()
cdef update_pprec(Prec_General pprec_general):
    global pprec
    pprec = pprec_general
## ----------------------------------- ##

## =======     AREA INTEGRAL
## =============================================================================
def pyAreaIntegral(y, Psi, p_prec):
    cdef int n_grid = p_prec['Nt']
    tau_grid = np.zeros(n_grid, dtype=np.double)
    It_grid = np.zeros(n_grid, dtype=np.double)

    cdef double[::1] ctau_grid = tau_grid
    cdef double[::1] cIt_grid = It_grid
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)
    cdef pAreaIntegral p
    cdef double t_min

    p.y = y
    p.tau_max = p_prec['tmax']
    p.n_rho = p_prec['n_rho']
    p.n_theta = p_prec['n_theta']
    p.pNLens = pNLens

    integrate_AreaIntegral(&t_min, &ctau_grid[0], &cIt_grid[0], n_grid, &p)

    free_pLens(pNLens)

    return t_min, tau_grid, It_grid

## =============================================================================
