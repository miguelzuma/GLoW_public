#
# GLoW - csingle_contour.pxd
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

from clenses cimport pNamedLens

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "ode_tools.h" nogil:
    ctypedef struct SolODE:
        int n_buffer
        int n_allocated
        int n_points
        int n_eqs
        double *t
        double **y

    SolODE *init_SolODE(int n_eqs, int n_buffer)
    void free_SolODE(SolODE *sol)

cdef extern from "single_contour_lib.h" nogil:
    ctypedef enum methods_contour: m_contour_std, m_contour_robust

    # compute I(tau)
    double driver_contour(double tau_ini, double x1_min, double x2_min, double y,
                          pNamedLens *pNLens, int method);

    # increase the density of points in the contour
    SolODE *interpolate_contour(int n_points, double x1_min, double x2_min, SolODE *sol)

    # store the contour for tau in SolODE
    int driver_get_contour(double tau, int n_points, double x1_min, double x2_min, double y,
                           pNamedLens *pNLens, int method, SolODE *sol)

    # integrate dR_dtau to find R(tau)
    int driver_dR_dtau(int n_points, double *R_grid, double *tau_grid,
                       double x1_min, double x2_min, double y,
                       pNamedLens *pNLens);

    # invert tau(R) to find R(tau)
    int driver_R_of_tau(int n_points, double *R_grid, double *tau_grid,
                        double x1_min, double x2_min, double y,
                        pNamedLens *pNLens);
