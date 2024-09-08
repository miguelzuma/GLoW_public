#
# GLoW - cspecial.pxd
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

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "special_lib.h" nogil:
    double f_fresnel(double x)
    double g_fresnel(double x)

    double Mtilde_Struve_PowerSeries(double x, double nu, double tol)
    double Mtilde_Struve_Asymptotic(double x, double nu, double tol)
    double Mtilde_Struve_PieceWise(double x, double nu, double tol)
    double Mtilde_Struve(double x, double nu, double tol)

    double complex F11_singlepoint(double u, double c, int *status, int *approx_flag)
    int F11_sorted(double *u, double c, int n_F, double complex *F11, int nthreads)

    int sorted_interpolation(double *x_subgrid, double *y_subgrid, int n_subgrid,
                             double *x_grid, double *y_grid, int n_grid)
