#
# GLoW - croots.pxd
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

from clenses cimport pNamedLens, Lens

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "roots_lib.h" nogil:
    ctypedef enum types_CritPoint: type_min, type_max, type_saddle, \
                                   type_singcusp_max, type_singcusp_min, \
                                   type_non_converged

    ctypedef struct CritPoint:
        int type
        double t
        double mag
        double x1
        double x2

    ctypedef struct pImage:
        double y
        CritPoint *point
        Lens *Psi

    int is_same_CritPoint(CritPoint *p_a, CritPoint *p_b)
    void swap_CritPoint(CritPoint *p_a, CritPoint *p_b)
    void sort_x_CritPoint(int n_points, CritPoint *p)
    void sort_t_CritPoint(int n_points, CritPoint *p)

    int find_CritPoint_root_2D(double x1guess, double x2guess, pImage *p)
    int check_only_min_CritPoint_2D(CritPoint *p, double y, pNamedLens *pNLens)
    CritPoint *driver_all_CritPoints_1D(int *n_cpoints, double y, pNamedLens *pNLens)
    CritPoint *driver_all_CritPoints_2D(int *n_cpoints, double y, pNamedLens *pNLens)


cpdef convert_CritPoint_to_pcrit(CritPoint p)
cdef CritPoint convert_pcrit_to_CritPoint(p_crit)
cpdef pyFind_all_CritPoints_1D(y, Psi)
cpdef pyFind_all_CritPoints_2D(y, Psi)
cpdef pyCheck_min(y, Psi)
