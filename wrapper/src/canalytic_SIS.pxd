#
# GLoW - canalytic_SIS.pxd
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

cdef extern from "complex.h":
    double complex I

# different precision available
cdef extern from "analytic_SIS_lib.h" nogil:
    ctypedef enum Fw_SIS_methods: sis_direct, sis_osc

    double It_SIS_DoublePrec(double tau, double y, double psi0) # double precision (around 2e-16)
    double It_SIS_SinglePrec(double tau, double y, double psi0) # single precision (around 1e-7)
    double It_SIS_ApproxPrec(double tau, double y, double psi0) # approx precision (around 5e-4)

    double complex Fw_SIS(double w, double y, double psi0, int method)
