#
# GLoW - canalytic_SIS.pyx
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
#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np

cimport cython
from cython.parallel import prange

from canalytic_SIS cimport It_SIS_DoublePrec, Fw_SIS, Fw_SIS_methods

## ----------  Precision ------------- ##
handle_GSL_errors()
cdef update_pprec(Prec_General pprec_general):
    global pprec
    pprec = pprec_general

try:
    import psutil
    max_num_threads = psutil.cpu_count(logical=False)
except ModuleNotFoundError:
    import multiprocessing
    max_num_threads = int(multiprocessing.cpu_count()/2)
## ----------------------------------- ##

## =======     ANALYTIC SIS
## =============================================================================

# main implementation with double precision
def pyIt_SIS(tau, double y, double psi0=1., bint parallel=True):
    cdef int i
    cdef double[:] ctau = np.ascontiguousarray(tau, dtype=np.double)
    cdef int i_max = ctau.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyIt = np.zeros(i_max, dtype=np.double)
    cdef double[:] It = pyIt

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='static'):
        It[i] = It_SIS_DoublePrec(ctau[i], y, psi0)

    if np.isscalar(tau):
        pyIt = pyIt.item()

    return pyIt

def pyFw_SIS(w, double y, double psi0=1., method='direct', bint parallel=True):
    cdef int i, cmethod
    cdef double[:] cw = np.ascontiguousarray(w, dtype=np.double)
    cdef int i_max = cw.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyFw = np.empty(i_max, dtype=np.cdouble)
    cdef double complex[:] Fw = pyFw

    if method == 'direct':
        cmethod = Fw_SIS_methods.sis_direct
    elif method == 'osc':
        cmethod = Fw_SIS_methods.sis_osc
    else:
        cmethod = Fw_SIS_methods.sis_direct

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        Fw[i] = Fw_SIS(cw[i], y, psi0, cmethod)

    if np.isscalar(w):
        pyFw = pyFw.item()

    return pyFw
