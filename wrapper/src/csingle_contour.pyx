#
# GLoW - csingle_contour.pyx
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
from libc.stdlib cimport malloc, free

cimport csingle_contour
from csingle_contour cimport SolODE, init_SolODE, free_SolODE
from clenses cimport pNamedLens, free_pLens, convert_pphys_to_pLens

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


## =======     CONTOUR INTEGRATION (SINGLE IMAGE)
## =============================================================================

def pyContour(tau, x1_min, x2_min, y, Psi, method='standard', parallel=True):
    cdef int i
    cdef int cmethod
    cdef double cy = y
    cdef double cx1_min = x1_min
    cdef double cx2_min = x2_min
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)

    cdef double[:] ctau = np.ascontiguousarray(tau, dtype=np.double)
    cdef int i_max = ctau.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyIt = np.zeros(i_max, dtype=np.double)
    cdef double[:] It = pyIt

    if method == 'standard':
        cmethod = methods_contour.m_contour_std
    elif method == 'robust':
        cmethod = methods_contour.m_contour_robust
    else:
        cmethod = methods_contour.m_contour_std

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        It[i] = csingle_contour.driver_contour(ctau[i], cx1_min, cx2_min, cy, pNLens, cmethod)

    if np.isscalar(tau):
        pyIt = pyIt.item()

    free_pLens(pNLens)

    return pyIt


## =======     GET CONTOUR (SINGLE IMAGE)
## =============================================================================

cdef convert_SolODE(SolODE *sol):
    cdef int i
    cdef Py_ssize_t i_max = sol.n_points

    py_sigma = np.zeros(i_max, dtype=np.double)
    py_alpha = np.zeros(i_max, dtype=np.double)
    py_R = np.zeros(i_max, dtype=np.double)
    py_x1 = np.zeros(i_max, dtype=np.double)
    py_x2 = np.zeros(i_max, dtype=np.double)

    for i in range(i_max):
        py_sigma[i] = sol.t[i]
        py_alpha[i] = sol.y[0][i]
        py_R[i] = sol.y[1][i]
        py_x1[i] = sol.y[2][i]
        py_x2[i] = sol.y[3][i]

    return {'sigma':py_sigma, 'alpha':py_alpha, 'R':py_R, 'x1':py_x1, 'x2':py_x2}

def pyGetContour(tau, x1_min, x2_min, y, Psi, method='standard', n_points=0, parallel=True):
    cdef int cmethod, status, i
    cdef int n_eqs = 4
    cdef int n_buffer = 200
    cdef int cn_points = n_points
    cdef double cy = y
    cdef double cx1_min = x1_min
    cdef double cx2_min = x2_min
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)

    cdef double[:] ctau = np.ascontiguousarray(tau, dtype=np.double)
    cdef int i_max = ctau.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1
    cdef SolODE **sols = <SolODE **>malloc(i_max*sizeof(SolODE*))

    if method == 'standard':
        cmethod = methods_contour.m_contour_std
    elif method == 'robust':
        cmethod = methods_contour.m_contour_robust
    else:
        cmethod = methods_contour.m_contour_std

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        sols[i] = init_SolODE(n_eqs, n_buffer)
        status = csingle_contour.driver_get_contour(ctau[i], cn_points, cx1_min, cx2_min, cy, pNLens, cmethod, sols[i])

    dat = np.array([convert_SolODE(sols[i]) for i in np.arange(i_max)])

    if np.isscalar(tau):
        dat = dat.item()

    for i in range(i_max):
        free_SolODE(sols[i])
    free(sols)
    free_pLens(pNLens)

    return dat
