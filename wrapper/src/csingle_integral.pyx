#
# GLoW - csingle_integral.pyx
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

from csingle_integral cimport driver_SingleIntegral, \
                              free_Contours, driver_get_contour_SingleIntegral, Contours
from clenses cimport pNamedLens, free_pLens, convert_pphys_to_pLens
from croots cimport CritPoint, convert_pcrit_to_CritPoint, sort_x_CritPoint, sort_t_CritPoint

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


## =======     SINGLE INTEGRAL
## =============================================================================

cdef get_m_integral(pymethod):
    cdef int cmethod

    if pymethod == 'qng':
        cmethod = methods_integral.m_integral_qng
    elif pymethod == 'qag15':
        cmethod = methods_integral.m_integral_g15
    elif pymethod == 'qag21':
        cmethod = methods_integral.m_integral_g21
    elif pymethod == 'qag31':
        cmethod = methods_integral.m_integral_g31
    elif pymethod == 'qag41':
        cmethod = methods_integral.m_integral_g41
    elif pymethod == 'qag51':
        cmethod = methods_integral.m_integral_g51
    elif pymethod == 'qag61':
        cmethod = methods_integral.m_integral_g61
    elif pymethod == 'direct':
        cmethod = methods_integral.m_integral_dir
    else:
        message = "integration method not recognized"
        raise ValueError(message)

    return cmethod

def pySingleIntegral(tau, y, Psi, p_crits, method='qag15', parallel=True):
    cdef int i
    cdef int cmethod  = get_m_integral(method)
    cdef int n_points = len(p_crits)
    cdef double tmin  = p_crits[0]['t']
    cdef double cy    = y
    cdef CritPoint *points  = <CritPoint *>malloc(n_points*sizeof(CritPoint))
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)

    cdef double[:] ctau = np.ascontiguousarray(tau, dtype=np.double)
    cdef int i_max = ctau.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyIt = np.zeros(i_max, dtype=np.double)
    cdef double[:] It = pyIt

    for i in range(n_points):
        points[i] = convert_pcrit_to_CritPoint(p_crits[i])

        if points[i].t < tmin:
            tmin = points[i].t

    # points need to be sorted before going in
    sort_x_CritPoint(n_points, points)

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        It[i] = driver_SingleIntegral(ctau[i], cy, tmin, n_points, points, pNLens, cmethod)

    if np.isscalar(tau):
        pyIt = pyIt.item()

    free(points)
    free_pLens(pNLens)

    return pyIt


## =======     LOW-LEVEL INFORMATION
## =============================================================================

def pyGetBrackets(tau, y, Psi, p_crits):
    cdef int n_points = len(p_crits)
    cdef double tmin  = p_crits[0]['t']
    cdef CritPoint *points  = <CritPoint *>malloc(n_points*sizeof(CritPoint))
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)
    cdef Lens lens = init_lens(pNLens)
    cdef pSIntegral p

    for i in range(n_points):
        points[i] = convert_pcrit_to_CritPoint(p_crits[i])

        if points[i].t < tmin:
            tmin = points[i].t

    # points need to be sorted before going in
    sort_x_CritPoint(n_points, points)

    p.y = y
    p.tau = tau
    p.t = tau + tmin
    p.n_points = n_points;
    p.points = points;
    p.Psi = &lens;
    # ----------------------

    find_Brackets(&p)

    brackets = [[p.brackets[i].a, p.brackets[i].b] for i in range(p.n_brackets)]

    # ---------------------
    free_Brackets(&p)
    free_pLens(pNLens)
    free(points)

    return brackets


def pyAlpha_SingleIntegral(r, tau, y, Psi, p_crits, parallel=True):
    cdef int i
    cdef int n_points = len(p_crits)
    cdef double tmin  = p_crits[0]['t']
    cdef CritPoint *points  = <CritPoint *>malloc(n_points*sizeof(CritPoint))
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)
    cdef Lens lens = init_lens(pNLens)
    cdef pSIntegral p

    cdef double[:] cr = np.ascontiguousarray(r, dtype=np.double)
    cdef int i_max = cr.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyAlpha = np.zeros(i_max, dtype=np.double)
    cdef double[:] alpha = pyAlpha

    for i in range(n_points):
        points[i] = convert_pcrit_to_CritPoint(p_crits[i])

        if points[i].t < tmin:
            tmin = points[i].t

    # points need to be sorted before going in
    sort_x_CritPoint(n_points, points)

    p.y = y
    p.tau = tau
    p.t = tau + tmin
    p.n_points = n_points;
    p.points = points;
    p.Psi = &lens;
    # ---------------------

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        alpha[i] = alpha_SingleIntegral(cr[i], &p)

    if np.isscalar(r):
        pyAlpha = pyAlpha.item()

    # ---------------------
    free_pLens(pNLens)
    free(points)

    return pyAlpha


def pyIntegrand_SingleIntegral(xi, tau, y, Psi, p_crits, parallel=True):
    cdef int i
    cdef int n_points = len(p_crits)
    cdef double tmin  = p_crits[0]['t']
    cdef CritPoint *points  = <CritPoint *>malloc(n_points*sizeof(CritPoint))
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)
    cdef Lens lens = init_lens(pNLens)
    cdef pSIntegral p

    cdef double[:] cxi = np.ascontiguousarray(xi, dtype=np.double)
    cdef int i_max = cxi.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyIntegrand = np.zeros(i_max, dtype=np.double)
    cdef double[:] integrand = pyIntegrand

    for i in range(n_points):
        points[i] = convert_pcrit_to_CritPoint(p_crits[i])

        if points[i].t < tmin:
            tmin = points[i].t

    # points need to be sorted before going in
    sort_x_CritPoint(n_points, points)

    p.y = y
    p.tau = tau
    p.t = tau + tmin
    p.n_points = n_points;
    p.points = points;
    p.Psi = &lens;

    find_Brackets(&p)
    # ---------------------

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        integrand[i] = integrand_SingleIntegral(cxi[i], &p)

    if np.isscalar(xi):
        pyIntegrand = pyIntegrand.item()

    # ---------------------
    free_Brackets(&p)
    free_pLens(pNLens)
    free(points)

    return pyIntegrand


## =======     GET CONTOUR
## =============================================================================

cdef convert_Contours(Contours *cnt):
    cdef int i, j
    cdef Py_ssize_t i_max = cnt.n_contours
    cdef Py_ssize_t j_max = cnt.n_points

    py_x1 = np.zeros((i_max, j_max), dtype=np.double)
    py_x2 = np.zeros((i_max, j_max), dtype=np.double)

    cdef double[:, :] x1 = py_x1
    cdef double[:, :] x2 = py_x2

    for i in range(i_max):
        for j in range(j_max):
            x1[i][j] = cnt.x1[i][j]
            x2[i][j] = cnt.x2[i][j]

    return {'x1':py_x1, 'x2':py_x2}

def pyGetContourSI(tau, y, Psi, p_crits, int n_points=100, parallel=True):
    cdef int i
    cdef int n_critpoints = len(p_crits)
    cdef double tmin = p_crits[0]['t']
    cdef double cy = y
    cdef CritPoint *crit_points = <CritPoint *>malloc(n_points*sizeof(CritPoint))
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)

    cdef double[:] ctau = np.ascontiguousarray(tau, dtype=np.double)
    cdef int i_max = ctau.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    cdef Contours **cnt = <Contours **>malloc(i_max*sizeof(Contours*))

    for i in range(n_critpoints):
        crit_points[i] = convert_pcrit_to_CritPoint(p_crits[i])

        if crit_points[i].t < tmin:
            tmin = crit_points[i].t

    # points need to be sorted before going in
    sort_x_CritPoint(n_critpoints, crit_points)

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        cnt[i] = driver_get_contour_SingleIntegral(ctau[i], n_points, cy, tmin, n_critpoints, crit_points, pNLens)

    dat = np.array([convert_Contours(cnt[i]) for i in np.arange(i_max)])

    if np.isscalar(tau):
        dat = dat.item()

    ## -----------------------------------------------------

    for i in range(i_max):
        free_Contours(cnt[i])

    free(cnt)
    free(crit_points)
    free_pLens(pNLens)

    return dat
