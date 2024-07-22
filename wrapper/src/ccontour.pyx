#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np

cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free

cimport ccontour
from ccontour cimport ctr_types, Center
from ccontour cimport SolODE, init_SolODE, free_SolODE
from croots cimport convert_pcrit_to_CritPoint
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


## =======     INIT CENTERS
## =============================================================================
cpdef convert_Center_to_pcenter(Center ctr):
    p_center = {}

    if ctr.type == ctr_types.ctr_type_min:
        p_center['type'] = 'min'
    if ctr.type == ctr_types.ctr_type_max:
        p_center['type'] = 'max'
    if ctr.type == ctr_types.ctr_type_saddle_8_maxmax:
        p_center['type'] = 'saddle 8 maxmax'
    if ctr.type == ctr_types.ctr_type_saddle_8_minmin:
        p_center['type'] = 'saddle 8 minmin'
    if ctr.type == ctr_types.ctr_type_saddle_O_min:
        p_center['type'] = 'saddle O min'
    if ctr.type == ctr_types.ctr_type_saddle_O_max:
        p_center['type'] = 'saddle O max'

    p_center['x10'] = ctr.x0[0]
    p_center['x20'] = ctr.x0[1]
    p_center['tau0'] = ctr.tau0
    p_center['t0'] = ctr.t0
    p_center['R_max'] = ctr.R_max
    p_center['alpha_out'] = ctr.alpha_out
    p_center['is_init_birthdeath'] = ctr.is_init_birthdeath
    p_center['tau_birth'] = ctr.tau_birth
    p_center['tau_death'] = ctr.tau_death

    return p_center

cdef Center convert_pcenter_to_Center(p_center):
    cdef Center ctr

    if p_center['type'] == 'min':
        ctr.type = ctr_types.ctr_type_min
    if p_center['type'] == 'max':
        ctr.type = ctr_types.ctr_type_max
    if p_center['type'] == 'saddle 8 maxmax':
        ctr.type = ctr_types.ctr_type_saddle_8_maxmax
    if p_center['type'] == 'saddle 8 minmin':
        ctr.type = ctr_types.ctr_type_saddle_8_minmin
    if p_center['type'] == 'saddle O min':
        ctr.type = ctr_types.ctr_type_saddle_O_min
    if p_center['type'] == 'saddle O max':
        ctr.type = ctr_types.ctr_type_saddle_O_max

    ctr.x0[0] = p_center['x10']
    ctr.x0[1] = p_center['x20']
    ctr.tau0 = p_center['tau0']
    ctr.t0 = p_center['t0']
    ctr.R_max = p_center['R_max']
    ctr.alpha_out = p_center['alpha_out']
    ctr.is_init_birthdeath = p_center['is_init_birthdeath']
    ctr.tau_birth = p_center['tau_birth']
    ctr.tau_death = p_center['tau_death']

    return ctr

cpdef pyInit_all_Centers(p_crits, y, Psi):
    cdef int i
    cdef int n_ctrs = len(p_crits)
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)
    cdef CritPoint *points = <CritPoint *>malloc(n_ctrs*sizeof(CritPoint))
    cdef Center *ctrs

    for i in range(n_ctrs):
        points[i] = convert_pcrit_to_CritPoint(p_crits[i])

    ctrs = ccontour.init_all_Center(&n_ctrs, points, y, pNLens)

    p_centers = []
    for i in range(n_ctrs):
        p_centers.append(convert_Center_to_pcenter(ctrs[i]))

    free_pLens(pNLens)
    free(points)
    ccontour.free_all_Center(ctrs)

    return p_centers


## =======     INTEGRATE ALL CONTOURS
## =============================================================================

def pyMultiContour(tau, p_centers, y, Psi, parallel=True):
    cdef int i
    cdef int n_ctrs = len(p_centers)
    cdef double cy = y
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)
    cdef Center *ctrs = <Center *>malloc(n_ctrs*sizeof(Center))

    cdef double[:] ctau = np.ascontiguousarray(tau, dtype=np.double)
    cdef int i_max = ctau.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyIt = np.zeros(i_max, dtype=np.double)
    cdef double[:] It = pyIt

    for i in range(n_ctrs):
        ctrs[i] = convert_pcenter_to_Center(p_centers[i])

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        It[i] = ccontour.driver_contour2d(ctau[i], n_ctrs, ctrs, cy, pNLens)

    if np.isscalar(tau):
        pyIt = pyIt.item()

    free_pLens(pNLens)
    free(ctrs)

    return pyIt


## =======     GET CONTOURS (w poins)
## =============================================================================

cdef convert_SolODE_multi(int n_ctrs, SolODE **sols):
    cdef int i, j
    cdef Py_ssize_t i_max, j_max

    cdef double[:] sigma
    cdef double[:] alpha
    cdef double[:] R
    cdef double[:] x1
    cdef double[:] x2

    dic = {'sigma':[], 'alpha':[], 'R':[], 'x1':[], 'x2':[]}

    i_max = 0
    for i in range(n_ctrs):
        if sols[i] != NULL:
            j_max = sols[i][0].n_points

            py_sigma = np.zeros(j_max, dtype=np.double)
            py_alpha = np.zeros(j_max, dtype=np.double)
            py_R     = np.zeros(j_max, dtype=np.double)
            py_x1    = np.zeros(j_max, dtype=np.double)
            py_x2    = np.zeros(j_max, dtype=np.double)

            sigma = py_sigma
            alpha = py_alpha
            R     = py_R
            x1    = py_x1
            x2    = py_x2

            for j in range(j_max):
                sigma[j] = sols[i][0].t[j]
                alpha[j] = sols[i][0].y[0][j]
                R[j]     = sols[i][0].y[1][j]
                x1[j]    = sols[i][0].y[2][j]
                x2[j]    = sols[i][0].y[3][j]

            dic['sigma'].append(py_sigma)
            dic['alpha'].append(py_alpha)
            dic['R'].append(py_R)
            dic['x1'].append(py_x1)
            dic['x2'].append(py_x2)

    return dic

def pyGetMultiContour(tau, p_centers, y, Psi, n_points=0, parallel=True):
    cdef int i
    cdef int n_ctrs = len(p_centers)
    cdef int cn_points = n_points
    cdef double cy = y
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)
    cdef Center *ctrs = <Center *>malloc(n_ctrs*sizeof(Center))

    cdef double[:] ctau = np.ascontiguousarray(tau, dtype=np.double)
    cdef int i_max = ctau.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1
    cdef SolODE ***sols = <SolODE ***>malloc(i_max*sizeof(SolODE**))

    for i in range(n_ctrs):
        ctrs[i] = convert_pcenter_to_Center(p_centers[i])

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        sols[i] = ccontour.driver_get_contour2d(ctau[i], cn_points, n_ctrs, ctrs, cy, pNLens)

    dat = np.array([convert_SolODE_multi(n_ctrs, sols[i]) for i in np.arange(i_max)])

    if np.isscalar(tau):
        dat = dat.item()

    for i in range(i_max):
        ccontour.free_SolODE_contour2d(n_ctrs, sols[i])
    free(sols)
    free_pLens(pNLens)
    free(ctrs)

    return dat


## =======     GET CARTESIAN CONTOUR
## =============================================================================

cdef convert_SolODE_cartesian(SolODE *sol):
    cdef int i
    cdef Py_ssize_t i_max = sol.n_points

    py_sigma = np.zeros(i_max, dtype=np.double)
    py_x1    = np.zeros(i_max, dtype=np.double)
    py_x2    = np.zeros(i_max, dtype=np.double)

    cdef double[:] c_sigma = py_sigma
    cdef double[:] c_x1    = py_x1
    cdef double[:] c_x2    = py_x2

    for i in range(i_max):
        c_sigma[i] = sol.t[i]
        c_x1[i] = sol.y[0][i]
        c_x2[i] = sol.y[1][i]

    return {'sigma':py_sigma, 'x1':py_x1, 'x2':py_x2}

def pyGetContour_x1x2(x10, x20, y, sigmaf, n_points, Psi, parallel=True):
    cdef int i, status
    cdef int n_eqs = 2
    cdef int cn_points = n_points
    cdef double cy = y
    cdef double csigmaf = sigmaf
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)

    x1s = x10
    x2s = x20
    if np.isscalar(x1s) is False:
        if np.isscalar(x2s):
            x2s = np.full_like(x1s, x2s)
    if np.isscalar(x2s) is False:
        if np.isscalar(x1s):
            x1s = np.full_like(x2s, x1s)

    cdef double[:] cx1 = np.ascontiguousarray(x1s, dtype=np.double)
    cdef double[:] cx2 = np.ascontiguousarray(x2s, dtype=np.double)
    cdef int i_max = cx1.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1
    cdef SolODE **sols = <SolODE **>malloc(i_max*sizeof(SolODE*))

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='dynamic', chunksize=1):
        sols[i] = init_SolODE(n_eqs, cn_points)
        status = ccontour.driver_get_contour2d_x1x2(cx1[i], cx2[i], cy, csigmaf, cn_points, pNLens, sols[i])

    dat = np.array([convert_SolODE_cartesian(sols[i]) for i in np.arange(i_max)])

    if np.isscalar(x1s):
        dat = dat.item()

    for i in range(i_max):
        free_SolODE(sols[i])
    free(sols)
    free_pLens(pNLens)

    return dat
