#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np

cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free

cimport croots
from croots cimport types_CritPoint, CritPoint, pImage
from clenses cimport pNamedLens, free_pLens, convert_pphys_to_pLens, Lens, init_lens

## ----------  Precision ------------- ##
handle_GSL_errors()
cdef update_pprec(Prec_General pprec_general):
    global pprec
    pprec = pprec_general
## ----------------------------------- ##

## =======     CRITICAL POINTS AND ROOTS
## =============================================================================
def pyFind_CritPoint_2D(x1guess, x2guess, y, Psi, keep_all=True):
    cdef int i
    cdef pImage im
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)
    cdef Lens cPsi = init_lens(pNLens)
    
    cdef double[:] cx1guess = np.ascontiguousarray(x1guess, dtype=np.double)
    cdef double[:] cx2guess = np.ascontiguousarray(x2guess, dtype=np.double)
    cdef int i_max = cx1guess.shape[0]
    cdef CritPoint *points = <CritPoint *>malloc(i_max*sizeof(CritPoint))
    
    im.y = y
    im.Psi = &cPsi
    for i in range(i_max):
        im.point = &points[i]    
        croots.find_CritPoint_root_2D(cx1guess[i], cx2guess[i], &im)
        
    if keep_all is False:
        i_max = filter_different_CritPoint_list(points, i_max)
        p = [convert_CritPoint_to_pcrit(points[k]) for k in range(i_max)]
    else:
        p = convert_CritPoint_list(points, i_max)
    
    free(points)
    free_pLens(pNLens)
    
    return p

cdef filter_different_CritPoint_list(CritPoint *points, int n_points):
    cdef int i, j, n
    cdef int ctrue  = pprec.ctrue
    cdef int cfalse = pprec.cfalse
    cdef int compare_flag
    
    n = 1
    compare_flag = cfalse
    for i in range(1, n_points):
        for j in range(n):
            if is_same_CritPoint(&points[i], &points[j]) == ctrue:
                compare_flag = ctrue
                continue
        
        if compare_flag == cfalse:
            swap_CritPoint(&points[n], &points[i])
            n += 1
        
        compare_flag = cfalse
        
    return n

cdef convert_CritPoint_list(CritPoint *points, int n_points):
    cdef int i
    
    pytypes = np.empty(n_points, dtype=np.intc)
    pyx1    = np.empty(n_points, dtype=np.double)
    pyx2    = np.empty(n_points, dtype=np.double)
    pyt     = np.empty(n_points, dtype=np.double)
    pymag   = np.empty(n_points, dtype=np.double)
    cdef int[:] ctypes  = pytypes
    cdef double[:] cx1  = pyx1
    cdef double[:] cx2  = pyx2
    cdef double[:] ct   = pyt
    cdef double[:] cmag = pymag
    
    dic_types = {<int>types_CritPoint.type_min : 'min',
                 <int>types_CritPoint.type_max : 'max',
                 <int>types_CritPoint.type_saddle : 'saddle',
                 <int>types_CritPoint.type_singcusp_max  : 'sing/cusp max',
                 <int>types_CritPoint.type_singcusp_min  : 'sing/cusp min',
                 <int>types_CritPoint.type_non_converged : 'non-converged'}
        
    for i in range(n_points):
        ctypes[i] = points[i].type
        cx1[i]    = points[i].x1
        cx2[i]    = points[i].x2
        ct[i]     = points[i].t
        cmag[i]   = points[i].mag
        
    pypoints = {'type' : pytypes,
                'x1'   : pyx1,
                'x2'   : pyx2,
                't'    : pyt,
                'mag'  : pymag}
                
    return pypoints, dic_types

cpdef convert_CritPoint_to_pcrit(CritPoint p):
    p_crit = {}

    if p.type == types_CritPoint.type_min:
        p_crit['type'] = 'min'
    if p.type == types_CritPoint.type_max:
        p_crit['type'] = 'max'
    if p.type == types_CritPoint.type_saddle:
        p_crit['type'] = 'saddle'
    if p.type == types_CritPoint.type_singcusp_max:
        p_crit['type'] = 'sing/cusp max'
    if p.type == types_CritPoint.type_singcusp_min:
        p_crit['type'] = 'sing/cusp min'
    if p.type == types_CritPoint.type_non_converged:
        p_crit['type'] = 'non-converged'

    p_crit['t']   = p.t
    p_crit['x1']  = p.x1
    p_crit['x2']  = p.x2
    p_crit['mag'] = p.mag

    return p_crit

cdef CritPoint convert_pcrit_to_CritPoint(p_crit):
    cdef CritPoint p

    if p_crit['type'] == 'min':
        p.type = types_CritPoint.type_min
    if p_crit['type'] == 'max':
        p.type = types_CritPoint.type_max
    if p_crit['type'] == 'saddle':
        p.type = types_CritPoint.type_saddle
    if p_crit['type'] == 'sing/cusp max':
        p.type = types_CritPoint.type_singcusp_max
    if p_crit['type'] == 'sing/cusp min':
        p.type = types_CritPoint.type_singcusp_min
    if p_crit['type'] == 'non-converged':
        p.type = types_CritPoint.type_non_converged

    p.t = p_crit['t']
    p.x1 = p_crit['x1']
    p.x2 = p_crit['x2']
    p.mag = p_crit['mag']

    return p

cpdef pyFind_all_CritPoints_1D(y, Psi):
    cdef int n_points, i
    cdef CritPoint *points
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)

    points = croots.driver_all_CritPoints_1D(&n_points, y, pNLens)
    free_pLens(pNLens)

    p_crits = [convert_CritPoint_to_pcrit(points[i]) for i in range(n_points)]
    free(points)

    return p_crits

cpdef pyFind_all_CritPoints_2D(y, Psi):
    cdef int n_points, i
    cdef CritPoint *points
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)

    points = croots.driver_all_CritPoints_2D(&n_points, y, pNLens)
    free_pLens(pNLens)

    p_crits = [convert_CritPoint_to_pcrit(points[i]) for i in range(n_points)]
    free(points)

    return p_crits

cpdef pyCheck_min(y, Psi):
    cdef int n_points
    cdef CritPoint p
    cdef pNamedLens *pNLens = convert_pphys_to_pLens(Psi)

    n_points = croots.check_only_min_CritPoint_2D(&p, y, pNLens)
    free_pLens(pNLens)

    return n_points, p.x1, p.x2, p.t, p.mag
