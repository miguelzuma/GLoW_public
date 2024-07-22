#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np

# optional dependency: needed for the numerical verification of the Struve function
try:
    import mpmath as mp
except ModuleNotFoundError:
    mp = None

cimport cython
from cython.parallel import prange
from libc.math cimport pow, fabs

cimport cspecial

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


## =======     FRESNEL FUNCTIONS
## =============================================================================

def pyFresnel_f(x, parallel=True):
    cdef int i
    cdef double[:] xs = np.ascontiguousarray(x, dtype=np.double)
    cdef int i_max = xs.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyM = np.zeros(i_max, dtype=np.double)
    cdef double[:] M = pyM

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='static'):
        M[i] = cspecial.f_fresnel(xs[i])

    if np.isscalar(x):
        pyM = pyM.item()

    return pyM

def pyFresnel_g(x, parallel=True):
    cdef int i
    cdef double[:] xs = np.ascontiguousarray(x, dtype=np.double)
    cdef int i_max = xs.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyM = np.zeros(i_max, dtype=np.double)
    cdef double[:] M = pyM

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='static'):
        M[i] = cspecial.g_fresnel(xs[i])

    if np.isscalar(x):
        pyM = pyM.item()

    return pyM


## =======     STRUVE FUNCTIONS
## =============================================================================

def pyMtilde_Struve_PowerSeries(x, double nu, double tol=1e-10, bint parallel=True):
    cdef int i
    cdef double[:] xs = np.ascontiguousarray(x, dtype=np.double)
    cdef int i_max = xs.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyM = np.zeros(i_max, dtype=np.double)
    cdef double[:] M = pyM

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='static'):
        M[i] = cspecial.Mtilde_Struve_PowerSeries(xs[i], nu, tol)

    if np.isscalar(x):
        pyM = pyM.item()

    return pyM

def pyMtilde_Struve_Asymptotic(x, double nu, double tol=1e-10, bint parallel=True):
    cdef int i
    cdef double[:] xs = np.ascontiguousarray(x, dtype=np.double)
    cdef int i_max = xs.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyM = np.zeros(i_max, dtype=np.double)
    cdef double[:] M = pyM

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='static'):
        M[i] = cspecial.Mtilde_Struve_Asymptotic(xs[i], nu, tol)

    if np.isscalar(x):
        pyM = pyM.item()

    return pyM

def pyMtilde_Struve_PieceWise(x, double nu, double tol=1e-10, bint parallel=True):
    cdef int i
    cdef double[:] xs = np.ascontiguousarray(x, dtype=np.double)
    cdef int i_max = xs.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyM = np.zeros(i_max, dtype=np.double)
    cdef double[:] M = pyM

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='static'):
        M[i] = cspecial.Mtilde_Struve_PieceWise(xs[i], nu, tol)

    if np.isscalar(x):
        pyM = pyM.item()

    return pyM

def pyMtilde_Struve(x, double nu, double tol=1e-10, bint parallel=True):
    cdef int i
    cdef double[:] xs = np.ascontiguousarray(x, dtype=np.double)
    cdef int i_max = xs.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyM = np.zeros(i_max, dtype=np.double)
    cdef double[:] M = pyM

    for i in prange(i_max, nogil=True, num_threads=nthreads, schedule='static'):
        M[i] = cspecial.Mtilde_Struve(xs[i], nu, tol)

    if np.isscalar(x):
        pyM = pyM.item()

    return pyM

## ----------------------------------------

def Mtilde_Struve_mpmath(x, nu, prec_dps=100):
    if mp is None:
        message = "It seems that the python module mpmath is not installed. Without it, the python "\
                  "version of the Struve function cannot be used."
        raise ModuleNotFoundError(message)

    mp.mp.dps = prec_dps

    if np.isscalar(x):
        return Mtilde_Struve_mpmath_novec(x, nu)
    else:
        return Mtilde_Struve_mpmath_vec(x, nu)

cdef double Mtilde_Struve_mpmath_novec(double x, double nu):
    return float(mp.struvel(nu, x)-mp.besseli(nu, x))*pow(2./x, nu)

cdef Mtilde_Struve_mpmath_vec(double[:] x, double nu):
    cdef int i
    cdef Py_ssize_t i_max = x.shape[0]

    pyM = np.zeros(i_max, dtype=np.double)
    cdef double[:] M = pyM

    for i in range(i_max):
        M[i] = Mtilde_Struve_mpmath_novec(x[i], nu)

    return pyM

cdef pyMtilde_Struve_PieceWise_mpmath(double x, double nu, double tol):
    if x < 10:
        return cspecial.Mtilde_Struve_PowerSeries(x, nu, tol)
    elif x > (fabs(nu) + 20):
        return cspecial.Mtilde_Struve_Asymptotic(x, nu, tol)
    else:
        return Mtilde_Struve_mpmath_novec(x, nu)

def pyMtilde_Struve_mpmath(x, nu, tol=1e-10):
    if tol is None:
        return Mtilde_Struve_mpmath(x, nu)

    if np.isscalar(x):
        return pyMtilde_Struve_PieceWise_mpmath(x, nu, tol)
    else:
        return pyMtilde_Struve_mpmath_vec(x, nu, tol)

cdef pyMtilde_Struve_mpmath_vec(double[:] x, double nu, double tol):
    cdef int i
    cdef Py_ssize_t i_max = x.shape[0]

    pyM = np.zeros(i_max, dtype=np.double)
    cdef double[:] M = pyM

    for i in range(i_max):
        M[i] = pyMtilde_Struve_PieceWise_mpmath(x[i], nu, tol)

    return pyM


## =======     HYPERGEOMETRIC FUNCTION
## =============================================================================

def pyF11(u, c, parallel=True):
    cdef double cc = c
    cdef double[:] cu = np.ascontiguousarray(u, dtype=np.double)
    cdef int imax = cu.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyF = np.empty(imax, dtype=np.cdouble)
    cdef double complex[:] F = pyF

    # u is assumed to be sorted, otherwise F11_sorted will raise an error
    F11_sorted(&cu[0], cc, imax, &F[0], nthreads)

    if np.isscalar(u):
        pyF = pyF.item()

    return pyF

def pyF11_singlepoint(u, c, parallel=True):
    cdef double[:] cu = np.ascontiguousarray(u, dtype=np.double)
    cdef double[:] cc

    if np.isscalar(c) and not np.isscalar(u):
        cc = np.full_like(u, c)
    else:
        cc = np.ascontiguousarray(c, dtype=np.double)

    cdef int i
    cdef int imax = cu.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyF = np.empty(imax, dtype=np.cdouble)
    pyn = np.empty(imax, dtype=np.intc)
    pyflag = np.empty(imax, dtype=np.intc)
    cdef double complex[:] F = pyF
    cdef int[:] cn = pyn
    cdef int[:] cflag = pyflag

    for i in prange(imax, nogil=True, num_threads=nthreads, schedule='static'):
        F[i] = F11_singlepoint(cu[i], cc[i], &cn[i], &cflag[i])

    if np.isscalar(u):
        pyF = pyF.item()
        pyn = pyn.item()
        pyflag = pyflag.item()

    return pyF, pyn, pyflag

## =======     INTERPOLATION
## =============================================================================

def pyInterpolate_sorted(x_subgrid, x_grid, y_grid):
    # if the arrays are not contiguous, make a contiguous copy
    cdef int status
    cdef double[:] cx_g  = np.ascontiguousarray(x_grid, dtype=np.double)
    cdef double[:] cy_g  = np.ascontiguousarray(y_grid, dtype=np.double)
    cdef double[:] cx_sg = np.ascontiguousarray(x_subgrid, dtype=np.double)

    y_subgrid = np.empty_like(x_subgrid)
    cdef double[:] cy_sg = y_subgrid

    status = sorted_interpolation(&cx_sg[0], &cy_sg[0], cx_sg.shape[0], &cx_g[0], &cy_g[0], cx_g.shape[0])

    return y_subgrid
