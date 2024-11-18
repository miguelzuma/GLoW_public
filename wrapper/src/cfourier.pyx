#
# GLoW - cfourier.pyx
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
cimport numpy as cnp

cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.math cimport M_PI, log2

cimport cfourier
from croots cimport CritPoint, convert_pcrit_to_CritPoint, sort_t_CritPoint

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


## =======     FOURIER
## =============================================================================

## --- Time domain
## ------------------------------------------------------------------------
def pyR0_reg(tau, double alpha, double beta, double sigma, bint parallel=True):
    cdef int i
    cdef double[:] ts = np.ascontiguousarray(tau, dtype=np.double)
    cdef int n_ts = ts.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyR = np.empty(n_ts, dtype=np.double)
    cdef double[:] R = pyR

    for i in prange(n_ts, nogil=True, num_threads=nthreads, schedule='static'):
        R[i] = cfourier.R0_reg(ts[i], alpha, beta, sigma)

    if np.isscalar(tau):
        pyR = pyR.item()

    return pyR

def pyR1_reg(tau, double alpha, double beta, double sigma, bint parallel=True):
    cdef int i
    cdef double[:] ts = np.ascontiguousarray(tau, dtype=np.double)
    cdef int n_ts = ts.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyR = np.empty(n_ts, dtype=np.double)
    cdef double[:] R = pyR

    for i in prange(n_ts, nogil=True, num_threads=nthreads, schedule='static'):
        R[i] = cfourier.R1_reg(ts[i], alpha, beta, sigma)

    if np.isscalar(tau):
        pyR = pyR.item()

    return pyR

def pySfull_reg(tau, double A, double B, parallel=True):
    cdef int i
    cdef double[:] ts = np.ascontiguousarray(tau, dtype=np.double)
    cdef int n_ts = ts.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyR = np.empty(n_ts, dtype=np.double)
    cdef double[:] R = pyR

    for i in prange(n_ts, nogil=True, num_threads=nthreads, schedule='static'):
        R[i] = cfourier.Sfull_reg(ts[i], A, B)

    if np.isscalar(tau):
        pyR = pyR.item()

    return pyR

def pyR0_step_reg(tau, double tau_scale, double I_asymp, double alpha=1., double sigma=3.2, bint parallel=True):
    cdef int i
    cdef double[:] ts = np.ascontiguousarray(tau, dtype=np.double)
    cdef int n_ts = ts.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyR = np.empty(n_ts, dtype=np.double)
    cdef double[:] R = pyR

    for i in prange(n_ts, nogil=True, num_threads=nthreads, schedule='static'):
        R[i] = cfourier.R0_step_reg(ts[i], tau_scale, I_asymp, alpha, sigma)

    if np.isscalar(tau):
        pyR = pyR.item()

    return pyR

## --- Frequency domain
## ------------------------------------------------------------------------
def pyR0_reg_FT(w, double alpha, double beta, double sigma, bint parallel=True):
    cdef int i
    cdef double[:] ws = np.ascontiguousarray(w, dtype=np.double)
    cdef int n_ws = ws.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyR = np.empty(n_ws, dtype=np.cdouble)
    cdef double complex[:] R = pyR

    for i in prange(n_ws, nogil=True, num_threads=nthreads, schedule='static'):
        R[i] = cfourier.R0_reg_FT(ws[i], alpha, beta, sigma)

    if np.isscalar(w):
        pyR = pyR.item()

    return pyR

def pyR1_reg_FT(w, double alpha, double beta, double sigma, bint parallel=True):
    cdef int i
    cdef double[:] ws = np.ascontiguousarray(w, dtype=np.double)
    cdef int n_ws = ws.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyR = np.empty(n_ws, dtype=np.cdouble)
    cdef double complex[:] R = pyR

    for i in prange(n_ws, nogil=True, num_threads=nthreads, schedule='static'):
        R[i] = cfourier.R1_reg_FT(ws[i], alpha, beta, sigma)

    if np.isscalar(w):
        pyR = pyR.item()

    return pyR

def pySfull_reg_FT(w, double A, double B, bint parallel=True):
    cdef int i
    cdef double[:] ws = np.ascontiguousarray(w, dtype=np.double)
    cdef int n_ws = ws.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyR = np.empty(n_ws, dtype=np.cdouble)
    cdef double complex[:] R = pyR

    for i in prange(n_ws, nogil=True, num_threads=nthreads, schedule='static'):
        R[i] = cfourier.Sfull_reg_FT(ws[i], A, B)

    if np.isscalar(w):
        pyR = pyR.item()

    return pyR

def pyR0_step_reg_FT(w, double tau_scale, double I_asymp, double alpha=1., double sigma=3.2, bint parallel=True):
    cdef int i
    cdef double[:] ws = np.ascontiguousarray(w, dtype=np.double)
    cdef int n_ws = ws.shape[0]
    cdef int nthreads = max_num_threads if parallel else 1

    pyR = np.empty(n_ws, dtype=np.cdouble)
    cdef double complex[:] R = pyR

    for i in prange(n_ws, nogil=True, num_threads=nthreads, schedule='static'):
        R[i] = cfourier.R0_step_reg_FT(ws[i], tau_scale, I_asymp, alpha, sigma)

    if np.isscalar(w):
        pyR = pyR.item()

    return pyR


## =============================================================================


## --- Windowing
## ------------------------------------------------------------------------
def pyApply_Tukey(cnp.ndarray f_in, double alpha):
    cdef double[:] cf_in = np.ascontiguousarray(f_in, dtype=np.double)
    cdef int size = cf_in.shape[0]

    apply_window_Tukey(&cf_in[0], size, alpha)

    return f_in

def pyApply_right_Tukey(cnp.ndarray f_in, double alpha):
    cdef double[:] cf_in = np.ascontiguousarray(f_in, dtype=np.double)
    cdef int size = cf_in.shape[0]

    apply_right_window_Tukey(&cf_in[0], size, alpha)

    return f_in

## --- FFTs
## ------------------------------------------------------------------------
def pyFFT_gsl(cnp.ndarray f_in, bint parallel=False):
    cdef int ndim = f_in.ndim

    f_in = np.ascontiguousarray(f_in) # makes a contiguous copy of the numpy array

    if ndim == 1:
        f_fft = pyFFT_gsl_novec(f_in)
    elif ndim == 2:
        f_fft = pyFFT_gsl_vec(f_in, parallel)
    else:
        message = 'too many dimensions for the input of the FT'
        raise ValueError(message)

    return f_fft

cdef pyFFT_gsl_novec(double[:] f_in):
    cdef int size = f_in.shape[0]
    cdef int size_freq = int(size/2 + 1)

    pyf_out = np.empty(size_freq, dtype=np.cdouble)
    cdef double complex[:] f_out = pyf_out

    compute_reFFT_gsl(&f_in[0], size, &f_out[0])

    return pyf_out

cdef pyFFT_gsl_vec(double[:,:] f_in, bint parallel):
    cdef int i
    cdef int n_batches = f_in.shape[0]
    cdef int size      = f_in.shape[1]
    cdef int size_freq = int(size/2 + 1)

    pyf_out = np.empty((n_batches, size_freq), dtype=np.cdouble)
    cdef double complex[:,:] f_out = pyf_out

    if parallel is True:
        for i in prange(n_batches, nogil=True, schedule='dynamic', chunksize=1):
            compute_reFFT_gsl(&f_in[i][0], size, &f_out[i][0])
    else:
        for i in range(n_batches):
            compute_reFFT_gsl(&f_in[i][0], size, &f_out[i][0])

    return pyf_out

## ------------------------------------------------------------------------

def pyFFT_pocketfft(cnp.ndarray f_in, bint parallel=False):
    cdef int ndim = f_in.ndim

    f_in = np.ascontiguousarray(f_in) # makes a contiguous copy of the numpy array

    if ndim == 1:
        f_fft = pyFFT_pocketfft_novec(f_in)
    elif ndim == 2:
        f_fft = pyFFT_pocketfft_vec(f_in, parallel)
    else:
        message = 'too many dimensions for the input of the FT'
        raise ValueError(message)

    return f_fft

cdef pyFFT_pocketfft_novec(double[:] f_in):
    cdef int size = f_in.shape[0]
    cdef int size_freq = int(size/2 + 1)

    pyf_out = np.empty(size_freq, dtype=np.cdouble)
    cdef double complex[:] f_out = pyf_out

    compute_reFFT_pocketfft(&f_in[0], size, &f_out[0])

    return pyf_out

cdef pyFFT_pocketfft_vec(double[:,:] f_in, bint parallel):
    cdef int i
    cdef int n_batches = f_in.shape[0]
    cdef int size      = f_in.shape[1]
    cdef int size_freq = int(size/2 + 1)

    pyf_out = np.empty((n_batches, size_freq), dtype=np.cdouble)
    cdef double complex[:,:] f_out = pyf_out

    if parallel is True:
        for i in prange(n_batches, nogil=True, schedule='dynamic', chunksize=1):
            compute_reFFT_pocketfft(&f_in[i][0], size, &f_out[i][0])
    else:
        for i in range(n_batches):
            compute_reFFT_pocketfft(&f_in[i][0], size, &f_out[i][0])

    return pyf_out

## ------------------------------------------------------------------------

cdef RegScheme* pyInit_RegScheme(tau_grid, It_reg_grid, reg_sch, parallel=False):
    p_crits = reg_sch['p_crits']
    cdef int n_points = len(p_crits)
    cdef RegScheme *sch = <RegScheme*>malloc(sizeof(RegScheme))
    cdef CritPoint *points = <CritPoint *>malloc(n_points*sizeof(CritPoint))

    cdef int num_threads
    cdef double[:] ctau = np.ascontiguousarray(tau_grid, dtype=np.double)
    cdef double[:] cIt  = np.ascontiguousarray(It_reg_grid, dtype=np.double)
    cdef int n_tau = ctau.shape[0]

    for i in range(n_points):
        points[i] = convert_pcrit_to_CritPoint(p_crits[i])

    sch.ps = points
    sch.n_ps = n_points
    sch.stage = reg_sch['stage']

    sch.slope = reg_sch['slope'] if reg_sch['slope'] is not None else 0.

    sch.det = reg_sch['det']

    for i, (amp, index) in enumerate(zip(reg_sch['amp'], reg_sch['index'])):
        sch.amp[i]   = amp   if amp   is not None else 0.
        sch.index[i] = index if index is not None else 0.

    if parallel is True:
        num_threads = max_num_threads
    else:
        num_threads = 1

    sch.nthreads = num_threads

    sch.n_grid = n_tau
    sch.tau_grid = &ctau[0]
    sch.It_reg_grid = &cIt[0]

    return sch

def pyUpdate_RegScheme(stage, tau_grid, It_grid, reg_sch, parallel=False):
    cdef int num_threads
    cdef int cstage = stage
    cdef RegScheme *sch = pyInit_RegScheme(tau_grid, np.empty(1), reg_sch, parallel)

    cdef double[:] ctau = np.ascontiguousarray(tau_grid, dtype=np.double)
    cdef double[:] cIt  = np.ascontiguousarray(It_grid, dtype=np.double)
    cdef int n_tau = ctau.shape[0]

    pyIt_reg = np.empty(n_tau, dtype=np.double)
    cdef double[:] cIt_reg = pyIt_reg

    sch.It_reg_grid = &cIt_reg[0]
    update_RegScheme(&cIt[0], cstage, sch)

    reg_sch['stage'] = sch.stage
    reg_sch['slope'] = sch.slope
    reg_sch['amp']   = [sch.amp[0], sch.amp[1]]
    reg_sch['index'] = [sch.index[0], sch.index[1]]

    pyFree_RegScheme(sch)

    return pyIt_reg

cdef pyFree_RegScheme(RegScheme *sch):
    free(sch.ps)
    free(sch)

cdef FreqTable* pyInit_FreqTable(tau_grid, It_reg_grid, p_prec, reg_sch):
    cdef FreqTable *ft
    cdef RegScheme *sch = pyInit_RegScheme(tau_grid, It_reg_grid, reg_sch, p_prec['parallel'])

    ft = init_FreqTable(p_prec['wmin'],\
                        p_prec['wmax'],\
                        sch,\
                        p_prec['N_keep'],\
                        p_prec['N_below_discard'],\
                        p_prec['N_above_discard'],\
                        p_prec['smallest_tau_max'],\
                        p_prec['window_transition'])
    return ft

cdef pyFree_FreqTable(FreqTable *ft):
    pyFree_RegScheme(ft.sch)
    free_FreqTable(ft)

def pyFreqTable_to_dic(tau_grid, It_reg_grid, p_prec, reg_sch):
    cdef int i, n_fft
    cdef double tau_max, dtau, df
    cdef FreqTable *ft = pyInit_FreqTable(tau_grid, It_reg_grid, p_prec, reg_sch)

    freq_table_dic= {}
    freq_table_dic['n_fft'] = []
    freq_table_dic['dtau']  = []
    freq_table_dic['df']    = []
    freq_table_dic['n_fft_keep'] = []
    freq_table_dic['wmin_real']  = []
    freq_table_dic['wmax_real']  = []
    freq_table_dic['wmin_batch'] = []
    freq_table_dic['wmax_batch'] = []
    freq_table_dic['tau_max']    = []

    for i in range(ft.n_batches):
        n_fft = ft.n_fft[i]
        tau_max = ft.tau_max[i]
        dtau = tau_max/(n_fft-1.)
        df = 1./dtau/n_fft

        freq_table_dic['n_fft'].append(n_fft)
        freq_table_dic['dtau'].append(dtau)
        freq_table_dic['df'].append(df)
        freq_table_dic['n_fft_keep'].append(ft.imax_batch[i]-ft.imin_batch[i])
        freq_table_dic['wmin_real'].append(2*np.pi*ft.fmin_real[i])
        freq_table_dic['wmax_real'].append(2*np.pi*ft.fmax_real[i])
        freq_table_dic['wmin_batch'].append(2*np.pi*ft.imin_batch[i]*df)
        freq_table_dic['wmax_batch'].append(2*np.pi*(ft.imax_batch[i]-1)*df)
        freq_table_dic['tau_max'].append(tau_max)

    pyFree_FreqTable(ft)
    return freq_table_dic

## ------------------------------------------------------------------------

def pyIt_sing(tau, reg_sch, stage=None, parallel=True):
    cdef RegScheme *sch = pyInit_RegScheme(np.empty(1), np.empty(1), reg_sch, parallel=parallel)

    cdef int cstage = sch.stage if stage is None else stage
    cdef int nthreads = max_num_threads if parallel else 1

    cdef double[:] ts = np.ascontiguousarray(tau, dtype=np.double)
    cdef int n_ts = ts.shape[0]

    pyIt_sing = np.empty(n_ts, dtype=np.double)
    cdef double[:] cIts = pyIt_sing

    fill_It_sing(n_ts, &ts[0], &cIts[0], cstage, sch, nthreads)
    pyFree_RegScheme(sch)

    if np.isscalar(tau) is True:
        pyIt_sing = pyIt_sing.item()

    return pyIt_sing

def pyFw_sing(w, reg_sch, stage=None, parallel=True):
    cdef RegScheme *sch = pyInit_RegScheme(np.empty(1), np.empty(1), reg_sch, parallel=parallel)
    cdef int cstage = sch.stage if stage is None else stage
    cdef int nthreads = max_num_threads if parallel else 1

    cdef double[:] ws = np.ascontiguousarray(w, dtype=np.double)
    cdef int n_ws = ws.shape[0]

    pyFw_sing = np.empty(n_ws, dtype=np.cdouble)
    cdef double complex[:] cFws = pyFw_sing

    fill_Fw_sing(n_ws, &ws[0], &cFws[0], cstage, sch, nthreads)
    pyFree_RegScheme(sch)

    if np.isscalar(w) is True:
        pyFw_sing = pyFw_sing.item()

    return pyFw_sing

## ------------------------------------------------------------------------

def pyCompute_Fw(tau_grid, It_reg_grid, p_prec, reg_sch):
    cdef int i, n_ws
    cdef int n_ts = tau_grid.shape[0]
    cdef FreqTable *ft = pyInit_FreqTable(tau_grid, It_reg_grid, p_prec, reg_sch)

    n_ws = ft.n_total + 2

    w_grid  = np.empty(n_ws, dtype=np.double)
    Fw_grid = np.empty(n_ws, dtype=np.cdouble)
    Fw_reg_grid = np.empty(n_ws, dtype=np.cdouble)

    cdef double[:] cws = w_grid
    cdef double complex[:] cFws = Fw_grid
    cdef double complex[:] cFws_reg = Fw_reg_grid

    compute_Fw(&cws[0], &cFws_reg[0], &cFws[0], ft)

    pyFree_FreqTable(ft)
    return w_grid, Fw_grid, Fw_reg_grid

def pyCompute_Fw_std(tau_grid, It_reg_grid, p_prec, reg_sch):
    cdef int i, n_tau, n_ts, n_ws, cstage
    cdef double dtau, df, tau_max, wmax, wmin, alpha
    cdef double *It
    cdef RegScheme *sch = pyInit_RegScheme(tau_grid, It_reg_grid, reg_sch, parallel=False)

    wmax = p_prec['wmax']
    wmin = p_prec['wmin']
    alpha = p_prec['window_transition']

    n_ts = 2**int(log2(wmax/wmin)+1)
    n_ws = int(1 + n_ts/2)

    w_grid  = np.empty(n_ws, dtype=np.double)
    Fw_grid = np.empty(n_ws, dtype=np.cdouble)
    Fw_reg_grid = np.empty(n_ws, dtype=np.cdouble)
    cdef double[:] cws = w_grid
    cdef double complex[:] cFws = Fw_grid
    cdef double complex[:] cFws_reg = Fw_reg_grid

    cdef double complex imag_unit = 1.j

    ## ---------------------------
    with nogil:
        tau_max = 2*M_PI*(n_ts-1)/n_ts/wmin
        dtau = tau_max/(n_ts-1)
        df = 1./dtau/n_ts

        It = <double *>malloc(n_ts*sizeof(double))

        for i in range(n_ts):
            It[i] = i*dtau

        sorted_interpolation(It, It, n_ts, sch.tau_grid, sch.It_reg_grid, sch.n_grid);
        apply_right_window_Tukey(It, n_ts, alpha)
        compute_reFFT_pocketfft(It, n_ts, &cFws_reg[0])

        free(It)

        for i in range(1, n_ws):
            cws[i] = 2*M_PI*i*df
            cFws_reg[i] = -imag_unit*i*df*dtau*cFws_reg[i].conjugate()
            cFws[i] = cFws_reg[i] + eval_Fw_sing(cws[i], sch.stage, sch)
    ## ---------------------------

    pyFree_RegScheme(sch)
    return w_grid[1:], Fw_grid[1:], Fw_reg_grid[1:]


## =============================================================================


## --- Direct FT
## ------------------------------------------------------------------------
def pyCompute_Fw_directFT(w, tau_grid, It_reg_grid, reg_sch, stage=None, parallel=True):
    cdef int i
    cdef RegScheme *sch = pyInit_RegScheme(tau_grid, It_reg_grid, reg_sch, parallel)
    cdef int cstage = sch.stage if stage is None else stage

    cdef double[:] ws = np.ascontiguousarray(w, dtype=np.double)
    cdef int n_ws = ws.shape[0]

    pyFw_reg = np.empty(n_ws, dtype=np.cdouble)
    pyFw     = np.empty(n_ws, dtype=np.cdouble)
    cdef double complex[:] Fw_reg = pyFw_reg
    cdef double complex[:] Fw = pyFw

    if cstage < 2:
        message = "reg stage=%d < 2" % cstage
        raise ValueError(message)

    # pure C parallel region
    ## -----------------------------------------------------------------
    for i in prange(n_ws, nogil=True, num_threads=sch.nthreads, schedule='dynamic'):
        Fw_reg[i] = compute_DirectFT(ws[i], sch.tau_grid, sch.It_reg_grid, sch.n_grid)
        Fw[i] = Fw_reg[i] + eval_Fw_sing(ws[i], cstage, sch)
    ## -----------------------------------------------------------------

    pyFree_RegScheme(sch)

    if np.isscalar(w) is True:
        pyFw = pyFw.item()
        pyFw_reg = pyFw_reg.item()

    return pyFw, pyFw_reg


## --- Point lens
## ------------------------------------------------------------------------
def pyFw_PointLens(y, w, parallel=True):
    cdef int nthreads = max_num_threads if parallel else 1
    cdef double[:] ws = np.ascontiguousarray(w, dtype=np.double)
    cdef int n_ws = ws.shape[0]

    pyFw = np.empty(n_ws, dtype=np.cdouble)
    cdef double complex[:] Fw = pyFw

    fill_Fw_PointLens(y, n_ws, &ws[0], &Fw[0], nthreads)

    if np.isscalar(w) is True:
        pyFw = pyFw.item()

    return pyFw


## =============================================================================
