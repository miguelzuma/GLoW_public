#
# GLoW - cfourier.pxd
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

from croots cimport CritPoint

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "special_lib.h" nogil:
    int sorted_interpolation(double *x_subgrid, double *y_subgrid, int n_subgrid,
                             double *x_grid, double *y_grid, int n_grid)

cdef extern from "fourier_lib.h" nogil:
    ctypedef struct RegScheme:
        int stage
        int n_ps
        double slope
        double amp[2]
        double index[2]
        int n_grid, nthreads
        double *tau_grid
        double *It_reg_grid
        CritPoint *ps

    ctypedef struct FreqTable:
        int n_grid, n_total;
        double wmin, wmax;
        double a_Tukey, smallest_tau_max;
        int n_batches, n_keep, n_below_discard, n_above_discard;
        int *imin_batch
        int *imax_batch
        int *n_fft
        int *n_cumbatch
        double *fmin_real
        double *fmax_real
        double *tau_max
        RegScheme *sch
        int nthreads
        double *tau_grid
        double *It_reg_grid

    double R0_reg(double tau, double alpha, double beta, double sigma)
    double R1_reg(double tau, double alpha, double beta, double sigma)
    double Sfull_reg(double tau, double A, double B)
    double It_sing_asymp(double tau, int n_points, CritPoint *ps, double asymp_A, double asymp_index)
    double It_sing_no_asymp(double tau, int n_points, CritPoint *ps)

    double complex R0_reg_FT(double w, double alpha, double beta, double sigma)
    double complex R1_reg_FT(double w, double alpha, double beta, double sigma)
    double complex Sfull_reg_FT(double w, double alpha, double beta)
    double complex Fw_sing_asymp(double w, int n_points, CritPoint *ps, double asymp_A, double asymp_index)
    double complex Fw_sing_no_asymp(double w, int n_points, CritPoint *ps)

    int apply_window_Tukey(double *wd, int n_wd, double alpha)
    int apply_right_window_Tukey(double *wd, int n_wd, double alpha)
    int compute_reFFT_gsl(double *f_in, int n, double complex *f_out)
    int compute_reFFT_pocketfft(double *f_in, int n, double complex *f_out)

    double eval_It_sing(double tau, int stage, RegScheme *sch)
    double complex eval_Fw_sing(double w, int stage, RegScheme *sch)
    void fill_It_reg(int n_tau, double *tau, double *It, double *It_reg, int stage, RegScheme *sch, int nthreads)
    void fill_It_sing(int n_tau, double *tau, double *It_sing, int stage, RegScheme *sch, int nthreads)
    void fill_Fw_sing(int n_w, double *w, double complex *Fw_sing, int stage, RegScheme *sch, int nthreads)

    FreqTable *init_FreqTable(double wmin, double wmax, RegScheme *sch,
                              int n_keep, int n_below_discard, int n_above_discard,
                              double smallest_tau_max, double a_Tukey)
    int update_RegScheme(double *It_grid, int stage, RegScheme *sch)
    int compute_Fw(double *w_grid, double complex *Fw_reg_grid, double complex *Fw_grid, FreqTable *ft)
    void display_FreqTable(FreqTable *ft)
    void free_FreqTable(FreqTable *ft)

    double complex compute_DirectFT(double w, double *tau_grid, double *It_grid, int n_grid)

    void fill_Fw_PointLens(double y, int n_w, double *w, double complex *Fw, int nthreads)
