/*
 * GLoW - fourier_lib.h
 *
 * Copyright (C) 2023, Hector Villarrubia-Rojo
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FOURIER_LIB_H
#define FOURIER_LIB_H

#include <complex.h>
#include "roots_lib.h"

typedef struct {
    int stage;
    int n_ps;
    double slope;
    double amp[2];
    double index[2];
    int n_grid, nthreads;
    double *tau_grid;
    double *It_reg_grid;
    CritPoint *ps;
} RegScheme;

typedef struct {
    int n_grid, n_total;
    double wmin, wmax;
    double a_Tukey, smallest_tau_max;
    int n_batches, n_keep, n_below_discard, n_above_discard;
    int *imin_batch, *imax_batch, *n_fft, *n_cumbatch;
    double *fmin_real, *fmax_real, *tau_max;
    RegScheme *sch;
    int nthreads;
    double *tau_grid, *It_reg_grid;
} FreqTable;

typedef struct {
    int n_grid;
    double w;
    double *tau_grid;
    double *It_grid;
} pFourierInt;

// =================================================================

// ======  Time domain regularization
// =================================================================
double R0_reg(double tau, double alpha, double beta, double sigma);
double R1_reg(double tau, double alpha, double beta, double sigma);
double RL_reg(double tau, double alpha, double beta);
double S_reg(double tau, double A, double B);
double Sfull_reg(double tau, double A, double B);
double It_sing_common(double tau, int n_points, CritPoint *ps, double *Cmax, double *Cmin);
double It_sing_asymp(double tau, int n_points, CritPoint *ps, double asymp_A, double asymp_index);
double It_sing_no_asymp(double tau, int n_points, CritPoint *ps);

// ======  Frequency domain regularization
// =================================================================
double complex R0_reg_FT(double w, double alpha, double beta, double sigma);
double complex R1_reg_FT(double w, double alpha, double beta, double sigma);
double complex RL_reg_FT(double w, double alpha, double beta);
double complex S_reg_FT(double w, double A, double B);
double complex Sfull_reg_FT(double w, double A, double B);
double complex Fw_sing_common(double w, int n_points, CritPoint *ps, double *Cmax, double *Cmin);
double complex Fw_sing_asymp(double w, int n_points, CritPoint *ps, double asymp_A, double asymp_index);
double complex Fw_sing_no_asymp(double w, int n_points, CritPoint *ps);

// ======  Fitting routines
// =================================================================
void fit_tail(int n_tau, double *tau, double *It, int n_max, double It_min, double *A, double *index);
double fit_slope(int n_tau, double *tau, double *It, int n_max);

// ======  FFTs
// =================================================================
int apply_window_Tukey(double *wd, int n_wd, double alpha);
int apply_right_window_Tukey(double *wd, int n_wd, double alpha);
int compute_reFFT_gsl(double *f_in, int n, double complex *f_out);
int compute_reFFT_pocketfft(double *f_in, int n, double complex *f_out);

double eval_It_sing(double tau, int stage, RegScheme *sch);
double complex eval_Fw_sing(double w, int stage, RegScheme *sch);
void fill_It_reg(int n_tau, double *tau, double *It, double *It_reg, int stage, RegScheme *sch, int nthreads);
void fill_It_sing(int n_tau, double *tau, double *It_sing, int stage, RegScheme *sch, int nthreads);
void fill_Fw_sing(int n_w, double *w, double complex *Fw_sing, int stage, RegScheme *sch, int nthreads);

FreqTable *init_FreqTable(double wmin, double wmax, RegScheme *sch,
                          int n_keep, int n_below_discard, int n_above_discard,
                          double smallest_tau_max, double a_Tukey);
int update_RegScheme(double *It_grid, int stage, RegScheme *sch);
int compute_batch(int i, double *w_grid, double complex *Fw_reg_grid, double complex *Fw_grid, FreqTable *ft);
int compute_Fw(double *w_grid, double complex *Fw_reg_grid, double complex *Fw_grid, FreqTable *ft);
void display_FreqTable(FreqTable *ft);
void free_FreqTable(FreqTable *ft);

// ======  Direct Fourier integral
// =================================================================
double complex compute_DirectFT(double w, double *tau_grid, double *It_grid, int n_grid);

// ======  Point lens
// =================================================================
void fill_Fw_PointLens(double y, int n_w, double *w, double complex *Fw, int nthreads);


// =================================================================
#endif  // FOURIER_LIB_H
