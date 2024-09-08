/*
 * GLoW - fourier_lib.c
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_fft_real.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "common.h"
#include "roots_lib.h"
#include "special_lib.h"
#include "pocketfft.h"
#include "fourier_lib.h"

#define EPS_SOFT 1e-15


// ======  Time domain regularization
// =================================================================

double R0_reg(double tau, double alpha, double beta, double sigma)
{
    double C;

    if(tau < 0)
        return 0;

    C = pow(beta/alpha, 1./sigma);

    return beta/pow(tau*tau + C*C, 0.5*sigma);
}

double R1_reg(double tau, double alpha, double beta, double sigma)
{
    return tau*R0_reg(tau, alpha, beta, sigma+1);
}

double RL_reg(double tau, double alpha, double beta)
{
    return R1_reg(tau, alpha, beta, 1);
}

double S_reg(double tau, double A, double B)
{
    if(tau < 0)
        return 0;

    return 0.5*A*B*gsl_sf_log_abs((tau+B)/(tau-B+EPS_SOFT));
}

double Sfull_reg(double tau, double A, double B)
{
    return S_reg(tau, A, B) - RL_reg(tau, A, A*B*B);
}

double It_sing_common(double tau, int n_points, CritPoint *ps, double *Cmax, double *Cmin)
{
    int i;
    double tmin, It, a, tau_i;

    tmin = ps[0].t;
    It = 1;
    *Cmax = 0;
    *Cmin = 0;

    for(i=1;i<n_points;i++)
    {
        tau_i = ps[i].t - tmin;

        if(ps[i].type == type_saddle)
        {
            a = 2*sqrt(ps[i].mag)/M_PI/tau_i;
            It += Sfull_reg(tau, a, tau_i);
        }
        else if(ps[i].type == type_max)
        {
            *Cmax += sqrt(ps[i].mag);
            if(tau < tau_i)
                It += sqrt(ps[i].mag);
        }
        else if(ps[i].type == type_min)
        {
            *Cmin += sqrt(ps[i].mag);
            if(tau < tau_i)
                It -= sqrt(ps[i].mag);
        }
    }

    return It;
}

double It_sing_asymp(double tau, int n_points, CritPoint *ps, double asymp_A, double asymp_index)
{
    double It, Cmax, Cmin;

    if(tau < 0)
        return 0;

    It = It_sing_common(tau, n_points, ps, &Cmax, &Cmin);
    It += R0_reg(tau, sqrt(ps[0].mag)-1-Cmax+Cmin, asymp_A, asymp_index);

    return M_2PI*It;
}

double It_sing_no_asymp(double tau, int n_points, CritPoint *ps)
{
    double It, Cmax, Cmin;

    if(tau < 0)
        return 0;

    It = It_sing_common(tau, n_points, ps, &Cmax, &Cmin);
    It += sqrt(ps[0].mag) - 1 - Cmax - Cmin;

    return M_2PI*It;
}

// =================================================================


// ======  Frequency domain regularization
// =================================================================

double complex R0_reg_FT(double w, double alpha, double beta, double sigma)
{
    double tol;
    double index, C, u;
    double prefactor, RK, RM, K;
    double complex R0;

    tol = 1e-10;
    index = 0.5*(1-sigma);
    C = pow(beta/alpha, 1./sigma);
    u = w*C;

    prefactor = M_SQRTPI*alpha*gsl_sf_gamma(1-0.5*sigma);

    // prevent underflow
    if(u < 80)
        K = gsl_sf_bessel_Knu(fabs(index), u);
    else
        K = 0;

    RK = 2./M_PI*pow(0.5*u, 0.5*(1+sigma))*K;
    RM = -0.5*u*Mtilde_Struve(u, index, tol);

    R0 = prefactor*(cos(M_PI_2*sigma)*RK + RM) - I*prefactor*sin(M_PI_2*sigma)*RK;

    return R0;
}

double complex R1_reg_FT(double w, double alpha, double beta, double sigma)
{
    double tol;
    double index, C, u;
    double RK, RM, K;
    double complex R1;

    tol = 1e-10;
    index = 1-0.5*sigma;
    C = pow(beta/alpha, 1./(sigma+1));
    u = w*C;

    // prevent underflow
    if(u < 80)
        K = gsl_sf_bessel_Knu(fabs(index), u);
    else
        K = 0;

    RK = 2*alpha*C/M_SQRTPI*pow(0.5*u, 1+0.5*sigma)\
            *gsl_sf_gamma(0.5*(1-sigma))*K;
    RM = alpha*C*u/(1-sigma)*(1 + 0.5*u*M_SQRTPI*gsl_sf_gamma(1.5-0.5*sigma)\
                                    *Mtilde_Struve(u, index, tol));

    R1 = cexp(-I*M_PI_2*sigma)*RK + I*RM;

    return R1;
}

double complex RL_reg_FT(double w, double alpha, double beta)
{
    double u      = w*sqrt(beta/alpha);
    double R_real = M_PI_2*beta*w*exp(-u);
    double R_imag = 0.5*beta*w*(gsl_sf_expint_Ei_scaled(u) - gsl_sf_expint_E1_scaled(u));

    return R_real + I*R_imag;
}

double complex S_reg_FT(double w, double A, double B)
{
    double u, f, cos_u, sin_u;
    double Si, ci, si, S_real, S_imag;

    u = w*B;

    Si = gsl_sf_Si(u);
    si = Si - M_PI_2;
    ci = gsl_sf_Ci(u);

    cos_u = cos(u);
    sin_u = sin(u);

    f = ci*sin_u - si*cos_u;

    S_real = M_PI_2*A*B*sin_u;
    S_imag = -A*B*(M_PI_2*cos_u - f);

    return S_real + I*S_imag;
}

double complex Sfull_reg_FT(double w, double A, double B)
{
    double complex Sfull, Rl;

    Sfull = S_reg_FT(w, A, B);
    Rl = RL_reg_FT(w, A, A*B*B);

    Sfull -= Rl;

    return Sfull;
}

double complex Fw_sing_common(double w, int n_points, CritPoint *ps, double *Cmax, double *Cmin)
{
    int i;
    double tmin, a, tau_i, sqr;
    double complex Fw, Sfull;

    tmin = ps[0].t;
    *Cmax = 0;
    *Cmin = 0;
    Fw = 0;

    for(i=1;i<n_points;i++)
    {
        tau_i = ps[i].t - tmin;
        sqr = sqrt(ps[i].mag);

        if(ps[i].type == type_saddle)
        {
            a = 2*sqr/M_PI/tau_i;
            Sfull = Sfull_reg_FT(w, a, tau_i);
            Fw += Sfull;
        }
        else if(ps[i].type == type_max)
        {
            *Cmax += sqr;
            Fw -= sqr*cexp(I*w*tau_i);
        }
        else if(ps[i].type == type_min)
        {
            *Cmin += sqr;
            Fw += sqr*cexp(I*w*tau_i);
        }
    }

    Fw += 1 - *Cmin + *Cmax;

    return Fw;
}

double complex Fw_sing_asymp(double w, int n_points, CritPoint *ps, double asymp_A, double asymp_index)
{
    double Cmax, Cmin;
    double complex Fw;

    Fw = Fw_sing_common(w, n_points, ps, &Cmax, &Cmin);
    Fw += R0_reg_FT(w, sqrt(ps[0].mag)-1-Cmax+Cmin, asymp_A, asymp_index);

    return Fw;
}

double complex Fw_sing_no_asymp(double w, int n_points, CritPoint *ps)
{
    double Cmax, Cmin;
    double complex Fw;

    Fw = Fw_sing_common(w, n_points, ps, &Cmax, &Cmin);

    Fw += sqrt(ps[0].mag) - 1 - Cmax + Cmin;

    return Fw;
}

// =================================================================


// ======  Fitting routines
// =================================================================

void fit_tail(int n_tau, double *tau, double *It, int n_max, double It_min, double *A, double *index)
{
    int i, j;
    int *ids;
    gsl_multifit_robust_workspace *wk;
    const gsl_multifit_robust_type *T;
    gsl_matrix *X;
    gsl_vector *y;
    gsl_vector *c;
    gsl_matrix *cov;

    T = gsl_multifit_robust_ols;
    //~ T = gsl_multifit_robust_bisquare;

    ids = (int *)malloc(n_max*sizeof(int));
    X   = gsl_matrix_alloc(n_max, 2);
    y   = gsl_vector_alloc(n_max);
    c   = gsl_vector_alloc(2);
    cov = gsl_matrix_alloc(2, 2);

    j = 0;
    for(i=n_tau-1;i>0;i--)
    {
        if(It[i] > It_min)
        {
            ids[n_max-1-j] = i;
            if(j == n_max-1)
                break;
            else
                j++;
        }
    }

    for(i=0;i<n_max;i++)
    {
        gsl_vector_set(y, i, log(It[ids[i]]));
        gsl_matrix_set(X, i, 0, 1.);
        gsl_matrix_set(X, i, 1, log(tau[ids[i]]));
    }

    wk = gsl_multifit_robust_alloc(T, X->size1, X->size2);
    gsl_multifit_robust(X, y, c, cov, wk);
    gsl_multifit_robust_free(wk);

    // HVR_DEBUG
    //~ printf("Results linear fit: c[0]=%g   c[1]=%g\n", gsl_vector_get(c, 0), gsl_vector_get(c, 1));

    *A = exp(gsl_vector_get(c, 0));
    *index = -gsl_vector_get(c, 1);

    gsl_matrix_free(cov);
    gsl_vector_free(c);
    gsl_vector_free(y);
    gsl_matrix_free(X);
    free(ids);
}

double fit_slope(int n_tau, double *tau, double *It, int n_max)
{
    int i;
    gsl_multifit_robust_workspace *wk;
    const gsl_multifit_robust_type *T;
    gsl_matrix *X;
    gsl_vector *y;
    gsl_vector *c;
    gsl_matrix *cov;

    T = gsl_multifit_robust_ols;
    //~ T = gsl_multifit_robust_bisquare;

    X   = gsl_matrix_alloc(n_max, 2);
    y   = gsl_vector_alloc(n_max);
    c   = gsl_vector_alloc(2);
    cov = gsl_matrix_alloc(2, 2);

    for(i=0;i<n_max;i++)
    {
        gsl_vector_set(y, i, It[i]);
        gsl_matrix_set(X, i, 0, 1.);
        gsl_matrix_set(X, i, 1, tau[i]);
    }

    wk = gsl_multifit_robust_alloc(T, X->size1, X->size2);
    gsl_multifit_robust(X, y, c, cov, wk);
    gsl_multifit_robust_free(wk);

    // HVR_DEBUG
    //~ printf("Results linear fit: c[0]=%g   c[1]=%g\n", gsl_vector_get(c, 0), gsl_vector_get(c, 1));

    gsl_matrix_free(cov);
    gsl_vector_free(c);
    gsl_vector_free(y);
    gsl_matrix_free(X);

    return gsl_vector_get(c, 1);
}

// =================================================================


// ======  FFTs
// =================================================================

int apply_window_Tukey(double *wd, int n_wd, double alpha)
{
    int i, width;

    // HVR -> add check for alpha

    width = (int)(alpha*(n_wd-1)/2.);

    for(i=0;i<=width;i++)
        wd[i] *= 0.5*(1 + cos(M_PI*(-1 + 2.*i/alpha/(n_wd-1))));

    for(i=n_wd-width-1;i<n_wd;i++)
        wd[i] *= 0.5*(1 + cos(M_PI*(-2.0/alpha + 1 + 2.*i/alpha/(n_wd-1))));

    return 0;
}

int apply_right_window_Tukey(double *wd, int n_wd, double alpha)
{
    int i, width;

    // HVR -> add check for alpha

    width = (int)(alpha*(n_wd-1)/2.);

    for(i=n_wd-width-1;i<n_wd;i++)
        wd[i] *= 0.5*(1 + cos(M_PI*(-2.0/alpha + 1 + 2.*i/alpha/(n_wd-1))));

    return 0;
}

int compute_reFFT_gsl(double *f_in, int n, double complex *f_out)
{
    int i;

    gsl_fft_real_radix2_transform(f_in, 1, n);

    f_out[0]   = f_in[0];
    f_out[n/2] = f_in[n-1];

    for(i=1;i<n/2;i++)
        f_out[i] = f_in[i] + I*f_in[n-i];

    return 0;
}

int compute_reFFT_pocketfft(double *f_in, int n, double complex *f_out)
{
    int i, n_freq = (int)(n/2 + 1);
    rfft_plan plan = make_rfft_plan(n);

    rfft_forward(plan, f_in, 1.);
    destroy_rfft_plan(plan);

    f_out[0] = f_in[0];
    f_out[n_freq-1] = f_in[n-1];
    for(i=1;i<n_freq-1;i++)
        f_out[i] = f_in[2*i-1] + I*f_in[2*i];

    return 0;
}


double eval_It_sing(double tau, int stage, RegScheme *sch)
{
    double It_sing;

    if(stage > 3)
    {
        PWARNING("reg_stage=%d > 3 is not valid. Using reg_stage=%d", stage, sch->stage);
        stage = sch->stage;
    }
    else if(stage > sch->stage)
    {
        PWARNING("reg_stage=%d > %d (max stage computed). Using reg_stage=%d", stage, sch->stage, sch->stage);
        stage = sch->stage;
    }

    if(stage == 0)
        It_sing = 0;
    else if(stage == 1)
        It_sing = It_sing_no_asymp(tau, sch->n_ps, sch->ps);
    else if(stage == 2)
        It_sing = It_sing_asymp(tau, sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
    else if(stage == 3)
    {
        It_sing = It_sing_asymp(tau, sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
        It_sing += R1_reg(tau, sch->slope, sch->amp[1], sch->index[1]);
    }

    return It_sing;
}

double complex eval_Fw_sing(double w, int stage, RegScheme *sch)
{
    double complex Fw_sing;

    if(stage > 3)
    {
        PWARNING("reg_stage=%d > 3 is not valid. Using reg_stage=%d", stage, sch->stage);
        stage = sch->stage;
    }
    else if(stage > sch->stage)
    {
        PWARNING("reg_stage=%d > %d (max stage computed). Using reg_stage=%d", stage, sch->stage, sch->stage);
        stage = sch->stage;
    }

    if(stage == 0)
        Fw_sing = 0;
    else if(stage == 1)
        Fw_sing = Fw_sing_no_asymp(w, sch->n_ps, sch->ps);
    else if(stage == 2)
        Fw_sing = Fw_sing_asymp(w, sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
    else if(stage == 3)
    {
        Fw_sing = Fw_sing_asymp(w, sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
        Fw_sing += R1_reg_FT(w, sch->slope, sch->amp[1], sch->index[1])/M_2PI;
    }

    return Fw_sing;
}

void fill_It_reg(int n_tau, double *tau, double *It, double *It_reg, int stage, RegScheme *sch, int nthreads)
{
    int i;
    double It_sing;

    if(stage > 3)
    {
        PWARNING("reg_stage=%d > 3 is not valid. Using reg_stage=%d", stage, sch->stage);
        stage = sch->stage;
    }
    else if(stage > sch->stage)
    {
        PWARNING("reg_stage=%d > %d (max stage computed). Using reg_stage=%d", stage, sch->stage, sch->stage);
        stage = sch->stage;
    }

    if(stage == 0)
        for(i=0;i<n_tau;i++)
            It_reg[i] = It[i];
    else if(stage == 1)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_tau;i++)
        {
            It_sing = It_sing_no_asymp(tau[i], sch->n_ps, sch->ps);
            It_reg[i] = It[i] - It_sing;
        }
    else if(stage == 2)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_tau;i++)
        {
            It_sing = It_sing_asymp(tau[i], sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
            It_reg[i] = It[i] - It_sing;
        }
    else if(stage == 3)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_tau;i++)
        {
            It_sing = It_sing_asymp(tau[i], sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
            It_reg[i] = It[i] - It_sing - R1_reg(tau[i], sch->slope, sch->amp[1], sch->index[1]);
        }
}

void fill_It_sing(int n_tau, double *tau, double *It_sing, int stage, RegScheme *sch, int nthreads)
{
    int i;

    if(stage > 3)
    {
        PWARNING("reg_stage=%d > 3 is not valid. Using reg_stage=%d", stage, sch->stage);
        stage = sch->stage;
    }
    else if(stage > sch->stage)
    {
        PWARNING("reg_stage=%d > %d (max stage computed). Using reg_stage=%d", stage, sch->stage, sch->stage);
        stage = sch->stage;
    }

    if(stage == 0)
        for(i=0;i<n_tau;i++)
            It_sing[i] = 0.;
    else if(stage == 1)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_tau;i++)
            It_sing[i] = It_sing_no_asymp(tau[i], sch->n_ps, sch->ps);
    else if(stage == 2)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_tau;i++)
            It_sing[i] = It_sing_asymp(tau[i], sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
    else if(stage == 3)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_tau;i++)
        {
            It_sing[i] = It_sing_asymp(tau[i], sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
            It_sing[i] += R1_reg(tau[i], sch->slope, sch->amp[1], sch->index[1]);
        }
}

void fill_Fw_sing(int n_w, double *w, double complex *Fw_sing, int stage, RegScheme *sch, int nthreads)
{
    int i;

    if(stage > sch->stage)
        stage = sch->stage;

    if(stage == 0)
        for(i=0;i<n_w;i++)
            Fw_sing[i] = 0.;
    else if(stage == 1)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_w;i++)
            Fw_sing[i] = Fw_sing_no_asymp(w[i], sch->n_ps, sch->ps);
    else if(stage == 2)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_w;i++)
            Fw_sing[i] = Fw_sing_asymp(w[i], sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
    else if(stage == 3)
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
        #endif
        for(i=0;i<n_w;i++)
        {
            Fw_sing[i] = Fw_sing_asymp(w[i], sch->n_ps, sch->ps, sch->amp[0], sch->index[0]);
            Fw_sing[i] += R1_reg_FT(w[i], sch->slope, sch->amp[1], sch->index[1])/M_2PI;
        }
}


FreqTable *init_FreqTable(double wmin, double wmax, RegScheme *sch,
                          int n_keep, int n_below_discard, int n_above_discard,
                          double smallest_tau_max, double a_Tukey)
{
    int i, n, n_fft;
    double Dlog2_w, dlog2_w, log2_wmin, log2_wmin_batch, log2_wmax_batch;
    double fmin_batch, fmax_batch, fmin, fmax, Dtau, tau_max, true_fmax, df;
    FreqTable *ft = (FreqTable *)malloc(sizeof(FreqTable));

    // direct input parameters
    ft->wmin = wmin;
    ft->wmax = wmax;
    ft->a_Tukey = a_Tukey;
    ft->smallest_tau_max = smallest_tau_max;
    ft->n_keep = n_keep;
    ft->n_below_discard = n_below_discard;
    ft->n_above_discard = n_above_discard;

    ft->sch = sch;
    ft->nthreads    = sch->nthreads;
    ft->n_grid      = sch->n_grid;
    ft->tau_grid    = sch->tau_grid;
    ft->It_reg_grid = sch->It_reg_grid;

    // find batches
    log2_wmin = log2(wmin);
    Dlog2_w = log2(wmax) - log2_wmin;
    n = (int)(Dlog2_w/n_keep + 1);
    dlog2_w = Dlog2_w/n;

    ft->n_batches  = n;
    ft->imin_batch = (int *)malloc(n*sizeof(int));
    ft->imax_batch = (int *)malloc(n*sizeof(int));
    ft->n_fft      = (int *)malloc(n*sizeof(int));
    ft->n_cumbatch = (int *)malloc(n*sizeof(int));
    ft->fmin_real  = (double *)malloc(n*sizeof(double));
    ft->fmax_real  = (double *)malloc(n*sizeof(double));
    ft->tau_max    = (double *)malloc(n*sizeof(double));

    ft->n_total = 0;
    ft->n_cumbatch[0] = 0;
    for(i=0;i<ft->n_batches;i++)
    {
        log2_wmin_batch = i*dlog2_w + log2_wmin;
        log2_wmax_batch = log2_wmin_batch + dlog2_w;

        fmin_batch = pow(2, log2_wmin_batch)/M_2PI;
        fmax_batch = pow(2, log2_wmax_batch)/M_2PI;

        fmin = fmin_batch/(1<<n_below_discard);
        fmax = fmax_batch*(1<<n_above_discard);

        // ---------------------------------------------

        Dtau = MAX(1, smallest_tau_max*fmin)/fmin;
        tau_max = Dtau;

        n_fft = 1<<(int)(log2(fmax/fmin)+1);
        df = (n_fft-1.)/tau_max/n_fft;

        true_fmax = n_fft/2*df;
        if(fmax_batch > true_fmax)
        {
            n_fft = 1<<(int)(log2(1 + 2*tau_max*fmax_batch) + 1);
            df = (n_fft-1.)/tau_max/n_fft;
        }

        // ---------------------------------------------

        ft->n_fft[i] = n_fft;
        ft->tau_max[i]    = tau_max;
        ft->imin_batch[i] = (int)(fmin_batch/df+1);
        ft->imax_batch[i] = (int)(fmax_batch/df+1);
        ft->fmin_real[i]  = df;
        ft->fmax_real[i]  = n_fft/2*df;

        ft->n_cumbatch[i] = ft->n_total;
        ft->n_total += ft->imax_batch[i] - ft->imin_batch[i];
    }

    return ft;
}

int update_RegScheme(double *It_grid, int stage, RegScheme *sch)
{
    int i, nmax_slope, nmax_tail;
    double It_min;

    nmax_slope = pprec.fo_updRegSch_nmax_slope;
    nmax_tail = pprec.fo_updRegSch_nmax_tail;
    It_min = pprec.fo_updRegSch_Itmin_tail;

    // Input checks
    // ******************************
    if(stage < 0)
    {
        PERROR("reg stage (=%d) must be positive", stage);
        fill_It_reg(sch->n_grid, sch->tau_grid, It_grid, sch->It_reg_grid, 0, sch, sch->nthreads);
        return 1;
    }

    if(sch->stage < 1)
    {
        PERROR("reg scheme must be preinitialized with at least stage 1");
        fill_It_reg(sch->n_grid, sch->tau_grid, It_grid, sch->It_reg_grid, 0, sch, sch->nthreads);
        return 1;
    }

    if(stage > 3)
    {
        PWARNING("reg_stage=%d > 3 is not valid. Using reg_stage=%d", stage, sch->stage);
        stage = sch->stage;
    }
    // ******************************

    // Actual computation
    if(stage <= sch->stage)
    {
        fill_It_reg(sch->n_grid, sch->tau_grid, It_grid, sch->It_reg_grid, stage, sch, sch->nthreads);
        return 0;
    }

    if(stage > sch->stage)
    {
        if(sch->stage == 1)
        {
            // fit for stage 2
            for(i=0;i<sch->n_grid;i++)
                sch->It_reg_grid[i] = It_grid[i]/M_2PI - 1;

            fit_tail(sch->n_grid, sch->tau_grid, sch->It_reg_grid, nmax_tail, It_min, sch->amp, sch->index);
            sch->stage = 2;
        }

        fill_It_reg(sch->n_grid, sch->tau_grid, It_grid, sch->It_reg_grid, sch->stage, sch, sch->nthreads);

        // we have ensured that It_reg has been computed up to stage 2
        // one more fit if we need stage 3
        if(stage == 3)
        {
            fit_tail(sch->n_grid, sch->tau_grid, sch->It_reg_grid, nmax_tail, It_min, sch->amp+1, sch->index+1);
            sch->slope = fit_slope(sch->n_grid, sch->tau_grid, sch->It_reg_grid, nmax_slope);
            sch->stage = 3;

            #ifdef _OPENMP
                #pragma omp parallel for num_threads(sch->nthreads) if(sch->nthreads > 1)
            #endif
            for(i=0;i<sch->n_grid;i++)
                sch->It_reg_grid[i] -= R1_reg(sch->tau_grid[i], sch->slope, sch->amp[1], sch->index[1]);
        }
    }

    return 0;
}

int compute_batch(int i, double *w_grid, double complex *Fw_reg_grid, double complex *Fw_grid, FreqTable *ft)
{
    // all the output arrays are assumed to have allocated ft->n_total+2
    int j, k, n_fft;
    int imin, imax;
    double df, dtau, tau_max;
    double *w, *It;
    double complex *Fw_reg, *Fw, *Fw_tmp;
    double w_tmp[2];
    double complex tmp[2];

    w  = w_grid + 1 + ft->n_cumbatch[i];
    Fw = Fw_grid + 1 + ft->n_cumbatch[i];
    Fw_reg = Fw_reg_grid + 1 + ft->n_cumbatch[i];

    imin = ft->imin_batch[i];
    imax = ft->imax_batch[i];
    n_fft = ft->n_fft[i];
    tau_max = ft->tau_max[i];

    It = (double *)malloc(n_fft*sizeof(double));
    Fw_tmp = (double complex *)malloc((n_fft/2+1)*sizeof(double complex));

    dtau = tau_max/(n_fft-1.);
    df = 1./dtau/n_fft;

    // use It as temporal storage for the tau grid
    for(j=0;j<n_fft;j++)
        It[j] = j*dtau;

    sorted_interpolation(It, It, n_fft, ft->tau_grid, ft->It_reg_grid, ft->n_grid);
    apply_right_window_Tukey(It, n_fft, ft->a_Tukey);

    compute_reFFT_pocketfft(It, n_fft, Fw_tmp);

    k = 0;
    for(j=imin;j<imax;j++)
    {
        w[k] = j*df*M_2PI;
        Fw_reg[k] = -I*j*df*conj(Fw_tmp[j])*dtau;
        Fw[k] = Fw_reg[k] + eval_Fw_sing(w[k], ft->sch->stage, ft->sch);

        k++;
    }

    // make sure that the first and last points correspond to wmin and wmax
    // interpolate linearly using the discarded data (otherwise extrapolate)
    if( (i==0) || (i==ft->n_batches-1) )
    {
        if(i == 0)
        {
            imin = (imin > 0) ? (imin-1) : 0;
            imax = imin + 1;
            j = 0;
            w_grid[j] = ft->wmin;
        }
        if(i == ft->n_batches-1)
        {
            imax = (imax < (n_fft/2+1)) ? imax : imax-1;
            imin = imax - 1;
            j = ft->n_total+1;
            w_grid[j] = ft->wmax;
        }

        w_tmp[0] = imin*df*M_2PI;
        w_tmp[1] = imax*df*M_2PI;

        tmp[0] = -I*imin*df*conj(Fw_tmp[imin])*dtau;
        tmp[1] = -I*imax*df*conj(Fw_tmp[imax])*dtau;

        Fw_reg_grid[j] = clinear_interpolation(w_grid[j], w_tmp, tmp, 0, 1);
        Fw_grid[j] = Fw_reg_grid[j] + eval_Fw_sing(w_grid[j], ft->sch->stage, ft->sch);
    }

    free(Fw_tmp);
    free(It);

    return 0;
}

int compute_Fw(double *w_grid, double complex *Fw_reg_grid, double complex *Fw_grid, FreqTable *ft)
{
    // all the output arrays are assumed to have allocated ft->n_total+2
    int i;

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(ft->nthreads) if(ft->nthreads > 1)
    #endif
    for(i=0;i<ft->n_batches;i++)
        compute_batch(i, w_grid, Fw_reg_grid, Fw_grid, ft);

    return 0;
}

void display_FreqTable(FreqTable *ft)
{
    int i;

    printf(" - Frequency table (n_total=%d):\n", ft->n_total);
    for(i=0;i<ft->n_batches;i++)
    {
        printf("(%.2d) n_fft=%d  imin=%d   imax=%d\n", i+1, ft->n_fft[i], ft->imin_batch[i], ft->imax_batch[i]);
        printf("     tau_max=%g   fmin=%g   fmax=%g\n", ft->tau_max[i], ft->fmin_real[i], ft->fmax_real[i]);
    }
}

void free_FreqTable(FreqTable *ft)
{
    free(ft->tau_max);
    free(ft->fmax_real);
    free(ft->fmin_real);
    free(ft->n_cumbatch);
    free(ft->n_fft);
    free(ft->imax_batch);
    free(ft->imin_batch);
    free(ft);
}

// =================================================================


// ======  Direct Fourier integral
// =================================================================

double complex compute_DirectFT(double w, double *tau_grid, double *It_grid, int n_grid)
{
    int i;
    double DI_i;
    double complex iwDtau_i, J_i, D_i, DFw, Fw;

    DI_i = It_grid[0];
    iwDtau_i = I*w*tau_grid[0];
    D_i = cexp(iwDtau_i);
    J_i = 1./M_2PI;

    Fw = J_i*((It_grid[0] - DI_i/iwDtau_i)*(1 - D_i) - DI_i);

    for(i=0;i<n_grid-1;i++)
    {
        J_i *= D_i;

        DI_i = It_grid[i+1] - It_grid[i];
        iwDtau_i = I*w*(tau_grid[i+1] - tau_grid[i]);
        D_i = cexp(iwDtau_i);

        DFw = J_i*((It_grid[i+1] - DI_i/iwDtau_i)*(1 - D_i) - DI_i);
        Fw += DFw;

        // HVR_DEBUG
        //~ printf("(%4.d/%d)   tau=%g   Fw=%g%+gi  |DFW|=%g   DFw=%g%+gi\n", i, n_grid, tau_grid[i], creal(Fw), cimag(Fw), cabs(DFw), creal(DFw), cimag(DFw));
    }

    return Fw;
}

// =================================================================


// ======  Point lens
// =================================================================

void fill_Fw_PointLens(double y, int n_w, double *w, double complex *Fw, int nthreads)
{
    int i;
    double x_min, t_min;
    double *u;
    double complex exponent;
    gsl_sf_result loggamma_mod, loggamma_arg;

    x_min = 0.5*(y + sqrt(y*y + 4));
    t_min = 0.5*(x_min-y)*(x_min-y) - log(x_min);

    u = (double *)malloc(n_w*sizeof(double));

    for(i=0;i<n_w;i++)
        u[i] = 0.5*w[i];

    F11_sorted(u, y*y, n_w, Fw, nthreads);

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) if(nthreads > 1) \
                                 private(loggamma_mod, loggamma_arg, exponent)
    #endif
    for(i=0;i<n_w;i++)
    {
        gsl_sf_lngamma_complex_e(1, -u[i], &loggamma_mod, &loggamma_arg);

        exponent = loggamma_mod.val + I*loggamma_arg.val;
        exponent += M_PI_2*u[i] + I*u[i]*(log(u[i]) - 2*t_min);

        Fw[i] *= cexp(exponent);
    }

    free(u);
}

// =================================================================
