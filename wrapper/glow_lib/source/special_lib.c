/*
 * GLoW - special_lib.c
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
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_result.h>
#include <gsl/gsl_errno.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "common.h"
#include "special_lib.h"

#define EPS_SOFT 1e-8

// =================================================================

// ======  Fresnel integrals
// =================================================================

// Coefficients for the approximation of the Fresnel integrals
static double an_fresnel[12] = {
     1.595769140,
    -0.000001702,
    -6.808568854,
    -0.000576361,
     6.920691902,
    -0.016898657,
    -3.050485660,
    -0.075752419,
     0.850663781,
    -0.025639041,
    -0.150230960,
     0.034404779
};

static double bn_fresnel[12] = {
    -0.000000033,
     4.255387524,
    -0.000092810,
    -7.780020400,
    -0.009520895,
     5.075161298,
    -0.138341947,
    -1.363729124,
    -0.403349276,
     0.702222016,
    -0.216195929,
     0.019547031
};

static double cn_fresnel[12] = {
     0.,
    -0.024933975,
     0.000003936,
     0.005770956,
     0.000689892,
    -0.009497136,
     0.011948809,
    -0.006748873,
     0.000246420,
     0.002102967,
    -0.001217930,
     0.000233939
};

static double dn_fresnel[12] = {
     0.199471140,
     0.000000023,
    -0.009351341,
     0.000023006,
     0.004851466,
     0.001903218,
    -0.017122914,
     0.029064067,
    -0.027928955,
     0.016497308,
    -0.005598515,
     0.000838386
};

// ------------------------------------------------------------------

double f_fresnel(double x)
{
    // Approximation for the Fresnel integrals from [Boersma, 1960]

    int i;
    double xn=1, sum=0, var=M_PI_2*x*x;

    if(x < 0)
        return M_SQRT2*cos(var + M_PI_4) - f_fresnel(-x);

    if(x >= 2*M_SQRT2/M_SQRTPI)
    {
        for(i=0;i<12;i++)
        {
            sum += dn_fresnel[i]*xn;
            xn = 4*xn/var;
        }
        return sqrt(4/var)*sum;
    }

    for(i=0;i<12;i++)
    {
        sum += bn_fresnel[i]*xn;
        xn *= 0.25*var;
    }

    return M_SQRT1_2*cos(var + M_PI_4) + sqrt(0.25*var)*sum;
}

double g_fresnel(double x)
{
    // Approximation for the Fresnel integrals from [Boersma, 1960]

    int i;
    double xn=1, sum=0, var=M_PI_2*x*x;

    if(x < 0)
        return M_SQRT2*sin(var + M_PI_4) - g_fresnel(-x);

    if(x >= 2*M_SQRT2/M_SQRTPI)
    {
        for(i=0;i<12;i++)
        {
            sum += cn_fresnel[i]*xn;
            xn = 4*xn/var;
        }
        return -sqrt(4/var)*sum;
    }

    for(i=0;i<12;i++)
    {
        sum += an_fresnel[i]*xn;
        xn *= 0.25*var;
    }

    return M_SQRT1_2*sin(var + M_PI_4) - sqrt(0.25*var)*sum;
}

// =================================================================


// ======  Struve functions
// ===================================================================

double Mtilde_Struve_PowerSeries(double x, double nu, double tol)
{
    int n, max_iter=1000;
    double z, z2;
    double Mtilde_nu, delta_M;
    double R_n, even_n;

    z = 0.5*x;
    z2 = z*z;

    even_n = 1./gsl_sf_gamma(1.+nu);
    R_n = gsl_sf_gamma(nu+1)/gsl_sf_gamma(1.5)/gsl_sf_gamma(nu+1.5);

    Mtilde_nu = 0;

    n=0;
    do
    {
        delta_M = even_n*(R_n*z-1);
        Mtilde_nu += delta_M;

        even_n *= z2/(n+1)/(n+1+nu);
        R_n *= (n+1)*(n+nu+1)/(n+1.5)/(n+nu+1.5);
        n++;
    }
    while((n < max_iter) && (ABS(delta_M/Mtilde_nu) > tol));

    return Mtilde_nu;
}

double Mtilde_Struve_Asymptotic(double x, double nu, double tol)
{
    int k=0, max_iter;
    double delta_M, Mtilde_nu;
    double z, z2, Ak;

    z = 0.5*x;
    z2 = z*z;

    max_iter = (int)z;

    Mtilde_nu = 0;

    Ak = -gsl_sf_gamma(0.5-nu)*gsl_sf_gamma(0.5)*cos(M_PI*nu)/M_PI;

    k = 0;
    do
    {
        delta_M = Ak;

        Mtilde_nu += delta_M;

        Ak *= (k+0.5)*(k-nu+0.5)/z2;
        k++;
    }
    while((k < max_iter) && (ABS(delta_M/Mtilde_nu) > tol));

    return Mtilde_nu/z/M_PI;
}

double Mtilde_Struve_PieceWise(double x, double nu, double tol)
{
    // enough to reach a precision of 1e-7 (relative) for the worst case
    // scenario nu=0 and up to nu=20
    double x_max = 16.5 + 0.6*ABS(nu);

    if(x < 0)
    {
        PERROR("negative values not supported for the implementation of the Struve function (x = %g)", x)
        return 0;
    }

    if(x < x_max)
        return Mtilde_Struve_PowerSeries(x, nu, tol);
    else
        return Mtilde_Struve_Asymptotic(x, nu, tol);
}

double Mtilde_Struve(double x, double nu, double tol)
{
    //~ double M, M1, M2;

    if(nu > 20)
    {
        PERROR("nu too large to ensure proper computation of Struve function (nu = %g)", nu)
        return 0;
    }

    // HVR: implementation of the recurrence relation, seems to improve the
    //      precision a little bit (it reaches a 'plateau') but doesn't seem
    //      worth it
    //~ if(nu < 0.)
    //~ {
        //~ M1 = Mtilde_Struve(x, nu+1, tol);
        //~ M2 = Mtilde_Struve(x, nu+2, tol);

        //~ M = 0.25*x*x*M2 + (nu+1)*M1 + 0.25*x*M_2_SQRTPI/gsl_sf_gamma(nu + 2.5);
    //~ }
    //~ else
        //~ M = Mtilde_Struve_PieceWise(x, nu, tol);

    return Mtilde_Struve_PieceWise(x, nu, tol);
}

// ===================================================================


// ======  Interpolation routines
// ===================================================================

int searchsorted_left(double x, double *x_grid, int n_grid)
{
    int i_min, i_mid, i_max;

    i_min = 0;
    i_max = n_grid-1;

    // return index of the item in the list to the left of x
    if(x < x_grid[i_min])
        return -1;
    else if(x > x_grid[i_max])
        return i_max;
    else
    {
        while(i_max >= i_min)
        {
            i_mid = i_min + (i_max-i_min)/2;

            if(x == x_grid[i_mid])
                return i_mid;
            else if(x < x_grid[i_mid])
                i_max = i_mid-1;
            else
                i_min = i_mid+1;
        }

        return i_max;
    }
}

int searchsorted_right(double x, double *x_grid, int n_grid)
{
    int i_min, i_mid, i_max;

    i_min = 0;
    i_max = n_grid-1;

    // return index of the item in the list to the right of x
    if(x < x_grid[i_min])
        return i_min;
    else if(x > x_grid[i_max])
        return -1;
    else
    {
        while(i_max >= i_min)
        {
            i_mid = i_min + (i_max-i_min)/2;

            if(x == x_grid[i_mid])
                return i_mid;
            else if(x < x_grid[i_mid])
                i_max = i_mid-1;
            else
                i_min = i_mid+1;
        }

        return i_min;
    }
}

int searchsorted(double x, double *x_grid, int n_grid, int side)
{
    // side = 1/-1  -> return index of the item in the list to the right/left of x
    if(side == -1)
        return searchsorted_left(x, x_grid, n_grid);
    else if(side == 1)
        return searchsorted_right(x, x_grid, n_grid);
    else
        return -1;
}

int find_subarray(double *grid, int *jmin, int *jmax, double *subgrid, int *imin, int *imax)
{
    int index;
    double *g  = grid + *jmin;
    double *sg = subgrid + *imin;

    // left side
    index = searchsorted_left(subgrid[*imin], g, *jmax+1);
    if(index != -1)
        *jmin = index;
    else
    {
        index = searchsorted_right(grid[*jmin], sg, *imax+1);
        if(index != -1)
            *imin = index;
        else
        {
            *imin = *imax+1;
            return -1;   // there is no overlap
        }
    }

    // right side
    index = searchsorted_right(subgrid[*imax], g, *jmax+1);
    if(index != -1)
        *jmax = index;
    else
    {
        index = searchsorted_left(grid[*jmax], sg, *imax+1);
        if(index != -1)
            *imax = index;
        else
        {
            *imax = *imin-1;
            return -1;   // there is no overlap
        }
    }

    return 0;
}

int sorted_interpolation(double *x_subgrid, double *y_subgrid, int n_subgrid,
                         double *x_grid, double *y_grid, int n_grid)
{
    int i, j, status;
    int imin, imax;   // subgrid
    int jmin, jmax;   // grid

    imin = 0;
    imax = n_subgrid-1;
    jmin = 0;
    jmax = n_grid-1;

    status = find_subarray(x_grid, &jmin, &jmax, x_subgrid, &imin, &imax);

    j = jmin+1;
    for(i=imin;i<=imax;i++)
    {
        while( (x_subgrid[i]>x_grid[j]) && (j<jmax) )
            j++;

        y_subgrid[i] = linear_interpolation(x_subgrid[i], x_grid, y_grid, j-1, j);
    }

    // left extrapolation -> zero
    for(i=0;i<imin;i++)
        y_subgrid[i] = 0.;

    // right extrapolation -> constant
    for(i=imax+1;i<n_subgrid;i++)
        y_subgrid[i] = y_grid[n_grid-1];

    return status;
}

double point_interpolation(double x, double *x_grid, double *y_grid, int n_grid)
{
    int i;

    if(x < x_grid[0])
        return 0;
    else if(x > x_grid[n_grid-1])
        return 0;
    else
    {
        i = searchsorted_right(x, x_grid, n_grid);
        return linear_interpolation(x, x_grid, y_grid, i-1, i);
    }
}

// ===================================================================


// ======  Confluent hypergeometric function
// ===================================================================

double complex F11_series(double u, double c, double tol, int *status)
{
    int i, i2, n_terms;
    double z_im, a_im;
    double F_re, F_im, c_re, c_im, F11_re, F11_im, tmp, Fabs2;

    // a = i*u
    // b = 1
    // z = c*i*u

    n_terms = 10*(int)(fabs(c*u*u));
    n_terms = MAX(n_terms, 20);

    z_im = c*u;
    a_im = u;

    c_re = 1;
    c_im = 0;
    F11_re = 1;
    F11_im = 0;
    for(i=1;i<n_terms;i++)
    {
        i2 = i*i;
        F_re = -z_im*a_im/i2;
        F_im = (i-1)*z_im/i2;

        tmp = c_re;
        c_re = F_re*tmp - F_im*c_im;
        c_im = F_im*tmp + F_re*c_im;

        F11_re += c_re;
        F11_im += c_im;

        Fabs2 = F_re*F_re + F_im*F_im;
        if(Fabs2 < 1)
        {
            if(((c_re*c_re + c_im*c_im))/Fabs2 < (tol*tol))
                break;
        }
    }

    if(i < n_terms)
        *status = i;
    else
        *status = -1;

    return F11_re + I*F11_im;
}

double complex F11_series_b(double u, int b, double c, double tol, int *status)
{
    int i, n_terms;
    double z_im, a_im;
    double F_re, F_im, c_re, c_im, F11_re, F11_im, tmp, Fabs2;

    // a = i*u
    // b = b
    // z = c*i*u

    n_terms = 10*(int)(fabs(u*u*c));
    n_terms = MAX(n_terms, 20);

    z_im = c*u;
    a_im = u;

    c_re = 1;
    c_im = 0;
    F11_re = 1;
    F11_im = 0;
    for(i=1;i<n_terms;i++)
    {
        F_re = -z_im*a_im/(i-1+b)/i;
        F_im = (i-1)*z_im/(i-1+b)/i;

        tmp = c_re;
        c_re = F_re*tmp - F_im*c_im;
        c_im = F_im*tmp + F_re*c_im;

        F11_re += c_re;
        F11_im += c_im;

        Fabs2 = F_re*F_re + F_im*F_im;
        if(Fabs2 < 1)
        {
            if(((c_re*c_re + c_im*c_im))/Fabs2 < (tol*tol))
                break;
        }
    }

    if(i < n_terms)
        *status = i;
    else
        *status = -1;

    return F11_re + I*F11_im;
}

double complex F11_recurrence(double u, double c, int n_up, double tol, int *status)
{
    int i, b, status1, status2;
    double complex f, F11_b_2, F11_b_1, F11_b;

    // a = i*u
    // b = 1
    // z = c*i*u

    F11_b_2 = F11_series_b(u, n_up+2, c, 1e-1*tol, &status1);
    F11_b_1 = F11_series_b(u, n_up+1, c, 1e-1*tol, &status2);

    *status = MAX(status1, status2);

    for(i=0;i<n_up;i++)
    {
        b = n_up-i;
        f = I*c*u/b;
        F11_b = (1 + f)*F11_b_1 - (1 - I*u/(b+1))*f*F11_b_2;

        F11_b_2 = F11_b_1;
        F11_b_1 = F11_b;
    }

    return F11_b;
}

double complex F11_DLMF_largez(double u, double c, int order)
{
    int i;
    gsl_sf_result loggamma_mod, loggamma_arg;
    double x_r, x_i;
    double complex a, z, s1, s2, S1, S2, F, gamma, F11;

    a = I*u;
    // b = 1
    z = I*c*u;

    S1 = s1 = 1;
    S2 = s2 = 1;
    for(i=0;i<order+1;i++)
    {
        F = (1-a+i)*(1-a+i)/(i+1)/z;
        s1 *= F;

        F = -(a+i)*(a+i)/(i+1)/z;
        s2 *= F;

        S1 += s1;
        S2 += s2;
    }

    x_r = creal(a);
    x_i = cimag(a);
    gsl_sf_lngamma_complex_e(x_r, x_i, &loggamma_mod, &loggamma_arg);
    gamma = exp(loggamma_mod.val)*cexp(I*loggamma_arg.val);
    F11 = cexp(z)*cpow(z, a-1)*S1/gamma;

    x_r = creal(1-a);
    x_i = cimag(1-a);
    gsl_sf_lngamma_complex_e(x_r, x_i, &loggamma_mod, &loggamma_arg);
    gamma = exp(loggamma_mod.val)*cexp(I*loggamma_arg.val);
    F11 += cexp(I*M_PI*a)*cpow(z, -a)*S2/gamma;

    return F11;
}

// -------------------------------------------------------------------

double complex F11_Temme_order2(double u, double c)
{
    // definition: F11(i*u, 1, c*i*u)

    double w0, beta, sqr_beta;
    double sinh_w0, sinh_w0_2, cosh_w0, coth_w0_2, exp_w0_pl, exp_w0_mn;
    double xi0_2, xi0_3, xi0_4, sqr_xi0_2, alpha0, alpha0_1, alpha0_2;
    double w0_dot, den, f0_pl, f0_1_pl, f0_2_pl, f0_mn, f0_1_mn, f0_2_mn;
    double gamma_1, t0_1, t0_2, a0, b0, C, a1, b1;

    double complex j0_term, j1_term, a_sum, b_sum, F11;

    w0 = 2*log(0.5*sqrt(c) + sqrt(1+0.25*c));

    // -----------------------------------------------------------------
    sinh_w0   = sinh(w0);
    sinh_w0_2 = sinh(w0/2);
    cosh_w0   = cosh(w0);
    coth_w0_2 = 1/tanh(w0/2);
    exp_w0_pl = exp(w0);
    exp_w0_mn = exp(-w0);

    beta = 0.5*(w0 + sinh_w0);
    sqr_beta = sqrt(beta);

    xi0_2 = 0.5*coth_w0_2;
    xi0_3 = -0.25*(2 + cosh_w0)/sinh_w0_2/sinh_w0_2;
    xi0_4 = 0.25*(5 + cosh_w0)*coth_w0_2/sinh_w0_2/sinh_w0_2;
    sqr_xi0_2 = sqrt(xi0_2);

    alpha0 = sqr_beta/sqr_xi0_2;
    alpha0_1 = -xi0_3*sqr_beta/3./sqr_xi0_2/xi0_2;
    alpha0_2 = (9*xi0_2*xi0_2*xi0_2 - 9*xi0_2*xi0_4*beta + 11*beta*xi0_3*xi0_3)/36./sqr_beta/sqr_xi0_2/xi0_2/xi0_2;
    w0_dot = alpha0/beta;

    den = 1/(1-exp_w0_mn);
    f0_pl = den*alpha0;
    f0_1_pl = den*(alpha0_1 - exp_w0_mn*f0_pl);
    f0_2_pl = den*(alpha0_2 + exp_w0_mn*(f0_pl - 2*f0_1_pl));

    den = 1/(1-exp_w0_pl);
    f0_mn = -den*alpha0;
    f0_1_mn = den*(alpha0_1 - exp_w0_pl*f0_mn);
    f0_2_mn = den*(-alpha0_2 + exp_w0_pl*(f0_mn - 2*f0_1_mn));

    gamma_1 = xi0_3/6/xi0_2;

    t0_1 = sqr_beta*sqr_xi0_2;
    t0_2 = xi0_2 + sqr_beta*xi0_3/3/sqr_xi0_2;
    // -----------------------------------------------------------------

    a0 = (f0_pl + f0_mn)/2;
    b0 = (f0_pl - f0_mn)/2/beta;

    C = beta*w0_dot/8/sqr_beta/sqr_xi0_2;
    a1 = C*(f0_2_pl - f0_2_mn - 2*b0*t0_2 - 2*gamma_1*(f0_1_pl + f0_1_mn - 2*b0*t0_1));
    b1 = C*(f0_2_pl + f0_2_mn - 2*gamma_1*(f0_1_pl - f0_1_mn))/beta;
    // -----------------------------------------------------------------

    j0_term = gsl_sf_bessel_J0(2*u*beta);
    j1_term = -1j*beta*gsl_sf_bessel_J1(2*u*beta);

    a_sum = a0 - I*a1/u;
    b_sum = b0 - I*b1/u;

    F11 = cexp(0.5*I*u*c)*(j0_term*a_sum + j1_term*b_sum);

    return F11;
}

HyperCoeffs *init_F11_Temme_coeffs(double c)
{
    int n, nmax, sign_i, sign_j, i, j, k, id;
    int *max_index;
    double w0, tmp, binom, factorial;
    double beta, sqr_beta, beta_2, beta_32, beta_52;
    double sinh_w0, sinh_w0_2, cosh_w0, coth_w0_2, exp_w0_pl, exp_w0_mn, cosh_2w0, cosh_3w0;
    double *gamma, *t, *alpha;
    double *D_pl, *D_mn, *g_pl, *g_mn, *g_dot_pl, *g_dot_mn;
    double *w_dot_pl, *w_dot_mn, *f_pl, *f_mn;
    HyperCoeffs *coeffs = (HyperCoeffs *)malloc(sizeof(HyperCoeffs));

    double sqr_xi2, xi2, xi3, xi4, xi5, xi6, xi7, xi8;
    double xi22, xi32, xi42, xi52;
    double xi23, xi33, xi43;
    double xi24, xi34;
    double xi25, xi35;
    double xi36;

    n = 3;
    nmax = 2*n;

    w0 = 2*log(0.5*sqrt(c) + sqrt(1+0.25*c));
    sinh_w0 = sinh(w0);
    beta = 0.5*(w0 + sinh_w0);

    coeffs->n = n;
    coeffs->c = c;
    coeffs->beta = beta;
    coeffs->a = (double *)calloc((n+1), sizeof(double));
    coeffs->b = (double *)calloc((n+1), sizeof(double));

    // -----------------------------------------------------
    sinh_w0_2 = sinh(w0/2);
    cosh_w0   = cosh(w0);
    cosh_2w0  = cosh(2*w0);
    cosh_3w0  = cosh(3*w0);
    coth_w0_2 = 1/tanh(w0/2);
    exp_w0_pl = exp(w0);
    exp_w0_mn = exp(-w0);

    xi2 = 0.5*coth_w0_2;
    tmp = sinh_w0_2*sinh_w0_2;
    xi3 = -0.25*(2 + cosh_w0)/tmp;
    xi4 = 0.25*(5 + cosh_w0)*coth_w0_2/tmp;
    tmp = tmp*tmp;
    xi5 = -1./16*(26*cosh_w0 + cosh_2w0 + 33)/tmp;
    xi6 = 1./16*(56*cosh_w0 + cosh_2w0 + 123)*coth_w0_2/tmp;
    tmp = sinh_w0_2*sinh_w0_2*tmp;
    xi7 = -1./64*(1191*cosh_w0 + 120*cosh_2w0 + cosh_3w0 + 1208)/tmp;
    xi8 = 1./64*(4047*cosh_w0 + 246*cosh_2w0 + cosh_3w0 + 5786)*coth_w0_2/tmp;

    sqr_xi2 = sqrt(xi2);
    sqr_beta = sqrt(beta);
    beta_2 = beta*beta;
    beta_32 = beta*sqr_beta;
    beta_52 = beta*beta_32;
    // -----------------------------------------------------

    // -----------------------------------------------------
    gamma = (double *)malloc(nmax*sizeof(double));
    t     = (double *)malloc((nmax+1)*sizeof(double));
    alpha = (double *)malloc((nmax+1)*sizeof(double));

    xi22 = xi2*xi2;
    xi32 = xi3*xi3;
    xi42 = xi4*xi4;
    xi52 = xi5*xi5;

    xi23 = xi22*xi2;
    xi33 = xi32*xi3;
    xi43 = xi42*xi4;

    xi24 = xi23*xi2;
    xi34 = xi33*xi3;

    xi25 = xi24*xi2;
    xi35 = xi34*xi3;

    xi36 = xi35*xi3;

    gamma[0] = 1;
    gamma[1] = (1.0/6.0)*xi3/xi2;
    gamma[2] = -1.0/24.0*xi4/xi2 + (1.0/24.0)*xi32/xi22 - 1.0/8.0*xi2/beta;
    gamma[3] = (1.0/120.0)*xi5/xi2 - 1.0/48.0*xi3*xi4/xi22 + (5.0/432.0)*xi33/xi23 + (1.0/48.0)*xi3/beta;
    gamma[4] = ((1.0/240.0)*xi3*xi5 + (1.0/384.0)*xi42)/xi22 - 5.0/576.0*xi32*xi4/xi23 + (35.0/10368.0)*xi34/xi24 - 1.0/192.0*xi4/beta + (-1.0/720.0*beta*xi6 + (1.0/576.0)*xi32)/(beta*xi2) + (3.0/128.0)*xi22/beta_2;
    gamma[5] = (1.0/2304.0)*xi3*(4*xi3*xi5 + 5*xi42)/xi23 - 35.0/10368.0*xi33*xi4/xi24 + (7.0/6912.0)*xi35/xi25 + (1.0/960.0)*xi5/beta + ((1.0/5040.0)*beta*xi7 - 1.0/1152.0*xi3*xi4)/(beta*xi2) + (-1.0/2880.0*beta*(2*xi3*xi6 + 3*xi4*xi5) + (1.0/3456.0)*xi33)/(beta*xi22) - 3.0/256.0*xi2*xi3/beta_2;

    tmp = sqr_beta*sqr_xi2;
    t[0] = beta;
    t[1] = tmp;
    t[2] = tmp*((1.0/3.0)*xi3/xi2 + sqr_xi2/sqr_beta);
    t[3] = tmp*((1.0/4.0)*xi4/xi2 - 1.0/12.0*xi32/xi22 + (3.0/4.0)*xi2/beta + xi3/(sqr_beta*sqr_xi2));
    t[4] = tmp*((1.0/5.0)*xi5/xi2 - 1.0/6.0*xi3*xi4/xi22 + (1.0/18.0)*xi33/xi23 + (3.0/2.0)*xi3/beta + xi4/(sqr_beta*sqr_xi2));
    t[5] = tmp*((1.0/6.0)*xi6/xi2 - 1.0/6.0*xi3*xi5/xi22 - 5.0/48.0*xi42/xi22 + (5.0/24.0)*xi32*xi4/xi23 - 25.0/432.0*xi34/xi24 + (15.0/8.0)*xi4/beta + (5.0/8.0)*xi32/(beta*xi2) - 15.0/16.0*xi22/beta_2 + xi5/(sqr_beta*sqr_xi2));
    t[6] = tmp*((1.0/7.0)*xi7/xi2 - 1.0/6.0*xi3*xi6/xi22 - 1.0/4.0*xi4*xi5/xi22 + (1.0/4.0)*xi32*xi5/xi23 + (5.0/16.0)*xi3*xi42/xi23 - 25.0/72.0*xi33*xi4/xi24 + (35.0/432.0)*xi35/xi25 + (9.0/4.0)*xi5/beta + (15.0/8.0)*xi3*xi4/(beta*xi2) - 5.0/24.0*xi33/(beta*xi22) - 75.0/16.0*xi2*xi3/beta_2 + xi6/(sqr_beta*sqr_xi2));

    tmp = 1./3/xi2/sqr_xi2;
    alpha[0] = tmp*(3*sqr_beta*xi2);
    alpha[1] = tmp*(-sqr_beta*xi3);
    alpha[2] = tmp*(sqr_beta*(-3.0/4.0*xi4 + (11.0/12.0)*xi32/xi2) + (3.0/4.0)*xi22/sqr_beta);
    alpha[3] = tmp*(sqr_beta*(-3.0/5.0*xi5 + 2*xi3*xi4/xi2 - 4.0/3.0*xi33/xi22));
    alpha[4] = tmp*(sqr_beta*(-1.0/2.0*xi6 + (21.0/10.0)*xi3*xi5/xi2 + (23.0/16.0)*xi42/xi2 - 137.0/24.0*xi32*xi4/xi22 + (379.0/144.0)*xi34/xi23) + (-3.0/8.0*xi2*xi4 + (3.0/8.0)*xi32)/sqr_beta - 9.0/16.0*xi23/beta_32);
    alpha[5] = tmp*(sqr_beta*(-3.0/7.0*xi7 + (13.0/6.0)*xi3*xi6/xi2 + (15.0/4.0)*xi4*xi5/xi2 - 89.0/12.0*xi32*xi5/xi22 - 485.0/48.0*xi3*xi42/xi22 + (445.0/24.0)*xi33*xi4/xi23 - 2825.0/432.0*xi35/xi24) + (-3.0/4.0*xi2*xi5 + (15.0/8.0)*xi3*xi4 - 25.0/24.0*xi33/xi2)/sqr_beta - 15.0/16.0*xi22*xi3/beta_32);
    alpha[6] = tmp*(sqr_beta*(-3.0/8.0*xi8 + (31.0/14.0)*xi3*xi7/xi2 + (37.0/8.0)*xi4*xi6/xi2 + (117.0/40.0)*xi52/xi2 - 73.0/8.0*xi32*xi6/xi22 - 251.0/8.0*xi3*xi4*xi5/xi22 - 455.0/64.0*xi43/xi22 + (689.0/24.0)*xi33*xi5/xi23 + (11225.0/192.0)*xi32*xi42/xi23 - 39445.0/576.0*xi34*xi4/xi24 + (3755.0/192.0)*xi36/xi25) + (-9.0/8.0*xi2*xi6 + (27.0/8.0)*xi3*xi5 + (165.0/64.0)*xi42 - 255.0/32.0*xi32*xi4/xi2 + (205.0/64.0)*xi34/xi22)/sqr_beta + (-45.0/64.0*xi22*xi4 - 45.0/64.0*xi2*xi32)/beta_32 + (135.0/64.0)*xi24/beta_52);
    // -----------------------------------------------------

    // -----------------------------------------------------
    max_index = (int *)malloc(n*sizeof(int));

    for(i=0;i<n;i++)
        max_index[i] = nmax - 2*i;

    D_pl = (double *)calloc(n*nmax, sizeof(double));
    D_mn = (double *)calloc(n*nmax, sizeof(double));
    g_pl = (double *)calloc(n*nmax, sizeof(double));
    g_mn = (double *)calloc(n*nmax, sizeof(double));
    g_dot_pl = (double *)calloc(n*nmax, sizeof(double));
    g_dot_mn = (double *)calloc(n*nmax, sizeof(double));

    w_dot_pl = (double *)calloc((nmax-1), sizeof(double));
    w_dot_mn = (double *)calloc((nmax-1), sizeof(double));

    f_pl = (double *)calloc((n+1)*(nmax+1), sizeof(double));
    f_mn = (double *)calloc((n+1)*(nmax+1), sizeof(double));
    // -----------------------------------------------------

    // -----------------------------------------------------
    sign_i = 1;
    for(i=0;i<(nmax-1);i++)
    {
        w_dot_pl[i] = alpha[i];
        w_dot_mn[i] = sign_i*alpha[i];

        sign_j = 1;
        for(j=0;j<i;j++)
        {
            binom = gsl_sf_choose(i, j);
            w_dot_pl[i] -= binom*w_dot_pl[j]*t[i-j];
            w_dot_mn[i] -= binom*w_dot_mn[j]*t[i-j]*sign_i*sign_j;
            sign_j = -sign_j;
        }

        w_dot_pl[i] /= beta;
        w_dot_mn[i] /= beta;

        sign_i = -sign_i;
    }

    sign_i = 1;
    for(i=0;i<(nmax+1);i++)
    {
        sign_j = sign_i;
        for(j=0;j<i;j++)
        {
            binom = gsl_sf_choose(i, j);
            f_pl[i] += sign_j*binom*f_pl[j];
            f_mn[i] += sign_j*binom*f_mn[j];
            sign_j = -sign_j;
        }

        f_pl[i] *= exp_w0_mn;
        f_mn[i] *= exp_w0_pl;

        f_pl[i] += alpha[i];
        f_mn[i] += -sign_i*alpha[i];

        f_pl[i] /= (1-exp_w0_mn);
        f_mn[i] /= (1-exp_w0_pl);

        sign_i = -sign_i;
    }

    coeffs->a[0] = 0.5*(f_pl[0] + f_mn[0]);
    coeffs->b[0] = 0.5*(f_pl[0] - f_mn[0])/beta;
    // -----------------------------------------------------

    // -----------------------------------------------------
    tmp = 2.*sqr_beta*sqr_xi2;
    for(k=0;k<n;k++)
    {
        id = k*nmax;

        sign_i = 1;
        for(i=0;i<max_index[k];i++)
        {
            D_pl[id+i] = f_pl[id+k+i+1] - coeffs->b[k]*t[i+1];
            D_mn[id+i] = f_mn[id+k+i+1] - sign_i*coeffs->b[k]*t[i+1];
            sign_i = -sign_i;
        }

        sign_i = 1;
        for(i=0;i<max_index[k];i++)
        {
            sign_j = 1;
            for(j=0;j<(i+1);j++)
            {
                factorial = gsl_sf_fact(j+1);
                g_pl[id+i] += sign_j/factorial*D_pl[id+j]*gamma[i-j];
                g_mn[id+i] += 1./factorial*D_mn[id+j]*gamma[i-j];
                sign_j = -sign_j;
            }

            factorial = gsl_sf_fact(i);
            g_pl[id+i] *= sign_i*factorial/tmp;
            g_mn[id+i] *= factorial/tmp;
            sign_i = -sign_i;
        }

        for(i=0;i<(max_index[k]-1);i++)
        {
            for(j=0;j<(i+1);j++)
            {
                binom = gsl_sf_choose(i, j);
                g_dot_pl[id+i] += binom*g_pl[id+j+1]*w_dot_pl[i-j];
                g_dot_mn[id+i] += binom*g_mn[id+j+1]*w_dot_mn[i-j];
            }
        }

        for(i=0;i<(max_index[k]-1);i++)
        {
            sign_j = 1;
            for(j=0;j<(i+1);j++)
            {
                binom = gsl_sf_choose(i, j);
                f_pl[(k+1)*(nmax+1)+i] += binom*t[j]*g_dot_pl[id+i-j];
                f_mn[(k+1)*(nmax+1)+i] -= sign_j*binom*t[j]*g_dot_mn[id+i-j];
                sign_j = -sign_j;
            }
        }

        coeffs->a[k+1] = 0.5*(f_pl[(k+1)*(nmax+1)] + f_mn[(k+1)*(nmax+1)]);
        coeffs->b[k+1] = 0.5*(f_pl[(k+1)*(nmax+1)] - f_mn[(k+1)*(nmax+1)])/beta;
    }
    // -----------------------------------------------------

    free(D_pl);
    free(D_mn);
    free(g_pl);
    free(g_mn);
    free(g_dot_pl);
    free(g_dot_mn);
    free(w_dot_pl);
    free(w_dot_mn);
    free(f_pl);
    free(f_mn);

    free(max_index);
    free(alpha);
    free(t);
    free(gamma);

    return coeffs;
}

void free_F11_Temme_coeffs(HyperCoeffs *coeffs)
{
    free(coeffs->a);
    free(coeffs->b);
    free(coeffs);
}

double complex F11_Temme(double u, HyperCoeffs *coeffs)
{
    // definition: F11(i*u, 1, c*i*u)

    int i;
    double j0_term, j1_term, x;
    double complex a_sum, b_sum, F;

    j0_term = gsl_sf_bessel_J0(2*u*coeffs->beta);
    j1_term = coeffs->beta*gsl_sf_bessel_J1(2*u*coeffs->beta);

    a_sum = coeffs->a[0];
    b_sum = coeffs->b[0];

    x = 1;
    F = -I;
    for(i=1;i<(coeffs->n+1);i++)
    {
        x *= u;

        a_sum += F*coeffs->a[i]/x;
        b_sum += F*coeffs->b[i]/x;

        F *= -I;
    }

    return cexp(0.5*I*u*coeffs->c)*(j0_term*a_sum - I*j1_term*b_sum);
}

double complex F11_Temme_full(double u, double c)
{
    // definition: F11(i*u, 1, c*i*u)

    int i;
    double j0_term, j1_term, x;
    double complex a_sum, b_sum, F;
    HyperCoeffs *coeffs = init_F11_Temme_coeffs(c);

    j0_term = gsl_sf_bessel_J0(2*u*coeffs->beta);
    j1_term = coeffs->beta*gsl_sf_bessel_J1(2*u*coeffs->beta);

    a_sum = coeffs->a[0];
    b_sum = coeffs->b[0];

    x = 1;
    F = -I;
    for(i=1;i<(coeffs->n+1);i++)
    {
        x *= u;

        a_sum += F*coeffs->a[i]/x;
        b_sum += F*coeffs->b[i]/x;

        F *= -I;
    }

    free_F11_Temme_coeffs(coeffs);

    return cexp(0.5*I*u*c)*(j0_term*a_sum - I*j1_term*b_sum);
}

// -------------------------------------------------------------------

double complex F11_singlepoint(double u, double c, int *status, int *approx_flag)
{
    // definition: F11(i*u, 1, c*i*u)

    int b;
    double z;
    double complex F11;
    HyperCoeffs *coeffs;

    z = u*c;
    *status = 0;
    *approx_flag = 0;

    if(u > 4)
    {
        *approx_flag = flag_F11_Temme;

        coeffs = init_F11_Temme_coeffs(c);
        F11 = F11_Temme(u, coeffs);
        free_F11_Temme_coeffs(coeffs);
    }
    else
    {
        if(z > 22)
        {
            *approx_flag = flag_F11_largez;

            F11 = F11_DLMF_largez(u, c, 16);
        }
        else
        {
            if( z < 5 )
            {
                *approx_flag = flag_F11_series;

                F11 = F11_series(u, c, 1e-5, status);
            }
            else
            {
                *approx_flag = flag_F11_recurrence;

                b = 22*(int)(z/5);
                F11 = F11_recurrence(u, c, b, 1e-6, status);
            }
        }
    }

    return F11;
}

int F11_sorted(double *u, double c, int n_F, double complex *F11, int nthreads)
{
    // definition: F11(i*u, 1, c*i*u)

    int i, id, nmax, status, b;
    double z;
    HyperCoeffs *coeffs;

    for(i=1;i<n_F;i++)
    {
        if(u[i] < u[i-1])
        {
            PERROR("u-values must be sorted (u[i]<u[i+1])")
            return 1;
        }
    }

    // use Temme for u > 4
    id = searchsorted_right(4, u, n_F);

    // never use Temme expression if id==-1
    if(id == -1)
        nmax = n_F;
    else
    {
        nmax = id;
        coeffs = init_F11_Temme_coeffs(c);
    }

    // series expansion and asymptotic for large arg
    #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) if(nthreads > 1) \
                                 private(z, b, status)
    #endif
    for(i=0;i<nmax;i++)
    {
        z = c*u[i];

        if(z > 22)
        {
            // HVR_DEBUG
            //~ printf("id=%d   u=%g   type=asymp\n", i, u[i]);
            F11[i] = F11_DLMF_largez(u[i], c, 16);
        }
        else
        {
            if(z < 5)
            {
                // HVR_DEBUG
                //~ printf("id=%d   u=%g   type=series\n", i, u[i]);
                F11[i] = F11_series(u[i], c, 1e-5, &status);
            }
            else
            {
                // HVR_DEBUG
                //~ printf("id=%d   u=%g   type=rec\n", i, u[i]);

                b = 22*(int)(z/5);
                F11[i] = F11_recurrence(u[i], c, b, 1e-6, &status);
            }
        }
    }

    // Temme asymptotic approximation
    #ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) if(nthreads > 1)
    #endif
    for(i=nmax;i<n_F;i++)
    {
        // HVR_DEBUG
        //~ printf("id=%d   u=%g   type=Temme\n", i, u[i]);
        F11[i] = F11_Temme(u[i], coeffs);
    }

    if(id != -1)
        free_F11_Temme_coeffs(coeffs);

    return 0;
}

// ===================================================================
