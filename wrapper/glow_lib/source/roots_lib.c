/*
 * GLoW - roots_lib.c
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

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_rng.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"

// =================================================================


// ======  Operate with critical points
// =================================================================

void display_CritPoint(CritPoint *p)
{
    if(p->type == type_min)
        printf("Min at ");
    if(p->type == type_max)
        printf("Max at ");
    if(p->type == type_saddle)
        printf("Saddle at ");
    if(p->type == type_singcusp_max)
        printf("Sing/cusp max at ");
    if(p->type == type_singcusp_min)
        printf("Sing/cusp min at  ");
    if(p->type == type_non_converged)
        printf("Point not converged at ");
    printf("(%e, %e), with t = %e, mu = %e\n", p->x1, p->x2, p->t, p->mag);
}

void copy_CritPoint(CritPoint *p_dest, const CritPoint *p_src)
{
    p_dest->type = p_src->type;
    p_dest->t    = p_src->t;
    p_dest->mag  = p_src->mag;
    p_dest->x1   = p_src->x1;
    p_dest->x2   = p_src->x2;
}

void swap_CritPoint(CritPoint *p_a, CritPoint *p_b)
{
    CritPoint p_tmp;

    copy_CritPoint(&p_tmp, p_a);
    copy_CritPoint(p_a, p_b);
    copy_CritPoint(p_b, &p_tmp);
}

int is_same_CritPoint(CritPoint *p_a, CritPoint *p_b)
{
    double dist;

    if(p_a->type == p_b->type)
    {
        dist = sqrt((p_a->x1-p_b->x1)*(p_a->x1-p_b->x1) + (p_a->x2-p_b->x2)*(p_a->x2-p_b->x2));
        if(dist < pprec.ro_issameCP_dist)
            return _TRUE_;
    }

    return _FALSE_;
}

void classify_CritPoint(CritPoint *p, double y, Lens *Psi)
{
    double d11, d22, d12;
    double trA, detA;
    double phi_derivs[N_derivs];

    phiFermat_2ndDeriv(phi_derivs, y, p->x1, p->x2, Psi);

    p->t = phi_derivs[i_0];
    p->mag = magnification(p->x1, p->x2, Psi);

    d11 = phi_derivs[i_dx1dx1];
    d12 = phi_derivs[i_dx1dx2];
    d22 = phi_derivs[i_dx2dx2];

    trA = d11 + d22;
    detA = d11*d22 - d12*d12;

    if(detA > 0)
    {
        if(trA > 0)
            p->type = type_min;
        else
            p->type = type_max;
    }
    else
        p->type = type_saddle;
}

int find_i_xmin_CritPoint(int n_points, CritPoint *p)
{
    int i, j;
    double xmin;

    j = 0;
    xmin = p[0].x1;
    for(i=0;i<n_points;i++)
    {
        if(p[i].x1 < xmin)
        {
            xmin = p[i].x1;
            j = i;
        }
    }

    return j;
}

int find_i_tmin_CritPoint(int n_points, CritPoint *p)
{
    int i, j;
    double tmin;

    tmin = p[0].t;
    j = 0;
    for(i=0;i<n_points;i++)
    {
        if(p[i].t < tmin)
        {
            tmin = p[i].t;
            j = i;
        }
    }

    return j;
}

void sort_x_CritPoint(int n_points, CritPoint *p)
{
    int i, j;

    for(i=0;i<n_points;i++)
    {
        j = find_i_xmin_CritPoint(n_points-i, p+i);
        swap_CritPoint(p+i, p+i+j);
    }
}

void sort_t_CritPoint(int n_points, CritPoint *p)
{
    int i, j;

    for(i=0;i<n_points;i++)
    {
        j = find_i_tmin_CritPoint(n_points-i, p+i);
        swap_CritPoint(p+i, p+i+j);
    }
}


// ======  1D functions
// =================================================================

double dphi_1D(double x1, void *pimage)
{
    pImage *p = (pImage *)pimage;
    double phi_derivs[N_derivs];

    phiFermat_1stDeriv(phi_derivs, p->y, x1, 0, p->Psi);

    return phi_derivs[i_dx1];
}

double ddphi_1D(double x1, void *pimage)
{
    pImage *p = (pImage *)pimage;
    double phi_derivs[N_derivs];

    phiFermat_2ndDeriv(phi_derivs, p->y, x1, 0, p->Psi);

    return phi_derivs[i_dx1dx1];
}

void dphi_ddphi_1D(double x1, void *pimage, double *y, double *dy)
{
    pImage *p = (pImage *)pimage;
    double phi_derivs[N_derivs];

    phiFermat_2ndDeriv(phi_derivs, p->y, x1, 0, p->Psi);

    *y = phi_derivs[i_dx1];
    *dy = phi_derivs[i_dx1dx1];
}

void find_CritPoint_1D(double xguess, pImage *p)
{
    int status;
    int iter, max_iter;
    double x, x0;
    double epsabs, epsrel;

    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;
    gsl_function_fdf FDF;

    FDF.f = &dphi_1D;
    FDF.df = &ddphi_1D;
    FDF.fdf = &dphi_ddphi_1D;
    FDF.params = p;

    x = xguess;
    T = get_fdfRoot(pprec.ro_findCP1D.id);
    s = gsl_root_fdfsolver_alloc(T);
    gsl_root_fdfsolver_set(s, &FDF, x);

    max_iter = pprec.ro_findCP1D.max_iter;
    epsabs   = pprec.ro_findCP1D.epsabs;
    epsrel   = pprec.ro_findCP1D.epsrel;

    iter = 0;
    do
    {
        status = gsl_root_fdfsolver_iterate(s);
        x0 = x;
        x = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x, x0, epsabs, epsrel);

        // HVR_DEBUG
        //~ printf("x=%g   dphi_1D=%g  ddphi_1D=%g  iter = %d/%d\n", x, dphi_1D(x0, p), ddphi_1D(x0, p), iter, max_iter);
        //~ if(status == GSL_SUCCESS)
            //~ printf ("Converged: x=%e\n", x);

        iter++;
    }
    while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fdfsolver_free(s);

    // initialize the crit point found
    p->point->x1 = x;
    p->point->x2 = 0;
    if(status == GSL_SUCCESS)
        classify_CritPoint(p->point, p->y, p->Psi);
    else
        p->point->type = type_non_converged;
}

void find_CritPoint_bracket_1D(double x_lo, double x_hi, pImage *p)
{
    int status;
    int iter, max_iter;
    double x;
    double epsabs, epsrel;

    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    gsl_function F;

    F.function = &dphi_1D;
    F.params = p;

    // swap
    if(x_lo > x_hi)
    {
        x = x_lo;
        x_lo = x_hi;
        x_hi = x;
    }

    T = get_fRoot(pprec.ro_findCP1D_bracket.id);
    s = gsl_root_fsolver_alloc(T);
    gsl_root_fsolver_set(s, &F, x_lo, x_hi);

    max_iter = pprec.ro_findCP1D_bracket.max_iter;
    epsabs   = pprec.ro_findCP1D_bracket.epsabs;
    epsrel   = pprec.ro_findCP1D_bracket.epsrel;

    iter = 0;
    do
    {
        status = gsl_root_fsolver_iterate(s);
        x = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s);
        x_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lo, x_hi, epsabs, epsrel);

        // HVR_DEBUG
        //~ printf("iter = %d/%d\n", iter, max_iter);
        //~ if(status == GSL_SUCCESS)
            //~ printf ("Converged: x=%e\n", x);

        iter++;
    }
    while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free(s);

    // initialize the crit point found
    p->point->x1 = x;
    p->point->x2 = 0;
    if(status == GSL_SUCCESS)
        classify_CritPoint(p->point, p->y, p->Psi);
    else
        p->point->type = type_non_converged;
}

int check_singcusp_1D(CritPoint *p, double y, Lens *Psi)
{
    int has_singcusp;
    double dx, eps;
    double dphi_le, dphi_ri;
    double phi_derivs[N_derivs];

    dx = pprec.ro_singcusp1D_dx;
    eps = pprec.ro_singcusp1D_eps;

    phiFermat_1stDeriv(phi_derivs, y, -dx, 0, Psi);
    dphi_le = phi_derivs[i_dx1];

    phiFermat_1stDeriv(phi_derivs, y,  dx, 0, Psi);
    dphi_ri = phi_derivs[i_dx1];

    has_singcusp = _FALSE_;
    if(SIGN(dphi_le) != SIGN(dphi_ri))
    {
        has_singcusp = _TRUE_;

        if(dphi_ri < 0)
            p->type  = type_singcusp_max;
        else
            p->type  = type_singcusp_min;

        p->x1  = 0.;
        p->x2  = 0.;
        p->t   = phiFermat(y, eps, 0, Psi);
        p->mag = magnification(eps, 0, Psi);
    }

    return has_singcusp;
}

CritPoint *find_all_CritPoints_1D(int *n_cpoints, double y, Lens *Psi)
{
    int i, sign;
    int n_points, n_brackets, n_buffer;
    double xmin, xmax, Delta;
    double x_lo, x_hi, dphi_lo, dphi_hi;
    CritPoint *ps;
    pImage p;

    p.y = y;
    p.Psi = Psi;

    // precision parameters
    xmax = MAX(pprec.ro_findallCP1D_xmax, 2*y);
    xmin = pprec.ro_findallCP1D_xmin;
    n_brackets = pprec.ro_findallCP1D_nbrackets;  // number per side

    n_buffer = 10;
    ps = (CritPoint *)malloc(n_buffer*sizeof(CritPoint));

    // find at least one crit point with Newton's method
    p.point = ps;
    find_CritPoint_1D(-xmax, &p);

    p.point = ps+1;
    find_CritPoint_1D(xmax, &p);

    if( (ps[0].type != type_non_converged) && (ps[1].type != type_non_converged) )
    {
        // if both converged keep the most distant
        if(ABS(ps[1].x1) > ABS(ps[0].x1))
            swap_CritPoint(ps, ps+1);
    }
    else if( (ps[0].type == type_non_converged) && (ps[1].type == type_non_converged) )
    {
        // if neither converged, error
        PERROR("Failed to find the first 1d critical point")
    }
    else if( (ps[0].type == type_non_converged) && (ps[1].type != type_non_converged) )
    {
        // if only the second one converged, swap them and use it
        swap_CritPoint(ps, ps+1);
    }

    // HVR_DEBUG
    //~ display_CritPoint(ps);
    //~ display_CritPoint(ps+1);

    // use this point to set a scale to bracket
    n_points = 1;
    xmax = ABS(ps[0].x1) - xmin;

    // find sing/cusp
    if( check_singcusp_1D(ps+1, y, Psi) == _TRUE_ )
        n_points++;

    // find brackets: change sign to check left and right sides
    Delta = pow(xmax/xmin, 1./n_brackets);

    for(sign=-1; sign<=1; sign+=2)
    {
        x_lo = sign*xmin;
        x_hi = x_lo*Delta;
        for(i=0;i<n_brackets;i++)
        {
            dphi_lo = dphi_1D(x_lo, &p);
            dphi_hi = dphi_1D(x_hi, &p);

            // HVR_DEBUG
            //~ printf("x_lo=%e   x_hi=%e    f_lo=%e   f_hi=%e\n", x_lo, x_hi, dphi_lo, dphi_hi);

            if(SIGN(dphi_lo) != SIGN(dphi_hi))
            {
                p.point = ps+n_points;
                find_CritPoint_bracket_1D(x_lo, x_hi, &p);
                n_points++;

                // buffer full
                if(n_points == n_buffer)
                {
                    n_buffer *= 2;
                    ps = realloc(ps, n_buffer*sizeof(CritPoint));
                }
            }

            x_lo = x_hi;
            x_hi *= Delta;
        }
    }

    // realloc to actual number of points
    ps = realloc(ps, n_points*sizeof(CritPoint));
    *n_cpoints = n_points;

    // sort them
    sort_t_CritPoint(n_points, ps);

    return ps;
}

CritPoint *driver_all_CritPoints_1D(int *n_cpoints, double y, pNamedLens *pNLens)
{
    Lens Psi = init_lens(pNLens);

    return find_all_CritPoints_1D(n_cpoints, y, &Psi);
}


// ======  2D functions
// =================================================================

// === Growing/shrinking circles (generic algorithm)

double Tmin_R(double R, void *ptarget)
{
    int status, iter, max_iter;
    double m, dT, min, dalpha;
    double eps, a, b, T_m;
    double epsabs, epsrel;
    pTarget *p = (pTarget *)ptarget;

    const gsl_min_fminimizer_type *T;
    gsl_min_fminimizer *s;
    gsl_function F;

    // ---------------------------

    p->R = R;
    F.function = p->T_func;
    F.params = p;

    max_iter = pprec.ro_TminR.max_iter;
    T = get_fMin(pprec.ro_TminR.id);
    s = gsl_min_fminimizer_alloc(T);

    // shift extrema to avoid symmetric situations
    eps = 0.0124937;
    a = eps;
    b = 2*M_PI+eps;

    for(iter=0;iter<max_iter;iter++)
    {
        // use the derivative to estimate the correct point
        dalpha = pprec.ro_TminR_dalpha;
        dT = p->dT_func_dalpha(b, ptarget);
        if(dT > 0)
            m = b - dalpha;
        else
            m = a + dalpha;

        // check that it is a healthy initial guess
        T_m = p->T_func(m, p);
        if( (p->T_func(a, p) > T_m) || (p->T_func(b, p) > T_m) )
            break;
        else
        {
            // otherwise keep rotating the interval to avoid the harmful dT=0
            eps = 2*M_PI/max_iter;
            a += eps;
            b += eps;
        }
    }

    // HVR_DEBUG
    //~ printf("dT = %e (y=%e)\n", dT, p->y);
    //~ printf("a=%e    m=%e    b=%e\n", a, m, b);
    //~ printf("T(a)=%e    T(m)=%e    T(b)=%e\n\n", p->T_func(a, p), p->T_func(m, p), p->T_func(b, p));

    gsl_min_fminimizer_set(s, &F, m, a, b);

    epsabs = pprec.ro_TminR.epsabs;
    epsrel = pprec.ro_TminR.epsrel;

    iter = 0;
    do
    {
        iter++;
        status = gsl_min_fminimizer_iterate(s);

        m = gsl_min_fminimizer_x_minimum(s);
        a = gsl_min_fminimizer_x_lower(s);
        b = gsl_min_fminimizer_x_upper(s);

        status = gsl_min_test_interval(a, b, epsabs, epsrel);

        // HVR_DEBUG
        //~ if (status == GSL_SUCCESS)
            //~ printf ("Converged:\n");
    }
    while(status == GSL_CONTINUE && iter < max_iter);

    gsl_min_fminimizer_free(s);

    // ----------------------------------------

    // bring it back into the [0, 2pi) range
    p->alpha = MOD_2PI(m);
    min = p->T_func(p->alpha, p);

    return min;
}

double dTmin_R(double R, void *ptarget)
{
    double deriv;
    pTarget *p = (pTarget *)ptarget;

    deriv = p->dT_func_dR(R, ptarget);

    // finite difference approx
    // deriv = (Tmin_R(R+pprec.ro_TminR_dR, ptarget)-Tmin_R(R, ptarget))/pprec.ro_TminR_dR;

    return deriv;
}

void TmindTmin_R(double R, void *ptarget, double *y, double *dy)
{
    double T, deriv;
    pTarget *p = (pTarget *)ptarget;

    T = Tmin_R(R, ptarget);
    deriv = p->dT_func_dR(R, ptarget);

    // finite difference approx
    // dR = pprec.ro_TminR_dR
    // deriv = (Tmin_R(R+dR, ptarget)-T)/dR;

    *y = T;
    *dy = deriv;
}

int Tmin(pTarget *ptarget)
{
    int status, iter, max_iter;
    double x, x0;
    double epsabs, epsrel;
    pTarget *p = ptarget;

    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;
    gsl_function_fdf FDF;

    FDF.f = &Tmin_R;
    FDF.df = &dTmin_R;
    FDF.fdf = &TmindTmin_R;
    FDF.params = p;

    x = p->R;
    T = get_fdfRoot(pprec.ro_Tmin.id);
    max_iter = pprec.ro_Tmin.max_iter;
    epsabs = pprec.ro_Tmin.epsabs;
    epsrel = pprec.ro_Tmin.epsrel;

    s = gsl_root_fdfsolver_alloc(T);
    gsl_root_fdfsolver_set(s, &FDF, x);

    iter = 0;
    do
    {
        iter++;
        status = gsl_root_fdfsolver_iterate(s);
        x0 = x;
        x = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x, x0, epsabs, epsrel);

        //~ printf("iter=%d  R=%g\n", iter, x);
    }
    while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fdfsolver_free (s);

    // ---------------

    // evaluate the minimum (should be zero)
    p->R = x;
    p->min = p->T_func(p->alpha, p);

    return status;
}


// === Critical points through minimization (with a threshold)

double dphisqr_func(double alpha, void *ptarget)
{
    pTarget *p = (pTarget *)ptarget;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double d1, d2;

    x1x2_def(x_vec, p->R, alpha, p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];

    return d1*d1 + d2*d2;
}

double ddphisqr_func_dalpha(double alpha, void *ptarget)
{
    pTarget *p = (pTarget *)ptarget;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double d1, d2, Dx1, Dx2, dT;
    double d11, d22, d12;

    x1x2_def(x_vec, p->R, alpha, p->x0_vec);
    phiFermat_2ndDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];
    d11 = phi_derivs[i_dx1dx1];
    d22 = phi_derivs[i_dx2dx2];
    d12 = phi_derivs[i_dx1dx2];

    dT = -2*Dx2*(d1*d11 + d2*d12) + 2*Dx1*(d1*d12 + d2*d22);

    return dT;
}

double ddphisqr_func_dR(double R, void *ptarget)
{
    pTarget *p = (pTarget *)ptarget;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double d1, d2, Dx1, Dx2, dT;
    double d11, d22, d12;

    x1x2_def(x_vec, R, p->alpha, p->x0_vec);
    phiFermat_2ndDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];
    d11 = phi_derivs[i_dx1dx1];
    d22 = phi_derivs[i_dx2dx2];
    d12 = phi_derivs[i_dx1dx2];

    dT = 2./R*(Dx1*(d1*d11 + d2*d12) + Dx2*(d1*d12 + d2*d22));

    return dT;
}

double dphisqr_f_CritPoints_2D(const gsl_vector *x, void *params)
{
    double x1, x2;
    pImage *p = (pImage *)params;
    double phi_derivs[N_derivs];
    double d1, d2;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    phiFermat_1stDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];

    return d1*d1 + d2*d2;
}

void dphisqr_df_CritPoints_2D(const gsl_vector *x, void *params, gsl_vector *df)
{
    double x1, x2;
    double d1, d2, d11, d12, d22;
    double phi_derivs[N_derivs];
    pImage *p = (pImage *)params;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    phiFermat_2ndDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];
    d11 = phi_derivs[i_dx1dx1];
    d12 = phi_derivs[i_dx1dx2];
    d22 = phi_derivs[i_dx2dx2];

    gsl_vector_set(df, i_x1, 2*d1*d11 + 2*d2*d12);
    gsl_vector_set(df, i_x2, 2*d1*d12 + 2*d2*d22);
}

void dphisqr_fdf_CritPoints_2D(const gsl_vector *x, void *params, double *f, gsl_vector *df)
{
    double x1, x2;
    double d1, d2, d11, d12, d22;
    double phi_derivs[N_derivs];
    pImage *p = (pImage *)params;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    phiFermat_2ndDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];
    d11 = phi_derivs[i_dx1dx1];
    d12 = phi_derivs[i_dx1dx2];
    d22 = phi_derivs[i_dx2dx2];

    *f = d1*d1 + d2*d2;
    gsl_vector_set(df, i_x1, 2*d1*d11 + 2*d2*d12);
    gsl_vector_set(df, i_x2, 2*d1*d12 + 2*d2*d22);
}

int find_CritPoint_min_2D(double x1guess, double x2guess, pImage *p)
{
    int iter, max_iter, status;
    double tol, first_step, epsabs;

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

    gsl_vector *x_vec;
    gsl_multimin_function_fdf F;

    F.n = N_dims;
    F.f = dphisqr_f_CritPoints_2D;
    F.df = dphisqr_df_CritPoints_2D;
    F.fdf = dphisqr_fdf_CritPoints_2D;
    F.params = p;

    x_vec = gsl_vector_alloc(N_dims);
    gsl_vector_set (x_vec, i_x1, x1guess);
    gsl_vector_set (x_vec, i_x2, x2guess);

    T = get_fdfMultimin(pprec.ro_findCP2D_min.id);
    s = gsl_multimin_fdfminimizer_alloc(T, N_dims);

    first_step = pprec.ro_findCP2D_min.first_step;
    max_iter   = pprec.ro_findCP2D_min.max_iter;
    tol        = pprec.ro_findCP2D_min.tol;     // needs lower tol because we are minimizing the square
    epsabs     = pprec.ro_findCP2D_min.epsabs;

    gsl_multimin_fdfminimizer_set(s, &F, x_vec, first_step, tol);

    iter = 0;
    do
    {
        status = gsl_multimin_fdfminimizer_iterate(s);

        if(status)
            break;

        status = gsl_multimin_test_gradient(s->gradient, epsabs);

        // HVR_DEBUG
        //~ if (status == GSL_SUCCESS)
            //~ printf ("Minimum found at:\n");

        //~ printf ("%5d %.5f %.5f %10.5f \n", iter,
                                           //~ gsl_vector_get (s->x, i_x1),
                                           //~ gsl_vector_get (s->x, i_x2),
                                           //~ s->f);

        iter++;
    }
    while(status==GSL_CONTINUE && iter<max_iter);

    // initialize the crit point found
    p->point->x1 = gsl_vector_get(s->x, i_x1);
    p->point->x2 = gsl_vector_get(s->x, i_x2);
    if(status == GSL_SUCCESS)
        classify_CritPoint(p->point, p->y, p->Psi);
    else
        p->point->type = type_non_converged;

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x_vec);

    return 0;
}


// === Critical points using 2d root finding

int phi_grad(const gsl_vector *x, void *params, gsl_vector *f)
{
    double x1, x2;
    double d1, d2;
    double phi_derivs[N_derivs];
    pImage *p = (pImage *)params;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    phiFermat_1stDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];

    gsl_vector_set(f, i_x1, d1);
    gsl_vector_set(f, i_x2, d2);

    return GSL_SUCCESS;
}

int phi_hess(const gsl_vector *x, void *params, gsl_matrix *J)
{
    double x1, x2;
    double d11, d12, d22;
    double phi_derivs[N_derivs];
    pImage *p = (pImage *)params;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    phiFermat_2ndDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    d11 = phi_derivs[i_dx1dx1];
    d12 = phi_derivs[i_dx1dx2];
    d22 = phi_derivs[i_dx2dx2];

    gsl_matrix_set(J, i_x1, i_x1, d11);
    gsl_matrix_set(J, i_x1, i_x2, d12);
    gsl_matrix_set(J, i_x2, i_x1, d12);
    gsl_matrix_set(J, i_x2, i_x2, d22);

    return GSL_SUCCESS;
}

int phi_grad_hess(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J)
{
    double x1, x2;
    double d1, d2, d11, d12, d22;
    double phi_derivs[N_derivs];
    pImage *p = (pImage *)params;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    phiFermat_2ndDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];
    d11 = phi_derivs[i_dx1dx1];
    d12 = phi_derivs[i_dx1dx2];
    d22 = phi_derivs[i_dx2dx2];

    gsl_vector_set(f, i_x1, d1);
    gsl_vector_set(f, i_x2, d2);

    gsl_matrix_set(J, i_x1, i_x1, d11);
    gsl_matrix_set(J, i_x1, i_x2, d12);
    gsl_matrix_set(J, i_x2, i_x1, d12);
    gsl_matrix_set(J, i_x2, i_x2, d22);

    return GSL_SUCCESS;
}

int find_CritPoint_root_2D(double x1guess, double x2guess, pImage *p)
{
    int status;
    int iter, max_iter;
    double epsabs;
    //~ double epsrel;

    const gsl_multiroot_fdfsolver_type *T;
    gsl_multiroot_fdfsolver *s;
    gsl_vector *x = gsl_vector_alloc(N_dims);
    gsl_multiroot_function_fdf f = {phi_grad,
                                    phi_hess,
                                    phi_grad_hess,
                                    N_dims, p};


    gsl_vector_set(x, i_x1, x1guess);
    gsl_vector_set(x, i_x2, x2guess);

    T = get_fdfMultiroot(pprec.ro_findCP2D_root.id);
    s = gsl_multiroot_fdfsolver_alloc(T, N_dims);
    gsl_multiroot_fdfsolver_set(s, &f, x);

    epsabs = pprec.ro_findCP2D_root.epsabs;
    //~ epsrel = pprec.ro_findCP2D_root.epsrel;
    max_iter = pprec.ro_findCP2D_root.max_iter;

    iter = 0;
    do
    {
        status = gsl_multiroot_fdfsolver_iterate(s);

        if(status)  // solver stuck
            break;

        // HVR -> test changing this to test_delta
        status = gsl_multiroot_test_residual(s->f, epsabs);

        iter++;
    }
    while(status == GSL_CONTINUE && iter < max_iter);

    // HVR_DEBUG
    //~ printf ("status = %s    iter = %d\n", gsl_strerror (status), iter);

    // initialize the crit point found
    p->point->x1 = gsl_vector_get(s->x, i_x1);
    p->point->x2 = gsl_vector_get(s->x, i_x2);
    if(status == GSL_SUCCESS)
        classify_CritPoint(p->point, p->y, p->Psi);
    else
        p->point->type = type_non_converged;

    gsl_multiroot_fdfsolver_free(s);
    gsl_vector_free(x);

    return 0;
}


// === Generic functions to work with lists of CritPoint

void add_CritPoint(CritPoint *p_single, int *n_list, CritPoint *p_list)
{
    // add p_single in p_list at the position n_list if different
    // and converged and then increase n_list
    char is_repeated;
    int i;
    int n_points = *n_list;

    if(p_single->type != type_non_converged)
    {
        is_repeated = _FALSE_;
        for(i=0;i<n_points;i++)
        {
            if(is_same_CritPoint(p_list+i, p_single) == _TRUE_)
            {
                is_repeated = _TRUE_;
                break;
            }
        }

        if(is_repeated == _FALSE_)
        {
            copy_CritPoint(p_list+n_points, p_single);
            *n_list = n_points+1;
        }
    }
}

CritPoint *filter_CritPoint(int *n, CritPoint *p)
{
    // keep only different and converged points in p
    int i, n_new, n_old;
    CritPoint *p_new;

    n_old = *n;
    n_new = 0;

    p_new = (CritPoint *)malloc(n_old*sizeof(CritPoint));

    for(i=0;i<n_old;i++)
        add_CritPoint(p+i, &n_new, p_new);

    // realloc to the actual number of points
    *n = n_new;
    p_new = realloc(p_new, n_new*sizeof(CritPoint));

    sort_t_CritPoint(n_new, p_new);

    free(p);

    return p_new;
}

CritPoint *merge_CritPoint(int n1, CritPoint *p1, int n2, CritPoint *p2, int *n_points)
{
    // combine p1 and p2 into a single list, keeping only different points
    // allocate a new *p and free *p1 and *p2
    int i;
    CritPoint *p;

    *n_points = 0;
    p = (CritPoint *)malloc((n1+n2)*sizeof(CritPoint));

    for(i=0;i<n1;i++)
        add_CritPoint(p1+i, n_points, p);

    for(i=0;i<n2;i++)
        add_CritPoint(p2+i, n_points, p);

    // realloc to the actual number of points
    p = realloc(p, (*n_points)*sizeof(CritPoint));

    sort_t_CritPoint(*n_points, p);

    free(p1);
    free(p2);

    return p;
}


// === Find cusps and singularities

void fill_CritPoint_near_singcusp(CritPoint *p_root, CritPoint *p_singcusp, double y, Lens *Psi)
{
    char is_bracketed, sign;
    int i, iter, max_iter, n_angles;
    double R, R0, scale, dphi, alpha;
    double x10, x20, x1, x2, Dx1, Dx2;
    double phi_derivs[N_derivs];
    pImage p;

    R0 = pprec.ro_initcusp_R;
    n_angles = pprec.ro_initcusp_n;

    max_iter = pprec.ro_findnearCritPoint_max_iter;
    scale = pprec.ro_findnearCritPoint_scale;

    x10 = p_singcusp->x1;
    x20 = p_singcusp->x2;
    if(p_singcusp->type == type_singcusp_max)
        sign = 1;
    else if(p_singcusp->type == type_singcusp_min)
        sign = -1;
    else
        PERROR("singular point not recognized")

    iter = 0;
    is_bracketed = _FALSE_;
    while( (is_bracketed == _FALSE_) && (iter < max_iter) )
    {
        R = R0*scale;

        for(i=0;i<n_angles;i++)
        {
            alpha = i*M_2PI/n_angles;
            Dx1 = R*cos(alpha);
            Dx2 = R*sin(alpha);
            x1 = x10 + Dx1;
            x2 = x20 + Dx2;

            phiFermat_1stDeriv(phi_derivs, y, x1, x2, Psi);
            dphi = Dx1*phi_derivs[i_dx1] + Dx2*phi_derivs[i_dx2];

            if(SIGN(dphi) == sign)
            {
                is_bracketed = _TRUE_;
                break;
            }

        }

        R0 = R;
        iter ++;
    }

    // try to find crit point with this guess
    p.y = y;
    p.point = p_root;
    p.Psi = Psi;
    find_CritPoint_root_2D(x1, x2, &p);

    // HVR_DEBUG
    //~ printf(" --- Found point? (iter=%d/%d):  ", iter, max_iter);
    //~ display_CritPoint(p_root);
}

CritPoint *find_singcusp(int *n_singcusp, double y, Lens *Psi, pNamedLens *pNLens)
{
    char is_center;
    int i, j, n_angles, n_vec, n_points, n_sc;
    double x1, x2, x10, x20, Dx1, Dx2;
    double alpha, R, dphi_R, dphi_R_old=0;
    double phi_derivs[N_derivs];
    double *xvec;
    CritPoint *points;

    R = pprec.ro_initcusp_R;
    n_angles = pprec.ro_initcusp_n;

    // find number of tentative candidates (center of the lens
    // plus center of sublenses if we have a composite lens)
    n_sc = 0;
    n_vec = 0;
    xvec = get_cusp_sing(&n_vec, pNLens);

    // actual number of possible sing_cusps = n_vec/2
    // we will look for the first crit point near so we allocate
    // twice as many points
    n_points = n_vec/2;
    points = (CritPoint *)malloc(n_vec*sizeof(CritPoint));

    for(i=0;i<n_points;i++)
    {
        x10 = xvec[2*i];
        x20 = xvec[2*i+1];

        is_center = _TRUE_;
        for(j=0;j<n_angles;j++)
        {
            alpha = j*M_2PI/n_angles;
            Dx1 = R*cos(alpha);
            Dx2 = R*sin(alpha);
            x1 = x10 + Dx1;
            x2 = x20 + Dx2;

            phiFermat_1stDeriv(phi_derivs, y, x1, x2, Psi);
            dphi_R = Dx1*phi_derivs[i_dx1] + Dx2*phi_derivs[i_dx2];

            //~ printf("i=%d  j=%d   dphi_R=%e   dphi_old=%e\n", i, j, dphi_R, dphi_R_old);

            // initialize in the first step
            if( (SIGN(dphi_R) != SIGN(dphi_R_old)) && (j>0) )
            {
                is_center = _FALSE_;
                break;
            }
            else
                dphi_R_old = dphi_R;
        }
        //~ printf("i=%d  flag=%d\n", i, is_center);

        if(is_center == _TRUE_)
        {
            if(SIGN(dphi_R) == -1)
                points[n_sc].type = type_singcusp_max;
            if(SIGN(dphi_R) == 1)
                points[n_sc].type = type_singcusp_min;

            //~ printf("x10=%e  x20=%e\n", x10, x20);

            points[n_sc].t   = phiFermat(y, x10, x20, Psi);
            points[n_sc].mag = magnification(x10, x20, Psi);
            points[n_sc].x1  = x10;
            points[n_sc].x2  = x20;

            // finally try to find a crit point near the sing/cusp
            fill_CritPoint_near_singcusp(points+n_sc+1, points+n_sc, y, Psi);
            n_sc += 2;
        }
    }
    free(xvec);

    *n_singcusp = n_sc;

    return points;
}


// === High level functions to find crit points

CritPoint *find_first_CritPoints_2D(int *n_points, double y, Lens *Psi)
{
    int i, n_extra_points;
    double R_in, R_out;
    double R, th, x1guess, x2guess;
    double x_vec[N_dims];
    pImage pim;
    pTarget pt;
    CritPoint *p;

    R_in = pprec.ro_findfirstCP2D_Rin;
    R_out = pprec.ro_findfirstCP2D_Rout + y;
    n_extra_points = 20;

    *n_points = 2 + n_extra_points;
    p = (CritPoint *)malloc((*n_points)*sizeof(CritPoint));

    pt.y = y;
    pt.alpha = 0;
    pt.Psi = Psi;
    pt.x0_vec[i_x1] = 0;
    pt.x0_vec[i_x2] = 0;
    pt.T_func = dphisqr_func;
    pt.dT_func_dalpha = ddphisqr_func_dalpha;
    pt.dT_func_dR = ddphisqr_func_dR;

    // find and initialize the interior point
    pt.R = R_in;
    Tmin(&pt);

    x1x2_def(x_vec, pt.R, pt.alpha, pt.x0_vec);
    p[0].x1 = x_vec[i_x1];
    p[0].x2 = x_vec[i_x2];
    classify_CritPoint(p, y, Psi);

    // find and initialize the exterior point
    pt.R = R_out;
    Tmin(&pt);

    x1x2_def(x_vec, pt.R, pt.alpha, pt.x0_vec);
    p[1].x1 = x_vec[i_x1];
    p[1].x2 = x_vec[i_x2];
    classify_CritPoint(p+1, y, Psi);

    // for better precision, readjust the crit points found
    pim.y = y;
    pim.Psi = Psi;

    pim.point = p;
    find_CritPoint_root_2D(pim.point->x1, pim.point->x2, &pim);

    pim.point = p+1;
    find_CritPoint_root_2D(pim.point->x1, pim.point->x2, &pim);

    // try to find a few more points using a standard 2d (min) root finder
    for(i=0;i<n_extra_points;i++)
    {
        R = R_out;
        th = M_2PI*i/(n_extra_points-1);

        x1guess = R*cos(th);
        x2guess = R*sin(th);

        pim.point = p+i+2;
        find_CritPoint_min_2D(x1guess, x2guess, &pim);

        x1guess = pim.point->x1;
        x2guess = pim.point->x2;
        find_CritPoint_root_2D(x1guess, x2guess, &pim);
    }

    // HVR_DEBUG
    //~ for(i=0;i<*n_points;i++)
    //~ {
        //~ printf("R=%g,  ", sqrt((p+i)->x1*(p+i)->x1 + (p+i)->x2*(p+i)->x2));
        //~ display_CritPoint(p+i);
    //~ }

    return p;
}

int check_only_min_CritPoint_2D(CritPoint *p, double y, pNamedLens *pNLens)
{
    int n1, n2, n_points;
    CritPoint *p1, *p2;
    Lens Psi = init_lens(pNLens);

    // find first points and check for sing/cusps
    p1 = find_first_CritPoints_2D(&n1, y, &Psi);
    p2 = find_singcusp(&n2, y, &Psi, pNLens);

    // keep only different, converged and sort them by t
    p1 = merge_CritPoint(n1, p1, n2, p2, &n_points);

    // copy minimum t stored at p1[0]
    copy_CritPoint(p, p1);

    free(p1);

    return n_points;
}

CritPoint *find_all_CritPoints_2D(int *n_cpoints, double Rmin, double Rmax, double y, Lens *Psi,
                                  int (*find_CritPoint)(double x1, double x2, pImage *p))
{
    // find critical points throwing n_guesses between Rmin and Rmax
    int i, n_guesses;
    double R, th, x1guess, x2guess;
    CritPoint *ps;
    pImage p;

    const gsl_rng_type * T;
    gsl_rng * r;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    // alloc maximum number of images
    n_guesses = pprec.ro_findallCP2D_npoints;
    *n_cpoints = n_guesses;
    ps = (CritPoint *)malloc(n_guesses*sizeof(CritPoint));

    if(pprec.ro_findallCP2D_force_search == _TRUE_)
        Rmin = 0.;

    p.y = y;
    p.Psi = Psi;

    // throw random guesses between the two circles
    for(i=0;i<n_guesses;i++)
    {
        R = (Rmax-Rmin)*sqrt(gsl_rng_uniform(r)) + Rmin;
        th = M_2PI*gsl_rng_uniform(r);

        x1guess = R*cos(th);
        x2guess = R*sin(th);

        p.point = ps + i;
        find_CritPoint(x1guess, x2guess, &p);
    }

    gsl_rng_free(r);

    return ps;
}

CritPoint *find_all_CritPoints_min_2D(int *n_cpoints, double Rmin, double Rmax, double y, Lens *Psi)
{
    return find_all_CritPoints_2D(n_cpoints, Rmin, Rmax, y, Psi, find_CritPoint_min_2D);
}

CritPoint *find_all_CritPoints_root_2D(int *n_cpoints, double Rmin, double Rmax, double y, Lens *Psi)
{
    return find_all_CritPoints_2D(n_cpoints, Rmin, Rmax, y, Psi, find_CritPoint_root_2D);
}

CritPoint *driver_all_CritPoints_2D(int *n_cpoints, double y, pNamedLens *pNLens)
{
    int i, n, n_tmp;
    double R, Rmin, Rmax;
    CritPoint *ps, *ps_tmp;
    Lens Psi;

    Psi = init_lens(pNLens);

    // find first points and check for sing/cusps
    // keep only different, converged and sort them by t
    ps = find_first_CritPoints_2D(&n, y, &Psi);
    ps_tmp = find_singcusp(&n_tmp, y, &Psi, pNLens);
    ps = merge_CritPoint(n, ps, n_tmp, ps_tmp, &n);

    // find innermost and outermost points and throw guesses inside
    R = sqrt(ps[0].x1*ps[0].x1 + ps[0].x2*ps[0].x2);
    Rmin = R;
    Rmax = R;
    for(i=1;i<n;i++)
    {
        R = sqrt(ps[i].x1*ps[i].x1 + ps[i].x2*ps[i].x2);

        if(R < Rmin)
            Rmin = R;

        if(R > Rmax)
            Rmax = R;
    }

    ps_tmp = find_all_CritPoints_root_2D(&n_tmp, Rmin, Rmax, y, &Psi);
    ps = merge_CritPoint(n, ps, n_tmp, ps_tmp, n_cpoints);

    return ps;
}


// ======  Direct 2d minimization
// =================================================================

double phi_f_Minimum_2D(const gsl_vector *x, void *params)
{
    double x1, x2;
    pImage *p = (pImage *)params;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    return phiFermat(p->y, x1, x2, p->Psi);
}

void phi_df_Minimum_2D(const gsl_vector *x, void *params, gsl_vector *df)
{
    double x1, x2;
    double d1, d2;
    double phi_derivs[N_derivs];
    pImage *p = (pImage *)params;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    phiFermat_1stDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];

    gsl_vector_set(df, i_x1, d1);
    gsl_vector_set(df, i_x2, d2);
}

void phi_fdf_Minimum_2D(const gsl_vector *x, void *params, double *f, gsl_vector *df)
{
    double x1, x2;
    double phi, d1, d2;
    double phi_derivs[N_derivs];
    pImage *p = (pImage *)params;

    x1 = gsl_vector_get(x, i_x1);
    x2 = gsl_vector_get(x, i_x2);

    phiFermat_1stDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    phi = phi_derivs[i_0];
    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];

    *f = phi;
    gsl_vector_set(df, i_x1, d1);
    gsl_vector_set(df, i_x2, d2);
}

int find_local_Minimum_2D(double x1guess, double x2guess, pImage *p)
{
    int iter, status, max_iter;
    double first_step, tol, epsabs;

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;
    gsl_vector *x_vec;
    gsl_multimin_function_fdf F;

    F.n = N_dims;
    F.f = phi_f_Minimum_2D;
    F.df = phi_df_Minimum_2D;
    F.fdf = phi_fdf_Minimum_2D;
    F.params = p;

    x_vec = gsl_vector_alloc(N_dims);
    gsl_vector_set(x_vec, i_x1, x1guess);
    gsl_vector_set(x_vec, i_x2, x2guess);

    T = get_fdfMultimin(pprec.ro_findlocMin2D.id);
    s = gsl_multimin_fdfminimizer_alloc(T, N_dims);

    max_iter   = pprec.ro_findlocMin2D.max_iter;
    first_step = pprec.ro_findlocMin2D.first_step;
    tol        = pprec.ro_findlocMin2D.tol;
    epsabs     = pprec.ro_findlocMin2D.epsabs;

    gsl_multimin_fdfminimizer_set(s, &F, x_vec, first_step, tol);

    iter = 0;
    do
    {
        status = gsl_multimin_fdfminimizer_iterate(s);

        if(status)
            break;

        status = gsl_multimin_test_gradient(s->gradient, epsabs);

        //~ if (status == GSL_SUCCESS)
            //~ printf ("Minimum found at:\n");

        //~ printf ("%5d %.5f %.5f %10.5f \n", iter,
                                           //~ gsl_vector_get (s->x, i_x1),
                                           //~ gsl_vector_get (s->x, i_x2),
                                           //~ s->f);

        iter++;
    }
    while(status==GSL_CONTINUE && iter<max_iter);

    p->point->x1 = gsl_vector_get(s->x, i_x1);
    p->point->x2 = gsl_vector_get(s->x, i_x2);
    p->point->t  = s->f;

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x_vec);

    return 0;
}

int find_global_Minimum_2D(pImage *p)
{
    int i, n_points, n_guesses;
    double R, alpha, Rmax, Rmin;
    double x1guess, x2guess;
    CritPoint pmin;
    CritPoint *ps;

    const gsl_rng_type * T;
    gsl_rng * r;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    n_guesses = pprec.ro_findglobMin2D_nguesses;

    // we use the first critical points as a guide to restrict the search
    ps = find_first_CritPoints_2D(&n_points, p->y, p->Psi);
    ps = filter_CritPoint(&n_points, ps);

    // first guess is the minimum ps[0] from the critical points
    copy_CritPoint(&pmin, ps);

    // if there is only a minimum, skip the rest
    if( (n_points != 1) || (ps[0].type != type_min) )
    {
        // find innermost and outermost points and throw guesses inside
        R = sqrt(ps[0].x1*ps[0].x1 + ps[0].x2*ps[0].x2);
        Rmin = R;
        Rmax = R;
        for(i=1;i<n_points;i++)
        {
            R = sqrt(ps[i].x1*ps[i].x1 + ps[i].x2*ps[i].x2);

            if(R < Rmin)
                Rmin = R;

            if(R > Rmax)
                Rmax = R;
        }

        // throw random points and try to improve the initial tmin
        for(i=0;i<n_guesses;i++)
        {
            R = (Rmax-Rmin)*sqrt(gsl_rng_uniform(r)) + Rmin;
            alpha = M_2PI*gsl_rng_uniform(r);

            x1guess = R*cos(alpha);
            x2guess = R*sin(alpha);

            find_local_Minimum_2D(x1guess, x2guess, p);
            //~ display_CritPoint(p->point);

            if(p->point->t < pmin.t)
                copy_CritPoint(&pmin, p->point);
        }
    }

    copy_CritPoint(p->point, &pmin);
    classify_CritPoint(p->point, p->y, p->Psi);

    gsl_rng_free(r);
    free(ps);

    return 0;
}
