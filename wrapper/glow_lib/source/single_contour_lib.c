/*
 * GLoW - single_contour_lib.c
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
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_roots.h>

#include "common.h"
#include "ode_tools.h"
#include "lenses_lib.h"
#include "single_contour_lib.h"

#define EPS_TOL 1e-12

// =================================================================

// ======   find R(tau) inverting tau(R)
// =================================================================

double f_root_R_of_tau(double R, void *params)
{
    double phi_0, phi_R;
    double x_vec[N_dims];
    pRoot *p = (pRoot *)params;

    x1x2_def(x_vec, R, 0, p->p->x0_vec);
    phi_R = phiFermat(p->p->y, x_vec[i_x1], x_vec[i_x2], p->p->Psi);

    x1x2_def(x_vec, 0, 0, p->p->x0_vec);
    phi_0 = phiFermat(p->p->y, x_vec[i_x1], x_vec[i_x2], p->p->Psi);

    return phi_R-phi_0-p->tau;
}

double df_root_R_of_tau(double R, void *params)
{
    double dphi_R, Dx1, Dx2;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    pRoot *p = (pRoot *)params;

    x1x2_def(x_vec, R, 0, p->p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->p->y, x_vec[i_x1], x_vec[i_x2], p->p->Psi);

    Dx1 = x_vec[i_x1] - p->p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->p->x0_vec[i_x2];
    dphi_R = Dx1/R*phi_derivs[i_dx1] + Dx2/R*phi_derivs[i_dx2];

    return dphi_R;
}

void fdf_root_R_of_tau(double R, void *params, double *f, double *df)
{
    double phi_0, phi_R;
    double dphi_R, Dx1, Dx2;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    pRoot *p = (pRoot *)params;

    x1x2_def(x_vec, 0, 0, p->p->x0_vec);
    phi_0 = phiFermat(p->p->y, x_vec[i_x1], x_vec[i_x2], p->p->Psi);

    x1x2_def(x_vec, R, 0, p->p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->p->y, x_vec[i_x1], x_vec[i_x2], p->p->Psi);

    phi_R = phi_derivs[i_0];

    Dx1 = x_vec[i_x1] - p->p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->p->x0_vec[i_x2];
    dphi_R = Dx1/R*phi_derivs[i_dx1] + Dx2/R*phi_derivs[i_dx2];

    *f = phi_R-phi_0-p->tau;
    *df = dphi_R;
}

double find_R_of_tau(double tau, pIntegral *p)
{
    int status, iter, max_iter;
    double R0, Dx1, epsabs, epsrel, R;
    pRoot params;
    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;
    gsl_function_fdf FDF;

    max_iter = pprec.sc_findRtau.max_iter;
    epsabs   = pprec.sc_findRtau.epsabs;
    epsrel   = pprec.sc_findRtau.epsrel;

    params.tau = tau;
    params.p = p;

    FDF.f = f_root_R_of_tau;
    FDF.df = df_root_R_of_tau;
    FDF.fdf = fdf_root_R_of_tau;
    FDF.params = &params;

    T = get_fdfRoot(pprec.sc_findRtau.id);
    s = gsl_root_fdfsolver_alloc(T);

    Dx1 = p->x0_vec[i_x1]-p->y;

    R = sqrt(2*tau + Dx1*Dx1) - Dx1;  // initial guess
    gsl_root_fdfsolver_set(s, &FDF, R);

    iter=0;
    do
    {
        iter++;
        status = gsl_root_fdfsolver_iterate(s);

        R0 = R;  // previous point
        R = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(R, R0, epsabs, epsrel);

        // if(status == GSL_SUCCESS)
        //     printf("Converged  iter=%d  (tau=%e,    R=%e)\n", iter, tau, R);
    }
    while(status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fdfsolver_free(s);

    return R;
}

int find_all_R_of_tau(int n_points, double *R_grid, double *tau_grid, pIntegral *p)
{
    int i, status, iter, max_iter;
    double R0, Dx1, epsabs, epsrel;
    pRoot params;
    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;
    gsl_function_fdf FDF;

    max_iter = pprec.sc_findRtau.max_iter;
    epsabs   = pprec.sc_findRtau.epsabs;
    epsrel   = pprec.sc_findRtau.epsrel;

    params.tau = EPS_TOL;
    params.p = p;

    FDF.f = f_root_R_of_tau;
    FDF.df = df_root_R_of_tau;
    FDF.fdf = fdf_root_R_of_tau;
    FDF.params = &params;

    T = get_fdfRoot(pprec.sc_findRtau.id);
    s = gsl_root_fdfsolver_alloc(T);

    Dx1 = p->x0_vec[i_x1]-p->y;
    for(i=0;i<n_points;i++)
    {
        params.tau = tau_grid[i];
        R_grid[i] = sqrt(2*tau_grid[i] + Dx1*Dx1) - Dx1;  // initial guess
        gsl_root_fdfsolver_set(s, &FDF, R_grid[i]);

        iter=0;
        do
        {
            iter++;
            status = gsl_root_fdfsolver_iterate(s);

            R0 = R_grid[i];  // previous point
            R_grid[i] = gsl_root_fdfsolver_root(s);
            status = gsl_root_test_delta(R_grid[i], R0, epsabs, epsrel);

            //~ if(status == GSL_SUCCESS)
                //~ printf("Converged  iter=%d  (tau=%e,    R=%e)\n", iter, tau_grid[i], R_grid[i]);
        }
        while(status == GSL_CONTINUE && iter < max_iter);
    }

    gsl_root_fdfsolver_free(s);
    return 0;
}

int driver_R_of_tau(int n_points, double *R_grid, double *tau_grid,
                   double x1_min, double x2_min, double y,
                   pNamedLens *pNLens)
{
    Lens Psi;
    pIntegral params;

    Psi = init_lens(pNLens);

    params.y = y;
    params.x0_vec[i_x1] = x1_min;
    params.x0_vec[i_x2] = x2_min;
    params.Psi = &Psi;

    find_all_R_of_tau(n_points, R_grid, tau_grid, &params);

    return 0;
}

double find_R_of_tau_bracket(double tau, double R_hi, pIntegral *p)
{
    int status, iter, max_iter;
    double r, R_lo, f_hi;
    double epsabs, epsrel;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    gsl_function F;
    pRoot pR;

    pR.tau = tau;
    pR.p = p;

    // precision parameters
    epsabs = pprec.sc_findRtau_bracket.epsabs;
    epsrel = pprec.sc_findRtau_bracket.epsrel;
    max_iter = pprec.sc_findRtau_bracket.max_iter;
    T = get_fRoot(pprec.sc_findRtau_bracket.id);

    // lower bracket
    R_lo = EPS_TOL;

    F.function = f_root_R_of_tau;
    F.params = &pR;

    // check bracket
    f_hi = F.function(R_hi, F.params);
    if(f_hi < 0)
    {
        PWARNING("wrong bracket. Finding new bracket");

        iter = 0;
        while( (f_hi < 0) && (iter < max_iter) )
        {
            R_hi *= 2.;
            f_hi = F.function(R_hi, F.params);
            iter++;
        }

        PWARNING(" -> bracket found [%g, %g]", R_lo, R_hi)
    }

    s = gsl_root_fsolver_alloc(T);
    gsl_root_fsolver_set(s, &F, R_lo, R_hi);

    iter = 0;
    do
    {
        status = gsl_root_fsolver_iterate(s);
        r = gsl_root_fsolver_root(s);
        R_lo = gsl_root_fsolver_x_lower(s);
        R_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(R_lo, R_hi, epsabs, epsrel);

        iter++;
    }
    while(status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free (s);

    return r;
}

int find_all_R_of_tau_bracket(int n_points, double *R_grid, double *tau_grid, pIntegral *p)
{
    int i;

    // find the first one using Newton's method
    R_grid[n_points-1] = find_R_of_tau(tau_grid[n_points-1], p);

    // find the first one using the same bracketing
    //~ R_grid[n_points-1] = find_R_of_tau_bracket(tau_grid[n_points-1], 100, p);

    // if taus are sorted this reduces the number of iterations
    for(i=0;i<n_points-1;i++)
        R_grid[n_points-2-i] = find_R_of_tau_bracket(tau_grid[n_points-2-i], R_grid[n_points-1-i], p);

    return 0;
}

int driver_R_of_tau_bracket(int n_points, double *R_grid, double *tau_grid,
                            double x1_min, double x2_min, double y,
                            pNamedLens *pNLens)
{
    Lens Psi;
    pIntegral params;

    Psi = init_lens(pNLens);

    params.y = y;
    params.x0_vec[i_x1] = x1_min;
    params.x0_vec[i_x2] = x2_min;
    params.Psi = &Psi;

    find_all_R_of_tau_bracket(n_points, R_grid, tau_grid, &params);

    return 0;
}


// ======   find R(tau) solving the differential equation
// =================================================================

int dR_dtau(double t, const double u[], double f[], void *params)
{
    double R;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double dphi_R, Dx1, Dx2;
    pIntegral *p = (pIntegral *)params;

    R = u[0];

    x1x2_def(x_vec, R, 0, p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];

    dphi_R = (Dx1*phi_derivs[i_dx1] + Dx2*phi_derivs[i_dx2])/R;

    if(ABS(dphi_R) < EPS_TOL)
    {
        PERROR("dphi_dR < %g  at  x=(%g, %g)", EPS_TOL, x_vec[i_x1], x_vec[i_x2])
        return GSL_EBADFUNC;
    }

    f[0] = 1./dphi_R;

    return GSL_SUCCESS;
}

int integrate_dR_dtau(int n_points, double *R_grid, double *tau_grid, pIntegral *p)
{
    int i, status;
    double R, tau, tau_f;;
    double epsabs, epsrel, h;
    gsl_odeiv2_system sys = {dR_dtau, NULL, 1, p};
    gsl_odeiv2_driver *d;
    const gsl_odeiv2_step_type *T;

    h = pprec.sc_intdRdtau.h;
    epsabs = pprec.sc_intdRdtau.epsabs;
    epsrel = pprec.sc_intdRdtau.epsrel;
    T = get_stepODE(pprec.sc_intdRdtau.id);

    d = gsl_odeiv2_driver_alloc_y_new (&sys, T, h, epsabs, epsrel);

    // Initial conditions
    R = pprec.sc_intdRdtau_R0;
    tau = R*(p->x0_vec[i_x1]-p->y);
    for(i=0;i<n_points;i++)
    {
        tau_f = tau_grid[i];
        status = gsl_odeiv2_driver_apply (d, &tau, tau_f, &R);

        if(status != GSL_SUCCESS)
            PERROR("integration of dR/dtau failed (%s)", gsl_strerror(status))

        R_grid[i] = R;
    }

    gsl_odeiv2_driver_free (d);

    return 0;
}

int driver_dR_dtau(int n_points, double *R_grid, double *tau_grid,
                   double x1_min, double x2_min, double y,
                   pNamedLens *pNLens)
{
    Lens Psi;
    pIntegral params;

    Psi = init_lens(pNLens);

    params.y = y;
    params.x0_vec[i_x1] = x1_min;
    params.x0_vec[i_x2] = x2_min;
    params.Psi = &Psi;

    integrate_dR_dtau(n_points, R_grid, tau_grid, &params);

    return 0;
}


// == integrate the contour, assuming only one critical point (i.e. minimum)
// ===========================================================================

int system_contour(double t, const double u[], double f[], void *params)
{
    double alpha, R;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double dphi_R, dphi_alpha, Dx1, Dx2;
    pIntegral *p = (pIntegral *)params;

    alpha = t;
    R = u[i_R];

    x1x2_def(x_vec, R, alpha, p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];

    dphi_R = (Dx1*phi_derivs[i_dx1] + Dx2*phi_derivs[i_dx2])/R;
    dphi_alpha = -Dx2*phi_derivs[i_dx1] + Dx1*phi_derivs[i_dx2];

    if(dphi_R < pprec.sc_syscontour_eps)
    {
        if(pprec.sc_warn_switch == _TRUE_)
            PWARNING("dphi_dR=%g < %g at x=(%g, %g)", dphi_R, pprec.sc_syscontour_eps, x_vec[i_x1], x_vec[i_x2])
        return GSL_EBADFUNC;
    }

    f[i_R] = -dphi_alpha/dphi_R;
    f[i_I] = R/dphi_R;

    return GSL_SUCCESS;
}

int integrate_contour(double R_ini, double *R_f, double *I, pIntegral *p)
{
    int status;
    double u[N_eqs-1];
    double epsabs, epsrel, alpha, h;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_driver *d;
    gsl_odeiv2_system sys = {system_contour, NULL, N_eqs-1, p};

    // stepper and tolerance
    h = pprec.sc_intContourStd.h;
    epsabs = pprec.sc_intContourStd.epsabs;
    epsrel = pprec.sc_intContourStd.epsrel;
    T = get_stepODE(pprec.sc_intContourStd.id);

    // initialize the driver
    d = gsl_odeiv2_driver_alloc_y_new(&sys, T, h, epsabs, epsrel);

    // initial conditions
    u[i_R] = R_ini;
    u[i_I] = 0;

    alpha = 0;
    status = gsl_odeiv2_driver_apply (d, &alpha, 2*M_PI, u);

    if (status != GSL_SUCCESS)
        if(pprec.sc_warn_switch == _TRUE_)
            PWARNING("standard single contour method failed, switching to robust")

    *I = u[i_I];
    *R_f = u[i_R];

    gsl_odeiv2_driver_free (d);

    return status;
}


// == integrate a single contour (robust version)
// ===========================================================================

double reach_2pi_contour(const double y[], const double dydt[], void *pCond)
{
    return 0.5*y[i_alpha]/M_PI - 1.;
}

char is_closed_contour(const double y[], const double dydt[], void *pCond)
{
    pCondODE *pc = (pCondODE *)pCond;
    pIntegral *p = (pIntegral *)pc->params;

    if( ABS(y[i_R]/p->R_ini - 1.) < pc->tol_add )
        return _TRUE_;
    else
        return _FALSE_;
}

int system_contour_robust(double t, const double u[], double f[], void *params)
{
    int status;
    double alpha, R;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double dphi_R, dphi_alpha, Dx1, Dx2;
    pIntegral *p = (pIntegral *)params;

    alpha = u[i_alpha];
    R = u[i_R];

    x1x2_def(x_vec, R, alpha, p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];

    dphi_R = (Dx1*phi_derivs[i_dx1] + Dx2*phi_derivs[i_dx2])/R;
    dphi_alpha = -Dx2*phi_derivs[i_dx1] + Dx1*phi_derivs[i_dx2];

    f[i_R] = -dphi_alpha;
    f[i_I] = R;
    f[i_alpha] = dphi_R;

    // check whether to end integration
    status = check_pCond(p->pCond);

    // safeguard condition
    if(ABS(alpha) > 100)
        status = GSL_FAILURE;

    return status;
}

int integrate_contour_robust(double R_ini, double *R_f, double *I, pIntegral *p)
{
    int status;
    double u[N_eqs];
    double h, sigma, sigmaf;
    double epsabs, epsrel;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_driver *d;
    gsl_odeiv2_system sys;

    // define system
    sys.function = system_contour_robust;
    sys.jacobian = NULL;
    sys.dimension = N_eqs;
    sys.params = p;

    // stepper and precision
    h = pprec.sc_intContourRob.h;
    epsabs = pprec.sc_intContourRob.epsabs;
    epsrel = pprec.sc_intContourRob.epsrel;
    T = get_stepODE(pprec.sc_intContourRob.id);
    sigmaf = pprec.sc_intContourRob_sigmaf;

    // initial conditions
    u[i_R] = R_ini;
    u[i_I] = 0;
    u[i_alpha] = 0;
    sigma = 0;

    // allocate driver
    d = gsl_odeiv2_driver_alloc_conditional_new(&sys, T, h, epsabs, epsrel, p->pCond);

    // integrate
    status = gsl_odeiv2_driver_apply(d, &sigma, sigmaf, u);

    if( (status != GSL_SUCCESS) && (status != ODE_COND_MET) )
        PERROR("integration failed (%s)", gsl_strerror(status))

    if(status != ODE_COND_MET)
        PWARNING("ending condition not met in contour")

    *I = u[i_I];
    *R_f = u[i_R];

    gsl_odeiv2_driver_free (d);

    return status;
}


// == driver for contour integration
// ===========================================================================

double driver_contour(double tau, double x1_min, double x2_min, double y,
                      pNamedLens *pNLens, int method)
{
    int status;
    double R, I, R_ini;
    double tau_real, tau_min, I0;
    Lens Psi = init_lens(pNLens);
    pIntegral params;
    pCondODE *pc;

    // built-in step function
    if(tau < 0)
        return 0;

    // if tau is too small, we interpolate between 0 and tau_min
    tau_real = tau;
    tau_min = pprec.sc_drivContour_taumin_over_y2*y*y;
    if(tau < tau_min)
        tau = tau_min;

    params.y = y;
    params.x0_vec[i_x1] = x1_min;
    params.x0_vec[i_x2] = x2_min;
    params.Psi = &Psi;

    R_ini = find_R_of_tau(tau, &params);
    params.R_ini = R_ini;

    // Sometimes the Newton algorithm jumps too much and goes to
    // negative R (which makes sense in terms of x1, x2 with alpha=pi)
    // The computation of I(tau) then gives nonsensical results.
    // We avoid this by using a bracketing algorithm
    if(R_ini < 0)
    {
        R_ini = find_R_of_tau_bracket(tau, 2*ABS(R_ini), &params);
        params.R_ini = R_ini;
    }

    // try the standard method first
    status = GSL_SUCCESS;
    if(method == m_contour_std)
    {
        params.pCond = NULL;
        status = integrate_contour(R_ini, &R, &I, &params);
    }

    // if the standard method fails, try the robust one
    if( (method == m_contour_robust) || (status != GSL_SUCCESS))
    {
        pc = init_pCondODE();
        pc->params = &params;
        params.pCond = pc;

        pc->brack_cond = reach_2pi_contour;
        pc->add_cond = is_closed_contour;

        pc->tol_brack = pprec.sc_intContour_tol_brack;
        pc->tol_add = pprec.sc_intContour_tol_add;

        integrate_contour_robust(R_ini, &R, &I, &params);

        free_pCondODE(pc);
    }

    // linear interpolation when tau is very small
    if(tau_real < tau_min)
    {
        I0 = M_2PI*sqrt(magnification(x1_min, x2_min, &Psi));
        I = I + (I0 - I)*(1 - tau_real/tau_min);
    }

    return I;
}


// == get contours
// ===========================================================================

int get_contour_points(double R_ini, SolODE *sol, pIntegral *p)
{
    int status;
    double u[N_eqs-1], contour[4];
    double epsabs, epsrel, alpha, h;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_step *s;
    gsl_odeiv2_control *c;
    gsl_odeiv2_evolve *e;
    gsl_odeiv2_system sys = {system_contour, NULL, N_eqs-1, p};

    // stepper and tolerance
    h = pprec.sc_getContourStd.h;
    epsabs = pprec.sc_getContourStd.epsabs;
    epsrel = pprec.sc_getContourStd.epsrel;
    T = get_stepODE(pprec.sc_getContourStd.id);

    s = gsl_odeiv2_step_alloc(T, N_eqs-1);
    c = gsl_odeiv2_control_y_new(epsabs, epsrel);
    e = gsl_odeiv2_evolve_alloc(N_eqs-1);

    // initial conditions
    u[i_R] = R_ini;
    u[i_I] = 0;

    alpha = 0;
    while(alpha < 2*M_PI)
    {
        status = gsl_odeiv2_evolve_apply (e, c, s,
                                          &sys,
                                          &alpha, 2*M_PI,
                                          &h, u);

        if(status != GSL_SUCCESS)
        {
            PERROR("couldn't get contour points with standard method (try robust) (%s)", gsl_strerror(status))
            break;
        }

        contour[0] = alpha;
        contour[1] = u[i_R];

        // fill contour[2], contour[3] with x1, x2
        x1x2_def(contour+2, u[i_R], alpha, p->x0_vec);

        fill_SolODE(alpha, contour, sol);
    }

    gsl_odeiv2_step_free(s);
    gsl_odeiv2_control_free(c);
    gsl_odeiv2_evolve_free(e);

    return status;
}

int get_contour(double R_ini, int n_points, SolODE *sol, pIntegral *p)
{
    int i, status;
    double u[N_eqs-1];
    double alpha, dalpha, contour[4];
    double epsabs, epsrel, h;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_driver *d;
    gsl_odeiv2_system sys = {system_contour, NULL, N_eqs-1, p};

    // stepper and tolerance
    h = pprec.sc_getContourStd.h;
    epsabs = pprec.sc_getContourStd.epsabs;
    epsrel = pprec.sc_getContourStd.epsrel;
    T = get_stepODE(pprec.sc_getContourStd.id);

    // initialize the driver
    d = gsl_odeiv2_driver_alloc_y_new(&sys, T, h, epsabs, epsrel);

    // initial conditions
    u[i_R] = R_ini;
    u[i_I] = 0;

    // store first point
    contour[0] = 0;
    contour[1] = R_ini;
    x1x2_def(contour+2, R_ini, 0, p->x0_vec);
    fill_SolODE(0, contour, sol);

    dalpha = 2*M_PI/(n_points-1);
    for(i=0;i<(n_points-1);i++)
    {
        alpha = i*dalpha;
        status = gsl_odeiv2_driver_apply(d, &alpha, alpha + dalpha, u);

        contour[0] = alpha;
        contour[1] = u[i_R];

        // fill contour[2], contour[3] with x1, x2
        x1x2_def(contour+2, u[i_R], alpha, p->x0_vec);

        fill_SolODE(alpha, contour, sol);

        if (status != GSL_SUCCESS)
            PERROR("couldn't get contour points with standard method (try robust) (%s)", gsl_strerror(status))
    }

    gsl_odeiv2_driver_free (d);

    return status;
}

int get_contour_points_robust(double R_ini, SolODE *sol, pIntegral *p)
{
    int status;
    double u[N_eqs], contour[4];
    double epsabs, epsrel, h, sigma, sigmaf;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_step *s;
    gsl_odeiv2_control *c;
    gsl_odeiv2_evolve *e;
    gsl_odeiv2_system sys = {system_contour_robust, NULL, N_eqs, p};

    // stepper and tolerance
    h = pprec.sc_getContourRob.h;
    epsabs = pprec.sc_getContourRob.epsabs;
    epsrel = pprec.sc_getContourRob.epsrel;
    T = get_stepODE(pprec.sc_getContourRob.id);
    sigmaf = pprec.sc_getContourRob_sigmaf;

    s = gsl_odeiv2_step_alloc(T, N_eqs);
    c = gsl_odeiv2_control_conditional_new(epsabs, epsrel, p->pCond);
    e = gsl_odeiv2_evolve_alloc(N_eqs);

    // initial conditions
    u[i_R] = R_ini;
    u[i_I] = 0;
    u[i_alpha] = 0;
    sigma = 0;

    while(sigma < sigmaf)
    {
        status = gsl_odeiv2_evolve_apply (e, c, s,
                                          &sys,
                                          &sigma, sigmaf,
                                          &h, u);

        if( (status != GSL_SUCCESS) && (status != ODE_COND_MET) )
        {
            PERROR("couldn't get contour points with robust method (%s)", gsl_strerror(status))
            break;
        }

        contour[0] = u[i_alpha];
        contour[1] = u[i_R];

        // fill contour[2], contour[3] with x1, x2
        x1x2_def(contour+2, u[i_R], u[i_alpha], p->x0_vec);

        fill_SolODE(sigma, contour, sol);

        if(status == ODE_COND_MET)
            break;
    }

    gsl_odeiv2_step_free(s);
    gsl_odeiv2_control_free(c);
    gsl_odeiv2_evolve_free(e);

    return 0;
}

int get_contour_robust(double R_ini, int n_points, SolODE *sol, pIntegral *p)
{
    int i, status;
    double u[N_eqs];
    double sigmaf, sigma, dsigma, contour[4];
    double epsabs, epsrel, h;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_driver *d;
    gsl_odeiv2_system sys = {system_contour_robust, NULL, N_eqs, p};
    pCondODE *pCond = (pCondODE *)p->pCond;

    // stepper and tolerance
    h = pprec.sc_getContourRob.h;
    epsabs = pprec.sc_getContourRob.epsabs;
    epsrel = pprec.sc_getContourRob.epsrel;
    T = get_stepODE(pprec.sc_getContourRob.id);
    sigmaf = pprec.sc_getContourRob_sigmaf;

    // initialize the driver
    d = gsl_odeiv2_driver_alloc_conditional_new(&sys, T, h, epsabs, epsrel, p->pCond);

    // initial conditions
    u[i_R] = R_ini;
    u[i_I] = 0;
    u[i_alpha] = 0;
    sigma = 0;

    // store first point
    contour[0] = u[i_alpha];
    contour[1] = u[i_R];
    x1x2_def(contour+2, u[i_R], u[i_alpha], p->x0_vec);
    fill_SolODE(sigma, contour, sol);

    // find final sigma integrating everything
    status = gsl_odeiv2_driver_apply(d, &sigma, sigmaf, u);

    if( (status != GSL_SUCCESS) && (status != ODE_COND_MET) )
        PERROR("couldn't get contour points with robust method (%s)", gsl_strerror(status))
    else
        sigmaf = sigma;

    // restart the driver
    gsl_odeiv2_driver_free(d);
    d = gsl_odeiv2_driver_alloc_y_new(&sys, T, h, epsabs, epsrel);

    // restart the initial conditions
    pCond->cond_met = _FALSE_;
    u[i_R] = R_ini;
    u[i_I] = 0;
    u[i_alpha] = 0;

    // finally, store the grid
    dsigma = sigmaf/(n_points-1);
    for(i=0;i<(n_points-1);i++)
    {
        sigma = i*dsigma;
        status = gsl_odeiv2_driver_apply(d, &sigma, sigma + dsigma, u);

        contour[0] = u[i_alpha];
        contour[1] = u[i_R];

        // fill contour[2], contour[3] with x1, x2
        x1x2_def(contour+2, u[i_R], u[i_alpha], p->x0_vec);
        fill_SolODE(sigma, contour, sol);

        if( (status != GSL_SUCCESS) && (status != ODE_COND_MET))
            PERROR("couldn't get contour points with robust method (%s)", gsl_strerror(status))
    }

    gsl_odeiv2_driver_free(d);

    return 0;
}

int driver_get_contour(double tau, int n_points,
                       double x1_min, double x2_min, double y,
                       pNamedLens *pNLens, int method,
                       SolODE *sol)
{
    double R_ini;
    Lens Psi = init_lens(pNLens);
    pIntegral params;
    pCondODE *pc;

    // HVR -> improve this
    // built-in step function
    //~ if(tau < 0)
        // do something here

    //~ if(tau < EPS_TAU_STEP)
        // do something approximate here

    params.y = y;
    params.x0_vec[i_x1] = x1_min;
    params.x0_vec[i_x2] = x2_min;
    params.Psi = &Psi;

    R_ini = find_R_of_tau(tau, &params);
    params.R_ini = R_ini;

    if(method == m_contour_std)
    {
        params.pCond = NULL;

        if(n_points > 1)
            get_contour(R_ini, n_points, sol, &params);
        else
            get_contour_points(R_ini, sol, &params);

    }
    else
    {
        pc = init_pCondODE();
        pc->params = &params;
        params.pCond = pc;

        pc->brack_cond = reach_2pi_contour;
        pc->add_cond = is_closed_contour;

        pc->tol_brack = pprec.sc_getContour_tol_brack;
        pc->tol_add = pprec.sc_getContour_tol_add;

        if(n_points > 1)
            get_contour_robust(R_ini, n_points, sol, &params);
        else
            get_contour_points_robust(R_ini, sol, &params);

        free_pCondODE(pc);
    }

    return 0;
}
