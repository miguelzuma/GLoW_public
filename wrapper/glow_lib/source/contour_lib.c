/*
 * GLoW - contour_lib.c
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
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_multimin.h>

#include "common.h"
#include "ode_tools.h"
#include "lenses_lib.h"
#include "roots_lib.h"
#include "contour_lib.h"

#define EPS_TOL 1e-12

// =====================================================================


// ======  Create and initialize the centers from the crit points
// =====================================================================

Center *init_all_Center(int *n_centers, CritPoint *points, double y, pNamedLens *pNLens)
{
    int i, n_ctr;
    Center *ctrs;
    pIntegral2d p;
    Lens Psi = init_lens(pNLens);

    // init pIntegral2d first
    p.y = y;
    p.tmin = points[0].t;
    p.Psi = &Psi;
    // ----------------------------------

    n_ctr = *n_centers; // shorthand

    if(n_ctr%2 == 0)
        PERROR("even number of centers found (n=%d)", n_ctr)

    ctrs = (Center *)malloc(n_ctr*sizeof(Center));

    for(i=0;i<n_ctr;i++)
    {
        // generic initialization for all the centers
        ctrs[i].tau0 = points[i].t - p.tmin;
        ctrs[i].t0   = points[i].t;
        ctrs[i].x0[i_x1] = points[i].x1;
        ctrs[i].x0[i_x2] = points[i].x2;
        ctrs[i].is_init_birthdeath = _FALSE_;
        ctrs[i].tau_birth = 0;
        ctrs[i].tau_death = 0;
        ctrs[i].R_max = 0;

        // specific now for each type
        if(points[i].type == type_saddle)
            fill_saddle_Center(ctrs+i, &p);
        else
        {
            if(points[i].type == type_min)
                ctrs[i].type = ctr_type_min;
            else if(points[i].type == type_max)
                ctrs[i].type = ctr_type_max;
            else if(points[i].type == type_singcusp_min)
                ctrs[i].type = ctr_type_min;
            else if(points[i].type == type_singcusp_max)
                ctrs[i].type = ctr_type_max;
            else
                PERROR("could not initialize center, unrecognized crit point")

            // HVR -> maybe change this to something clever
            ctrs[i].alpha_out = 0;
        }
    }

    // initialize birth/death of centers
    find_birth_death(n_ctr, ctrs, &p);

    // HVR_DEBUG
    //~ for(i=0;i<n_ctr;i++)
        //~ display_Center(ctrs+i);

    // find maximum radius for each center (i.e. radius to death)
    for(i=0;i<n_ctr;i++)
    {
        // outermost center is not initialized and doesn't have R_max
        if(ctrs[i].is_init_birthdeath == _FALSE_)
            continue;

        update_pIntegral2d(ctrs+i, &p);
        if( (ctrs[i].type == ctr_type_min) ||
            (ctrs[i].type == ctr_type_saddle_8_minmin) ||
            (ctrs[i].type == ctr_type_saddle_O_max) )
        {
            ctrs[i].R_max = R_of_tau_bracketing_small(ctrs[i].tau_death, &p);
        }
        else if( (ctrs[i].type == ctr_type_max) ||
                 (ctrs[i].type == ctr_type_saddle_8_maxmax) ||
                 (ctrs[i].type == ctr_type_saddle_O_min) )
        {
            p.sign = -1;
            ctrs[i].R_max = R_of_tau_bracketing_small(ctrs[i].tau_birth, &p);
        }
    }

    return ctrs;
}

void free_all_Center(Center *ctrs)
{
    free(ctrs);
}

void display_Center(Center *ctr)
{
    char *ctr_type_names[] = {"min", "max", "saddle 8 maxmax", "saddle 8 minmin",
                                            "saddle O max",    "saddle O min"};

    printf(" Center (%s)\n", ctr_type_names[(int)ctr->type]);
    printf(" ** x0 = (%g, %g)\n", ctr->x0[i_x1], ctr->x0[i_x2]);
    printf(" ** tau0 = %g\n", ctr->tau0);
    printf(" ** alpha_out/pi = %g\n", ctr->alpha_out/M_PI);

    if(ctr->is_init_birthdeath == _TRUE_)
    {
        printf(" ** tau_birth = %g      tau_death = %g\n", ctr->tau_birth, ctr->tau_death);
        printf(" ** R_max = %g\n", ctr->R_max);
    }
    else
        printf(" ** Birth/death not initialized\n");
}


// ======  Find R(tau) for each contours
// =====================================================================

double small_R_guess(double dtau, double alpha, double x10, double x20, pIntegral2d *p2d)
{
    double cth, sth, dR;
    double dphiRR, d11, d12, d22;
    double phi_derivs[N_derivs];

    cth = cos(alpha);
    sth = sin(alpha);

    phiFermat_2ndDeriv(phi_derivs, p2d->y, x10, x20, p2d->Psi);

    d11 = phi_derivs[i_dx1dx1];
    d12 = phi_derivs[i_dx1dx2];
    d22 = phi_derivs[i_dx2dx2];

    dphiRR = d11*cth*cth + 2*d12*sth*cth + d22*sth*sth;

    // HVR: check that the signs are all correct
    dR = sqrt(2*dtau/dphiRR);

    return dR;
}

int dR_dtau_contour2d(double t, const double u[], double f[], void *params)
{
    double R;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double Rdphi_R, Dx1, Dx2;
    pIntegral2d *p = (pIntegral2d *)params;

    R = u[0];

    x1x2_def(x_vec, R, p->alpha_ini, p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];

    //~ dphi_R = (Dx1*phi_derivs[i_dx1] + Dx2*phi_derivs[i_dx2])/R;
    Rdphi_R = Dx1*phi_derivs[i_dx1] + Dx2*phi_derivs[i_dx2];

    if(ABS(Rdphi_R) < EPS_TOL*R)
    {
        PERROR("|dphi_dR|= < %g  at  x=(%g, %g)", EPS_TOL, x_vec[i_x1], x_vec[i_x2])
        return GSL_EBADFUNC;
    }

    //~ f[0] = 1./dphi_R;
    f[0] = p->sign*R/Rdphi_R;

    return GSL_SUCCESS;
}

double R_of_tau_integrate(double tau, pIntegral2d *p)
{
    int status;
    double tau0, epsabs, epsrel, h, R;
    double x_vec[N_dims];
    gsl_odeiv2_system sys = {dR_dtau_contour2d, NULL, 1, p};
    gsl_odeiv2_driver *d;
    const gsl_odeiv2_step_type *T;

    h = pprec.mc_intRtau.h;
    epsabs = pprec.mc_intRtau.epsabs;
    epsrel = pprec.mc_intRtau.epsrel;
    T = get_stepODE(pprec.mc_intRtau.id);

    d = gsl_odeiv2_driver_alloc_y_new (&sys, T, h, epsabs, epsrel);

    // initial conditions
    R = p->R_ini;
    x1x2_def(x_vec, p->R_ini, p->alpha_ini, p->x0_vec);
    tau0 = phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi) - p->tmin;

    // trick to integrate when tau < tau0
    tau = p->sign*tau;
    tau0 = p->sign*tau0;

    status = gsl_odeiv2_driver_apply (d, &tau0, tau, &R);

    if (status != GSL_SUCCESS)
        PERROR("integration of dR/dt failed (%s)", gsl_strerror(status))

    gsl_odeiv2_driver_free (d);

    return R;
}

// HVR: come back to this, clumsy
// use the same function as in the bracket part, do not reimplement it
double R_of_tau_bracketing_small(double tau, pIntegral2d *p)
{
    char is_bracketed;
    int i, max_iter, n_brackets;
    double t, Dt0, Dt1;
    double R0, R1, dR, dlogR, R_min_bracket;
    double R, R_lo, R_hi;
    double x_vec[N_dims];

    t = tau + p->tmin;

    max_iter = pprec.mc_brackRtau_small_maxiter;
    R_min_bracket = pprec.mc_brackRtau_small_Rmin;
    n_brackets = pprec.mc_brackRtau_small_nbrackets;

    R0 = R_min_bracket;
    x1x2_def(x_vec, R0, p->alpha_ini, p->x0_vec);
    Dt0 = t - phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    dR = 100*small_R_guess(Dt0, p->alpha_ini, p->x0_vec[i_x1], p->x0_vec[i_x2], p);
    dR = MIN(dR, 1);

    R1 = R0 + dR;
    x1x2_def(x_vec, R1, p->alpha_ini, p->x0_vec);
    Dt1 = t - phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    if( SIGN(Dt0) != SIGN(Dt1) )
        is_bracketed = _TRUE_;
    else
        is_bracketed = _FALSE_;

    i = 0;
    while( (i<max_iter) && (is_bracketed == _FALSE_) )
    {
        R0 = R1;
        Dt0 = Dt1;

        R1 += dR;
        x1x2_def(x_vec, R1, p->alpha_ini, p->x0_vec);
        Dt1 = t - phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

        //~ printf("iter=%d   R1=%f   Dt0=%f   Dt1=%f\n", i, R1, Dt0, Dt1);

        if( SIGN(Dt0) != SIGN(Dt1) )
            is_bracketed = _TRUE_;
        else
            is_bracketed = _FALSE_;

        i++;
    }

    if(is_bracketed == _TRUE_)
    {
        R0 = R_min_bracket;

        dlogR = (log(R1) - log(R0))/(n_brackets - 1.);
        R_lo = R0;
        R_hi = R0;
        Dt1 = Dt0;

        //~ printf("R0=%f   R1=%f\n", R0, R1);

        for(i=1;i<n_brackets;i++)
        {
            R_lo = R_hi;
            Dt0 = Dt1;

            R_hi = R0*exp(i*dlogR);
            x1x2_def(x_vec, R_hi, p->alpha_ini, p->x0_vec);
            Dt1 = t - phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

            //~ printf("i=%d   R_hi=%f    Dt0=%f    Dt1=%f\n", i, R_hi, Dt0, Dt1);

            if( SIGN(Dt0) != SIGN(Dt1) )
            {
                is_bracketed = _TRUE_;
                break;
            }
            else
                is_bracketed = _FALSE_;
        }

        if(is_bracketed == _FALSE_)   // we couldn't refine -> keep the old
        {
            R_lo = R0;
            R_hi = R1;
        }

        // refine using a proper bracketing algorithm
        R = find_R_in_bracket(tau, R_lo, R_hi, p);
    }
    else
    {
        // use the differential equation as fallback
        PWARNING("could not find bracket for R(tau), solving dR/dtau")

        p->R_ini = pprec.mc_brackRtau_small_Rini;
        R = R_of_tau_integrate(tau, p);
    }

    return R;
}

double R_of_tau_bracketing_large(double tau, pIntegral2d *p)
{
    char is_bracketed;
    int i, max_iter;
    double t, Dt0, Dt1;
    double R, R0, R1, dR, tmp;
    double x_vec[N_dims];

    t = tau + p->tmin;

    max_iter = pprec.mc_brackRtau_large_maxiter;

    R0 = 0;
    x1x2_def(x_vec, R0, p->alpha_ini, p->x0_vec);
    Dt0 = t - phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    tmp = p->y*cos(p->alpha_ini);
    dR = tmp + sqrt(ABS(tmp*tmp + 2*(t - 0.5*p->y*p->y)));
    dR = MAX(dR, 1);

    R1 = R0 + dR;
    x1x2_def(x_vec, R1, p->alpha_ini, p->x0_vec);
    Dt1 = t - phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    if( SIGN(Dt0) != SIGN(Dt1) )
        is_bracketed = _TRUE_;
    else
        is_bracketed = _FALSE_;

    i = 0;
    while( (i<max_iter) && (is_bracketed == _FALSE_) )
    {
        R0 = R1;
        Dt0 = Dt1;

        R1 *= 2.;
        x1x2_def(x_vec, R1, p->alpha_ini, p->x0_vec);
        Dt1 = t - phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

        //~ printf("iter=%d   R1=%f   Dt0=%f   Dt1=%f\n", i, R1, Dt0, Dt1);

        if( SIGN(Dt0) != SIGN(Dt1) )
            is_bracketed = _TRUE_;
        else
            is_bracketed = _FALSE_;
    }

    if(is_bracketed == _TRUE_)
    {
        // refine using a proper bracketing algorithm
        R = find_R_in_bracket(tau, R0, R1, p);
    }
    else
    {
        // use the differential equation as fallback
        PWARNING("could not find bracket for R(tau), solving dR/dtau")

        p->R_ini = pprec.mc_brackRtau_large_Rini;
        R = R_of_tau_integrate(tau, p);
    }

    return R;
}

double phi_minus_tau_func(double R, void *params)
{
    double t;
    double x_vec[N_dims];
    pRoot2d *p = (pRoot2d *)params;

    x1x2_def(x_vec, R, p->p->alpha_ini, p->p->x0_vec);
    t = phiFermat(p->p->y, x_vec[i_x1], x_vec[i_x2], p->p->Psi);

    return p->t - t;
}

double find_R_in_bracket(double tau, double R0, double R1, pIntegral2d *p)
{
    int status, iter, max_iter;
    double r, R_lo, R_hi;
    double epsabs, epsrel;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    gsl_function F;
    pRoot2d pR;

    pR.t = tau + p->tmin;
    pR.p = p;

    F.function = phi_minus_tau_func;
    F.params = &pR;

    if(R0 > R1)
    {
        R_lo = R1;
        R_hi = R0;
    }
    else
    {
        R_lo = R0;
        R_hi = R1;
    }

    epsabs = pprec.mc_findRbracket.epsabs;
    epsrel = pprec.mc_findRbracket.epsrel;
    max_iter = pprec.mc_findRbracket.max_iter;
    T = get_fRoot(pprec.mc_findRbracket.id);

    // HVR_DEBUG: test that we have a healthy bracket
    //~ double f_lo = F.function(R_lo, F.params);
    //~ double f_hi = F.function(R_hi, F.params);
    //~ printf("tau=%e    f_lo=%e    f_hi=%e\n", tau, f_lo, f_hi);

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


// ======  Helper functions to init the params structs
// =====================================================================

void update_pIntegral2d(Center *ctr, pIntegral2d *p)
{
    p->sign = 1;
    p->alpha_ini = ctr->alpha_out;
    p->x0_vec[i_x1] = ctr->x0[i_x1];
    p->x0_vec[i_x2] = ctr->x0[i_x2];
}

void update_pCondODE(pCondODE *pc, pIntegral2d *p)
{
    pc->params = p;
    pc->brack_cond = reach_2pi_contour2d;
    pc->tol_brack = pprec.mc_updCondODE_tol_brack;

    pc->add_cond = is_closed_contour2d;
    //~ pc->add_cond = NULL;

    //~ pc->tol_add = 50*pc->tol_brack;
    pc->tol_add = pprec.mc_updCondODE_tol_add;

    pc->cond_met = _FALSE_;

    p->pCond = pc;
}


// ======  ODE system
// =====================================================================

double reach_2pi_contour2d(const double y[], const double dydt[], void *pCond)
{
    pCondODE *pc = (pCondODE *)pCond;
    pIntegral2d *p = (pIntegral2d *)pc->params;

    // HVR_DEBUG
    //~ printf("alpha_ini/pi = %f     alpha/pi = %f     cond = %f\n", p->alpha_ini/M_PI, y[j_alpha]/M_PI, (y[j_alpha] - p->alpha_ini)/M_2PI - 1.);

    return ABS(y[j_alpha] - p->alpha_ini)/M_2PI - 1.;
}

char is_closed_contour2d(const double y[], const double dydt[], void *pCond)
{
    pCondODE *pc = (pCondODE *)pCond;
    pIntegral2d *p = (pIntegral2d *)pc->params;

    // HVR_DEBUG
    //~ printf("R_ini = %f     R = %f     cond = %f\n", p->R_ini, y[j_R], ABS(y[j_R]/p->R_ini - 1.));

    if( ABS(y[j_R]/p->R_ini - 1.) < pc->tol_add )
        return _TRUE_;
    else
        return _FALSE_;
}

int system_contour2d_robust(double t, const double u[], double f[], void *params)
{
    int status;
    double alpha, R;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double dphi_R, dphi_alpha, Dx1, Dx2;
    pIntegral2d *p = (pIntegral2d *)params;

    alpha = u[j_alpha];
    R = u[j_R];

    x1x2_def(x_vec, R, alpha, p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);

    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];

    dphi_R = (Dx1*phi_derivs[i_dx1] + Dx2*phi_derivs[i_dx2])/R;
    dphi_alpha = -Dx2*phi_derivs[i_dx1] + Dx1*phi_derivs[i_dx2];

    f[j_R] = -dphi_alpha;
    f[j_I] = R;
    f[j_alpha] = dphi_R;

    // check whether to end integration
    status = check_pCond(p->pCond);

    return status;
}


// ======  Initialize the saddle points
// =====================================================================

int integrate_contour2d_saddle(double sigmaf, double *sigma0, double *u0, pIntegral2d *p)
{
    int status;
    double u[N_eqs2d];
    double h0, sigma;
    double epsabs, epsrel;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_driver *d;
    gsl_odeiv2_system sys;

    // define system
    sys.function = system_contour2d_robust;
    sys.jacobian = NULL;
    sys.dimension = N_eqs2d;
    sys.params = p;

    // stepper and precision
    h0 = pprec.mc_intContourSaddle.h;
    epsabs = pprec.mc_intContourSaddle.epsabs;
    epsrel = pprec.mc_intContourSaddle.epsrel;
    T = get_stepODE(pprec.mc_intContourSaddle.id);

    // initial conditions
    sigma = *sigma0;
    u[j_R] = u0[j_R];
    u[j_I] = u0[j_I];
    u[j_alpha] = u0[j_alpha];

    // allocate driver
    d = gsl_odeiv2_driver_alloc_conditional_new(&sys, T, h0, epsabs, epsrel, p->pCond);

    // integrate
    status = gsl_odeiv2_driver_apply(d, &sigma, sigmaf, u);

    if( (status != GSL_SUCCESS) && (status != ODE_COND_MET) )
        PERROR("saddle point integration failed (%s)", gsl_strerror(status))

    gsl_odeiv2_driver_free(d);

    *sigma0 = sigma;
    u0[j_R] = u[j_R];
    u0[j_alpha] = u[j_alpha];
    u0[j_I] = u[j_I];

    return status;
}

double principal_direction_saddle(Center *ctr, pIntegral2d *p2d)
{
    double vmax_x1, vmax_x2, alpha_max;
    double phi_derivs[N_derivs];
    gsl_matrix *hessian = gsl_matrix_alloc(N_dims, N_dims);
    gsl_vector *eval = gsl_vector_alloc(N_dims);
    gsl_matrix *evec = gsl_matrix_alloc(N_dims, N_dims);
    gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(N_dims);

    phiFermat_2ndDeriv(phi_derivs, p2d->y, ctr->x0[i_x1], ctr->x0[i_x2], p2d->Psi);
    gsl_matrix_set(hessian, i_x1, i_x1, phi_derivs[i_dx1dx1]);
    gsl_matrix_set(hessian, i_x2, i_x1, phi_derivs[i_dx1dx2]);
    gsl_matrix_set(hessian, i_x1, i_x2, phi_derivs[i_dx1dx2]);
    gsl_matrix_set(hessian, i_x2, i_x2, phi_derivs[i_dx2dx2]);

    gsl_eigen_symmv(hessian, eval, evec, w);
    gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_ASC);

    // HVR_DEBUG
    //~ for(int i=0;i<N_dims;i++)
    //~ {
        //~ double eval_i = gsl_vector_get(eval, i);
        //~ gsl_vector_view evec_i = gsl_matrix_column(evec, i);

        //~ printf("eigenvalue = %g\n", eval_i);
        //~ printf("eigenvector = \n");
        //~ gsl_vector_fprintf(stdout, &evec_i.vector, "%g");
    //~ }

    // assuming that one eigenvalue is negative and the other positive
    vmax_x1 = gsl_matrix_get(evec, i_x1, 0);
    vmax_x2 = gsl_matrix_get(evec, i_x2, 0);

    // HVR: double check this
    alpha_max = atan(vmax_x2/(vmax_x1 + EPS_TOL));
    alpha_max = MOD_2PI(alpha_max);

    // HVR_DEBUG
    //~ {
        //~ double dR = 5e-2;
        //~ double x10 = ctr->x0[i_x1];
        //~ double dx1 = dR*cos(alpha_max);
        //~ double x20 = ctr->x0[i_x2];
        //~ double dx2 = dR*sin(alpha_max);
        //~ double phi0 = phiFermat(p2d->y, x10, x20, p2d->Psi);
        //~ double phi1 = phiFermat(p2d->y, x10 + dx1, x20 + dx2, p2d->Psi);

        //~ printf("diff = %g\n", phi1-phi0);
    //~ }

    gsl_eigen_symmv_free(w);
    gsl_matrix_free(evec);
    gsl_vector_free(eval);
    gsl_matrix_free(hessian);

    return alpha_max;
}

void fill_saddle_Center(Center *ctr, pIntegral2d *p2d)
{
    int i, j;
    int status, n_out, n_sigma;
    int i_max1, i_max2, i_min1, i_min2, i_out;
    double alpha_max, dR;
    double sigma0, sigma, sigmaf, dsigma;
    double alphas[4];
    double u[4][N_eqs2d];
    pCondODE *pc;

    // ---------------------------------------------
    // Initialize parameters
    pc = init_pCondODE();
    update_pCondODE(pc, p2d);

    p2d->x0_vec[i_x1] = ctr->x0[i_x1];
    p2d->x0_vec[i_x2] = ctr->x0[i_x2];
    // ---------------------------------------------

    // Find principal directions of the saddle
    alpha_max = principal_direction_saddle(ctr, p2d);

    i_max1 = 0;
    i_max2 = 1;
    i_min1 = 2;
    i_min2 = 3;
    alphas[i_max1] = MOD_2PI(alpha_max);
    alphas[i_max2] = MOD_2PI(alpha_max + M_PI);
    alphas[i_min1] = MOD_2PI(alpha_max + M_PI_2);
    alphas[i_min2] = MOD_2PI(alpha_max + 3*M_PI_2);

    // prec params
    dR      = pprec.mc_fillSaddleCenter_dR;
    n_sigma = pprec.mc_fillSaddleCenter_nsigma;
    sigmaf  = pprec.mc_fillSaddleCenter_sigmaf;
    dsigma  = sigmaf/(n_sigma-2);

    // number of directions to exit the saddle (updated later)
    n_out = 0;

    // initial conditions
    p2d->R_ini = dR;
    for(j=0;j<4;j++)
    {
        u[j][j_R] = dR;
        u[j][j_alpha] = alphas[j];
    }

    // integrate chunks of the curve in each of the principal directions
    for(i=0;i<n_sigma;i++)
    {
        sigma0 = i*dsigma;
        sigmaf = sigma0 + dsigma;

        // HVR_DEBUG
        printf("i=%d/%d   sigma0=%f     sigmaf=%f\n", i, n_sigma, sigma0, sigmaf);

        for(j=0;j<4;j++)
        {
            sigma = sigma0;
            u[j][j_I] = 0;

            p2d->alpha_ini = alphas[j];
            pc->cond_met = _FALSE_;

            status = integrate_contour2d_saddle(sigmaf, &sigma, u[j], p2d);

            // this direction has closed -> it points out of the saddle
            if(status == ODE_COND_MET)
            {
                n_out++;
                i_out = j;

                // HVR_DEBUG
                printf("n_out=%d   alpha[%d]/pi=%f  sigmaf=%f\n", n_out, j, alphas[j]/M_PI, sigma);
            }
        }

        // if the opposite side of the saddle seems near to closing, try a bit harder
        if(n_out==1)
        {
            // find the opposite side
            if(i_out == i_max1)
                j = i_max2;
            if(i_out == i_max2)
                j = i_max1;
            if(i_out == i_min1)
                j = i_min2;
            if(i_out == i_min2)
                j = i_min1;

            // check that it is nearly closed and then integrate more
            if(u[j][j_alpha]/M_2PI > 0.9)
            {
                sigma0 = sigma;

                for(i=0;i<10;i++)
                {
                    sigma  = sigma0 + i*dsigma;
                    sigmaf = sigma + dsigma;
                    u[j][j_I] = 0;

                    // HVR_DEBUG
                    printf("i=%d   sigma0=%f     sigmaf=%f\n", i, sigma, sigmaf);

                    p2d->alpha_ini = alphas[j];
                    pc->cond_met = _FALSE_;

                    status = integrate_contour2d_saddle(sigmaf, &sigma, u[j], p2d);

                    if(status == ODE_COND_MET)
                    {
                        n_out++;

                        // HVR_DEBUG
                        printf("n_out=%d   alpha[%d]/pi=%f  sigmaf=%f\n", n_out, j, alphas[j]/M_PI, sigma);

                        break;
                    }
                }
            }
        }

        if(n_out > 0)
        {
            ctr->alpha_out = alphas[i_out];
            break;
        }
    }

    if(n_out == 1)
    {
        if( (i_out == i_max1) || (i_out == i_max2) )
            ctr->type = ctr_type_saddle_O_min;

        if( (i_out == i_min1) || (i_out == i_min2) )
            ctr->type = ctr_type_saddle_O_max;
    }
    else if(n_out == 2)
    {
        if( (i_out == i_max1) || (i_out == i_max2) )
            ctr->type = ctr_type_saddle_8_maxmax;

        if( (i_out == i_min1) || (i_out == i_min2) )
            ctr->type = ctr_type_saddle_8_minmin;
    }
    else
        PERROR("problem identifying the saddle point at tau0=%g", ctr->tau0)

    // ---------------------------------------------
    free_pCondODE(pc);
}


// ======  Contour integration
// =====================================================================

double integrate_contour2d(pIntegral2d *p)
{
    int status;
    double u[N_eqs2d];
    double h0, sigma, sigmaf;
    double epsabs, epsrel;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_driver *d;
    gsl_odeiv2_system sys;

    // define system
    sys.function = system_contour2d_robust;
    sys.jacobian = NULL;
    sys.dimension = N_eqs2d;
    sys.params = p;

    // stepper and precision
    h0 = pprec.mc_intContour.h;
    epsabs = pprec.mc_intContour.epsabs;
    epsrel = pprec.mc_intContour.epsrel;
    T = get_stepODE(pprec.mc_intContour.id);
    sigmaf = pprec.mc_intContour_sigmaf;

    // initial conditions
    sigma = 0;
    u[j_R] = p->R_ini;
    u[j_I] = 0;
    u[j_alpha] = p->alpha_ini;

    // allocate driver
    d = gsl_odeiv2_driver_alloc_conditional_new(&sys, T, h0, epsabs, epsrel, p->pCond);

    // integrate
    status = gsl_odeiv2_driver_apply(d, &sigma, sigmaf, u);

    if( (status != GSL_SUCCESS) && (status != ODE_COND_MET) )
        PERROR("integration failed (%s)", gsl_strerror(status))

    if(status != ODE_COND_MET)
        PWARNING("ending condition not met in contour")

    gsl_odeiv2_driver_free(d);

    return u[j_I];
}

double integrate_all_contour2d(double tau, int n_ctrs, Center *ctrs, pIntegral2d *p)
{
    char is_alive;
    int i;
    double I=0, I_ctr;
    pCondODE *pc = (pCondODE *)p->pCond;

    for(i=0;i<n_ctrs;i++)
    {
        is_alive = _FALSE_;
        update_pIntegral2d(ctrs+i, p);

        if(ctrs[i].is_init_birthdeath == _FALSE_)
        {
            // outermost critical point -> no death
            if(tau > ctrs[i].tau0)
            {
                is_alive = _TRUE_;
                p->R_ini = R_of_tau_bracketing_large(tau, p);

                if(p->R_ini < 0)
                    PWARNING("R_ini<0 for the outermost contour at tau=%g", tau)
            }
        }
        else
        {
            if( (tau > ctrs[i].tau_birth) && (tau < ctrs[i].tau_death) )
            {
                is_alive = _TRUE_;
                p->R_ini = find_R_in_bracket(tau, 0, ctrs[i].R_max, p);

                if(p->R_ini < 0)
                    PWARNING("R_ini<0 for an internal contour at tau=%g", tau)
            }
        }

        // integrate and add up each live contour
        if(is_alive == _TRUE_)
        {
            pc->cond_met = _FALSE_;         // make sure to reset it
            I_ctr = integrate_contour2d(p);

            // HVR_DEBUG
            //~ printf("i=%d    I_ctr=%f\n", i, I_ctr/M_2PI);

            I += I_ctr;
        }
    }

    return I;
}

double driver_contour2d(double tau, int n_ctrs, Center *ctrs, double y, pNamedLens *pNLens)
{
    double I, I0, tau_real, tau_min;
    Lens Psi = init_lens(pNLens);
    pIntegral2d p;
    pCondODE *pc;

    // built-in step function
    if(tau < 0)
        return 0;

    // if tau is too small, we interpolate between 0 and tau_min
    tau_real = tau;
    tau_min = pprec.mc_drivContour_taumin_over_y2*y*y;
    if(tau < tau_min)
        tau = tau_min;

    // initialize the common parameters in pIntegral2d
    pc = init_pCondODE();
    update_pCondODE(pc, &p);

    p.y = y;
    p.tmin = ctrs[0].t0;
    p.Psi = &Psi;
    // ---------------------------------------------

    // integration
    I = integrate_all_contour2d(tau, n_ctrs, ctrs, &p);

    free_pCondODE(pc);

    // linear interpolation when tau is very small
    if(tau_real < tau_min)
    {
        I0 = M_2PI*sqrt(magnification(ctrs[0].x0[i_x1], ctrs[0].x0[i_x2], &Psi));
        I = I + (I0 - I)*(1 - tau_real/tau_min);
    }

    return I;
}


// ======  Apply diferent rules to initialize the centers
// =====================================================================

TableCandidates *init_TableCandidates(int n_ctrs, Center *ctrs)
{
    int i, j, k, id;
    char type;
    TableCandidates *t = (TableCandidates *)malloc(sizeof(TableCandidates));

    t->n_ctrs = n_ctrs;
    t->ctrs = ctrs;

    t->n_up  = (int *)malloc(n_ctrs*sizeof(int));
    t->n_dw  = (int *)malloc(n_ctrs*sizeof(int));
    t->id_up = (int **)malloc(n_ctrs*sizeof(int *));
    t->id_dw = (int **)malloc(n_ctrs*sizeof(int *));

    for(i=0;i<n_ctrs;i++)
    {
        t->n_up[i] = 0;
        t->n_dw[i] = 0;

        type = ctrs[i].type;

        // all above
        if( (type == ctr_type_saddle_8_maxmax) ||
            (type == ctr_type_saddle_O_max) ||
            (type == ctr_type_saddle_O_min) )
            t->n_up[i] = n_ctrs - 1 - i;

        // all below
        if( (type == ctr_type_saddle_8_minmin) ||
            (type == ctr_type_saddle_O_max) ||
            (type == ctr_type_saddle_O_min) )
            t->n_dw[i] = i;

        k = 0;
        t->id_up[i] = (int *)malloc(t->n_up[i]*sizeof(int));
        for(j=0;j<t->n_up[i];j++)
        {
            id = j + i + 1;
            type = ctrs[id].type;

            if( (type == ctr_type_max) ||
                (type == ctr_type_saddle_8_maxmax) ||
                (type == ctr_type_saddle_O_min) )
            {
                t->id_up[i][k] = id;
                k++;
            }
        }
        t->n_up[i] = k;

        k = 0;
        t->id_dw[i] = (int *)malloc(t->n_dw[i]*sizeof(int));
        for(j=0;j<t->n_dw[i];j++)
        {
            id = j;
            type = ctrs[id].type;

            if( (type == ctr_type_min) ||
                (type == ctr_type_saddle_8_minmin) ||
                (type == ctr_type_saddle_O_max) )
            {
                t->id_dw[i][k] = id;
                k++;
            }
        }
        t->n_dw[i] = k;
    }

    return t;
}

void display_TableCandidates(TableCandidates *t)
{
    int i, j;

    printf("Candidates table:\n");
    for(i=0;i<t->n_ctrs;i++)
    {
        printf(" (%3.1u)  down: [", i);
        for(j=0;j<t->n_dw[i];j++)
        {
            printf("%2.1d", t->id_dw[i][j]);
            if(j < t->n_dw[i] - 1)
                printf(", ");
        }
        printf("]\n");

        printf("          up: [");
        for(j=0;j<t->n_up[i];j++)
        {
            printf("%2.1d", t->id_up[i][j]);
            if(j < t->n_up[i] - 1)
                printf(", ");
        }
        printf("]\n");
    }
}

void free_TableCandidates(TableCandidates *t)
{
    int i;

    for(i=0;i<t->n_ctrs;i++)
    {
        free(t->id_dw[i]);
        free(t->id_up[i]);
    }

    free(t->id_dw);
    free(t->id_up);
    free(t->n_dw);
    free(t->n_up);
    free(t);
}


int find_birth_death(int n_ctrs, Center *ctrs, pIntegral2d *p)
{
    int i, j, n, status;
    TableCandidates *t = init_TableCandidates(n_ctrs, ctrs);

    status = 0;

    // initialize either birth or death for all centers
    for(i=0;i<n_ctrs;i++)
    {
        if( (ctrs[i].type == ctr_type_min) ||
            (ctrs[i].type == ctr_type_saddle_8_minmin) ||
            (ctrs[i].type == ctr_type_saddle_O_min) )
        {
            ctrs[i].tau_birth = ctrs[i].tau0;
        }
        else if( (ctrs[i].type == ctr_type_max) ||
                 (ctrs[i].type == ctr_type_saddle_8_maxmax) ||
                 (ctrs[i].type == ctr_type_saddle_O_max) )
        {
            ctrs[i].tau_death = ctrs[i].tau0;
        }
    }

    // HVR_DEBUG
    //~ display_TableCandidates(t);

    // perform the obvious assignments of candidates
    update_reduce_Table(t);

    // HVR_DEBUG
    //~ printf("\nGOING FOR MINIMA:\n");

    // try to minimize/maximize inside the saddle_O to associate min/max with them
    update_minimize_Table(t, p);

    // check that initialization is ok
    // only saddle_O_max should remain unitialized (the outer saddle)
    n = 0;
    for(i=0;i<n_ctrs;i++)
    {
        if(ctrs[i].is_init_birthdeath == _FALSE_)
        {
            j = i;
            n++;
        }
    }
    if( (n != 1) ||
        (n_ctrs > 2  && (ctrs[j].type != ctr_type_saddle_O_max) && (ctrs[j].type != ctr_type_saddle_8_minmin)) ||
        (n_ctrs == 1 && ctrs[j].type != ctr_type_min) )
    {
        PERROR("initialization of centers was unsuccessful")
        status = 1;
    }

    free_TableCandidates(t);

    return status;
}

void update_reduce_Table(TableCandidates *t)
{
    char is_updated;
    int i, j, k_up, k_dw;
    int id1, id2;
    Center *ctr, *ctr1, *ctr2;

    do
    {
        is_updated = _FALSE_;

        for(i=0;i<t->n_ctrs;i++)
        {
            j = 0;
            while(j < t->n_up[i])
            {
                k_up = t->n_up[i];
                id1 = t->id_up[i][j];
                id2 = t->id_up[i][k_up-1];

                if(t->ctrs[id1].is_init_birthdeath == _TRUE_)
                {
                    // swap with the last index and reduce the size
                    // i.e. eliminate this entry
                    t->id_up[i][j] = id2;
                    t->id_up[i][k_up-1] = id1;
                    t->n_up[i]--;
                    j = 0;
                }
                else
                    j++;
            }

            j = 0;
            while(j < t->n_dw[i])
            {
                k_dw = t->n_dw[i];
                id1 = t->id_dw[i][j];
                id2 = t->id_dw[i][k_dw-1];

                if(t->ctrs[id1].is_init_birthdeath == _TRUE_)
                {
                    // swap with the last index and reduce the size
                    // i.e. eliminate this entry
                    t->id_dw[i][j] = id2;
                    t->id_dw[i][k_dw-1] = id1;
                    t->n_dw[i]--;
                    j = 0;
                }
                else
                    j++;
            }
        }

        // HVR_DEBUG
        //~ display_TableCandidates(t);

        for(i=0;i<t->n_ctrs;i++)
        {
            k_up = t->n_up[i];
            k_dw = t->n_dw[i];
            ctr = t->ctrs+i;

            if( (k_dw == 2) && (ctr->type == ctr_type_saddle_8_minmin) )
            {
                // HVR_DEBUG
                //~ printf(" * center %d can be initialized\n", i);

                ctr1 = t->ctrs + t->id_dw[i][0];
                ctr2 = t->ctrs + t->id_dw[i][1];

                ctr1->tau_death = ctr->tau0;
                ctr2->tau_death = ctr->tau0;

                ctr1->is_init_birthdeath = _TRUE_;
                ctr2->is_init_birthdeath = _TRUE_;

                t->n_dw[i] = 0;

                is_updated = _TRUE_;
            }
            else if( (k_up == 2) && (ctr->type == ctr_type_saddle_8_maxmax) )
            {
                // HVR_DEBUG
                //~ printf(" * center %d can be initialized\n", i);

                ctr1 = t->ctrs + t->id_up[i][0];
                ctr2 = t->ctrs + t->id_up[i][1];

                ctr1->tau_birth = ctr->tau0;
                ctr2->tau_birth = ctr->tau0;

                ctr1->is_init_birthdeath = _TRUE_;
                ctr2->is_init_birthdeath = _TRUE_;

                t->n_up[i] = 0;

                is_updated = _TRUE_;
            }
            else if( (k_up == 1) || (k_dw == 1) )
            {
                // HVR_DEBUG
                //~ printf(" * center %d can be initialized\n", i);

                if( (ctr->type == ctr_type_saddle_O_max) || (ctr->type == ctr_type_saddle_O_min) )
                {
                    if(k_up == 1)
                    {
                        ctr1 = t->ctrs + t->id_up[i][0];
                        ctr1->tau_birth = ctr->tau0;
                        ctr1->is_init_birthdeath = _TRUE_;

                        ctr->tau_death = ctr->tau0;

                        t->n_up[i] = 0;

                        is_updated = _TRUE_;
                    }
                    else
                    {
                        ctr1 = t->ctrs + t->id_dw[i][0];
                        ctr1->tau_death = ctr->tau0;
                        ctr1->is_init_birthdeath = _TRUE_;

                        ctr->tau_birth = ctr->tau0;

                        t->n_dw[i] = 0;

                        is_updated = _TRUE_;
                    }
                }
            }
        }
    }
    while(is_updated == _TRUE_);
}

void update_minimize_Table(TableCandidates *t, pIntegral2d *p)
{
    char flag;
    int i, j, id, id_min, n_ctrs;
    double dist, dist_min, dx1, dx2;
    double x_min[N_dims];
    Center *ctrs;

    n_ctrs = t->n_ctrs;
    ctrs = t->ctrs;

    for(i=0;i<n_ctrs;i++)
    {
        // HVR: implement the min finder only for saddle_O for now
        //      but it can also be applied to saddle_8
        if( (ctrs[i].type != ctr_type_saddle_O_max) && (ctrs[i].type != ctr_type_saddle_O_min) )
            continue;

        //______________________________________________________________
        flag = _FALSE_;
        for(j=0;j<t->n_up[i];j++)
        {
            flag = _TRUE_;
            id = t->id_up[i][j];

            if( (ctrs[id].type != ctr_type_min) && (ctrs[id].type != ctr_type_max) )
                flag = _FALSE_;
        }
        if(flag == _TRUE_)
        {
            minimize_in_saddle_O(x_min, ctrs+i, p);

            for(j=0;j<t->n_up[i];j++)
            {
                id = t->id_up[i][j];
                dx1 = x_min[i_x1] - ctrs[id].x0[i_x1];
                dx2 = x_min[i_x2] - ctrs[id].x0[i_x2];
                dist = sqrt(dx1*dx1 + dx2*dx2);

                //~ printf("id=%d   xmin=(%f, %f)   dist=%f\n", id, x_min[i_x1], x_min[i_x2], dist);

                if(j == 0)
                {
                    id_min = id;
                    dist_min = dist;
                    continue;
                }

                if(dist < dist_min)
                {
                    dist_min = dist;
                    id_min = id;
                }
            }

            //~ printf(" * center %d  inside of saddle %d\n", id_min, i);
            if( (ctrs[i].type == ctr_type_saddle_O_max) && (ctrs[id_min].type == ctr_type_max) )
            {
                ctrs[id_min].tau_birth = ctrs[i].tau0;
                ctrs[id_min].is_init_birthdeath = _TRUE_;
                t->n_up[i] = 0;
                update_reduce_Table(t);
            }
            if( (ctrs[i].type == ctr_type_saddle_O_min) && (ctrs[id_min].type == ctr_type_min) )
            {
                ctrs[id_min].tau_death = ctrs[i].tau0;
                ctrs[id_min].is_init_birthdeath = _TRUE_;
                t->n_up[i] = 0;
                update_reduce_Table(t);
            }
        }
        // *************************************************************

        //______________________________________________________________
        flag = _FALSE_;
        for(j=0;j<t->n_dw[i];j++)
        {
            flag = _TRUE_;
            id = t->id_dw[i][j];

            if( (ctrs[id].type != ctr_type_min) && (ctrs[id].type != ctr_type_max) )
                flag = _FALSE_;
        }
        if(flag == _TRUE_)
        {
            minimize_in_saddle_O(x_min, ctrs+i, p);

            for(j=0;j<t->n_dw[i];j++)
            {
                id = t->id_dw[i][j];
                dx1 = x_min[i_x1] - ctrs[id].x0[i_x1];
                dx2 = x_min[i_x2] - ctrs[id].x0[i_x2];
                dist = sqrt(dx1*dx1 + dx2*dx2);

                //~ printf("id=%d   xmin=(%f, %f)   dist=%f\n", id, x_min[i_x1], x_min[i_x2], dist);

                if(j == 0)
                {
                    id_min = id;
                    dist_min = dist;
                    continue;
                }

                if(dist < dist_min)
                {
                    dist_min = dist;
                    id_min = id;
                }
            }

            //~ printf(" * center %d  inside of saddle %d\n", id_min, i);

            if( (ctrs[i].type == ctr_type_saddle_O_max) && (ctrs[id_min].type == ctr_type_max) )
            {
                ctrs[id_min].tau_birth = ctrs[i].tau0;
                ctrs[id_min].is_init_birthdeath = _TRUE_;
                t->n_dw[i] = 0;
                update_reduce_Table(t);
            }
            if( (ctrs[i].type == ctr_type_saddle_O_min) && (ctrs[id_min].type == ctr_type_min) )
            {
                ctrs[id_min].tau_death = ctrs[i].tau0;
                ctrs[id_min].is_init_birthdeath = _TRUE_;
                t->n_dw[i] = 0;
                update_reduce_Table(t);
            }
        }
        // *************************************************************
    }
}


// ======  Find min/max inside a saddle point
// =====================================================================

double f_phi_multimin(const gsl_vector *v, void *params)
{
    double x1, x2, phi;
    pIntegral2d *p = (pIntegral2d *)params;

    x1 = gsl_vector_get(v, i_x1);
    x2 = gsl_vector_get(v, i_x2);

    phi = phiFermat(p->y, x1, x2, p->Psi);

    return p->sign*phi;
}

void df_phi_multimin(const gsl_vector *v, void *params, gsl_vector *df)
{
    double x1, x2;
    double phi_derivs[N_derivs];
    pIntegral2d *p = (pIntegral2d *)params;

    x1 = gsl_vector_get(v, i_x1);
    x2 = gsl_vector_get(v, i_x2);

    phiFermat_1stDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    gsl_vector_set(df, i_x1, p->sign*phi_derivs[i_dx1]);
    gsl_vector_set(df, i_x2, p->sign*phi_derivs[i_dx2]);
}

void fdf_phi_multimin(const gsl_vector *v, void *params,
                      double *f, gsl_vector *df)
{
    double x1, x2;
    double phi_derivs[N_derivs];
    pIntegral2d *p = (pIntegral2d *)params;

    x1 = gsl_vector_get(v, i_x1);
    x2 = gsl_vector_get(v, i_x2);

    phiFermat_1stDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    *f = p->sign*phi_derivs[i_0];
    gsl_vector_set(df, i_x1, p->sign*phi_derivs[i_dx1]);
    gsl_vector_set(df, i_x2, p->sign*phi_derivs[i_dx2]);
}

int minimize_in_saddle_O(double *x_min, Center *ctr, pIntegral2d *p)
{
    int iter, max_iter, status;
    double dR, tol, tol_gradient, h;
    double x_vec[N_dims];

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

    gsl_vector *x;
    gsl_multimin_function_fdf F;

    F.n = N_dims;
    F.f = f_phi_multimin;
    F.df = df_phi_multimin;
    F.fdf = fdf_phi_multimin;
    F.params = p;

    // starting point
    if(ctr->type == ctr_type_saddle_O_min)
        p->sign = 1;
    else if(ctr->type == ctr_type_saddle_O_max)
        p->sign = -1;

    max_iter = pprec.mc_minInSaddle.max_iter;
    h = pprec.mc_minInSaddle.first_step;
    tol = pprec.mc_minInSaddle.tol;
    tol_gradient = pprec.mc_minInSaddle.epsabs;
    T = get_fdfMultimin(pprec.mc_minInSaddle.id);
    dR = pprec.mc_minInSaddle_dR;

    x1x2_def(x_vec, dR, ctr->alpha_out + M_PI, ctr->x0);

    x = gsl_vector_alloc(N_dims);
    gsl_vector_set(x, i_x1, x_vec[i_x1]);
    gsl_vector_set(x, i_x2, x_vec[i_x2]);

    s = gsl_multimin_fdfminimizer_alloc(T, N_dims);

    gsl_multimin_fdfminimizer_set(s, &F, x, h, tol);

    iter = 0;
    do
    {
        status = gsl_multimin_fdfminimizer_iterate(s);

        if(status)
            break;

        status = gsl_multimin_test_gradient(s->gradient, tol_gradient);

        // HVR_DEBUG
        //~ if(status == GSL_SUCCESS)
            //~ printf ("Minimum found at:\n");
        //~ printf ("%5d %.5f %.5f %10.5f \n", iter,
                                           //~ gsl_vector_get(s->x, i_x1),
                                           //~ gsl_vector_get(s->x, i_x2),
                                           //~ s->f);

        iter++;
    }
    while( (status == GSL_CONTINUE) && (iter < max_iter) );

    x_min[i_x1] = gsl_vector_get(s->x, i_x1);
    x_min[i_x2] = gsl_vector_get(s->x, i_x2);

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);

    return 0;
}


// ======  Get parametric contours
// =====================================================================

int get_contour2d_points_robust(SolODE *sol, pIntegral2d *p)
{
    int status;
    double u[N_eqs2d], contour[4];
    double epsabs, epsrel, h, sigma, sigmaf;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_step *s;
    gsl_odeiv2_control *c;
    gsl_odeiv2_evolve *e;
    gsl_odeiv2_system sys = {system_contour2d_robust, NULL, N_eqs2d, p};

    // stepper and tolerance
    h = pprec.mc_getContour.h;
    epsabs = pprec.mc_getContour.epsabs;
    epsrel = pprec.mc_getContour.epsrel;
    T = get_stepODE(pprec.mc_getContour.id);
    sigmaf = pprec.mc_getContour_sigmaf;

    s = gsl_odeiv2_step_alloc(T, N_eqs2d);
    c = gsl_odeiv2_control_conditional_new(epsabs, epsrel, p->pCond);
    e = gsl_odeiv2_evolve_alloc(N_eqs2d);

    // initial conditions
    u[j_R] = p->R_ini;
    u[j_I] = 0;
    u[j_alpha] = p->alpha_ini;

    sigma = 0;

    while(sigma < sigmaf)
    {
        status = gsl_odeiv2_evolve_apply (e, c, s,
                                          &sys,
                                          &sigma, sigmaf,
                                          &h, u);

        if( (status != GSL_SUCCESS) && (status != ODE_COND_MET) )
        {
            PERROR("integration of contour failed (%s)", gsl_strerror(status))
            break;
        }

        contour[0] = u[j_alpha];
        contour[1] = u[j_R];

        // fill contour[2], contour[3] with x1, x2
        x1x2_def(contour+2, u[j_R], u[j_alpha], p->x0_vec);

        fill_SolODE(sigma, contour, sol);

        if(status == ODE_COND_MET)
            break;
    }

    gsl_odeiv2_step_free(s);
    gsl_odeiv2_control_free(c);
    gsl_odeiv2_evolve_free(e);

    return 0;
}

int get_contour2d_robust(int n_points, SolODE *sol, pIntegral2d *p)
{
    int i, status;
    double u[N_eqs2d];
    double sigmaf, sigma, dsigma, contour[4];
    double epsabs, epsrel, h0;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_driver *d;
    gsl_odeiv2_system sys = {system_contour2d_robust, NULL, N_eqs2d, p};
    pCondODE *pCond = (pCondODE *)p->pCond;

    // stepper and tolerance
    h0 = pprec.mc_getContour.h;
    epsabs = pprec.mc_getContour.epsabs;
    epsrel = pprec.mc_getContour.epsrel;
    T = get_stepODE(pprec.mc_getContour.id);
    sigmaf = pprec.mc_getContour_sigmaf;

    // initialize the driver
    d = gsl_odeiv2_driver_alloc_conditional_new(&sys, T, h0, epsabs, epsrel, p->pCond);

    // initial conditions
    u[j_R] = p->R_ini;
    u[j_I] = 0;
    u[j_alpha] = p->alpha_ini;
    sigma = 0;

    // store first point
    contour[0] = u[j_alpha];
    contour[1] = u[j_R];
    x1x2_def(contour+2, u[j_R], u[j_alpha], p->x0_vec);
    fill_SolODE(sigma, contour, sol);

    // find final sigma integrating everything
    status = gsl_odeiv2_driver_apply(d, &sigma, sigmaf, u);

    if( (status != GSL_SUCCESS) && (status != ODE_COND_MET) )
        PERROR("integration of contour failed (%s)", gsl_strerror(status))
    else
        sigmaf = sigma;

    // restart the driver
    gsl_odeiv2_driver_free(d);
    //~ d = gsl_odeiv2_driver_alloc_conditional_new(&sys, T, h0, epsabs, epsrel, p->pCond);
    d = gsl_odeiv2_driver_alloc_y_new(&sys, T, h0, epsabs, epsrel);

    // restart the initial conditions
    pCond->cond_met = _FALSE_;
    u[j_R] = p->R_ini;
    u[j_I] = 0;
    u[j_alpha] = p->alpha_ini;

    // finally, store the grid
    dsigma = sigmaf/(n_points-1);
    for(i=0;i<(n_points-1);i++)
    {
        sigma = i*dsigma;
        status = gsl_odeiv2_driver_apply(d, &sigma, sigma + dsigma, u);

        contour[0] = u[j_alpha];
        contour[1] = u[j_R];

        // fill contour[2], contour[3] with x1, x2
        x1x2_def(contour+2, u[j_R], u[j_alpha], p->x0_vec);
        fill_SolODE(sigma, contour, sol);

        if( (status != GSL_SUCCESS) && (status != ODE_COND_MET))
            PERROR("integration of contour failed (%s)", gsl_strerror(status))
    }

    gsl_odeiv2_driver_free(d);

    return 0;
}

SolODE **driver_get_contour2d(double tau, int n_points, int n_ctrs, Center *ctrs,
                              double y, pNamedLens *pNLens)
{
    int i, n_eqs, n_buffer;
    pIntegral2d p;
    pCondODE *pc;
    Lens Psi = init_lens(pNLens);

    char is_alive;
    SolODE **sols = (SolODE **)malloc(n_ctrs*sizeof(SolODE *));

    n_eqs = 4;
    n_buffer = 200;
    for(i=0;i<n_ctrs;i++)
        sols[i] = NULL;

    // initialize the common parameters in pIntegral2d
    pc = init_pCondODE();
    update_pCondODE(pc, &p);

    p.y = y;
    p.tmin = ctrs[0].t0;
    p.Psi = &Psi;
    // ---------------------------------------------

    // HVR -> improve this for small tau
    //~ if(tau < tau_min)
        // do something here

    for(i=0;i<n_ctrs;i++)
    {
        is_alive = _FALSE_;
        update_pIntegral2d(ctrs+i, &p);

        if(ctrs[i].is_init_birthdeath == _FALSE_)
        {
            // outermost critical point -> no death
            if(tau > ctrs[i].tau0)
            {
                is_alive = _TRUE_;
                p.R_ini = R_of_tau_bracketing_large(tau, &p);
            }
        }
        else
        {
            if( (tau > ctrs[i].tau_birth) && (tau < ctrs[i].tau_death) )
            {
                is_alive = _TRUE_;
                p.R_ini = find_R_in_bracket(tau, 0, ctrs[i].R_max, &p);
            }
        }

        // integrate and add up each live contour
        if(is_alive == _TRUE_)
        {
            sols[i] = init_SolODE(n_eqs, n_buffer);

            pc->cond_met = _FALSE_;         // make sure to reset it

            if(n_points > 1)
                get_contour2d_robust(n_points, sols[i], &p);
            else
                get_contour2d_points_robust(sols[i], &p);
        }
    }

    free_pCondODE(pc);

    return sols;
}

void free_SolODE_contour2d(int n_sols, SolODE **sols)
{
    int i;

    for(i=0;i<n_sols;i++)
    {   if(sols[i] != NULL)
            free_SolODE(sols[i]);
    }

    free(sols);
}


// ======  Get contours with x1, x2 parameterization
// =====================================================================

int system_contour2d_x1x2(double t, const double u[], double f[], void *params)
{
    double x1, x2;
    double phi_derivs[N_derivs];
    pContour *p = (pContour *)params;

    x1 = u[i_x1];
    x2 = u[i_x2];

    phiFermat_1stDeriv(phi_derivs, p->y, x1, x2, p->Psi);

    f[i_x1] = -phi_derivs[i_dx2];
    f[i_x2] =  phi_derivs[i_dx1];

    return GSL_SUCCESS;
}

int get_contour2d_x1x2(int n_points, SolODE *sol, pContour *p)
{
    int i, status;
    double u[N_dims];
    double sigmaf, sigma, dsigma;
    double epsabs, epsrel, h0;
    const gsl_odeiv2_step_type *T;
    gsl_odeiv2_driver *d;
    gsl_odeiv2_system sys = {system_contour2d_x1x2, NULL, N_dims, p};

    // stepper and tolerance
    h0 = pprec.mc_getContour_x1x2.h;
    epsabs = pprec.mc_getContour_x1x2.epsabs;
    epsrel = pprec.mc_getContour_x1x2.epsrel;
    T = get_stepODE(pprec.mc_getContour_x1x2.id);

    // initialize the driver
    d = gsl_odeiv2_driver_alloc_y_new(&sys, T, h0, epsabs, epsrel);

    // initial conditions
    u[i_x1] = p->x10;
    u[i_x2] = p->x20;
    sigma = 0;
    sigmaf = p->sigmaf;

    // store first point
    fill_SolODE(sigma, u, sol);

    // store the grid
    dsigma = sigmaf/(n_points-1);
    for(i=0;i<(n_points-1);i++)
    {
        sigma = i*dsigma;

        status = gsl_odeiv2_driver_apply(d, &sigma, sigma + dsigma, u);

        fill_SolODE(sigma, u, sol);

        if(status != GSL_SUCCESS)
            PERROR("integration of contour failed (%s)", gsl_strerror(status))
    }

    gsl_odeiv2_driver_free(d);

    return 0;
}

int driver_get_contour2d_x1x2(double x10, double x20, double y,
                              double sigmaf, int n_points,
                              pNamedLens *pNLens,
                              SolODE *sol)
{
    pContour params;
    Lens Psi = init_lens(pNLens);

    params.y = y;
    params.x10 = x10;
    params.x20 = x20;
    params.sigmaf = sigmaf;
    params.Psi = &Psi;

    get_contour2d_x1x2(n_points, sol, &params);

    return 0;
}

