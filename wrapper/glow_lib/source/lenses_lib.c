/*
 * GLoW - lenses_lib.c
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
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_exp.h>

#include "common.h"
#include "lenses_lib.h"

#define EPS_SOFT 1e-15
#define EPS_SMALL_X_NFW 1e-2


// =================================================================

char *names_lenses[] = {"SIS",
                        "CIS",
                        "point lens",
                        "ball",
                        "NFW",
                        "tSIS",
                        "off-center SIS",
                        "off-center CIS",
                        "off-center point lens"
                        "off-center ball",
                        "off-center NFW",
                        "combined lens",
                        "grid 1d",
                        "eSIS",
                        "ext"};

Lens (*init_func_lenses[N_lenses])(void *) = { init_lens_SIS,
                                               init_lens_CIS,
                                               init_lens_PointLens,
                                               init_lens_Ball,
                                               init_lens_NFW,
                                               init_lens_tSIS,
                                               init_lens_offcenterSIS,
                                               init_lens_offcenterCIS,
                                               init_lens_offcenterPointLens,
                                               init_lens_offcenterBall,
                                               init_lens_offcenterNFW,
                                               init_lens_CombinedLens,
                                               init_lens_Grid1d,
                                               init_lens_eSIS,
                                               init_lens_Ext };

void (*free_func_lenses[N_lenses])(pNamedLens *) = { free_pLens_SIS,
                                                     free_pLens_CIS,
                                                     free_pLens_PointLens,
                                                     free_pLens_Ball,
                                                     free_pLens_NFW,
                                                     free_pLens_tSIS,
                                                     free_pLens_offcenterSIS,
                                                     free_pLens_offcenterCIS,
                                                     free_pLens_offcenterPointLens,
                                                     free_pLens_offcenterBall,
                                                     free_pLens_offcenterNFW,
                                                     free_pLens_CombinedLens,
                                                     free_pLens_Grid1d,
                                                     free_pLens_eSIS,
                                                     free_pLens_Ext };


// =================================================================

int x1x2_def(double *x_vec, double R, double alpha, double *x0_vec)
{
    x_vec[i_x1] = x0_vec[i_x1] + R*cos(alpha);
    x_vec[i_x2] = x0_vec[i_x2] + R*sin(alpha);

    return 0;
}

double phiFermat(double y, double x1, double x2, Lens *Psi)
{
    double phi, psi;

    psi = Psi->psi(x1, x2, Psi->pLens);
    phi = 0.5*x1*x1 + 0.5*x2*x2 - x1*y + 0.5*y*y - psi;

    return phi;
}

int phiFermat_1stDeriv(double *phi_derivs, double y, double x1, double x2, Lens *Psi)
{
    double phi, d1, d2;
    double psi, dpsi_dx1, dpsi_dx2;

    // fill psi derivatives reusing phi_derivs
    Psi->psi_1stDerivs(phi_derivs, x1, x2, Psi->pLens);

    psi = phi_derivs[i_0];
    dpsi_dx1 = phi_derivs[i_dx1];
    dpsi_dx2 = phi_derivs[i_dx2];

    phi = 0.5*x1*x1 + 0.5*x2*x2 - x1*y + 0.5*y*y - psi;
    d1 = x1 - y - dpsi_dx1;
    d2 = x2 - dpsi_dx2;

    phi_derivs[i_0] = phi;
    phi_derivs[i_dx1] = d1;
    phi_derivs[i_dx2] = d2;

    return 0;
}

int phiFermat_2ndDeriv(double *phi_derivs, double y, double x1, double x2, Lens *Psi)
{
    double phi, d1, d2, d11, d22, d12;
    double psi, dpsi_dx1, dpsi_dx2, ddpsi_dx1dx1, ddpsi_dx2dx2, ddpsi_dx1dx2;

    // fill psi derivatives reusing phi_derivs
    Psi->psi_2ndDerivs(phi_derivs, x1, x2, Psi->pLens);

    psi = phi_derivs[i_0];
    dpsi_dx1 = phi_derivs[i_dx1];
    dpsi_dx2 = phi_derivs[i_dx2];
    ddpsi_dx1dx1 = phi_derivs[i_dx1dx1];
    ddpsi_dx2dx2 = phi_derivs[i_dx2dx2];
    ddpsi_dx1dx2 = phi_derivs[i_dx1dx2];

    phi = 0.5*x1*x1 + 0.5*x2*x2 - x1*y + 0.5*y*y - psi;
    d1 = x1 - y - dpsi_dx1;
    d2 = x2 - dpsi_dx2;
    d11 = 1 - ddpsi_dx1dx1;
    d22 = 1 - ddpsi_dx2dx2;
    d12 = - ddpsi_dx1dx2;

    phi_derivs[i_0] = phi;
    phi_derivs[i_dx1] = d1;
    phi_derivs[i_dx2] = d2;
    phi_derivs[i_dx1dx1] = d11;
    phi_derivs[i_dx2dx2] = d22;
    phi_derivs[i_dx1dx2] = d12;

    return 0;
}

double magnification(double x1, double x2, Lens *Psi)
{
    double d11, d22, d12;
    double kappa, gamma1, gamma2, gamma_sq, inv_mag;
    double psi_derivs[N_derivs];

    Psi->psi_2ndDerivs(psi_derivs, x1, x2, Psi->pLens);

    d11 = psi_derivs[i_dx1dx1];
    d22 = psi_derivs[i_dx2dx2];
    d12 = psi_derivs[i_dx1dx2];

    kappa = 0.5*(d11 + d22);
    gamma1 = 0.5*(d11 - d22);
    gamma2 = d12;
    gamma_sq = gamma1*gamma1 + gamma2*gamma2;

    inv_mag = (1-kappa)*(1-kappa) - gamma_sq;

    return 1./ABS(inv_mag);
}


void rotate_vector(double *x_vec, double cos_th, double sin_th)
{
    double x1, x2;

    x1 = x_vec[i_x1];
    x2 = x_vec[i_x2];

    x_vec[i_x1] = x1*cos_th - x2*sin_th;
    x_vec[i_x2] = x1*sin_th + x2*cos_th;
}

void rotate_gradient(double *f_derivs, double cos_th, double sin_th)
{
    double d1, d2;

    d1 = f_derivs[i_dx1];
    d2 = f_derivs[i_dx2];

    f_derivs[i_dx1] = d1*cos_th + d2*sin_th;
    f_derivs[i_dx2] = -d1*sin_th + d2*cos_th;
}

void rotate_gradient_hessian(double *f_derivs, double cos_th, double sin_th)
{
    double a, b;
    double d1, d2, d11, d12, d22;

    d1  = f_derivs[i_dx1];
    d2  = f_derivs[i_dx2];
    d11 = f_derivs[i_dx1dx1];
    d12 = f_derivs[i_dx1dx2];
    d22 = f_derivs[i_dx2dx2];

    f_derivs[i_dx1] = d1*cos_th + d2*sin_th;
    f_derivs[i_dx2] = -d1*sin_th + d2*cos_th;

    a = 2*sin_th*cos_th*d12;
    b = sin_th*(d11-d22);
    f_derivs[i_dx1dx1] = d11 + a - sin_th*b;
    f_derivs[i_dx2dx2] = d22 - a + sin_th*b;
    f_derivs[i_dx1dx2] = d12*(1 - 2*sin_th*sin_th) - cos_th*b;
}


double *add_cusp_sing(int *n, double *xvec, double x1, double x2)
{
    char is_same_point;
    int i;
    double dx1, dx2, dist;

    // keep the point only if it is different from the rest in the list
    is_same_point = _FALSE_;
    for(i=0;i<*n;i+=2)
    {
        dx1 = x1 - xvec[i];
        dx2 = x2 - xvec[i+1];

        dist = sqrt(dx1*dx1 + dx2*dx2);
        if(dist < pprec.ro_issameCP_dist)
            is_same_point = _TRUE_;
    }

    if(is_same_point == _FALSE_)
    {
        xvec = (double *)realloc(xvec, (*n+2)*sizeof(double));
        xvec[*n]   = x1;
        xvec[*n+1] = x2;
        *n += 2;
    }

    return xvec;
}

// works recursively and can handle combinations of combined lenses as well
// x=(0, 0) is always included
double *get_cusp_sing(int *n, pNamedLens *pNLens)
{
    int i, lens_type;
    double x1, x2;
    static double *xvec = NULL;

    if(*n == 0)
    {
        xvec = (double *)malloc(2*sizeof(double));
        xvec[0] = 0;
        xvec[1] = 0;
        *n = 2;
    }

    lens_type = pNLens->lens_type;

    if(lens_type == i_offcenterSIS)
    {
        x1 = ((pLens_offcenterSIS *)(pNLens->pLens))->xc1;
        x2 = ((pLens_offcenterSIS *)(pNLens->pLens))->xc2;
        xvec = add_cusp_sing(n, xvec, x1, x2);
    }
    else if(lens_type == i_offcenterPointLens)
    {
        x1 = ((pLens_offcenterPointLens *)(pNLens->pLens))->xc1;
        x2 = ((pLens_offcenterPointLens *)(pNLens->pLens))->xc2;
        xvec = add_cusp_sing(n, xvec, x1, x2);
    }
    else if(lens_type == i_eSIS)
    {
        x1 = ((pLens_eSIS *)(pNLens->pLens))->xc1;
        x2 = ((pLens_eSIS *)(pNLens->pLens))->xc2;
        xvec = add_cusp_sing(n, xvec, x1, x2);
    }
    else if(lens_type == i_CombinedLens)
    {
        for(i=0;i<((pLens_CombinedLens*)(pNLens->pLens))->n_sublenses_added;i++)
            xvec = get_cusp_sing(n, ((pLens_CombinedLens*)(pNLens->pLens))->psublenses[i]);
    }

    return xvec;
}


double call_psi(double x1, double x2, Lens *Psi)
{
    return Psi->psi(x1, x2, Psi->pLens);
}

int call_psi_1stDerivs(double *psi_derivs, double x1, double x2, Lens *Psi)
{
    return Psi->psi_1stDerivs(psi_derivs, x1, x2, Psi->pLens);
}

int call_psi_2ndDerivs(double *psi_derivs, double x1, double x2, Lens *Psi)
{
    return Psi->psi_2ndDerivs(psi_derivs, x1, x2, Psi->pLens);
}


// =================================================================

Lens init_lens(pNamedLens* pNLens)
{
    return init_func_lenses[pNLens->lens_type](pNLens->pLens);
}

void free_pLens(pNamedLens* pNLens)
{
    free_func_lenses[pNLens->lens_type](pNLens);
}

// =================================================================


// =================================================================

pNamedLens* create_pLens_SIS(double psi0)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_SIS *pLens = (pLens_SIS*)malloc(sizeof(pLens_SIS));

    pLens->psi0 = psi0;

    pNLens->lens_type = i_SIS;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_SIS(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_SIS(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_SIS;
    Psi.psi_1stDerivs = psi_1stDerivs_SIS;
    Psi.psi_2ndDerivs = psi_2ndDerivs_SIS;
    Psi.pLens = pLens;

    return Psi;
}

double psi_SIS(double x1, double x2, void *pLens)
{
    double r;
    pLens_SIS *p = (pLens_SIS *)pLens;

    r = sqrt(x1*x1 + x2*x2);

    return p->psi0*r;
}

int psi_1stDerivs_SIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double r, psi, dpsi_dr, d1, d2;
    pLens_SIS *p = (pLens_SIS *)pLens;

    r = sqrt(x1*x1 + x2*x2);

    psi = p->psi0*r;
    dpsi_dr = p->psi0;

    d1 = dpsi_dr*x1/(r+EPS_SOFT);
    d2 = dpsi_dr*x2/(r+EPS_SOFT);

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_SIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double r, psi, dpsi_dr, ddpsi_drdr;
    double d1, d2, d11, d22, d12;
    double R1, R2;
    pLens_SIS *p = (pLens_SIS *)pLens;

    r = sqrt(x1*x1 + x2*x2);
    R1 = x1/(r+EPS_SOFT);
    R2 = x2/(r+EPS_SOFT);

    psi = p->psi0*r;
    dpsi_dr = p->psi0;
    ddpsi_drdr = 0;

    d1 = dpsi_dr*R1;
    d2 = dpsi_dr*R2;

    d11 = ddpsi_drdr*R1*R1 + dpsi_dr*(1-R1*R1)/(r+EPS_SOFT);
    d22 = ddpsi_drdr*R2*R2 + dpsi_dr*(1-R2*R2)/(r+EPS_SOFT);
    d12 = (ddpsi_drdr - dpsi_dr/(r+EPS_SOFT))*R1*R2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================

pNamedLens* create_pLens_offcenterSIS(double psi0, double xc1, double xc2)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_offcenterSIS *pLens = (pLens_offcenterSIS*)malloc(sizeof(pLens_offcenterSIS));

    pLens->pSIS.psi0 = psi0;
    pLens->xc1 = xc1;
    pLens->xc2 = xc2;

    pNLens->lens_type = i_offcenterSIS;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_offcenterSIS(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_offcenterSIS(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_offcenterSIS;
    Psi.psi_1stDerivs = psi_1stDerivs_offcenterSIS;
    Psi.psi_2ndDerivs = psi_2ndDerivs_offcenterSIS;
    Psi.pLens = pLens;

    return Psi;
}

double psi_offcenterSIS(double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterSIS *p = (pLens_offcenterSIS *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    return psi_SIS(x1_rel, x2_rel, &(p->pSIS));
}

int psi_1stDerivs_offcenterSIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterSIS *p = (pLens_offcenterSIS *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_1stDerivs_SIS(psi_derivs, x1_rel, x2_rel, &(p->pSIS));

    return 0;
}

int psi_2ndDerivs_offcenterSIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterSIS *p = (pLens_offcenterSIS *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_2ndDerivs_SIS(psi_derivs, x1_rel, x2_rel, &(p->pSIS));

    return 0;
}


// =================================================================

pNamedLens* create_pLens_CIS(double psi0, double rc)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_CIS *pLens = (pLens_CIS*)malloc(sizeof(pLens_CIS));

    pLens->psi0 = psi0;
    pLens->rc = rc;

    pNLens->lens_type = i_CIS;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_CIS(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_CIS(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_CIS;
    Psi.psi_1stDerivs = psi_1stDerivs_CIS;
    Psi.psi_2ndDerivs = psi_2ndDerivs_CIS;
    Psi.pLens = pLens;

    return Psi;
}

double psi_CIS(double x1, double x2, void *pLens)
{
    double x, sqr;
    pLens_CIS *p = (pLens_CIS *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    sqr = sqrt(x*x + p->rc*p->rc);

    return p->psi0*(sqr + p->rc*log(2.*p->rc/(sqr + p->rc)));
}

int psi_1stDerivs_CIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, sqr, psi, dpsi_dr, d1, d2;
    pLens_CIS *p = (pLens_CIS *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    sqr = sqrt(x*x + p->rc*p->rc);

    psi = p->psi0*(sqr + p->rc*log(2.*p->rc/(sqr + p->rc)));
    dpsi_dr = p->psi0*x/sqr*(1. - p->rc/(sqr + p->rc));

    d1 = dpsi_dr*x1/(x+EPS_SOFT);
    d2 = dpsi_dr*x2/(x+EPS_SOFT);

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_CIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, sqr, psi, dpsi_dr, ddpsi_drdr;
    double d1, d2, d11, d22, d12;
    double R1, R2, R, tmp1, tmp2;
    pLens_CIS *p = (pLens_CIS *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    sqr = sqrt(x*x + p->rc*p->rc);
    R = (x/sqr)*(x/sqr);
    R1 = x1/(x+EPS_SOFT);
    R2 = x2/(x+EPS_SOFT);

    psi = p->psi0*(sqr + p->rc*log(2.*p->rc/(sqr + p->rc)));
    dpsi_dr = p->psi0*x/sqr*(1. - p->rc/(sqr + p->rc));

    tmp1 = p->psi0*R*p->rc/(sqr+p->rc)/(sqr+p->rc);
    tmp2 = p->psi0*(1 - p->rc/(sqr+p->rc))*(1-R)/sqr;
    ddpsi_drdr = tmp1+tmp2;

    d1 = dpsi_dr*R1;
    d2 = dpsi_dr*R2;

    d11 = ddpsi_drdr*R1*R1 + dpsi_dr*(1-R1*R1)/(x+EPS_SOFT);
    d22 = ddpsi_drdr*R2*R2 + dpsi_dr*(1-R2*R2)/(x+EPS_SOFT);
    d12 = (ddpsi_drdr - dpsi_dr/(x+EPS_SOFT))*R1*R2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================

pNamedLens* create_pLens_offcenterCIS(double psi0, double rc, double xc1, double xc2)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_offcenterCIS *pLens = (pLens_offcenterCIS*)malloc(sizeof(pLens_offcenterCIS));

    pLens->pCIS.psi0 = psi0;
    pLens->pCIS.rc = rc;
    pLens->xc1 = xc1;
    pLens->xc2 = xc2;

    pNLens->lens_type = i_offcenterCIS;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_offcenterCIS(pNamedLens* pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_offcenterCIS(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_offcenterCIS;
    Psi.psi_1stDerivs = psi_1stDerivs_offcenterCIS;
    Psi.psi_2ndDerivs = psi_2ndDerivs_offcenterCIS;
    Psi.pLens = pLens;

    return Psi;
}

double psi_offcenterCIS(double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterCIS *p = (pLens_offcenterCIS *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    return psi_CIS(x1_rel, x2_rel, &(p->pCIS));
}

int psi_1stDerivs_offcenterCIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterCIS *p = (pLens_offcenterCIS *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_1stDerivs_CIS(psi_derivs, x1_rel, x2_rel, &(p->pCIS));

    return 0;
}

int psi_2ndDerivs_offcenterCIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterCIS *p = (pLens_offcenterCIS *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_2ndDerivs_CIS(psi_derivs, x1_rel, x2_rel, &(p->pCIS));

    return 0;
}


// =================================================================

pNamedLens* create_pLens_CombinedLens(int n_sublenses)
{
    pNamedLens *pNLens = (pNamedLens *)malloc(sizeof(pNamedLens));
    pLens_CombinedLens *pLens= (pLens_CombinedLens *)malloc(sizeof(pLens_CombinedLens));

    // note that, after creating pLens_CombinedLens, the individual pLens for
    // the sublenses must still be created manually, filling the array psublenses
    // Once this is done, the full object is ready to be created with init_lens
    pLens->n_sublenses_added = 0;
    pLens->n_sublenses = n_sublenses;
    pLens->psublenses = (pNamedLens **)malloc(n_sublenses*sizeof(pNamedLens*));
    pLens->sublenses = (Lens *)malloc(n_sublenses*sizeof(Lens));

    pNLens->lens_type = i_CombinedLens;
    pNLens->pLens = pLens;

    return pNLens;
}

int add_lens_CombinedLens(pNamedLens* new_pNLens, pNamedLens* combined_pNLens)
{
    pLens_CombinedLens *p = (pLens_CombinedLens *)combined_pNLens->pLens;

    if(p->n_sublenses_added < p->n_sublenses)
    {
        p->psublenses[p->n_sublenses_added] = new_pNLens;
        p->n_sublenses_added++;
    }
    else
    {
        PERROR("impossible to add new sublens (n_added=%d, n_sublenses=%d)", p->n_sublenses_added, p->n_sublenses)
        return 1;
    }

    return 0;
}

void free_pLens_CombinedLens(pNamedLens* pNLens)
{
    int i;
    pLens_CombinedLens *p = (pLens_CombinedLens *)pNLens->pLens;

    for(i=0;i<p->n_sublenses_added;i++)
        free_pLens(p->psublenses[i]);

    free(p->psublenses);
    free(p->sublenses);

    free(p);
    free(pNLens);
}

Lens init_lens_CombinedLens(void *pLens)
{
    int i;
    Lens Psi;
    pLens_CombinedLens *p = (pLens_CombinedLens *)pLens;

    Psi.psi = psi_CombinedLens;
    Psi.psi_1stDerivs = psi_1stDerivs_CombinedLens;
    Psi.psi_2ndDerivs = psi_2ndDerivs_CombinedLens;
    Psi.pLens = pLens;

    if(p->n_sublenses_added != p->n_sublenses)
        PWARNING("not all the required lenses have been added (n_added=%d, n_sublenses=%d)", p->n_sublenses_added, p->n_sublenses)

    // create all the sublenses with the parameters provided
    for(i=0;i<p->n_sublenses_added;i++)
        p->sublenses[i] = init_lens(p->psublenses[i]);

    return Psi;
}

double psi_CombinedLens(double x1, double x2, void *pLens)
{
    int i;
    double psi=0;
    Lens sublens;
    pLens_CombinedLens *p = (pLens_CombinedLens *)pLens;

    for(i=0;i<p->n_sublenses_added;i++)
    {
        sublens = p->sublenses[i];
        psi += sublens.psi(x1, x2, sublens.pLens);
    }

    return psi;
}

int psi_1stDerivs_CombinedLens(double *psi_derivs, double x1, double x2, void *pLens)
{
    int i;
    double psi=0, d1=0, d2=0;
    Lens sublens;
    pLens_CombinedLens *p = (pLens_CombinedLens *)pLens;

    for(i=0;i<p->n_sublenses_added;i++)
    {
        sublens = p->sublenses[i];
        sublens.psi_1stDerivs(psi_derivs, x1, x2, sublens.pLens);
        psi += psi_derivs[i_0];
        d1 += psi_derivs[i_dx1];
        d2 += psi_derivs[i_dx2];
    }

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_CombinedLens(double *psi_derivs, double x1, double x2, void *pLens)
{
    int i;
    double psi=0, d1=0, d2=0, d11=0, d22=0, d12=0;
    Lens sublens;
    pLens_CombinedLens *p = (pLens_CombinedLens *)pLens;

    for(i=0;i<p->n_sublenses_added;i++)
    {
        sublens = p->sublenses[i];
        sublens.psi_2ndDerivs(psi_derivs, x1, x2, sublens.pLens);
        psi += psi_derivs[i_0];
        d1 += psi_derivs[i_dx1];
        d2 += psi_derivs[i_dx2];
        d11 += psi_derivs[i_dx1dx1];
        d22 += psi_derivs[i_dx2dx2];
        d12 += psi_derivs[i_dx1dx2];
    }

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================

double F_NFW(double u)
{
    double sqr;
    double F;

    if(u > 1 + EPS_SOFT)
    {
        sqr = sqrt((u-1)*(u+1));
        F = atan(sqr)/sqr;
    }
    else
    {
        if(u < 1 - EPS_SOFT)
        {
            sqr = sqrt((1-u)*(1+u));
            F = atanh(sqr)/sqr;
        }
        else
            F = 1 + 2*ABS(u-1);  // slight improvement with first correction
    }

    return F;
}

pNamedLens* create_pLens_NFW(double psi0, double xs)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_NFW *pLens = (pLens_NFW*)malloc(sizeof(pLens_NFW));

    pLens->psi0 = psi0;
    pLens->xs = xs;

    pNLens->lens_type = i_NFW;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_NFW(pNamedLens* pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_NFW(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_NFW;
    Psi.psi_1stDerivs = psi_1stDerivs_NFW;
    Psi.psi_2ndDerivs = psi_2ndDerivs_NFW;
    Psi.pLens = pLens;

    return Psi;
}

double psi_NFW(double x1, double x2, void *pLens)
{
    double u, psi, lg, F;
    double u2, u4, u6, u8;
    pLens_NFW *p = (pLens_NFW *)pLens;

    u = sqrt(x1*x1 + x2*x2)/p->xs;
    u2 = u*u;

    lg = log(0.5*u + EPS_SOFT);

    // needed for accurate results
    if(u > EPS_SMALL_X_NFW)
    {
        F = F_NFW(u);
        psi = p->psi0/2.*(lg*lg + (u2-1)*F*F);
    }
    else
    {
        u4=u2*u2;  u6=u4*u2;  u8=u4*u4;
        psi = -p->psi0/4.*(u2*lg + u4/8.*(1+3*lg) + u6/32.*(3+20/3.*lg) + u8/128*(107./12+35/2.*lg));
    }

    return psi;
}

int psi_1stDerivs_NFW(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, u, psi, lg, F;
    double u2, u3, u4, u5, u6, u7, u8;
    double dpsi_dx, d1, d2;
    pLens_NFW *p = (pLens_NFW *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    u = x/p->xs;
    u2 = u*u;

    lg = log(0.5*u + EPS_SOFT);

    if(u > EPS_SMALL_X_NFW)
    {
        F = F_NFW(u);
        psi = p->psi0/2.*(lg*lg + (u2-1)*F*F);
        dpsi_dx = p->psi0/p->xs*(lg + F)/u;
    }
    else
    {
        u3=u*u2;  u4=u2*u2;  u5=u4*u; u6=u4*u2;  u7=u6*u; u8=u4*u4;
        psi = -p->psi0/4.*(u2*lg + u4/8.*(1+3*lg) + u6/32.*(3+20/3.*lg) + u8/128*(107./12+35/2.*lg));
        dpsi_dx = -0.5*p->psi0/p->xs*(u*(0.5+lg) + u3*(7./16+3./4*lg) + u5*(37./96+5./8*lg) + u7*(533./1536+35./64*lg));
    }

    d1 = dpsi_dx*x1/(x+EPS_SOFT);
    d2 = dpsi_dx*x2/(x+EPS_SOFT);

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_NFW(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, u, psi, lg, F;
    double u2, u3, u4, u5, u6, u7, u8;
    double dpsi_dx, d1, d2, R1, R2;
    double ddpsi_dxdx, d11, d12, d22;
    pLens_NFW *p = (pLens_NFW *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    u = x/p->xs;
    u2 = u*u;

    R1 = x1/(x+EPS_SOFT);
    R2 = x2/(x+EPS_SOFT);

    lg = log(0.5*u + EPS_SOFT);

    if(u > EPS_SMALL_X_NFW)
    {
        F = F_NFW(u);
        psi = p->psi0/2.*(lg*lg + (u2-1)*F*F);
        dpsi_dx = p->psi0/p->xs*(lg + F)/u;
        ddpsi_dxdx = -p->psi0/p->xs/p->xs*( lg + (u2 + F*(1 - 2*u2))/((1-u)*(1+u)+EPS_SOFT) )/u2;
    }
    else
    {
        u3=u*u2;  u4=u2*u2;  u5=u4*u; u6=u4*u2;  u7=u6*u; u8=u4*u4;
        psi = -p->psi0/4.*(u2*lg + u4/8.*(1+3*lg) + u6/32.*(3+20/3.*lg) + u8/128*(107./12+35/2.*lg));
        dpsi_dx = -0.5*p->psi0/p->xs*(u*(0.5+lg) + u3*(7./16+3./4*lg) + u5*(37./96+5./8*lg) + u7*(533./1536+35./64*lg));
        ddpsi_dxdx = -0.5*p->psi0/p->xs/p->xs*(1.5 + lg + u2*(33./16+9./4*lg) + u4*(245./96+25./8*lg) + u6*(4571./1536+245./64*lg));
    }

    d1 = dpsi_dx*R1;
    d2 = dpsi_dx*R2;

    d11 = ddpsi_dxdx*R1*R1 + dpsi_dx*(1-R1*R1)/(x+EPS_SOFT);
    d22 = ddpsi_dxdx*R2*R2 + dpsi_dx*(1-R2*R2)/(x+EPS_SOFT);
    d12 = (ddpsi_dxdx - dpsi_dx/(x+EPS_SOFT))*R1*R2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================

pNamedLens* create_pLens_offcenterNFW(double psi0, double xs, double xc1, double xc2)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_offcenterNFW *pLens = (pLens_offcenterNFW*)malloc(sizeof(pLens_offcenterNFW));

    pLens->pNFW.psi0 = psi0;
    pLens->pNFW.xs = xs;
    pLens->xc1 = xc1;
    pLens->xc2 = xc2;

    pNLens->lens_type = i_offcenterNFW;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_offcenterNFW(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_offcenterNFW(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_offcenterNFW;
    Psi.psi_1stDerivs = psi_1stDerivs_offcenterNFW;
    Psi.psi_2ndDerivs = psi_2ndDerivs_offcenterNFW;
    Psi.pLens = pLens;

    return Psi;
}

double psi_offcenterNFW(double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterNFW *p = (pLens_offcenterNFW *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    return psi_NFW(x1_rel, x2_rel, &(p->pNFW));
}

int psi_1stDerivs_offcenterNFW(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterNFW *p = (pLens_offcenterNFW *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_1stDerivs_NFW(psi_derivs, x1_rel, x2_rel, &(p->pNFW));

    return 0;
}

int psi_2ndDerivs_offcenterNFW(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterNFW *p = (pLens_offcenterNFW *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_2ndDerivs_NFW(psi_derivs, x1_rel, x2_rel, &(p->pNFW));

    return 0;
}


// =================================================================

pNamedLens* create_pLens_PointLens(double psi0, double xc)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_PointLens *pLens = (pLens_PointLens*)malloc(sizeof(pLens_PointLens));

    pLens->psi0 = psi0;
    pLens->xc = xc;

    pNLens->lens_type = i_PointLens;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_PointLens(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_PointLens(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_PointLens;
    Psi.psi_1stDerivs = psi_1stDerivs_PointLens;
    Psi.psi_2ndDerivs = psi_2ndDerivs_PointLens;
    Psi.pLens = pLens;

    return Psi;
}

double psi_PointLens(double x1, double x2, void *pLens)
{
    double xx, r2;
    pLens_PointLens *p = (pLens_PointLens *)pLens;

    xx = x1*x1 + x2*x2;
    r2 = xx + p->xc*p->xc;

    return 0.5*p->psi0*log(r2);
}

int psi_1stDerivs_PointLens(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, xx, r2, psi, dpsi_dr, d1, d2;
    pLens_PointLens *p = (pLens_PointLens *)pLens;

    xx = x1*x1 + x2*x2;
    x  = sqrt(xx);
    r2 = xx + p->xc*p->xc;

    psi = 0.5*p->psi0*log(r2);
    dpsi_dr = p->psi0*x/r2;

    d1 = dpsi_dr*x1/(x+EPS_SOFT);
    d2 = dpsi_dr*x2/(x+EPS_SOFT);

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_PointLens(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, xx, r2, psi, dpsi_dr, ddpsi_drdr;
    double d1, d2, d11, d22, d12;
    double R1, R2;
    pLens_PointLens *p = (pLens_PointLens *)pLens;

    xx = x1*x1 + x2*x2;
    x  = sqrt(xx);
    r2 = xx + p->xc*p->xc;

    R1 = x1/(x+EPS_SOFT);
    R2 = x2/(x+EPS_SOFT);

    psi = 0.5*p->psi0*log(r2);
    dpsi_dr = p->psi0*x/r2;
    ddpsi_drdr = p->psi0*(p->xc*p->xc - xx)/r2/r2;

    d1 = dpsi_dr*R1;
    d2 = dpsi_dr*R2;

    d11 = ddpsi_drdr*R1*R1 + dpsi_dr*(1-R1*R1)/(x+EPS_SOFT);
    d22 = ddpsi_drdr*R2*R2 + dpsi_dr*(1-R2*R2)/(x+EPS_SOFT);
    d12 = (ddpsi_drdr - dpsi_dr/(x+EPS_SOFT))*R1*R2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================

pNamedLens* create_pLens_offcenterPointLens(double psi0, double xc, double xc1, double xc2)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_offcenterPointLens *pLens = (pLens_offcenterPointLens*)malloc(sizeof(pLens_offcenterPointLens));

    pLens->pPointLens.psi0 = psi0;
    pLens->pPointLens.xc = xc;
    pLens->xc1 = xc1;
    pLens->xc2 = xc2;

    pNLens->lens_type = i_offcenterPointLens;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_offcenterPointLens(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_offcenterPointLens(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_offcenterPointLens;
    Psi.psi_1stDerivs = psi_1stDerivs_offcenterPointLens;
    Psi.psi_2ndDerivs = psi_2ndDerivs_offcenterPointLens;
    Psi.pLens = pLens;

    return Psi;
}

double psi_offcenterPointLens(double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterPointLens *p = (pLens_offcenterPointLens *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    return psi_PointLens(x1_rel, x2_rel, &(p->pPointLens));
}

int psi_1stDerivs_offcenterPointLens(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterPointLens *p = (pLens_offcenterPointLens *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_1stDerivs_PointLens(psi_derivs, x1_rel, x2_rel, &(p->pPointLens));

    return 0;
}

int psi_2ndDerivs_offcenterPointLens(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterPointLens *p = (pLens_offcenterPointLens *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_2ndDerivs_PointLens(psi_derivs, x1_rel, x2_rel, &(p->pPointLens));

    return 0;
}


// =================================================================

pNamedLens* create_pLens_Ball(double psi0, double b)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_Ball *pLens = (pLens_Ball*)malloc(sizeof(pLens_Ball));

    pLens->psi0 = psi0;
    pLens->b = b;

    pNLens->lens_type = i_Ball;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_Ball(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_Ball(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_Ball;
    Psi.psi_1stDerivs = psi_1stDerivs_Ball;
    Psi.psi_2ndDerivs = psi_2ndDerivs_Ball;
    Psi.pLens = pLens;

    return Psi;
}

double psi_Ball(double x1, double x2, void *pLens)
{
    double r, psi, X;
    pLens_Ball *p = (pLens_Ball *)pLens;

    r = sqrt(x1*x1 + x2*x2);

    if(r < p->b)
    {
        X = sqrt(1-r*r/p->b/p->b);
        psi = p->psi0*(log(p->b*(1+X)) - X*(1+X*X/3.));
    }
    else
    {
        psi = p->psi0*log(r);
    }

    return psi;
}

int psi_1stDerivs_Ball(double *psi_derivs, double x1, double x2, void *pLens)
{
    double r, psi, X, dpsi_dr, d1, d2;
    pLens_Ball *p = (pLens_Ball *)pLens;

    r = sqrt(x1*x1 + x2*x2);

    if(r < p->b)
    {
        X = sqrt(1-r*r/p->b/p->b);
        psi = p->psi0*(log(p->b*(1+X)) - X*(1+X*X/3.));
        dpsi_dr = p->psi0*r*(1 + X*X/(X+1))/p->b/p->b;
    }
    else
    {
        psi = p->psi0*log(r);
        dpsi_dr = p->psi0/r;
    }

    d1 = dpsi_dr*x1/(r+EPS_SOFT);
    d2 = dpsi_dr*x2/(r+EPS_SOFT);

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_Ball(double *psi_derivs, double x1, double x2, void *pLens)
{
    double r, X, psi, dpsi_dr, ddpsi_drdr;
    double d1, d2, d11, d22, d12;
    double R1, R2;
    pLens_Ball *p = (pLens_Ball *)pLens;

    r = sqrt(x1*x1 + x2*x2);
    R1 = x1/(r+EPS_SOFT);
    R2 = x2/(r+EPS_SOFT);

    if(r < p->b)
    {
        X = sqrt(1-r*r/p->b/p->b);
        psi = p->psi0*(log(p->b*(1+X)) - X*(1+X*X/3.));
        dpsi_dr = p->psi0*r*(1 + X*X/(X+1))/p->b/p->b;
        ddpsi_drdr = p->psi0*(2*X*X + 2*X - 1)/(X+1)/p->b/p->b;
    }
    else
    {
        psi = p->psi0*log(r);
        dpsi_dr = p->psi0/r;
        ddpsi_drdr = -p->psi0/r/r;
    }

    d1 = dpsi_dr*R1;
    d2 = dpsi_dr*R2;

    d11 = ddpsi_drdr*R1*R1 + dpsi_dr*(1-R1*R1)/(r+EPS_SOFT);
    d22 = ddpsi_drdr*R2*R2 + dpsi_dr*(1-R2*R2)/(r+EPS_SOFT);
    d12 = (ddpsi_drdr - dpsi_dr/(r+EPS_SOFT))*R1*R2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================

pNamedLens* create_pLens_offcenterBall(double psi0, double b, double xc1, double xc2)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_offcenterBall *pLens = (pLens_offcenterBall*)malloc(sizeof(pLens_offcenterBall));

    pLens->pBall.psi0 = psi0;
    pLens->pBall.b = b;
    pLens->xc1 = xc1;
    pLens->xc2 = xc2;

    pNLens->lens_type = i_offcenterBall;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_offcenterBall(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_offcenterBall(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_offcenterBall;
    Psi.psi_1stDerivs = psi_1stDerivs_offcenterBall;
    Psi.psi_2ndDerivs = psi_2ndDerivs_offcenterBall;
    Psi.pLens = pLens;

    return Psi;
}

double psi_offcenterBall(double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterBall *p = (pLens_offcenterBall *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    return psi_Ball(x1_rel, x2_rel, &(p->pBall));
}

int psi_1stDerivs_offcenterBall(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterBall *p = (pLens_offcenterBall *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_1stDerivs_Ball(psi_derivs, x1_rel, x2_rel, &(p->pBall));

    return 0;
}

int psi_2ndDerivs_offcenterBall(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x1_rel, x2_rel;
    pLens_offcenterBall *p = (pLens_offcenterBall *)pLens;

    x1_rel = x1 - p->xc1;
    x2_rel = x2 - p->xc2;

    psi_2ndDerivs_Ball(psi_derivs, x1_rel, x2_rel, &(p->pBall));

    return 0;
}


// =================================================================

double F_tSIS(double u)
{
    double F, u2=u*u;

    if(u > 1e-2)
    {
        if(u < 5)
            F = (2*log(u) + gsl_sf_expint_E1(u2) + M_EULER)/u;
        else
            F = (2*log(u) + exp(-u2)*(1/u - 1/u2) + M_EULER)/u;
    }
    else
        F = u*(1 - u2/4. + u2*u2/18.);

    return F;
}

pNamedLens* create_pLens_tSIS(double psi0, double xb)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_tSIS *pLens = (pLens_tSIS*)malloc(sizeof(pLens_tSIS));

    pLens->psi0 = psi0;
    pLens->xb = xb;

    pNLens->lens_type = i_tSIS;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_tSIS(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_tSIS(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_tSIS;
    Psi.psi_1stDerivs = psi_1stDerivs_tSIS;
    Psi.psi_2ndDerivs = psi_2ndDerivs_tSIS;
    Psi.pLens = pLens;

    return Psi;
}

double psi_tSIS(double x1, double x2, void *pLens)
{
    double x, u, u2, psi, error_f, exprel;
    pLens_tSIS *p = (pLens_tSIS *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    u = x/p->xb;
    u2 = u*u;

    error_f = gsl_sf_erfc(u);
    exprel = gsl_sf_exprel(-u2)/M_SQRTPI;

    psi = p->psi0*x*(error_f + u*exprel + 0.5*F_tSIS(u)/M_SQRTPI);

    return psi;
}

int psi_1stDerivs_tSIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, u, u2, psi, error_f, exprel;
    double d1, d2, dpsi_dx;
    pLens_tSIS *p = (pLens_tSIS *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    u = x/p->xb;
    u2 = u*u;

    error_f = gsl_sf_erfc(u);
    exprel = gsl_sf_exprel(-u2)/M_SQRTPI;

    psi = p->psi0*x*(error_f + u*exprel + 0.5*F_tSIS(u)/M_SQRTPI);
    dpsi_dx = p->psi0*(error_f + u*exprel);

    d1 = dpsi_dx*x1/(x+EPS_SOFT);
    d2 = dpsi_dx*x2/(x+EPS_SOFT);

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_tSIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, u, u2, psi, error_f, exprel;
    double d1, d2, d11, d22, d12, dpsi_dx, ddpsi_dxdx;
    double R1, R2;
    pLens_tSIS *p = (pLens_tSIS *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    u = x/p->xb;
    u2 = u*u;

    error_f = gsl_sf_erfc(u);
    exprel = gsl_sf_exprel(-u2)/M_SQRTPI;

    psi = p->psi0*x*(error_f + u*exprel + 0.5*F_tSIS(u)/M_SQRTPI);
    dpsi_dx = p->psi0*(error_f + u*exprel);
    ddpsi_dxdx = -p->psi0/p->xb*exprel;

    R1 = x1/(x+EPS_SOFT);
    R2 = x2/(x+EPS_SOFT);

    d1 = dpsi_dx*R1;
    d2 = dpsi_dx*R2;

    d11 = ddpsi_dxdx*R1*R1 + dpsi_dx*(1-R1*R1)/(x+EPS_SOFT);
    d22 = ddpsi_dxdx*R2*R2 + dpsi_dx*(1-R2*R2)/(x+EPS_SOFT);
    d12 = (ddpsi_dxdx - dpsi_dx/(x+EPS_SOFT))*R1*R2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================

double eval_Interp1d(double x, Interp1d *f)
{
    // check bounds, if outside perform constant extrapolation
    if(x > f->x_grid[f->n_grid-1])
        x = f->x_grid[f->n_grid-1];
    else if(x < f->x_grid[0])
        x = f->x_grid[0];

    return gsl_spline_eval((gsl_spline *)f->spline, x, (gsl_interp_accel *)f->acc);
}

Interp1d* init_Interp1d(char *fname, int n_grid)
{
    double *x, *y;
    gsl_interp_accel *acc;
    gsl_spline *spline;
    const gsl_interp_type *interp_kind;
    Interp1d *f = (Interp1d *)malloc(sizeof(Interp1d));

    // load grid
    x = (double *)calloc(n_grid, sizeof(double));
    y = (double *)calloc(n_grid, sizeof(double));

    load_Grid1d(fname, x, y, n_grid);

    // init interpolant
    interp_kind = gsl_interp_linear;
    //~ interp_kind = gsl_interp_cspline;

    acc = gsl_interp_accel_alloc();
    spline = gsl_spline_alloc(interp_kind, n_grid);
    gsl_spline_init(spline, x, y, n_grid);

    f->n_grid = n_grid;
    f->x_grid = x;
    f->y_grid = y;
    f->acc = acc;
    f->spline = spline;

    return f;
}

void free_Interp1d(Interp1d *f)
{
    gsl_spline *spline    = (gsl_spline *)f->spline;
    gsl_interp_accel *acc = (gsl_interp_accel *)f->acc;

    free(f->x_grid);
    free(f->y_grid);
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);

    free(f);
}

void load_Grid1d(char *fname, double *xvals, double *yvals, int nvals)
{
    int i=0;
    FILE *fp;

    if( (fp = fopen(fname, "r")) == NULL )
        PERROR("file %s not found, lens grid will be empty", fname)
    else
    {
        while( (fscanf(fp, "%lf %lf", xvals+i, yvals+i) == 2) && (i<nvals) )
            i++;
        fclose(fp);
    }
}


pNamedLens* create_pLens_Grid1d(char *fname, int n_grid)
{
    char full_name[1000];
    pNamedLens *pNLens  = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_Grid1d *pLens = (pLens_Grid1d*)malloc(sizeof(pLens_Grid1d));

    pNLens->lens_type = i_Grid1d;
    pNLens->pLens = pLens;

    sprintf(full_name, "%s_psi_lens.dat", fname);
    pLens->psi_interp = init_Interp1d(full_name, n_grid);

    sprintf(full_name, "%s_dpsi_lens.dat", fname);
    pLens->dpsi_interp = init_Interp1d(full_name, n_grid);

    sprintf(full_name, "%s_ddpsi_lens.dat", fname);
    pLens->ddpsi_interp = init_Interp1d(full_name, n_grid);

    return pNLens;
}

void free_pLens_Grid1d(pNamedLens *pNLens)
{
    pLens_Grid1d *p = (pLens_Grid1d *)pNLens->pLens;

    free_Interp1d(p->psi_interp);
    free_Interp1d(p->dpsi_interp);
    free_Interp1d(p->ddpsi_interp);

    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_Grid1d(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_Grid1d;
    Psi.psi_1stDerivs = psi_1stDerivs_Grid1d;
    Psi.psi_2ndDerivs = psi_2ndDerivs_Grid1d;
    Psi.pLens = pLens;

    return Psi;
}

double psi_Grid1d(double x1, double x2, void *pLens)
{
    double x, psi;
    pLens_Grid1d *p = (pLens_Grid1d *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    psi = eval_Interp1d(x, p->psi_interp);

    return psi;
}

int psi_1stDerivs_Grid1d(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, psi, dpsi_dr, d1, d2;
    pLens_Grid1d *p = (pLens_Grid1d *)pLens;

    x = sqrt(x1*x1 + x2*x2);

    psi = eval_Interp1d(x, p->psi_interp);
    dpsi_dr = eval_Interp1d(x, p->dpsi_interp);

    d1 = dpsi_dr*x1/(x+EPS_SOFT);
    d2 = dpsi_dr*x2/(x+EPS_SOFT);

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_Grid1d(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x, psi, dpsi_dr, ddpsi_drdr;
    double d1, d2, d11, d22, d12;
    double R1, R2;
    pLens_Grid1d *p = (pLens_Grid1d *)pLens;

    x = sqrt(x1*x1 + x2*x2);
    R1 = x1/(x+EPS_SOFT);
    R2 = x2/(x+EPS_SOFT);

    psi = eval_Interp1d(x, p->psi_interp);
    dpsi_dr = eval_Interp1d(x, p->dpsi_interp);
    ddpsi_drdr = eval_Interp1d(x, p->ddpsi_interp);

    d1 = dpsi_dr*R1;
    d2 = dpsi_dr*R2;

    d11 = ddpsi_drdr*R1*R1 + dpsi_dr*(1-R1*R1)/(x+EPS_SOFT);
    d22 = ddpsi_drdr*R2*R2 + dpsi_dr*(1-R2*R2)/(x+EPS_SOFT);
    d12 = (ddpsi_drdr - dpsi_dr/(x+EPS_SOFT))*R1*R2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================

pNamedLens* create_pLens_eSIS(double psi0, double q, double alpha, double xc1, double xc2)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_eSIS *pLens = (pLens_eSIS*)malloc(sizeof(pLens_eSIS));

    pLens->psi0 = psi0;
    pLens->q = q;
    pLens->alpha = alpha;
    pLens->ca = cos(alpha);
    pLens->sa = sin(alpha);
    pLens->xc1 = xc1;
    pLens->xc2 = xc2;

    pNLens->lens_type = i_eSIS;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_eSIS(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_eSIS(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_eSIS;
    Psi.psi_1stDerivs = psi_1stDerivs_eSIS;
    Psi.psi_2ndDerivs = psi_2ndDerivs_eSIS;
    Psi.pLens = pLens;

    return Psi;
}

double psi_a0_eSIS(double x1, double x2, void *pLens)
{
    double r;
    pLens_eSIS *p = (pLens_eSIS *)pLens;
    double q2 = p->q*p->q;

    r = sqrt(x1*x1 + x2*x2/q2);

    return p->psi0*r;
}

int psi_1stDerivs_a0_eSIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double r, psi, dpsi_dr, d1, d2;
    pLens_eSIS *p = (pLens_eSIS *)pLens;
    double q2 = p->q*p->q;

    r = sqrt(x1*x1 + x2*x2/q2);

    psi = p->psi0*r;
    dpsi_dr = p->psi0;

    d1 = dpsi_dr*x1/(r+EPS_SOFT);
    d2 = dpsi_dr*x2/(r+EPS_SOFT)/q2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_a0_eSIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double r, psi, dpsi_dr, ddpsi_drdr;
    double d1, d2, d11, d22, d12;
    double R1, R2;
    pLens_eSIS *p = (pLens_eSIS *)pLens;
    double q2 = p->q*p->q;

    r = sqrt(x1*x1 + x2*x2/q2);
    R1 = x1/(r+EPS_SOFT);
    R2 = x2/(r+EPS_SOFT)/q2;

    psi = p->psi0*r;
    dpsi_dr = p->psi0;
    ddpsi_drdr = 0;

    d1 = dpsi_dr*R1;
    d2 = dpsi_dr*R2;

    d11 = ddpsi_drdr*R1*R1 + dpsi_dr*(1-R1*R1)/(r+EPS_SOFT);
    d22 = ddpsi_drdr*R2*R2 + dpsi_dr*(1./q2-R2*R2)/(r+EPS_SOFT);
    d12 = (ddpsi_drdr - dpsi_dr/(r+EPS_SOFT))*R1*R2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}

double psi_eSIS(double x1, double x2, void *pLens)
{
    double x_vec[N_dims];
    pLens_eSIS *p = (pLens_eSIS *)pLens;

    x_vec[i_x1] = x1 - p->xc1;
    x_vec[i_x2] = x2 - p->xc2;
    rotate_vector(x_vec, p->ca, p->sa);

    return psi_a0_eSIS(x_vec[i_x1], x_vec[i_x2], pLens);
}

int psi_1stDerivs_eSIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x_vec[N_dims];
    pLens_eSIS *p = (pLens_eSIS *)pLens;

    x_vec[i_x1] = x1 - p->xc1;
    x_vec[i_x2] = x2 - p->xc2;
    rotate_vector(x_vec, p->ca, p->sa);

    psi_1stDerivs_a0_eSIS(psi_derivs, x_vec[i_x1], x_vec[i_x2], pLens);
    rotate_gradient(psi_derivs, p->ca, p->sa);

    return 0;
}

int psi_2ndDerivs_eSIS(double *psi_derivs, double x1, double x2, void *pLens)
{
    double x_vec[N_dims];
    pLens_eSIS *p = (pLens_eSIS *)pLens;

    x_vec[i_x1] = x1 - p->xc1;
    x_vec[i_x2] = x2 - p->xc2;
    rotate_vector(x_vec, p->ca, p->sa);

    psi_2ndDerivs_a0_eSIS(psi_derivs, x_vec[i_x1], x_vec[i_x2], pLens);
    rotate_gradient_hessian(psi_derivs, p->ca, p->sa);

    return 0;
}


// =================================================================

pNamedLens* create_pLens_Ext(double kappa, double gamma1, double gamma2)
{
    pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
    pLens_Ext *pLens   = (pLens_Ext*)malloc(sizeof(pLens_Ext));

    pLens->kappa  = kappa;
    pLens->gamma1 = gamma1;
    pLens->gamma2 = gamma2;

    pNLens->lens_type = i_Ext;
    pNLens->pLens = pLens;

    return pNLens;
}

void free_pLens_Ext(pNamedLens *pNLens)
{
    free(pNLens->pLens);
    free(pNLens);
}

Lens init_lens_Ext(void *pLens)
{
    Lens Psi;

    Psi.psi = psi_Ext;
    Psi.psi_1stDerivs = psi_1stDerivs_Ext;
    Psi.psi_2ndDerivs = psi_2ndDerivs_Ext;
    Psi.pLens = pLens;

    return Psi;
}

double psi_Ext(double x1, double x2, void *pLens)
{
    pLens_Ext *p = (pLens_Ext *)pLens;

    return 0.5*((p->kappa + p->gamma1)*x1*x1 + (p->kappa - p->gamma1)*x2*x2) + p->gamma2*x1*x2;
}

int psi_1stDerivs_Ext(double *psi_derivs, double x1, double x2, void *pLens)
{
    double psi, d1, d2;
    pLens_Ext *p = (pLens_Ext *)pLens;

    psi = 0.5*( (p->kappa + p->gamma1)*x1*x1 + (p->kappa - p->gamma1)*x2*x2 ) + p->gamma2*x1*x2;

    d1 = (p->kappa + p->gamma1)*x1 + p->gamma2*x2;
    d2 = (p->kappa - p->gamma1)*x2 + p->gamma2*x1;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;

    return 0;
}

int psi_2ndDerivs_Ext(double *psi_derivs, double x1, double x2, void *pLens)
{
    double psi;
    double d1, d2, d11, d22, d12;
    pLens_Ext *p = (pLens_Ext *)pLens;

    psi = 0.5*( (p->kappa + p->gamma1)*x1*x1 + (p->kappa - p->gamma1)*x2*x2 ) + p->gamma2*x1*x2;

    d1 = (p->kappa + p->gamma1)*x1 + p->gamma2*x2;
    d2 = (p->kappa - p->gamma1)*x2 + p->gamma2*x1;

    d11 = p->kappa + p->gamma1;
    d22 = p->kappa - p->gamma1;
    d12 = p->gamma2;

    psi_derivs[i_0] = psi;
    psi_derivs[i_dx1] = d1;
    psi_derivs[i_dx2] = d2;
    psi_derivs[i_dx1dx1] = d11;
    psi_derivs[i_dx2dx2] = d22;
    psi_derivs[i_dx1dx2] = d12;

    return 0;
}


// =================================================================
//           New lens here
// =================================================================
