#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"
#include "area_lib.h"

#define EPS_SOFT 1e-8

// =================================================================

// find R such that over the circle min(phi-tmin)=tau_max
double Rmax_func(double alpha, void *ptarget)
{
    pTarget *p = (pTarget *)ptarget;
    double x_vec[N_dims];
    double phi, tmax;
    
    tmax = *(double *)p->params;
    
    x1x2_def(x_vec, p->R, alpha, p->x0_vec);
    phi = phiFermat(p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);
    
    return phi - tmax;
}

double dRmax_func_dalpha(double alpha, void *ptarget)
{
    pTarget *p = (pTarget *)ptarget;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double d1, d2, Dx1, Dx2, dT;
    
    x1x2_def(x_vec, p->R, alpha, p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);
    
    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];
    
    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];
    
    dT = -Dx2*d1 + Dx1*d2;
    
    return dT;
}

double dRmax_func_dR(double R, void *ptarget)
{
    pTarget *p = (pTarget *)ptarget;
    double x_vec[N_dims];
    double phi_derivs[N_derivs];
    double d1, d2, Dx1, Dx2, dT; 
    
    x1x2_def(x_vec, R, p->alpha, p->x0_vec);
    phiFermat_1stDeriv(phi_derivs, p->y, x_vec[i_x1], x_vec[i_x2], p->Psi);
    
    Dx1 = x_vec[i_x1] - p->x0_vec[i_x1];
    Dx2 = x_vec[i_x2] - p->x0_vec[i_x2];
    
    d1 = phi_derivs[i_dx1];
    d2 = phi_derivs[i_dx2];
    
    dT = (Dx1*d1 + Dx2*d2)/R;
    
    return dT;
}

double find_Rmax_AreaIntegral(double y, double tau_max, double tmin, Lens *Psi)
{
    double Rmax, tmax;
    pTarget pt;
    
    tmax = tau_max + tmin;
    
    pt.y = y;
    pt.R = 2*(y + sqrt(2*tau_max));
    pt.alpha = 0;
    pt.Psi = Psi;
    pt.x0_vec[i_x1] = 0;
    pt.x0_vec[i_x2] = 0;
    pt.T_func = Rmax_func;
    pt.dT_func_dalpha = dRmax_func_dalpha;
    pt.dT_func_dR = dRmax_func_dR;
    pt.params = &tmax;
    
    Tmin(&pt);
    
    Rmax = pt.R;
    
    return Rmax;
}

// =================================================================

int integrate_AreaIntegral(double *t_min, double *tau_result, double *It_result, int n_result, pAreaIntegral *p)
{   
    int i, j, i_t, n_total; 
    double y = p->y; 
    double Rmin, Rmax, x1, x2, phi;
    Lens Psi = init_lens(p->pNLens);
    CritPoint point_min, point_live_min;
    pImage pimage;
    
    int n_rho, n_theta;
    double rho_min, rho_max, Drho;
    double theta_min, theta_max, Dtheta;
    double rho_i, r_i, theta_j;
    //~ double *psi_i;  // possible if psi is axisymmetric (much faster)
    double *stheta_j, *ctheta_j;
    
    int nt;
    int *t_grid;
    double dt, tmin, tmax, t0;
    double phi_tmp;
    
    ////////////////////////////////////
    //////        SETUP         ////////
    ////////////////////////////////////
    
    // first find the minimum
    pimage.y = y;
    pimage.point = &point_min;
    pimage.Psi = &Psi;
    
    find_global_Minimum_2D(&pimage);
    tmin = point_min.t;
    *t_min = tmin;
    
    copy_CritPoint(&point_live_min, &point_min);  // updated in runtime 
    
    // define temporal grid
    nt = n_result;
    dt = p->tau_max/nt;
    tmax = p->tau_max + tmin;
    
    t_grid = (int *)malloc(nt*sizeof(int));
    for(i=0;i<nt;i++)
        t_grid[i] = 0;
    
    // restrict the area of integration
    Rmin = EPS_SOFT;
    Rmax = find_Rmax_AreaIntegral(y, p->tau_max, tmin, &Psi);
    
    // define spatial grid
    rho_min   = Rmin*Rmin;
    rho_max   = Rmax*Rmax;
    theta_min = -M_PI;
    theta_max = M_PI;
    n_rho     = p->n_rho;
    n_theta   = p->n_theta;
    Drho   = (rho_max - rho_min)/(n_rho-1);
    Dtheta = (theta_max - theta_min)/(n_theta-1);
    
    stheta_j = (double *)malloc(n_theta*sizeof(double));
    ctheta_j = (double *)malloc(n_theta*sizeof(double));
    
    for(j=0;j<n_theta;j++)
    {
        theta_j = j*Dtheta + theta_min;
        stheta_j[j] = sin(theta_j);
        ctheta_j[j] = cos(theta_j);
    }
    
    ////////////////////////////////////
    //////     COMPUTATION      ////////
    ////////////////////////////////////
    
    n_total = 0;
    t0 = tmin;
    for(i=0;i<n_rho;i++)
    {   
        rho_i = i*Drho + rho_min;
        r_i = sqrt(rho_i);
        
        phi_tmp = 0.5*rho_i + 0.5*y*y;
        
        for(j=0;j<n_theta;j++)
        {
            x1 = r_i*ctheta_j[j];
            x2 = r_i*stheta_j[j];
            
            //~ phi = phiFermat(y, x1, x2, &Psi);
            phi = phi_tmp - x1*y - Psi.psi(x1, x2, Psi.pLens);   // slightly faster
            
            if(phi<t0)
            {
                point_live_min.x1 = x1;
                point_live_min.x2 = x2;
                t0 = phi;
            }
                  
            if((phi>tmin) && (phi<tmax))
            {
                i_t = (int)((phi-tmin)/dt);                
                
                t_grid[i_t]++;
                n_total++;                
            }
        }
    }
    
    // use the magnification for tau=0
    tau_result[0] = 0;
    It_result[0] = 2*M_PI*sqrt(point_min.mag);
    for(i=1;i<nt;i++)
    {
        tau_result[i] = i*dt + 0.5*dt;
        It_result[i] = 0.5*Dtheta*Drho*t_grid[i]/dt;
    }
    
    // check that the real minimum is not off
    classify_CritPoint(&point_live_min, y, &Psi);
    if( is_same_CritPoint(&point_min, &point_live_min) == _FALSE_ )
    {
        PWARNING("real minimum do not agree with the one provided:"\
                 "   * Using: t=%g  x=(%g, %g)\n"\
                 "   * Found: t=%g  x=(%g, %g)",\
                 point_min.t, point_min.x1, point_min.x2,\
                 point_live_min.t, point_live_min.x1, point_live_min.x2)
    }
    
    // =======================================
        
    free(t_grid);
    free(ctheta_j);
    free(stheta_j);
    
    return 0;
}
