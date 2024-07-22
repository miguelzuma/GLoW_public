#ifndef AREA_LIB_H
#define AREA_LIB_H

typedef struct
{
    int n_rho, n_theta;
    double y, tau_max;
    pNamedLens *pNLens;
} pAreaIntegral;

// =================================================================

// find R such that over the circle min(phi-tmin)=tau_max
double Rmax_func(double alpha, void *ptarget);
double dRmax_func_dalpha(double alpha, void *ptarget);
double dRmax_func_dR(double R, void *ptarget);
double find_Rmax_AreaIntegral(double y, double tau_max, double tmin, Lens *Psi);

// area/grid/binning method
int integrate_AreaIntegral(double *t_min, double *tau_result, double *It_result, int n_result, pAreaIntegral *p);

// =================================================================

#endif  // AREA_LIB_H
