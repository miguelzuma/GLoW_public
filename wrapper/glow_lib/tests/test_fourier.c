#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"
#include "single_integral_lib.h"
#include "special_lib.h"
#include "fourier_lib.h"

#ifdef _OPENMP
    #include <omp.h>

    #define TIME_INIT double msec; double diff, start;
    #define TIME(func, n_iter) start=omp_get_wtime(); for(int l=0;l<n_iter;l++){func;} diff=omp_get_wtime()-start; msec=diff*1000; printf("Time taken %.3g miliseconds\n", msec/n_iter);
#endif

#ifndef _OPENMP
    #define TIME_INIT double msec; clock_t diff, start;
    #define TIME(func, n_iter) start=clock(); for(int l=0;l<n_iter;l++){func;} diff=clock()-start; msec=diff*1000/CLOCKS_PER_SEC; printf("Time taken %.2g miliseconds\n", msec/n_iter);
#endif

// =================================================================

int main(int argc, char *argv[])
{
    int i;

    int n_points;
    Lens Psi;
    pNamedLens *p;
    CritPoint *points;

    int n_taus;
    double y, tau;
    double tau_min, tau_max, tmin, Delta;
    double *tau_grid, *It_grid, *It_reg_grid;

    int n_ws;
    int n_keep, n_below, n_above;
    double wmin, wmax;
    double smallest_tau_max;
    double *w_grid;
    double complex *Fw_grid, *Fw_reg_grid;
    FreqTable *ft;
    RegScheme sch;

    if(argc==2)
        y = atof(argv[1]);
    else
    {
        printf("Usage: %s y\n", argv[0]);
        return 1;
    }

    handle_GSL_errors();
    TIME_INIT

    //==== Create lens
    //=========================================
    p = create_pLens_SIS(1);
    //~ p = create_pLens_PointLens(1);
    //~ p = create_pLens_CIS(1, 0.1);
    //~ p = create_pLens_NFW(1, 0.01);
    Psi = init_lens(p);

    //==== Find roots
    //=========================================
    points = find_all_CritPoints_1D(&n_points, y, &Psi);
    tmin = points[0].t;

    //==== Compute time-domain integral
    //=========================================
    n_taus = 1000;
    tau_min = 1e-2;
    tau_max = 1e6;

    tau_grid    = (double *)malloc(n_taus*sizeof(double));
    It_grid     = (double *)malloc(n_taus*sizeof(double));
    It_reg_grid = (double *)malloc(n_taus*sizeof(double));

    sort_x_CritPoint(n_points, points);

TIME(
    tau = tau_min;
    Delta = pow(tau_max/tau_min, 1./(n_taus-1));
    for(i=0;i<n_taus;i++)
    {
        tau_grid[i] = tau;
        It_grid[i]  = driver_SingleIntegral(tau, y, tmin, n_points, points, p, m_integral_g15);
        tau *= Delta;
    }
, 100)

// HVR_DEBUG
// **********************************************
int n=1000;
double c, w;
double *u, *ws;
double complex *F;

u = (double *)malloc(n*sizeof(double));
ws = (double *)malloc(n*sizeof(double));
F = (double complex *)malloc(n*sizeof(double complex));

wmin = 1e-3;
wmax = 1e3;

c = pow(y, 2);
w = wmin;
Delta = pow(wmax/wmin, 1./(n-1));
for(i=0;i<n;i++)
{
    ws[i] = w;
    u[i] = w*0.5;
    w *= Delta;
}

TIME(
    F11_sorted(u, c, n, F, 1);
, 1000);

TIME(
    F11_sorted(u, c, n, F, 2);
, 1000);

TIME(
    F11_sorted(u, c, n, F, 4);
, 1000);

printf("---------------------------------------------\n");

TIME(
    fill_Fw_PointLens(y, n, ws, F, 1);
, 1000);

TIME(
    fill_Fw_PointLens(y, n, ws, F, 2);
, 1000);

TIME(
    fill_Fw_PointLens(y, n, ws, F, 4);
, 1000);

free(u);
free(ws);
free(F);

exit(1);
// **********************************************

    sort_t_CritPoint(n_points, points);

    //==== Compute amplification factor
    //=========================================
    wmin = 1e-2;
    wmax = 1e2;
    smallest_tau_max = 10;
    n_keep = 2;
    n_below = 4;
    n_above = 6;

    // general
    sch.stage = 1;
    sch.n_ps  = n_points;
    sch.ps    = points;

    sch.nthreads = 4;
    sch.n_grid = n_taus;
    sch.tau_grid = tau_grid;
    sch.It_reg_grid = It_reg_grid;

    // SIS
    sch.stage = 2;
    sch.index[0] = 0.5;
    sch.amp[0] = 1./sqrt(2);

TIME(
    update_RegScheme(It_grid, sch.stage, &sch);
    ft = init_FreqTable(wmin, wmax, &sch, n_keep, n_below, n_above, smallest_tau_max, 0.2);
, 2000)
    //display_FreqTable(ft);

    n_ws = ft->n_total + 2;
    w_grid = (double *)malloc(n_ws*sizeof(double));
    Fw_grid = (double complex *)malloc(n_ws*sizeof(double complex));
    Fw_reg_grid = (double complex *)malloc(n_ws*sizeof(double complex));

TIME(
    compute_Fw(w_grid, Fw_reg_grid, Fw_grid, ft);
, 1000)

    //==== Clean up
    //=========================================
    free(Fw_reg_grid);
    free(Fw_grid);
    free(w_grid);
    free_FreqTable(ft);

    free(It_reg_grid);
    free(It_grid);
    free(tau_grid);

    free(points);
    free_pLens(p);

    return 0;
}
