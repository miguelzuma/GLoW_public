#ifndef SPECIAL_LIB_H
#define SPECIAL_LIB_H

#include <complex.h>

typedef struct hypercoeffs {
    int n;
    double c;
    double beta;
    double *a, *b;
} HyperCoeffs;

enum flags_F11_approx {flag_F11_series, flag_F11_recurrence, flag_F11_largez, flag_F11_Temme, N_flags_F11};

// =================================================================
static inline double linear_interpolation(double x_val, double *x, double *y, int i0, int i1)
    {return ((x_val-x[i0])*y[i1] - (x_val-x[i1])*y[i0])/(x[i1]-x[i0]);}

static inline double complex clinear_interpolation(double x_val, double *x, double complex *y, int i0, int i1)
    {return ((x_val-x[i0])*y[i1] - (x_val-x[i1])*y[i0])/(x[i1]-x[i0]);}
// =================================================================

// ======  Fresnel integrals
// =================================================================
double f_fresnel(double x);
double g_fresnel(double x);

// ======  Struve function
// =================================================================
double Mtilde_Struve_PowerSeries(double x, double nu, double tol);
double Mtilde_Struve_Asymptotic(double x, double nu, double tol);
double Mtilde_Struve_PieceWise(double x, double nu, double tol);
double Mtilde_Struve(double x, double nu, double tol);

// ======  Interpolation routines
// =================================================================
int searchsorted_left(double x, double *x_grid, int n_grid);
int searchsorted_right(double x, double *x_grid, int n_grid);
int searchsorted(double x, double *x_grid, int n_grid, int side);
int find_subarray(double *grid, int *jmin, int *jmax, double *subgrid, int *imin, int *imax);
int sorted_interpolation(double *x_subgrid, double *y_subgrid, int n_subgrid,
                         double *x_grid, double *y_grid, int n_grid);
double point_interpolation(double x, double *x_grid, double *y_grid, int n_grid);

// ======  Confluent hypergeometric function
// =================================================================
double complex F11_series(double u, double c, double tol, int *status);
double complex F11_series_b(double u, int b, double c, double tol, int *status);
double complex F11_recurrence(double u, double c, int n_up, double tol, int *status);
double complex F11_DLMF_largez(double u, double c, int order);

double complex F11_Temme_order2(double u, double c);
HyperCoeffs *init_F11_Temme_coeffs(double c);
void free_F11_Temme_coeffs(HyperCoeffs *coeffs);
double complex F11_Temme(double u, HyperCoeffs *coeffs);
double complex F11_Temme_full(double u, double c);

double complex F11_singlepoint(double u, double c, int *status, int *approx_flag);
int F11_sorted(double *u, double c, int n_F, double complex *F11, int nthreads);

// =================================================================
#endif  // SPECIAL_LIB_H
