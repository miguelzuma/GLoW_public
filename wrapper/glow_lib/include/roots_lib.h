/*
 * GLoW - roots_lib.h
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

#ifndef ROOTS_LIB_H
#define ROOTS_LIB_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "lenses_lib.h"

typedef struct {
    char type;
    double t, mag;
    double x1, x2;
} CritPoint;

typedef struct {
    double y;
    CritPoint *point;
    Lens *Psi;
} pImage;

typedef struct {
    double y;
    double min;
    double R, alpha;
    double x0_vec[N_dims];
    double (*T_func)(double alpha, void *ptarget);
    double (*dT_func_dalpha)(double alpha, void *ptarget);
    double (*dT_func_dR)(double R, void *ptarget);
    Lens *Psi;
    void *params;
} pTarget;

enum types_CritPoint {type_min, type_max, type_saddle,
                      type_singcusp_max, type_singcusp_min,
                      type_non_converged};

// =================================================================


// ======  Operate with critical points
// =================================================================
void display_CritPoint(CritPoint *p);
void copy_CritPoint(CritPoint *p_dest, const CritPoint *p_src);
void swap_CritPoint(CritPoint *p_a, CritPoint *p_b);
int is_same_CritPoint(CritPoint *p_a, CritPoint *p_b);
void classify_CritPoint(CritPoint *p, double y, Lens *Psi);

int find_i_xmin_CritPoint(int n_points, CritPoint *p);
int find_i_tmin_CritPoint(int n_points, CritPoint *p);
void sort_x_CritPoint(int n_points, CritPoint *p);
void sort_t_CritPoint(int n_points, CritPoint *p);


// ======  1D functions
// =================================================================
double dphi_1D(double x1, void *pimage);
double ddphi_1D(double x1, void *pimage);
void dphi_ddphi_1D(double x1, void *pimage, double *y, double *dy);
void find_CritPoint_1D(double xguess, pImage *p);
void find_CritPoint_bracket_1D(double x_lo, double x_hi, pImage *p);
int check_singcusp_1D(CritPoint *p, double y, Lens *Psi);
CritPoint *find_all_CritPoints_1D(int *n_cpoints, double y, Lens *Psi);
CritPoint *driver_all_CritPoints_1D(int *n_cpoints, double y, pNamedLens *pNLens);


// ======  2D functions
// =================================================================

// === Growing/shrinking circles (generic algorithm)
double Tmin_R(double R, void *ptarget);      // min(T_func) wrt alpha in a circle R
double dTmin_R(double R, void *ptarget);     // finite difference approx
void TmindTmin_R(double R, void *ptarget, double *y, double *dy);
int Tmin(pTarget *ptarget);                 // find the point where min(T_func)=0

// === Critical points through minimization (with a threshold)
double dphisqr_func(double alpha, void *ptarget);    // zero at critical points
double ddphisqr_func_dalpha(double alpha, void *ptarget);
double ddphisqr_func_dR(double R, void *ptarget);
double dphisqr_f_CritPoints_2D(const gsl_vector *x, void *params);
void dphisqr_df_CritPoints_2D(const gsl_vector *x, void *params, gsl_vector *df);
void dphisqr_fdf_CritPoints_2D(const gsl_vector *x, void *params, double *f, gsl_vector *df);
int find_CritPoint_min_2D(double x1guess, double x2guess, pImage *p);

// === Critical points using 2d root finding
int phi_grad(const gsl_vector *x, void *params, gsl_vector *f);
int phi_hess(const gsl_vector *x, void *params, gsl_matrix *J);
int phi_grad_hess(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J);
int find_CritPoint_root_2D(double x1guess, double x2guess, pImage *p);

// === Generic functions to work with lists of CritPoint
void add_CritPoint(CritPoint *p_single, int *n_list, CritPoint *p_list);
CritPoint *filter_CritPoint(int *n, CritPoint *p);
CritPoint *merge_CritPoint(int n1, CritPoint *p1, int n2, CritPoint *p2, int *n_points);

// === Find cusps and singularities
void fill_CritPoint_near_singcusp(CritPoint *p_root, CritPoint *p_singcusp, double y, Lens *Psi);
CritPoint *find_singcusp(int *n_singcusp, double y, Lens *Psi, pNamedLens *pNLens);
CritPoint *add_singcusp(int *n_cpoints, CritPoint *ps, double y, Lens *Psi, pNamedLens *pNLens);

// === High level functions to find critical points

// start with initial conditions R=0 and R=Large and look for crit points using Tmin
CritPoint *find_first_CritPoints_2D(int *n_points, double y, Lens *Psi);

// easy access for the wrapper
int check_only_min_CritPoint_2D(CritPoint *p, double y, pNamedLens *pNLens);

CritPoint *find_all_CritPoints_2D(int *n_cpoints, double Rmin, double Rmax, double y, Lens *Psi,
                                  int (*find_CritPoint)(double x1, double x2, pImage *p));
CritPoint *find_all_CritPoints_min_2D(int *n_cpoints, double Rmin, double Rmax, double y, Lens *Psi);
CritPoint *find_all_CritPoints_root_2D(int *n_cpoints, double Rmin, double Rmax, double y, Lens *Psi);
CritPoint *driver_all_CritPoints_2D(int *n_cpoints, double y, pNamedLens *pNLens);


// ======  Direct 2d minimization
// =================================================================
double phi_f_Minimum_2D(const gsl_vector *x, void *params);
void phi_df_Minimum_2D(const gsl_vector *x, void *params, gsl_vector *df);
void phi_fdf_Minimum_2D(const gsl_vector *x, void *params, double *f, gsl_vector *df);
int find_local_Minimum_2D(double x1guess, double x2guess, pImage *p);
int find_global_Minimum_2D(pImage *p);


// =================================================================

#endif  // ROOTS_LIB_H
