/*
 * GLoW - contour_lib.h
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

#ifndef CONTOUR_LIB_H
#define CONTOUR_LIB_H

#include <gsl/gsl_vector.h>
#include "roots_lib.h"

typedef struct {
    double y;
    char sign;
    double R_ini;
    double alpha_ini;
    double tmin;
    double x0_vec[N_dims];
    Lens *Psi;
    void *pCond;
} pIntegral2d;

typedef struct {
    double t;
    pIntegral2d *p;
} pRoot2d;

typedef struct {
    char type;
    double x0[N_dims];
    double tau0;
    double t0;
    double R_max;
    double alpha_out;
    char is_init_birthdeath;
    double tau_birth;
    double tau_death;
} Center;

typedef struct {
    int n_ctrs;
    Center *ctrs;
    int *n_up, *n_dw;
    int **id_up, **id_dw;
} TableCandidates;

typedef struct {
    double y;
    double x10, x20;
    double sigmaf;
    Lens *Psi;
    void *pCond;
} pContour;

enum indices_eqs2d {j_R, j_I, j_alpha, N_eqs2d};
enum ctr_types {ctr_type_min,
                ctr_type_max,
                ctr_type_saddle_8_maxmax,
                ctr_type_saddle_8_minmin,
                ctr_type_saddle_O_max,
                ctr_type_saddle_O_min,
                N_ctr_type};

// =====================================================================

// ======  Create and initialize the centers from the crit points
// =====================================================================
Center *init_all_Center(int *n_points, CritPoint *points, double y, pNamedLens *pNLens);
void free_all_Center(Center *ctrs);
void display_Center(Center *ctr);


// ======  Find R(tau) for each contours
// =====================================================================
double small_R_guess(double dtau, double alpha, double x10, double x20, pIntegral2d *p2d);
int dR_dtau_contour2d(double t, const double u[], double f[], void *params);
double R_of_tau_integrate(double tau, pIntegral2d *p2d);
double R_of_tau_bracketing_small(double tau, pIntegral2d *p);
double R_of_tau_bracketing_large(double tau, pIntegral2d *p);

double phi_minus_tau_func(double R, void *params);
double find_R_in_bracket(double tau, double R0, double R1, pIntegral2d *p);


// ======  Helper functions to init the params structs
// =====================================================================
void update_pIntegral2d(Center *ctr, pIntegral2d *p);
void update_pCondODE(pCondODE *pc, pIntegral2d *p);


// ======  ODE system
// =====================================================================
double reach_2pi_contour2d(const double y[], const double dydt[], void *pCond);
char is_closed_contour2d(const double y[], const double dydt[], void *pCond);
int system_contour2d_robust(double t, const double u[], double f[], void *params);


// ======  Initialize the saddle points
// =====================================================================
int integrate_contour2d_saddle(double sigmaf, double *sigma0, double *u0, pIntegral2d *p);
double principal_direction_saddle(Center *ctr, pIntegral2d *p2d);
void fill_saddle_Center(Center *ctr, pIntegral2d *p2d);


// ======  Contour integration
// =====================================================================
double integrate_contour2d(pIntegral2d *p);
double integrate_all_contour2d(double tau, int n_ctrs, Center *ctrs, pIntegral2d *p);
double driver_contour2d(double tau, int n_ctrs, Center *ctrs, double y, pNamedLens *pNLens);


// ======  Apply diferent rules to initialize the centers
// =====================================================================
TableCandidates *init_TableCandidates(int n_ctrs, Center *ctrs);
void display_TableCandidates(TableCandidates *t);
void free_TableCandidates(TableCandidates *t);

int find_birth_death(int n_ctrs, Center *ctrs, pIntegral2d *p2d);
void update_reduce_Table(TableCandidates *t);
void update_minimize_Table(TableCandidates *t, pIntegral2d *p);


// ======  Find min/max inside a saddle point
// =====================================================================
double f_phi_multimin(const gsl_vector *v, void *params);
void df_phi_multimin(const gsl_vector *v, void *params, gsl_vector *df);
void fdf_phi_multimin(const gsl_vector *v, void *params, double *f, gsl_vector *df);
int minimize_in_saddle_O(double *x_min, Center *ctr, pIntegral2d *p);


// ======  Get parametric contours
// =====================================================================
int get_contour2d_points_robust(SolODE *sol, pIntegral2d *p);
int get_contour2d_robust(int n_points, SolODE *sol, pIntegral2d *p);
SolODE **driver_get_contour2d(double tau, int n_points, int n_ctrs, Center *ctrs,
                              double y, pNamedLens *pNLens);
void free_SolODE_contour2d(int n_sols, SolODE **sols);


// ======  Get contours with x1, x2 parameterization
// =====================================================================
int system_contour2d_x1x2(double t, const double u[], double f[], void *params);
int get_contour2d_x1x2(int n_points, SolODE *sol, pContour *p);

int driver_get_contour2d_x1x2(double x10, double x20, double y,
                              double sigmaf, int n_points,
                              pNamedLens *pNLens,
                              SolODE *sol);


// =====================================================================

#endif  // CONTOUR_LIB_H
