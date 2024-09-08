/*
 * GLoW - single_contour_lib.h
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

#ifndef SINGLE_CONTOUR_H
#define SINGLE_CONTOUR_H

#include "ode_tools.h"
#include "lenses_lib.h"

typedef struct {
    double y;
    double R_ini;
    double x0_vec[N_dims];
    Lens *Psi;
    void *pCond;
} pIntegral;

typedef struct {
    double tau;
    pIntegral *p;
} pRoot;

enum indices_eqs {i_R, i_I, i_alpha, N_eqs};
enum methods_contour {m_contour_std, m_contour_robust};

// =================================================================

// ======   find R(tau) inverting tau(R)
// =================================================================
double f_root_R_of_tau(double R, void *params);
double df_root_R_of_tau(double R, void *params);
void fdf_root_R_of_tau(double R, void *params, double *f, double *df);

double find_R_of_tau(double tau, pIntegral *p);
int find_all_R_of_tau(int n_points, double *R_grid, double *tau_grid, pIntegral *p);
int driver_R_of_tau(int n_points, double *R_grid, double *tau_grid,
                    double x1_min, double x2_min, double y,
                    pNamedLens *pNLens);

double find_R_of_tau_bracket(double tau, double R_hi, pIntegral *p);
int find_all_R_of_tau_bracket(int n_points, double *R_grid, double *tau_grid, pIntegral *p);
int driver_R_of_tau_bracket(int n_points, double *R_grid, double *tau_grid,
                            double x1_min, double x2_min, double y,
                            pNamedLens *pNLens);

// ======   find R(tau) solving the differential equation
// =================================================================
int dR_dtau(double t, const double u[], double f[], void *params);
int integrate_dR_dtau(int n_points, double *R_grid, double *tau_grid, pIntegral *p);
int driver_dR_dtau(int n_points, double *R_grid, double *tau_grid,
                   double x1_min, double x2_min, double y,
                   pNamedLens *pNLens);

// == integrate the contour, assuming only one critical point (i.e. minimum)
// ===========================================================================
int system_contour(double t, const double u[], double f[], void *params);
int integrate_contour(double R_ini, double *R_f, double *I, pIntegral *p);

// == integrate a single contour (robust version)
// ===========================================================================
double reach_2pi_contour(const double y[], const double dydt[], void *pCond);
char is_closed_contour(const double y[], const double dydt[], void *pCond);
int system_contour_robust(double t, const double u[], double f[], void *params);
int integrate_contour_robust(double R_ini, double *R_f, double *I, pIntegral *p);

// == driver for contour integration
// ===========================================================================
double driver_contour(double tau,
                      double x1_min, double x2_min, double y,
                      pNamedLens *pNLens, int method);

// == get contours
// ===========================================================================
int get_contour_points(double R_ini, SolODE *sol, pIntegral *p);
int get_contour(double R_ini, int n_points, SolODE *sol, pIntegral *p);
int get_contour_points_robust(double R_ini, SolODE *sol, pIntegral *p);
int get_contour_robust(double R_ini, int n_points, SolODE *sol, pIntegral *p);
int driver_get_contour(double tau, int n_points,
                       double x1_min, double x2_min, double y,
                       pNamedLens *pNLens, int method,
                       SolODE *sol);


// ===========================================================================

#endif  // SINGLE_CONTOUR_H
