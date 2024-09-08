/*
 * GLoW - ode_tools.h
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

#ifndef ODE_TOOLS_H
#define ODE_TOOLS_H

#include <gsl/gsl_odeiv2.h>

#define ODE_COND_MET 37   // flag to identify when condition is met

typedef struct {
    char cond_met, cond_initialized;
    double cond_old, tol_brack, tol_add, n_reduce_h;
    double (*brack_cond)(const double y[], const double dydt[], void *pCond);
    char (*add_cond)(const double y[], const double dydt[], void *pCond);
    void *params;
} pCondODE;

typedef struct {
    int n_buffer;
    int n_allocated;
    int n_points;           // total number of points computed (several buffers)
    int n_eqs;
    double *t;
    double **y;             // n_eqs arrays of len(y[i]) = n_points
} SolODE;

// ========================================================

pCondODE *init_pCondODE(void);
void free_pCondODE(pCondODE *p);
int check_pCond(void *pCond);

gsl_odeiv2_control*
gsl_odeiv2_control_conditional_new(double eps_abs, double eps_rel,
                                   pCondODE *p);

gsl_odeiv2_driver *
gsl_odeiv2_driver_alloc_conditional_new(const gsl_odeiv2_system * sys,
                                        const gsl_odeiv2_step_type * T,
                                        const double hstart,
                                        const double epsabs, const double epsrel,
                                        pCondODE *p);

// ========================================================

SolODE *init_SolODE(int n_eqs, int n_buffer);
void free_SolODE(SolODE *sol);
void realloc_SolODE(SolODE *sol);
int fill_SolODE(double t, double *y, SolODE *sol);
SolODE *interpolate_SolODE(int n_points, SolODE *sol);

// ========================================================

#endif  // ODE_TOOLS_H
