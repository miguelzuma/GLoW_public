/*
 * GLoW - analytic_SIS_lib.h
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

#ifndef ANALYTIC_SIS_H
#define ANALYTIC_SIS_H

#include <gsl/gsl_sf_ellint.h>
#include <complex.h>

enum Fw_SIS_methods {sis_direct, sis_osc};

// =================================================================

// options include GSL_PREC_DOUBLE, GSL_PREC_SINGLE, GSL_PREC_APPROX
double I_func(double a, double b, double c, double d, gsl_mode_t prec_mode);
double It_SIS_VarPrec(double tau, double y, double psi0, gsl_mode_t prec_mode);

// different versions with the flags above
double It_SIS_DoublePrec(double tau, double y, double psi0);
double It_SIS_SinglePrec(double tau, double y, double psi0);
double It_SIS_ApproxPrec(double tau, double y, double psi0);

// =================================================================

double integrand_f0_Direct(double theta, void *param);
double integrand_g0_Direct(double theta, void *param);
double integrand_If_Direct(double theta, void *param);
double integrand_Ig_Direct(double theta, void *param);
double integrand_Is_Direct(double theta, void *param);
double integrand_Ic_Direct(double theta, void *param);
double integrand_Isc_OscWeight(double u, void *param);
double integrand_Jsc_OscWeight(double u, void *param);

double complex Fw_SIS(double w, double y, double psi0, int method);
double complex Fw_SIS_direct(double alpha0, double r);
double complex Fw_SIS_SL_osc(double alpha0, double r, double Delta);
double complex Fw_SIS_WL_osc(double alpha0, double r, double Delta);

// =================================================================

#endif  // ANALYTIC_SIS_H
