/*
 * GLoW - area_lib.h
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
