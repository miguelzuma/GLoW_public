/*
 * GLoW - common.h
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

#ifndef COMMON_H
#define COMMON_H

#include <gsl/gsl_min.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_odeiv2.h>

// =====================================================================

#define _TRUE_ 1
#define _FALSE_ 0
#define M_2PI 6.2831853071795864769252867665590057684
#define MOD_2PI(x) ((x) - M_2PI*floor((x)/M_2PI))
#define ABS(x) (((x) >= 0) ? (x) : -(x))
#define SIGN(x) (((x) >= 0) ? (1) : (-1))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define PWARNING(str, ...) {\
    if( (pprec.no_warnings == _FALSE_) && (pprec.no_output == _FALSE_) )\
        fprintf(stderr, "WARNING: "str" [%s:%d]\n", ##__VA_ARGS__, __FUNCTION__, __LINE__);\
}

#define PERROR(str, ...) {\
    if( (pprec.no_errors == _FALSE_) && (pprec.no_output == _FALSE_) )\
        fprintf(stderr, "ERROR: "str" [%s:%d]\n", ##__VA_ARGS__, __FUNCTION__, __LINE__);\
}

#define PGSLERROR(str, ...) {\
    if( (pprec.no_gslerrors == _FALSE_) && (pprec.no_output == _FALSE_) )\
        fprintf(stderr, "GSL ERROR: "str"\n", ##__VA_ARGS__);\
}

//#define NO_OUT
#ifdef NO_OUT
    #undef PWARNING
    #define PWARNING(str, ...) {}

    #undef PERROR
    #define PERROR(str, ...) {}

    #undef PGSLERROR
    #define PGSLERROR(str, ...) {}
#endif

// =====================================================================

void glow_error_handler(const char *reason, const char *file, int line, int gsl_errno);
void handle_GSL_errors();

// =====================================================================

typedef struct {
    int id, max_iter;
    double epsabs, epsrel;
} Prec_Base;

typedef struct {
    int id;
    double h, epsabs, epsrel;
} Prec_Solver;

typedef struct {
    int n;
    double epsabs, epsrel;
} Prec_Int;

typedef struct {
    int id, max_iter;
    double first_step, tol, epsabs;
} Prec_Multimin;

typedef struct {
    char ctrue;
    char cfalse;
    char debug_flag;
    char no_warnings;
    char no_errors;
    char no_gslerrors;
    char no_output;

           double ro_issameCP_dist;
        Prec_Base ro_findCP1D;
        Prec_Base ro_findCP1D_bracket;
           double ro_singcusp1D_dx;
           double ro_singcusp1D_eps;
           double ro_findallCP1D_xmin;
           double ro_findallCP1D_xmax;
              int ro_findallCP1D_nbrackets;
        Prec_Base ro_TminR;
           double ro_TminR_dalpha;
           double ro_TminR_dR;
        Prec_Base ro_Tmin;
    Prec_Multimin ro_findCP2D_min;
        Prec_Base ro_findCP2D_root;
              int ro_findfirstCP2D_nextra;
           double ro_findfirstCP2D_Rin;
           double ro_findfirstCP2D_Rout;
              int ro_findallCP2D_npoints;
             char ro_findallCP2D_force_search;
           double ro_initcusp_R;
           double ro_initcusp_n;
           double ro_findnearCritPoint_max_iter;
           double ro_findnearCritPoint_scale;
    Prec_Multimin ro_findlocMin2D;
              int ro_findglobMin2D_nguesses;

        Prec_Base sc_findRtau;
        Prec_Base sc_findRtau_bracket;
      Prec_Solver sc_intdRdtau;
           double sc_intdRdtau_R0;
           double sc_syscontour_eps;
      Prec_Solver sc_intContourStd;
      Prec_Solver sc_intContourRob;
           double sc_intContourRob_sigmaf;
           double sc_intContour_tau_smallest;
           double sc_intContour_tol_brack;
           double sc_intContour_tol_add;
           double sc_drivContour_taumin_over_y2;
      Prec_Solver sc_getContourStd;
      Prec_Solver sc_getContourRob;
           double sc_getContourRob_sigmaf;
           double sc_getContour_tol_brack;
           double sc_getContour_tol_add;
             char sc_warn_switch;

      Prec_Solver mc_intRtau;
              int mc_brackRtau_small_maxiter;
              int mc_brackRtau_small_nbrackets;
           double mc_brackRtau_small_Rmin;
           double mc_brackRtau_small_Rini;
              int mc_brackRtau_large_maxiter;
           double mc_brackRtau_large_Rini;
           double mc_brackRtau_large_scale;
           double mc_updCondODE_tol_brack;
           double mc_updCondODE_tol_add;
        Prec_Base mc_findRbracket;
      Prec_Solver mc_intContourSaddle;
              int mc_fillSaddleCenter_nsigma;
           double mc_fillSaddleCenter_dR;
           double mc_fillSaddleCenter_sigmaf;
      Prec_Solver mc_intContour;
           double mc_intContour_sigmaf;
           double mc_drivContour_taumin_over_y2;
    Prec_Multimin mc_minInSaddle;
           double mc_minInSaddle_dR;
      Prec_Solver mc_getContour;
           double mc_getContour_sigmaf;
      Prec_Solver mc_getContour_x1x2;

        Prec_Base si_findBrackBracket;
        Prec_Base si_findRootBracket;
              int si_findMovBracket_maxiter;
           double si_findMovBracket_scale;
         Prec_Int si_dirSingInt;
           double si_qngSingInt_epsabs;
           double si_qngSingInt_epsrel;
           double si_qngSingInt_ximin;
         Prec_Int si_qagSingInt;
           double si_qagSingInt_ximin;
           double si_drivContour_taumin_over_y2;

           double as_eps_soft;
              int as_FwSIS_n;
              int as_FwSIS_nmax_switch;
         Prec_Int as_FwDirect;
         Prec_Int as_slFwOsc_Direct;
         Prec_Int as_slFwOsc_Osc;
         Prec_Int as_wlFwOsc_Direct;
         Prec_Int as_wlFwOsc_Osc;

              int fo_updRegSch_nmax_slope;
              int fo_updRegSch_nmax_tail;
           double fo_updRegSch_Itmin_tail;
} Prec_General;

extern Prec_General pprec;

// =====================================================================

enum id_fRoot {id_fRoot_brent,
               N_id_fRoot};
extern char *names_fRoot[];
const gsl_root_fsolver_type *get_fRoot(int id);


enum id_fdfRoot {id_fdfRoot_newton,
                 id_fdfRoot_secant,
                 N_id_fdfRoot};
extern char *names_fdfRoot[];
const gsl_root_fdfsolver_type *get_fdfRoot(int id);


enum id_fMin {id_fMin_brent,
              N_id_fMin};
extern char *names_fMin[];
const gsl_min_fminimizer_type *get_fMin(int id);


enum id_fdfMultimin {id_fdfMultimin_conjugate_fr,
                     id_fdfMultimin_vector_bfgs,
                     N_id_fdfMultimin};
extern char *names_fdfMultimin[];
const gsl_multimin_fdfminimizer_type *get_fdfMultimin(int id);


enum id_fdfMultiroot {id_fdfMultiroot_newton,
                      id_fdfMultiroot_hybridsj,
                      id_fdfMultiroot_hybridj,
                      id_fdfMultiroot_gnewton,
                      N_id_fdfMultiroot};
extern char *names_fdfMultiroot[];
const gsl_multiroot_fdfsolver_type *get_fdfMultiroot(int id);


enum id_stepODE {id_stepODE_rkf45,
                 id_stepODE_rk8pd,
                 N_id_stepODE};
extern char *names_stepODE[];
const gsl_odeiv2_step_type *get_stepODE(int id);

// =====================================================================

#endif  // COMMON_H
