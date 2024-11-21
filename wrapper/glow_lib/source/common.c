/*
 * GLoW - common.c
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

#include <gsl/gsl_errno.h>
#include "common.h"

// =====================================================================

void glow_error_handler(const char *reason, const char *file, int line, int gsl_errno)
{
    // error explanation: gsl_strerror(gsl_errno)
    PGSLERROR("%s [%s:%d]", reason, file, line);
}

void handle_GSL_errors()
{
    // turn off GSL err
    //~ gsl_set_error_handler_off();

    gsl_set_error_handler(glow_error_handler);
}

// =====================================================================

char *names_fRoot[] = {"brent"};
const gsl_root_fsolver_type *get_fRoot(int id)
{
    const gsl_root_fsolver_type *T[] = {gsl_root_fsolver_brent};

    return T[id];
}

char *names_fdfRoot[] = {"newton",
                         "secant"};
const gsl_root_fdfsolver_type *get_fdfRoot(int id)
{
    const gsl_root_fdfsolver_type *T[] = {gsl_root_fdfsolver_newton,
                                          gsl_root_fdfsolver_secant};

    return T[id];
}

char *names_fMin[] = {"brent"};
const gsl_min_fminimizer_type *get_fMin(int id)
{
    const gsl_min_fminimizer_type *T[] = {gsl_min_fminimizer_brent};

    return T[id];
}

char *names_fdfMultimin[] = {"conjugate_fr",
                             "vector_bfgs"};
const gsl_multimin_fdfminimizer_type *get_fdfMultimin(int id)
{
    const gsl_multimin_fdfminimizer_type *T[] = {gsl_multimin_fdfminimizer_conjugate_fr,
                                                 gsl_multimin_fdfminimizer_vector_bfgs};

    return T[id];
}

char *names_fdfMultiroot[] = {"newton",
                              "hybridsj",
                              "hybridj",
                              "gnewton"};
const gsl_multiroot_fdfsolver_type *get_fdfMultiroot(int id)
{
    const gsl_multiroot_fdfsolver_type *T[] = {gsl_multiroot_fdfsolver_newton,
                                               gsl_multiroot_fdfsolver_hybridsj,
                                               gsl_multiroot_fdfsolver_hybridj,
                                               gsl_multiroot_fdfsolver_gnewton};

    return T[id];
}

char *names_stepODE[] = {"rkf45",
                         "rk8pd"};
const gsl_odeiv2_step_type *get_stepODE(int id)
{
    const gsl_odeiv2_step_type *T[] = {gsl_odeiv2_step_rkf45,
                                       gsl_odeiv2_step_rk8pd};

    return T[id];
}

Prec_General pprec =
{
    .ctrue        = _TRUE_,
    .cfalse       = _FALSE_,
    .debug_flag   = _FALSE_,
    .no_warnings  = _FALSE_,
    .no_errors    = _FALSE_,
    .no_gslerrors = _FALSE_,
    .no_output    = _FALSE_,

    ////////////////////////////////////
    ////////     root_lib.c     ////////
    ////////////////////////////////////
    .ro_issameCP_dist = 1e-5,

    .ro_findCP1D.id       = id_fdfRoot_newton,
    .ro_findCP1D.max_iter = 100,
    .ro_findCP1D.epsabs   = 0,
    .ro_findCP1D.epsrel   = 1e-8,

    .ro_findCP1D_bracket.id       = id_fRoot_brent,
    .ro_findCP1D_bracket.max_iter = 100,
    .ro_findCP1D_bracket.epsabs   = 1e-8,
    .ro_findCP1D_bracket.epsrel   = 1e-8,

    .ro_singcusp1D_dx = 1e-6,
    .ro_singcusp1D_eps = 1e-14,

    .ro_findallCP1D_xmin         = 1e-6,
    .ro_findallCP1D_xmax         = 10,
    .ro_findallCP1D_nbrackets    = 100,

    .ro_TminR.id       = id_fMin_brent,
    .ro_TminR.max_iter = 100,
    .ro_TminR.epsabs   = 1e-3,
    .ro_TminR.epsrel   = 0,
    .ro_TminR_dalpha = 1e-5,
    .ro_TminR_dR     = 1e-3,

    .ro_Tmin.id       = id_fdfRoot_newton,
    .ro_Tmin.max_iter = 100,
    .ro_Tmin.epsabs   = 0,
    .ro_Tmin.epsrel   = 1e-8,

    .ro_findCP2D_min.id         = id_fdfMultimin_conjugate_fr,
    .ro_findCP2D_min.max_iter   = 100,
    .ro_findCP2D_min.first_step = 0.01,
    .ro_findCP2D_min.tol        = 1e-12,
    .ro_findCP2D_min.epsabs     = 1e-12,

    .ro_findCP2D_root.id       = id_fdfMultiroot_newton,
    .ro_findCP2D_root.max_iter = 1000,
    .ro_findCP2D_root.epsabs   = 1e-8,
    .ro_findCP2D_root.epsrel   = 0,

    .ro_findfirstCP2D_nextra = 20,
    .ro_findfirstCP2D_Rin    = 1e-3,
    .ro_findfirstCP2D_Rout   = 10,

    .ro_findallCP2D_npoints      = 500,
    .ro_findallCP2D_force_search = _FALSE_,

    .ro_initcusp_R = 1e-6,
    .ro_initcusp_n = 100,

    .ro_findnearCritPoint_max_iter = 100,
    .ro_findnearCritPoint_scale = 1.2,

    .ro_findlocMin2D.id         = id_fdfMultimin_conjugate_fr,
    .ro_findlocMin2D.max_iter   = 100,
    .ro_findlocMin2D.first_step = 0.01,
    .ro_findlocMin2D.tol        = 1e-4,
    .ro_findlocMin2D.epsabs     = 1e-5,

    .ro_findglobMin2D_nguesses = 4,

    ////////////////////////////////////////////
    ////////   single_contour_lib.c     ////////
    ////////////////////////////////////////////
    .sc_findRtau.id       = id_fdfRoot_newton,
    .sc_findRtau.max_iter = 100,
    .sc_findRtau.epsabs   = 1e-5,
    .sc_findRtau.epsrel   = 1e-5,

    .sc_findRtau_bracket.id       = id_fRoot_brent,
    .sc_findRtau_bracket.max_iter = 100,
    .sc_findRtau_bracket.epsabs   = 1e-5,
    .sc_findRtau_bracket.epsrel   = 1e-5,

    .sc_intdRdtau.id     = id_stepODE_rk8pd,
    .sc_intdRdtau.h      = 1e-6,
    .sc_intdRdtau.epsabs = 1e-5,
    .sc_intdRdtau.epsrel = 0,
    .sc_intdRdtau_R0 = 1e-12,

    .sc_syscontour_eps = 1e-6,

    .sc_intContourStd.id     = id_stepODE_rk8pd,
    .sc_intContourStd.h      = 1e-6,
    .sc_intContourStd.epsabs = 1e-5,
    .sc_intContourStd.epsrel = 0,

    .sc_intContourRob.id     = id_stepODE_rk8pd,
    .sc_intContourRob.h      = 1e-6,
    .sc_intContourRob.epsabs = 1e-5,
    .sc_intContourRob.epsrel = 0,
    .sc_intContourRob_sigmaf = 1e10,

    .sc_intContour_tau_smallest = 1e-6,
    .sc_intContour_tol_brack    = 1e-4,
    .sc_intContour_tol_add      = 1e-1,

    .sc_drivContour_taumin_over_y2 = 1e-4,

    .sc_getContourStd.id     = id_stepODE_rk8pd,
    .sc_getContourStd.h      = 1e-6,
    .sc_getContourStd.epsabs = 1e-5,
    .sc_getContourStd.epsrel = 0,

    .sc_getContourRob.id     = id_stepODE_rk8pd,
    .sc_getContourRob.h      = 1e-6,
    .sc_getContourRob.epsabs = 1e-6,
    .sc_getContourRob.epsrel = 0,
    .sc_getContourRob_sigmaf = 1e10,

    .sc_getContour_tol_brack = 1e-6,
    .sc_getContour_tol_add   = 1e-1,

    .sc_warn_switch = _FALSE_,

    ////////////////////////////////////////////
    ////////   multi_contour_lib.c     /////////
    ////////////////////////////////////////////
    .mc_intRtau.id     = id_stepODE_rk8pd,
    .mc_intRtau.h      = 1e-6,
    .mc_intRtau.epsabs = 1e-5,
    .mc_intRtau.epsrel = 0,

    .mc_brackRtau_small_maxiter   = 1000,
    .mc_brackRtau_small_nbrackets = 100,
    .mc_brackRtau_small_Rmin      = 1e-8,
    .mc_brackRtau_small_Rini      = 1e-3,

    .mc_brackRtau_large_maxiter = 100,
    .mc_brackRtau_large_Rini    = 1e-3,
    .mc_brackRtau_large_scale   = 2.,

    .mc_updCondODE_tol_brack = 1e-4,
    .mc_updCondODE_tol_add   = 1e-1,

    .mc_findRbracket.id       = id_fRoot_brent,
    .mc_findRbracket.max_iter = 100,
    .mc_findRbracket.epsabs   = 1e-5,
    .mc_findRbracket.epsrel   = 1e-5,

    .mc_intContourSaddle.id     = id_stepODE_rk8pd,
    .mc_intContourSaddle.h      = 1e-6,
    .mc_intContourSaddle.epsabs = 1e-5,
    .mc_intContourSaddle.epsrel = 0,

    .mc_fillSaddleCenter_nsigma = 100,
    .mc_fillSaddleCenter_dR     = 5e-2,
    .mc_fillSaddleCenter_sigmaf = 1000,

    .mc_intContour.id     = id_stepODE_rk8pd,
    .mc_intContour.h      = 1e-6,
    .mc_intContour.epsabs = 1e-6,
    .mc_intContour.epsrel = 0,
    .mc_intContour_sigmaf = 1e4,

    .mc_drivContour_taumin_over_y2 = 1e-4,

    .mc_minInSaddle.id         = id_fdfMultimin_conjugate_fr,
    .mc_minInSaddle.max_iter   = 100,
    .mc_minInSaddle.first_step = 1e-3,
    .mc_minInSaddle.tol        = 1e-4,
    .mc_minInSaddle.epsabs     = 1e-3,
    .mc_minInSaddle_dR     = 1e-3,

    .mc_getContour.id     = id_stepODE_rk8pd,
    .mc_getContour.h      = 1e-6,
    .mc_getContour.epsabs = 1e-6,
    .mc_getContour.epsrel = 0,
    .mc_getContour_sigmaf = 1e4,

    .mc_getContour_x1x2.id     = id_stepODE_rk8pd,
    .mc_getContour_x1x2.h      = 1e-6,
    .mc_getContour_x1x2.epsabs = 1e-7,
    .mc_getContour_x1x2.epsrel = 0,

    ////////////////////////////////////////////
    ///////   single_integral_lib.c     ////////
    ////////////////////////////////////////////
    .si_findBrackBracket.id       = id_fRoot_brent,
    .si_findBrackBracket.max_iter = 100,
    .si_findBrackBracket.epsabs   = 0,
    .si_findBrackBracket.epsrel   = 1e-8,

    .si_findRootBracket.id       = id_fdfRoot_newton,
    .si_findRootBracket.max_iter = 100,
    .si_findRootBracket.epsabs   = 0,
    .si_findRootBracket.epsrel   = 1e-8,

    .si_findMovBracket_maxiter = 100,
    .si_findMovBracket_scale = 1.5,

    .si_dirSingInt.n      = 1000,
    .si_dirSingInt.epsabs = 1e-4,
    .si_dirSingInt.epsrel = 1e-4,

    .si_qngSingInt_epsabs = 1e-4,
    .si_qngSingInt_epsrel = 1e-4,
    .si_qngSingInt_ximin = 0.,

    .si_qagSingInt.n      = 100,
    .si_qagSingInt.epsabs = 1e-4,
    .si_qagSingInt.epsrel = 1e-4,
    .si_qagSingInt_ximin = 0.,

    .si_drivContour_taumin_over_y2 = 1e-4,

    ////////////////////////////////////////////
    ////////   analytic_SIS_lib.c     //////////
    ////////////////////////////////////////////
    .as_eps_soft = 0.,

    .as_FwSIS_n = 5,
    .as_FwSIS_nmax_switch = 5,

    .as_FwDirect.n      = 5000,
    .as_FwDirect.epsabs = 0,
    .as_FwDirect.epsrel = 1e-5,

    .as_slFwOsc_Direct.n      = 5000,
    .as_slFwOsc_Direct.epsabs = 0,
    .as_slFwOsc_Direct.epsrel = 1e-5,

    .as_slFwOsc_Osc.n      = 10,
    .as_slFwOsc_Osc.epsabs = 0,
    .as_slFwOsc_Osc.epsrel = 1e-5,

    .as_wlFwOsc_Direct.n      = 5000,
    .as_wlFwOsc_Direct.epsabs = 0,
    .as_wlFwOsc_Direct.epsrel = 1e-5,

    .as_wlFwOsc_Osc.n      = 30,
    .as_wlFwOsc_Osc.epsabs = 0,
    .as_wlFwOsc_Osc.epsrel = 1e-5,

    ////////////////////////////////////////////
    ////////      fourier_lib.c       //////////
    ////////////////////////////////////////////
    .fo_updRegSch_nmax_slope = 20,
    .fo_updRegSch_nmax_tail = 20,
    .fo_updRegSch_Itmin_tail = 5e-3
};
