#
# GLoW - ccommon.pxd
#
# Copyright (C) 2023, Hector Villarrubia-Rojo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

#cython: language_level=3

cdef update_pprec(Prec_General pprec_general)

cdef extern from "common.h" nogil:
    void handle_GSL_errors()

    ctypedef struct Prec_Base:
        int id, max_iter
        double epsabs, epsrel

    ctypedef struct Prec_Solver:
        int id
        double h, epsabs, epsrel

    ctypedef struct Prec_Int:
        int n
        double epsabs, epsrel

    ctypedef struct Prec_Multimin:
        int id, max_iter
        double first_step, tol, epsabs

    ctypedef struct Prec_General:
        char ctrue
        char cfalse
        char debug_flag
        char no_warnings
        char no_errors
        char no_gslerrors
        char no_output

        double ro_issameCP_dist
        Prec_Base ro_findCP1D
        Prec_Base ro_findCP1D_bracket
        double ro_singcusp1D_dx
        double ro_singcusp1D_eps
        double ro_findallCP1D_xmin
        double ro_findallCP1D_xmax
        int ro_findallCP1D_nbrackets
        Prec_Base ro_TminR
        double ro_TminR_dalpha
        double ro_TminR_dR
        Prec_Base ro_Tmin
        Prec_Multimin ro_findCP2D_min
        Prec_Base ro_findCP2D_root
        int ro_findfirstCP2D_nextra
        double ro_findfirstCP2D_Rin
        double ro_findfirstCP2D_Rout
        int ro_findallCP2D_npoints
        char ro_findallCP2D_force_search
        double ro_initcusp_R
        double ro_initcusp_n
        double ro_findnearCritPoint_max_iter
        double ro_findnearCritPoint_scale
        Prec_Multimin ro_findlocMin2D
        int ro_findglobMin2D_nguesses

        Prec_Base sc_findRtau
        Prec_Base sc_findRtau_bracket
        Prec_Solver sc_intdRdtau
        double sc_intdRdtau_R0
        double sc_syscontour_eps
        Prec_Solver sc_intContourStd
        Prec_Solver sc_intContourRob
        double sc_intContourRob_sigmaf
        double sc_intContour_tau_smallest
        double sc_intContour_tol_brack
        double sc_intContour_tol_add
        double sc_drivContour_taumin_over_y2
        Prec_Solver sc_getContourStd
        Prec_Solver sc_getContourRob
        double sc_getContourRob_sigmaf
        double sc_getContour_tol_brack
        double sc_getContour_tol_add
        char sc_warn_switch

        Prec_Solver mc_intRtau
        int mc_brackRtau_small_maxiter
        int mc_brackRtau_small_nbrackets
        double mc_brackRtau_small_Rmin
        double mc_brackRtau_small_Rini
        int mc_brackRtau_large_maxiter
        double mc_brackRtau_large_Rini
        double mc_brackRtau_large_scale
        double mc_updCondODE_tol_brack
        double mc_updCondODE_tol_add
        Prec_Base mc_findRbracket
        Prec_Solver mc_intContourSaddle
        int mc_fillSaddleCenter_nsigma
        double mc_fillSaddleCenter_dR
        double mc_fillSaddleCenter_sigmaf
        Prec_Solver mc_intContour
        double mc_intContour_sigmaf
        double mc_drivContour_taumin_over_y2
        Prec_Multimin mc_minInSaddle
        double mc_minInSaddle_dR
        Prec_Solver mc_getContour
        double mc_getContour_sigmaf
        Prec_Solver mc_getContour_x1x2

        Prec_Base si_findBrackBracket
        Prec_Base si_findRootBracket
        int si_findMovBracket_maxiter
        double si_findMovBracket_scale
        Prec_Int si_dirSingInt
        double si_qngSingInt_epsabs
        double si_qngSingInt_epsrel
        double si_qngSingInt_ximin
        Prec_Int si_qagSingInt
        double si_qagSingInt_ximin
        double si_drivContour_taumin_over_y2

        double as_eps_soft
        int as_FwSIS_n
        int as_FwSIS_nmax_switch
        Prec_Int as_FwDirect
        Prec_Int as_slFwOsc_Direct
        Prec_Int as_slFwOsc_Osc
        Prec_Int as_wlFwOsc_Direct
        Prec_Int as_wlFwOsc_Osc

        int fo_updRegSch_nmax_slope
        int fo_updRegSch_nmax_tail
        double fo_updRegSch_Itmin_tail


    ctypedef enum id_fRoot: id_fRoot_newton, \
                            N_id_fRoot

    ctypedef enum id_fdfRoot: id_fdfRoot_newton, \
                              N_id_fdfRoot

    ctypedef enum id_fMin: id_fMin_brent, \
                           N_id_fMin

    ctypedef enum id_fdfMultimin: id_fdfMultimin_conjugate_fr, \
                                  id_fdfMultimin_vector_bfgs, \
                                  N_id_fdfMultimin

    ctypedef enum id_fdfMultiroot: id_fdfMultimin_newton, \
                                   id_fdfMultiroot_hybridsj, \
                                   id_fdfMultiroot_hybridj, \
                                   id_fdfMultiroot_gnewton, \
                                   N_id_fdfMultiroot

    ctypedef enum id_stepODE: id_stepODE_rkf45, \
                              id_stepODE_rk8pd, \
                              N_id_stepODE

    Prec_General pprec
    char *names_fRoot[]
    char *names_fdfRoot[]
    char *names_fMin[]
    char *names_fdfMultimin[]
    char *names_fdfMultiroot[]
    char *names_stepODE[]
