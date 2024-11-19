#
# GLoW - ccommon.pyx
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

from libc.stdio cimport printf
cimport cython

cimport ccommon, croots, csingle_contour, ccontour, csingle_integral, carea, canalytic_SIS, clenses, cspecial, cfourier

## ---------------------------------------------------------------------

# needed to restart the values for new instances in the same session
cdef Prec_General pprec_default = pprec

## ----------  Precision ------------- ##
handle_GSL_errors()
cdef update_pprec(Prec_General pprec_general):
    global pprec
    pprec = pprec_general
    handle_GSL_errors()
## ----------------------------------- ##

## ---------------------------------------------------------------------

cdef get_name_id_corr(char **names, int i_max):
    cdef int i

    id_to_name = {i : names[i].decode('UTF-8') for i in range(i_max)}
    name_to_id = {names[i].decode('UTF-8') : i for i in range(i_max)}
    return id_to_name, name_to_id

fRoot_id_to_name,   fRoot_name_to_id   = get_name_id_corr(names_fRoot,   N_id_fRoot)
fdfRoot_id_to_name, fdfRoot_name_to_id = get_name_id_corr(names_fdfRoot, N_id_fdfRoot)
fMin_id_to_name,    fMin_name_to_id    = get_name_id_corr(names_fMin,    N_id_fMin)
fMultimin_id_to_name,  fMultimin_name_to_id  = get_name_id_corr(names_fdfMultimin,  N_id_fdfMultimin)
fMultiroot_id_to_name, fMultiroot_name_to_id = get_name_id_corr(names_fdfMultiroot, N_id_fdfMultiroot)
stepODE_id_to_name,    stepODE_name_to_id    = get_name_id_corr(names_stepODE,      N_id_stepODE)


id_to_name = {}
id_to_name['ro_findCP1D']         = fdfRoot_id_to_name
id_to_name['ro_findCP1D_bracket'] = fRoot_id_to_name
id_to_name['ro_TminR']            = fMin_id_to_name
id_to_name['ro_Tmin']             = fdfRoot_id_to_name
id_to_name['ro_findCP2D_min']     = fMultimin_id_to_name
id_to_name['ro_findCP2D_root']    = fMultiroot_id_to_name
id_to_name['ro_findlocMin2D']     = fMultimin_id_to_name

id_to_name['sc_findRtau']         = fdfRoot_id_to_name
id_to_name['sc_findRtau_bracket'] = fRoot_id_to_name
id_to_name['sc_intdRdtau']        = stepODE_id_to_name
id_to_name['sc_intContourStd']    = stepODE_id_to_name
id_to_name['sc_intContourRob']    = stepODE_id_to_name
id_to_name['sc_getContourStd']    = stepODE_id_to_name
id_to_name['sc_getContourRob']    = stepODE_id_to_name

id_to_name['mc_intRtau']          = stepODE_id_to_name
id_to_name['mc_findRbracket']     = fRoot_id_to_name
id_to_name['mc_intContourSaddle'] = stepODE_id_to_name
id_to_name['mc_intContour']       = stepODE_id_to_name
id_to_name['mc_minInSaddle']      = fMultimin_id_to_name
id_to_name['mc_getContour']       = stepODE_id_to_name
id_to_name['mc_getContour_x1x2']  = stepODE_id_to_name

id_to_name['si_findBrackBracket'] = fRoot_id_to_name
id_to_name['si_findRootBracket']  = fdfRoot_id_to_name


name_to_id = {}
name_to_id['ro_findCP1D']         = fdfRoot_name_to_id
name_to_id['ro_findCP1D_bracket'] = fdfRoot_name_to_id
name_to_id['ro_TminR']            = fMin_name_to_id
name_to_id['ro_Tmin']             = fdfRoot_name_to_id
name_to_id['ro_findCP2D_min']     = fMultimin_name_to_id
name_to_id['ro_findCP2D_root']    = fMultiroot_name_to_id
name_to_id['ro_findlocMin2D']     = fMultimin_name_to_id

name_to_id['sc_findRtau']         = fdfRoot_name_to_id
name_to_id['sc_findRtau_bracket'] = fRoot_name_to_id
name_to_id['sc_intdRdtau']        = stepODE_name_to_id
name_to_id['sc_intContourStd']    = stepODE_name_to_id
name_to_id['sc_intContourRob']    = stepODE_name_to_id
name_to_id['sc_getContourStd']    = stepODE_name_to_id
name_to_id['sc_getContourRob']    = stepODE_name_to_id

name_to_id['mc_intRtau']          = stepODE_name_to_id
name_to_id['mc_findRbracket']     = fRoot_name_to_id
name_to_id['mc_intContourSaddle'] = stepODE_name_to_id
name_to_id['mc_intContour']       = stepODE_name_to_id
name_to_id['mc_minInSaddle']      = fMultimin_name_to_id
name_to_id['mc_getContour']       = stepODE_name_to_id
name_to_id['mc_getContour_x1x2']  = stepODE_name_to_id

name_to_id['si_findBrackBracket'] = fRoot_name_to_id
name_to_id['si_findRootBracket']  = fdfRoot_name_to_id

## ---------------------------------------------------------------------

def check_input_dic(dic, dic_input):
    keys = dic.keys()
    for key, val in dic_input.items():
        if key not in keys:
            message = "key '%s' not found in C_prec, use display_Cprec()"\
                      " to show the default values" % key
            raise KeyError(message)

        # check keys of nested dictionaries
        if isinstance(val, dict):
            for key2, val2 in val.items():
                if key2 == 'type':
                    try:
                        val['id'] = name_to_id[key][val2]
                    except KeyError:
                        options = list(name_to_id[key].keys())
                        message = "'{val2}' is not a valid option for C_prec['{key}']['type'], "\
                                  "valid options: {options}".format(val2=val2, key=key, options=options)
                        raise KeyError(message)
                    del val[key2]
                    continue

                if key2 not in dic[key].keys():
                    message = "key '%s' not found in C_prec['%s'], use display_Cprec()"\
                              " to show the default values" % (key2, key)
                    raise KeyError(message)

def get_Cprec():
    Cprec = dict(pprec)

    for key, val in Cprec.items():
        if isinstance(val, dict):
            try:
                val['type'] = id_to_name[key][val['id']]
                del val['id']
            except KeyError:
                pass
    return Cprec

def display_Cprec():
    Cprec = dict(pprec)

    prefixes = ["ro_", "sc_", "mc_", "si_", "as_", "fo_"]
    output_dic = {p:"" for p in prefixes}
    output_dic['other'] = ""

    for key, val in Cprec.items():
        if key[:3] in prefixes:
            prefix = key[:3]
        else:
            prefix = 'other'

        if isinstance(val, dict):
            output_dic[prefix] += " - %s :\n" % key
            for key2, val2 in val.items():
                if key2 == 'id':
                    options = name_to_id[key].keys()
                    output_dic[prefix] += "   -- type = {ty}     options: {opt}"\
                              "\n".format(ty=id_to_name[key][val2], opt=list(options))
                else:
                    output_dic[prefix] += "   -- %s = %g\n" % (key2, val2)
        else:
            output_dic[prefix] += " - %s = %g\n" % (key, val)


    print("----------------------------------")
    print("           roots_lib.c            ")
    print("----------------------------------")
    print(output_dic['ro_'])

    print("----------------------------------")
    print("      single_contour_lib.c        ")
    print("----------------------------------")
    print(output_dic['sc_'])

    print("----------------------------------")
    print("      multi_contour_lib.c         ")
    print("----------------------------------")
    print(output_dic['mc_'])

    print("----------------------------------")
    print("      single_integral_lib.c       ")
    print("----------------------------------")
    print(output_dic['si_'])

    print("----------------------------------")
    print("       analytic_SIS_lib.c         ")
    print("----------------------------------")
    print(output_dic['as_'])

    print("----------------------------------")
    print("         fourier_lib.c            ")
    print("----------------------------------")
    print(output_dic['fo_'])

    print("----------------------------------")
    print("        other parameters          ")
    print("----------------------------------")
    print(output_dic['other'])

    print("----------------------------------")
    print("   NOTE: True = 1,  False = 0     ")
    print("----------------------------------")


cdef update_all_modules_Cprec(Prec_General pr):
    ccommon.update_pprec(pr)
    clenses.update_pprec(pr)
    croots.update_pprec(pr)
    csingle_contour.update_pprec(pr)
    ccontour.update_pprec(pr)
    csingle_integral.update_pprec(pr)
    carea.update_pprec(pr)
    canalytic_SIS.update_pprec(pr)
    cspecial.update_pprec(pr)
    cfourier.update_pprec(pr)

def update_Cprec(Cprec_input):
    # first create the python dictionary
    d = dict(pprec)

    # check input
    check_input_dic(d, Cprec_input)

    # update the python dictionary
    for key, val in Cprec_input.items():
        if isinstance(val, dict):
            d[key].update(val)
        else:
            d[key] = val

    # manually update the dictionary and use to update the global
    # variable in each module
    # HVR: far from ideal. We need to define a small snippet in
    #      each module defining prec quantities and update_pprec. The
    #      ultimate reason is that each of them is linked independently
    #      so each of them contains an independent pprec global struct
    #      (I guess this could be avoided by either getting rid of the
    #      global struct or compiling everything together, this is the
    #      lesser evil for now)
    manual_update_pprec(d, &pprec)
    update_all_modules_Cprec(pprec)

def set_default_Cprec():
    global pprec
    pprec = pprec_default
    update_all_modules_Cprec(pprec)

## ---------------------------------------------------------------------

cdef manual_update_pprec(d, Prec_General *pr):
    pr.ctrue  = d['ctrue']
    pr.cfalse = d['cfalse']
    pr.debug_flag   = d['debug_flag']
    pr.no_warnings  = d['no_warnings']
    pr.no_errors    = d['no_errors']
    pr.no_gslerrors = d['no_gslerrors']
    pr.no_output    = d['no_output']

    #////////////////////////////////////
    #////////     root_lib.c     ////////
    #////////////////////////////////////
    pr.ro_issameCP_dist = d['ro_issameCP_dist']

    pr.ro_findCP1D.id       = d['ro_findCP1D']['id']
    pr.ro_findCP1D.max_iter = d['ro_findCP1D']['max_iter']
    pr.ro_findCP1D.epsabs   = d['ro_findCP1D']['epsabs']
    pr.ro_findCP1D.epsrel   = d['ro_findCP1D']['epsrel']

    pr.ro_findCP1D_bracket.id       = d['ro_findCP1D_bracket']['id']
    pr.ro_findCP1D_bracket.max_iter = d['ro_findCP1D_bracket']['max_iter']
    pr.ro_findCP1D_bracket.epsabs   = d['ro_findCP1D_bracket']['epsabs']
    pr.ro_findCP1D_bracket.epsrel   = d['ro_findCP1D_bracket']['epsrel']

    pr.ro_singcusp1D_dx  = d['ro_singcusp1D_dx']
    pr.ro_singcusp1D_eps = d['ro_singcusp1D_eps']

    pr.ro_findallCP1D_xmin         = d['ro_findallCP1D_xmin']
    pr.ro_findallCP1D_xmax         = d['ro_findallCP1D_xmax']
    pr.ro_findallCP1D_nbrackets    = d['ro_findallCP1D_nbrackets']

    pr.ro_TminR.id       = d['ro_TminR']['id']
    pr.ro_TminR.max_iter = d['ro_TminR']['max_iter']
    pr.ro_TminR.epsabs   = d['ro_TminR']['epsabs']
    pr.ro_TminR.epsrel   = d['ro_TminR']['epsrel']
    pr.ro_TminR_dalpha = d['ro_TminR_dalpha']
    pr.ro_TminR_dR     = d['ro_TminR_dR']

    pr.ro_Tmin.id       = d['ro_Tmin']['id']
    pr.ro_Tmin.max_iter = d['ro_Tmin']['max_iter']
    pr.ro_Tmin.epsabs   = d['ro_Tmin']['epsabs']
    pr.ro_Tmin.epsrel   = d['ro_Tmin']['epsrel']

    pr.ro_findCP2D_min.id         = d['ro_findCP2D_min']['id']
    pr.ro_findCP2D_min.max_iter   = d['ro_findCP2D_min']['max_iter']
    pr.ro_findCP2D_min.first_step = d['ro_findCP2D_min']['first_step']
    pr.ro_findCP2D_min.tol        = d['ro_findCP2D_min']['tol']
    pr.ro_findCP2D_min.epsabs     = d['ro_findCP2D_min']['epsabs']

    pr.ro_findCP2D_root.id       = d['ro_findCP2D_root']['id']
    pr.ro_findCP2D_root.max_iter = d['ro_findCP2D_root']['max_iter']
    pr.ro_findCP2D_root.epsabs   = d['ro_findCP2D_root']['epsabs']
    pr.ro_findCP2D_root.epsrel   = d['ro_findCP2D_root']['epsrel']

    pr.ro_findfirstCP2D_nextra = d['ro_findfirstCP2D_nextra']
    pr.ro_findfirstCP2D_Rin    = d['ro_findfirstCP2D_Rin']
    pr.ro_findfirstCP2D_Rout   = d['ro_findfirstCP2D_Rout']

    pr.ro_findallCP2D_npoints      = d['ro_findallCP2D_npoints']
    pr.ro_findallCP2D_force_search = d['ro_findallCP2D_force_search']

    pr.ro_initcusp_R = d['ro_initcusp_R']
    pr.ro_initcusp_n = d['ro_initcusp_n']

    pr.ro_findnearCritPoint_max_iter = d['ro_findnearCritPoint_max_iter']
    pr.ro_findnearCritPoint_scale    = d['ro_findnearCritPoint_scale']

    pr.ro_findlocMin2D.id         = d['ro_findlocMin2D']['id']
    pr.ro_findlocMin2D.max_iter   = d['ro_findlocMin2D']['max_iter']
    pr.ro_findlocMin2D.first_step = d['ro_findlocMin2D']['first_step']
    pr.ro_findlocMin2D.tol        = d['ro_findlocMin2D']['tol']
    pr.ro_findlocMin2D.epsabs     = d['ro_findlocMin2D']['epsabs']

    pr.ro_findglobMin2D_nguesses = d['ro_findglobMin2D_nguesses']


    #////////////////////////////////////////////
    #////////   single_contour_lib.c     ////////
    #////////////////////////////////////////////
    pr.sc_findRtau.id       = d['sc_findRtau']['id']
    pr.sc_findRtau.max_iter = d['sc_findRtau']['max_iter']
    pr.sc_findRtau.epsabs   = d['sc_findRtau']['epsabs']
    pr.sc_findRtau.epsrel   = d['sc_findRtau']['epsrel']

    pr.sc_findRtau_bracket.id       = d['sc_findRtau_bracket']['id']
    pr.sc_findRtau_bracket.max_iter = d['sc_findRtau_bracket']['max_iter']
    pr.sc_findRtau_bracket.epsabs   = d['sc_findRtau_bracket']['epsabs']
    pr.sc_findRtau_bracket.epsrel   = d['sc_findRtau_bracket']['epsrel']

    pr.sc_intdRdtau.id     = d['sc_intdRdtau']['id']
    pr.sc_intdRdtau.h      = d['sc_intdRdtau']['h']
    pr.sc_intdRdtau.epsabs = d['sc_intdRdtau']['epsabs']
    pr.sc_intdRdtau.epsrel = d['sc_intdRdtau']['epsrel']
    pr.sc_intdRdtau_R0 = d['sc_intdRdtau_R0']

    pr.sc_syscontour_eps = d['sc_syscontour_eps']

    pr.sc_intContourStd.id     = d['sc_intContourStd']['id']
    pr.sc_intContourStd.h      = d['sc_intContourStd']['h']
    pr.sc_intContourStd.epsabs = d['sc_intContourStd']['epsabs']
    pr.sc_intContourStd.epsrel = d['sc_intContourStd']['epsrel']

    pr.sc_intContourRob.id     = d['sc_intContourRob']['id']
    pr.sc_intContourRob.h      = d['sc_intContourRob']['h']
    pr.sc_intContourRob.epsabs = d['sc_intContourRob']['epsabs']
    pr.sc_intContourRob.epsrel = d['sc_intContourRob']['epsrel']
    pr.sc_intContourRob_sigmaf = d['sc_intContourRob_sigmaf']

    pr.sc_intContour_tau_smallest = d['sc_intContour_tau_smallest']
    pr.sc_intContour_tol_brack    = d['sc_intContour_tol_brack']
    pr.sc_intContour_tol_add      = d['sc_intContour_tol_add']

    pr.sc_drivContour_taumin_over_y2 = d['sc_drivContour_taumin_over_y2']

    pr.sc_getContourStd.id     = d['sc_getContourStd']['id']
    pr.sc_getContourStd.h      = d['sc_getContourStd']['h']
    pr.sc_getContourStd.epsabs = d['sc_getContourStd']['epsabs']
    pr.sc_getContourStd.epsrel = d['sc_getContourStd']['epsrel']

    pr.sc_getContourRob.id     = d['sc_getContourRob']['id']
    pr.sc_getContourRob.h      = d['sc_getContourRob']['h']
    pr.sc_getContourRob.epsabs = d['sc_getContourRob']['epsabs']
    pr.sc_getContourRob.epsrel = d['sc_getContourRob']['epsrel']
    pr.sc_getContourRob_sigmaf = d['sc_getContourRob_sigmaf']

    pr.sc_getContour_tol_brack = d['sc_getContour_tol_brack']
    pr.sc_getContour_tol_add   = d['sc_getContour_tol_add']

    pr.sc_warn_switch   = d['sc_warn_switch']


    #////////////////////////////////////////////
    #////////   multi_contour_lib.c     /////////
    #////////////////////////////////////////////
    pr.mc_intRtau.id     = d['mc_intRtau']['id']
    pr.mc_intRtau.h      = d['mc_intRtau']['h']
    pr.mc_intRtau.epsabs = d['mc_intRtau']['epsabs']
    pr.mc_intRtau.epsrel = d['mc_intRtau']['epsrel']

    pr.mc_brackRtau_small_maxiter   = d['mc_brackRtau_small_maxiter']
    pr.mc_brackRtau_small_nbrackets = d['mc_brackRtau_small_nbrackets']
    pr.mc_brackRtau_small_Rmin      = d['mc_brackRtau_small_Rmin']
    pr.mc_brackRtau_small_Rini      = d['mc_brackRtau_small_Rini']

    pr.mc_brackRtau_large_maxiter = d['mc_brackRtau_large_maxiter']
    pr.mc_brackRtau_large_Rini    = d['mc_brackRtau_large_Rini']
    pr.mc_brackRtau_large_scale    = d['mc_brackRtau_large_scale']

    pr.mc_updCondODE_tol_brack = d['mc_updCondODE_tol_brack']
    pr.mc_updCondODE_tol_add   = d['mc_updCondODE_tol_add']

    pr.mc_findRbracket.id       = d['mc_findRbracket']['id']
    pr.mc_findRbracket.max_iter = d['mc_findRbracket']['max_iter']
    pr.mc_findRbracket.epsabs   = d['mc_findRbracket']['epsabs']
    pr.mc_findRbracket.epsrel   = d['mc_findRbracket']['epsrel']

    pr.mc_intContourSaddle.id     = d['mc_intContourSaddle']['id']
    pr.mc_intContourSaddle.h      = d['mc_intContourSaddle']['h']
    pr.mc_intContourSaddle.epsabs = d['mc_intContourSaddle']['epsabs']
    pr.mc_intContourSaddle.epsrel = d['mc_intContourSaddle']['epsrel']

    pr.mc_fillSaddleCenter_nsigma = d['mc_fillSaddleCenter_nsigma']
    pr.mc_fillSaddleCenter_dR     = d['mc_fillSaddleCenter_dR']
    pr.mc_fillSaddleCenter_sigmaf = d['mc_fillSaddleCenter_sigmaf']

    pr.mc_intContour.id     = d['mc_intContour']['id']
    pr.mc_intContour.h      = d['mc_intContour']['h']
    pr.mc_intContour.epsabs = d['mc_intContour']['epsabs']
    pr.mc_intContour.epsrel = d['mc_intContour']['epsrel']
    pr.mc_intContour_sigmaf = d['mc_intContour_sigmaf']

    pr.mc_drivContour_taumin_over_y2 = d['mc_drivContour_taumin_over_y2']

    pr.mc_minInSaddle.id         = d['mc_minInSaddle']['id']
    pr.mc_minInSaddle.max_iter   = d['mc_minInSaddle']['max_iter']
    pr.mc_minInSaddle.first_step = d['mc_minInSaddle']['first_step']
    pr.mc_minInSaddle.tol        = d['mc_minInSaddle']['tol']
    pr.mc_minInSaddle.epsabs     = d['mc_minInSaddle']['epsabs']
    pr.mc_minInSaddle_dR     = d['mc_minInSaddle_dR']

    pr.mc_getContour.id     = d['mc_getContour']['id']
    pr.mc_getContour.h      = d['mc_getContour']['h']
    pr.mc_getContour.epsabs = d['mc_getContour']['epsabs']
    pr.mc_getContour.epsrel = d['mc_getContour']['epsrel']
    pr.mc_getContour_sigmaf = d['mc_getContour_sigmaf']

    pr.mc_getContour_x1x2.id     = d['mc_getContour_x1x2']['id']
    pr.mc_getContour_x1x2.h      = d['mc_getContour_x1x2']['h']
    pr.mc_getContour_x1x2.epsabs = d['mc_getContour_x1x2']['epsabs']
    pr.mc_getContour_x1x2.epsrel = d['mc_getContour_x1x2']['epsrel']


    #////////////////////////////////////////////
    #///////   single_integral_lib.c     ////////
    #////////////////////////////////////////////
    pr.si_findBrackBracket.id       = d['si_findBrackBracket']['id']
    pr.si_findBrackBracket.max_iter = d['si_findBrackBracket']['max_iter']
    pr.si_findBrackBracket.epsabs   = d['si_findBrackBracket']['epsabs']
    pr.si_findBrackBracket.epsrel   = d['si_findBrackBracket']['epsrel']

    pr.si_findRootBracket.id       = d['si_findRootBracket']['id']
    pr.si_findRootBracket.max_iter = d['si_findRootBracket']['max_iter']
    pr.si_findRootBracket.epsabs   = d['si_findRootBracket']['epsabs']
    pr.si_findRootBracket.epsrel   = d['si_findRootBracket']['epsrel']

    pr.si_findMovBracket_maxiter = d['si_findMovBracket_maxiter']
    pr.si_findMovBracket_scale   = d['si_findMovBracket_scale']

    pr.si_dirSingInt.n      = d['si_dirSingInt']['n']
    pr.si_dirSingInt.epsabs = d['si_dirSingInt']['epsabs']
    pr.si_dirSingInt.epsrel = d['si_dirSingInt']['epsrel']

    pr.si_qngSingInt_epsabs = d['si_qngSingInt_epsabs']
    pr.si_qngSingInt_epsrel = d['si_qngSingInt_epsrel']
    pr.si_qngSingInt_ximin  = d['si_qngSingInt_ximin']

    pr.si_qagSingInt.n      = d['si_qagSingInt']['n']
    pr.si_qagSingInt.epsabs = d['si_qagSingInt']['epsabs']
    pr.si_qagSingInt.epsrel = d['si_qagSingInt']['epsrel']
    pr.si_qagSingInt_ximin  = d['si_qagSingInt_ximin']

    pr.si_drivContour_taumin_over_y2 = d['si_drivContour_taumin_over_y2']


    #////////////////////////////////////////////
    #////////   analytic_SIS_lib.c     //////////
    #////////////////////////////////////////////
    pr.as_eps_soft = d['as_eps_soft']

    pr.as_FwSIS_n = d['as_FwSIS_n']
    pr.as_FwSIS_nmax_switch = d['as_FwSIS_nmax_switch']

    pr.as_FwDirect.n      = d['as_FwDirect']['n']
    pr.as_FwDirect.epsabs = d['as_FwDirect']['epsabs']
    pr.as_FwDirect.epsrel = d['as_FwDirect']['epsrel']

    pr.as_slFwOsc_Direct.n      = d['as_slFwOsc_Direct']['n']
    pr.as_slFwOsc_Direct.epsabs = d['as_slFwOsc_Direct']['epsabs']
    pr.as_slFwOsc_Direct.epsrel = d['as_slFwOsc_Direct']['epsrel']

    pr.as_slFwOsc_Osc.n      = d['as_slFwOsc_Osc']['n']
    pr.as_slFwOsc_Osc.epsabs = d['as_slFwOsc_Osc']['epsabs']
    pr.as_slFwOsc_Osc.epsrel = d['as_slFwOsc_Osc']['epsrel']

    pr.as_wlFwOsc_Direct.n      = d['as_wlFwOsc_Direct']['n']
    pr.as_wlFwOsc_Direct.epsabs = d['as_wlFwOsc_Direct']['epsabs']
    pr.as_wlFwOsc_Direct.epsrel = d['as_wlFwOsc_Direct']['epsrel']

    pr.as_wlFwOsc_Osc.n      = d['as_wlFwOsc_Osc']['n']
    pr.as_wlFwOsc_Osc.epsabs = d['as_wlFwOsc_Osc']['epsabs']
    pr.as_wlFwOsc_Osc.epsrel = d['as_wlFwOsc_Osc']['epsrel']


    #////////////////////////////////////////////
    #////////      fourier_lib.c       //////////
    #////////////////////////////////////////////
    pr.fo_updRegSch_nmax_slope = d['fo_updRegSch_nmax_slope']
    pr.fo_updRegSch_nmax_tail  = d['fo_updRegSch_nmax_tail']
    pr.fo_updRegSch_Itmin_tail = d['fo_updRegSch_Itmin_tail']
