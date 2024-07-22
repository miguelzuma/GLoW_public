#cython: language_level=3

from clenses cimport pNamedLens

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "area_lib.h":
    ctypedef struct pAreaIntegral:
        int n_rho
        int n_theta
        double y
        double tau_max
        pNamedLens *pNLens
    
    int integrate_AreaIntegral(double *t_min, double *tau_result, double *It_result, int n_result, pAreaIntegral *p)
