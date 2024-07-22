#cython: language_level=3

from clenses cimport init_lens, pNamedLens, Lens
from croots cimport CritPoint

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "single_integral_lib.h" nogil:
    ctypedef enum methods_integral: m_integral_g15, m_integral_g21, m_integral_g31, \
                                    m_integral_g41, m_integral_g51, m_integral_g61, \
                                    m_integral_dir, m_integral_qng
    
    ctypedef struct Bracket:
        double a
        double b
    
    ctypedef struct pSIntegral:
        double y
        double tau
        double t
        int n_points
        CritPoint *points
        int n_brackets
        Bracket *brackets
        Lens *Psi
    
    ctypedef struct Contours:
        int n_contours
        int n_points
        double **x1
        double **x2
    
    int find_Brackets(pSIntegral *p)
    void free_Brackets(pSIntegral *p)
    
    double alpha_SingleIntegral(double r, void *pintegral)
    double integrand_SingleIntegral(double xi, void *pintegral)
    
    double driver_SingleIntegral(double tau, double y, double tmin, int n_points, CritPoint *points,
                                 pNamedLens *pNLens, int method)
                                 
    void free_Contours(Contours *cnt)
    Contours *driver_get_contour_SingleIntegral(double tau, int n_cpoints, double y, double tmin,
                                                int n_points, CritPoint *points, 
                                                pNamedLens *pNLens)
