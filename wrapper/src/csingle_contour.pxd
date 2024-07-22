#cython: language_level=3

from clenses cimport pNamedLens

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "ode_tools.h" nogil:
    ctypedef struct SolODE:
        int n_buffer
        int n_allocated
        int n_points
        int n_eqs
        double *t
        double **y
        
    SolODE *init_SolODE(int n_eqs, int n_buffer)
    void free_SolODE(SolODE *sol)

cdef extern from "single_contour_lib.h" nogil:
    ctypedef enum methods_contour: m_contour_std, m_contour_robust
    
    # compute I(tau)    
    double driver_contour(double tau_ini, double x1_min, double x2_min, double y,
                          pNamedLens *pNLens, int method);
    
    # increase the density of points in the contour
    SolODE *interpolate_contour(int n_points, double x1_min, double x2_min, SolODE *sol)
    
    # store the contour for tau in SolODE
    int driver_get_contour(double tau, int n_points, double x1_min, double x2_min, double y,
                           pNamedLens *pNLens, int method, SolODE *sol)
    
    # integrate dR_dtau to find R(tau)
    int driver_dR_dtau(int n_points, double *R_grid, double *tau_grid,
                       double x1_min, double x2_min, double y,
                       pNamedLens *pNLens);
    
    # invert tau(R) to find R(tau)
    int driver_R_of_tau(int n_points, double *R_grid, double *tau_grid,
                        double x1_min, double x2_min, double y,
                        pNamedLens *pNLens);
