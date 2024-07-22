#cython: language_level=3

from clenses cimport pNamedLens
from croots cimport CritPoint

cdef int _TRUE_ = 1
cdef int _FALSE_ = 0

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

cdef extern from "contour_lib.h" nogil:
    ctypedef enum ctr_types: ctr_type_min, \
                             ctr_type_max, \
                             ctr_type_saddle_8_maxmax, \
                             ctr_type_saddle_8_minmin, \
                             ctr_type_saddle_O_min, \
                             ctr_type_saddle_O_max, \
                             N_ctr_type
                             
    ctypedef struct Center:
        char type
        double x0[2]
        double tau0
        double t0
        double R_max
        double alpha_out
        char is_init_birthdeath
        double tau_birth
        double tau_death
    
    Center *init_all_Center(int *n_points, 
                            CritPoint *points, 
                            double y, 
                            pNamedLens *pNLens)
                            
    void free_all_Center(Center *ctrs)
    
    double driver_contour2d(double tau, 
                            int n_ctrs, 
                            Center *ctrs, 
                            double y, 
                            pNamedLens *pNLens)
    
    SolODE **driver_get_contour2d(double tau, 
                                  int n_points, 
                                  int n_ctrs, 
                                  Center *ctrs, 
                                  double y, 
                                  pNamedLens *pNLens)
                            
    void free_SolODE_contour2d(int n_sols, SolODE **sols)
    
    int driver_get_contour2d_x1x2(double x10, 
                                  double x20, 
                                  double y,
                                  double sigmaf, 
                                  int n_points, 
                                  pNamedLens *pNLens, 
                                  SolODE *sol)
