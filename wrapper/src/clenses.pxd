#cython: language_level=3

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "lenses_lib.h" nogil:
    ctypedef struct pNamedLens:
        int lens_type
        void *pLens

    ctypedef struct Lens:
        double (*psi)(double x1, double x2, void *pLens)
        int (*psi_1stDerivs)(double *psi_derivs, double x1, double x2, void *pLens)
        int (*psi_2ndDerivs)(double *psi_derivs, double x1, double x2, void *pLens)
        void *pLens

    ctypedef enum indices_spatial: i_x1, i_x2, N_dims
    ctypedef enum indices_derivs: i_0, i_dx1, i_dx2, i_dx1dx1, i_dx2dx2, i_dx1dx2, N_derivs

    double phiFermat(double y, double x1, double x2, Lens *Psi)
    int phiFermat_1stDeriv(double *phi_derivs, double y, double x1, double x2, Lens *Psi)
    int phiFermat_2ndDeriv(double *phi_derivs, double y, double x1, double x2, Lens *Psi)

    double call_psi(double x1, double x2, Lens *Psi)
    int call_psi_1stDerivs(double *psi_derivs, double x1, double x2, Lens *Psi)
    int call_psi_2ndDerivs(double *psi_derivs, double x1, double x2, Lens *Psi)

    Lens init_lens(pNamedLens *pNLens)
    void free_pLens(pNamedLens *pNLens)

    pNamedLens* create_pLens_SIS(double psi0)
    pNamedLens* create_pLens_CIS(double psi0, double rc)
    pNamedLens* create_pLens_PointLens(double psi0, double xc)
    pNamedLens* create_pLens_Ball(double psi0, double b)
    pNamedLens* create_pLens_NFW(double psi0, double xs)
    pNamedLens* create_pLens_tSIS(double psi0, double xb)
    pNamedLens* create_pLens_offcenterSIS(double psi0, double xc1, double xc2)
    pNamedLens* create_pLens_offcenterCIS(double psi0, double rc, double xc1, double xc2)
    pNamedLens* create_pLens_offcenterPointLens(double psi0, double xc, double xc1, double xc2)
    pNamedLens* create_pLens_offcenterBall(double psi0, double b, double xc1, double xc2)
    pNamedLens* create_pLens_offcenterNFW(double psi0, double xs, double xc1, double xc2)
    pNamedLens* create_pLens_CombinedLens(int n_sublenses)
    int add_lens_CombinedLens(pNamedLens* new_pNLens, pNamedLens* combined_pNLens)
    pNamedLens* create_pLens_Grid1d(char *fname, int n_grid)
    pNamedLens* create_pLens_SIE(double psi0, double q, double alpha, double xc1, double xc2)
    pNamedLens* create_pLens_Ext(double kappa, double gamma1, double gamma1)

cdef pNamedLens* convert_pphys_to_pLens(Psi)
