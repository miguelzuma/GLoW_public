#cython: language_level=3

## ----------  Precision ------------- ##
from ccommon cimport Prec_General, handle_GSL_errors

cdef update_pprec(Prec_General pprec_general)
cdef extern from "common.h" nogil:
    Prec_General pprec
## ----------------------------------- ##

cdef extern from "special_lib.h" nogil:
    double f_fresnel(double x)
    double g_fresnel(double x)

    double Mtilde_Struve_PowerSeries(double x, double nu, double tol)
    double Mtilde_Struve_Asymptotic(double x, double nu, double tol)
    double Mtilde_Struve_PieceWise(double x, double nu, double tol)
    double Mtilde_Struve(double x, double nu, double tol)

    double complex F11_singlepoint(double u, double c, int *status, int *approx_flag)
    int F11_sorted(double *u, double c, int n_F, double complex *F11, int nthreads)

    int sorted_interpolation(double *x_subgrid, double *y_subgrid, int n_subgrid,
                             double *x_grid, double *y_grid, int n_grid)
