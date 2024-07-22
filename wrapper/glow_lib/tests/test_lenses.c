#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "common.h"
#include "lenses_lib.h"

// =================================================================

int main(int argc, char *argv[])
{
    char *root = "external/gSIS";
    pNamedLens *p;
    Lens psi;
    double x;
    double psi_derivs[N_derivs];
    
    p = create_pLens_Grid1d(root, 10000);
    
    psi = init_lens(p);
    
    x = 1.39875;
    psi.psi_2ndDerivs(psi_derivs, x, 0, psi.pLens);
    printf("%f    %f    %f\n", psi_derivs[i_0], psi_derivs[i_dx1], psi_derivs[i_dx2]);
    printf("%f    %f    %f\n", psi_derivs[i_dx1dx1], psi_derivs[i_dx1dx2], psi_derivs[i_dx2dx2]);
    
    free_pLens(p);
    
    return 0;
}
