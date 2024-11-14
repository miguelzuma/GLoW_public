#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"
#include "ode_tools.h"
#include "contour_lib.h"

#define N_points 100

// =================================================================

// DEBUGGING SADDLE INTEGRATION

int main(int argc, char *argv[])
{
    int i;
    int n_points, n_centers;
    double y, xc1, xc2;
    pNamedLens *p;
    CritPoint *points;
    Center *centers;

    handle_GSL_errors();

    y = 0;

    // default pprec
    // pprec.mc_fillSaddleCenter_nsigma = 100;
    // pprec.mc_fillSaddleCenter_dR     = 5e-2;
    // pprec.mc_fillSaddleCenter_sigmaf = 1000;

    // new pprec
    pprec.mc_fillSaddleCenter_nsigma = 100;
    pprec.mc_fillSaddleCenter_dR     = 5e-2;
    pprec.mc_fillSaddleCenter_sigmaf = 100;

    //==== Create lens
    //=========================================
    xc1 = 8.714285714285715;
    xc2 = -0.1;

    p = create_pLens_CombinedLens(2);
    add_lens_CombinedLens(create_pLens_offcenterPointLens(1, 1e-10, xc1, xc2), p);
    add_lens_CombinedLens(create_pLens_Ext(0, 0.9, 0), p);

    //~ Psi = init_lens(p);

    //==== Find roots
    //=========================================
    printf("\n----------------------------------\n");
    printf("Finding images:\n");
    points = driver_all_CritPoints_2D(&n_points, y, p);
    for(i=0;i<n_points;i++)
        display_CritPoint(points+i);

    //==== Compute the saddles
    //=========================================
    printf("\n----------------------------------\n");
    printf("Finding centers:\n");
    n_centers = n_points;
    centers = init_all_Center(&n_centers, points, y, p);
    for(i=0;i<n_centers;i++)
    {
        //~ display_Center(centers+i);
        //~ printf("\n");
    }

    // HVR: start debugging fill_saddle_Center

    // ------------------------------------------------------

    free_all_Center(centers);
    free(points);
    free_pLens(p);

    return 0;
}

// =================================================================
