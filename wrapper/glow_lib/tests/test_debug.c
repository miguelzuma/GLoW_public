#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"

#define N_points 100

// =================================================================

// DEBUGGING SHEAR

int main(int argc, char *argv[])
{
    int i, n_points;
    double psi0, y, xc;
    pNamedLens *p;

    Lens Psi;
    pImage pimage;
    CritPoint point1, point2;
    CritPoint *points;

    handle_GSL_errors();

    y = 0;
    psi0 = 1.;

    // point lens + shear
    //~ xc = 0.9183673469387754;
    xc = -2.7551020408163263;
    //~ xc = 9.;

    pprec.ro_findallCP2D_force_search = _TRUE_;
    //~ pprec.ro_findfirstCP2D_Rout = 100;
    //~ pprec.ro_initcusp_n = 2000;
    //~ pprec.ro_initcusp_R = 1e-4;
    //~ pprec.ro_findnearCritPoint_scale = 1.5;

    p = create_pLens_CombinedLens(2);
    add_lens_CombinedLens(create_pLens_offcenterPointLens(psi0, 1e-10, xc, 1), p);
    add_lens_CombinedLens(create_pLens_Ext(0, 0.9, 0), p);

    Psi = init_lens(p);

    // -----------------------------
    // ----- MINIMIZATION
    // -----------------------------
    printf("\nFinding first two roots in 2D:\n");

    n_points = find_first_CritPoints_2D(y, &point1, &point2, &Psi);

    printf("\nn_points = %d\n", n_points);
    display_CritPoint(&point1);
    if(n_points > 1)
        display_CritPoint(&point2);

    // -----------------------------
    // ----- CRIT POINTS
    // -----------------------------
    printf("\n----------------------------------\n");
    i = check_only_min_CritPoint_2D(&point1, y, p);
    if(i==1)
    {
        printf("\nMinimum found:\n");
        display_CritPoint(&point1);
    }

    // -----------------------------
    // ----- GENERAL IMAGES
    // -----------------------------
    printf("\n----------------------------------\n");
    printf("General 2D image finder:\n");

    printf("\n  - Multidim minimization:\n");
    points = find_all_CritPoints_min_2D(&n_points, y, &Psi);
    for(i=0;i<n_points;i++)
    {
        printf("i = %d   ", i);
        display_CritPoint(points+i);
    }
    free(points);

    printf("\n  - Multidim root finding:\n");
    points = find_all_CritPoints_root_2D(&n_points, y, &Psi);
    for(i=0;i<n_points;i++)
    {
        printf("i = %d   ", i);
        display_CritPoint(points+i);
    }
    free(points);

    printf("\n  - Driver 2D:\n");
    points = driver_all_CritPoints_2D(&n_points, y, p);
    for(i=0;i<n_points;i++)
    {
        printf("i = %d   ", i);
        display_CritPoint(points+i);
    }
    free(points);

    free_pLens(p);

    return 0;
}

// =================================================================
