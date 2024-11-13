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
    double psi0, y, xc1, xc2;
    pNamedLens *p;

    Lens Psi;
    pImage pimage;
    CritPoint point1, point2;
    CritPoint *points;

    int n1, n2;
    CritPoint *p1, *p2;

    handle_GSL_errors();

    y = 0;
    psi0 = 1.;

    // point lens + shear
    //~ xc1 = 0.9183673469387754;
    //~ xc1 = -2.7551020408163263;
    xc1 = 7.428571428571429;
    //~ xc1 = 9.;
    xc2 = -0.1;

    pprec.ro_findallCP2D_force_search = _TRUE_;
    //~ pprec.ro_findfirstCP2D_Rout = 100;
    //~ pprec.ro_initcusp_n = 2000;
    //~ pprec.ro_initcusp_R = 1e-4;
    pprec.ro_findnearCritPoint_scale = 1.5;
    pprec.ro_findallCP2D_npoints = 500;

    p = create_pLens_CombinedLens(2);
    add_lens_CombinedLens(create_pLens_offcenterPointLens(psi0, 1e-10, xc1, xc2), p);
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

    // -----------------------------
    // ----- DEBUG COMBINE IMAGES
    // -----------------------------
    //~ printf("\n  - Combine CritPoint:\n");
    //~ p1 = driver_all_CritPoints_2D(&n1, y, p);
    //~ p2 = find_all_CritPoints_min_2D(&n2, y, &Psi);
    //~ points = merge_CritPoint(n1, p1, n2, p2, &n_points);
    //~ for(i=0;i<n_points;i++)
    //~ {
        //~ printf("i = %d   ", i);
        //~ display_CritPoint(points+i);
    //~ }
    //~ free(points);

    //~ printf("\n  - Filter CritPoint:\n");
    //~ p1 = driver_all_CritPoints_2D(&n1, y, p);
    //~ p2 = find_all_CritPoints_min_2D(&n2, y, &Psi);

    //~ n_points = n1+n2;
    //~ points = (CritPoint *)malloc(n_points*sizeof(CritPoint));
    //~ for(i=0;i<n1;i++)
        //~ copy_CritPoint(points+i, p1+i);
    //~ for(i=0;i<n2;i++)
        //~ copy_CritPoint(points+n1+i, p2+i);
    //~ points[0].type = type_non_converged;

    //~ printf("    -- Old list:\n");
    //~ for(i=0;i<n_points;i++)
    //~ {
        //~ printf("i = %d   ", i);
        //~ display_CritPoint(points+i);
    //~ }

    //~ printf("    -- Filtered list:\n");
    //~ points = filter_CritPoint(&n_points, points);
    //~ for(i=0;i<n_points;i++)
    //~ {
        //~ printf("i = %d   ", i);
        //~ display_CritPoint(points+i);
    //~ }

    //~ free(p1);
    //~ free(p2);
    //~ free(points);

    free_pLens(p);

    return 0;
}

// =================================================================
