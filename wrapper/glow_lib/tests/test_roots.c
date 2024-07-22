#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"

#define N_points 100

// =================================================================

int main(int argc, char *argv[])
{
    int i, n_points;
    double psi0, y;
    //~ double rc;
    pNamedLens *p;

    Lens Psi;
    pImage pimage;
    CritPoint point1, point2;
    CritPoint *points;

    //~ handle_GSL_errors();

    if(argc==2)
        y = atof(argv[1]);
    else
    {
        printf("Usage: %s y\n", argv[0]);
        return 1;
    }

    psi0 = 1.;

    // Different lenses only appear here
    //~ p = create_pLens_SIS(psi0);
    //~ p = create_pLens_CIS(psi0, 0.1);
    //~ p = create_pLens_offcenterCIS(psi0, 0.1, 0, 0);
    p = create_pLens_NFW(psi0, 10.);
    //~ p = create_pLens_PointLens(psi0);

    //~ rc = 0.05;
    //~ psi0 = 1./4.;
    //~ p = create_pLens_CombinedLens(4);
    //~ add_lens_CombinedLens(create_pLens_offcenterCIS(psi0, rc,  0.3,    0), p);
    //~ add_lens_CombinedLens(create_pLens_offcenterCIS(psi0, rc, -0.6,  0.3), p);
    //~ add_lens_CombinedLens(create_pLens_offcenterCIS(psi0, rc,  0.3, -0.3), p);
    //~ add_lens_CombinedLens(create_pLens_offcenterCIS(psi0, rc,    0,    0), p);

    //~ add_lens_CombinedLens(create_pLens_offcenterSIS(psi0/4.,  0.3,    0), p);
    //~ add_lens_CombinedLens(create_pLens_offcenterSIS(psi0/4., -0.6,  0.3), p);
    //~ add_lens_CombinedLens(create_pLens_offcenterSIS(psi0/4.,  0.3, -0.3), p);
    //~ add_lens_CombinedLens(create_pLens_offcenterSIS(psi0/4.,    0,    0), p);

    // problematic lens
    //~ p = create_pLens_CombinedLens(2);
    //~ add_lens_CombinedLens(create_pLens_offcenterSIS(0.8, 1.4989722544146864, -0.05551738912226567), p);
    //~ add_lens_CombinedLens(create_pLens_offcenterSIS(0.2, -1.4989722544146864, 0.05551738912226567), p);

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
    // ----- 1D VERSION
    // -----------------------------
    printf("\n----------------------------------\n");
    printf("Finding all roots in 1D:\n");

    points = find_all_CritPoints_1D(&n_points, y, &Psi);

    printf("\nn_points = %d\n", n_points);
    for(i=0;i<n_points;i++)
        display_CritPoint(points+i);

    printf("\nSorted points:\n");
    sort_x_CritPoint(n_points, points);
    for(i=0;i<n_points;i++)
        display_CritPoint(points+i);

    free(points);

    // -----------------------------
    // ----- 2D MINIMIZATION
    // -----------------------------
    printf("\n----------------------------------\n");
    printf("Finding minimum through 2D minimization:\n");

    pimage.y = y;
    pimage.point = &point1;
    pimage.Psi = &Psi;

    printf("\nUsing arbitrary initial guess:\n");
    find_local_Minimum_2D(-4, -8, &pimage);
    display_CritPoint(&point1);

    printf("\nRobust minimum:\n");
    find_global_Minimum_2D(&pimage);
    display_CritPoint(&point1);

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

    PWARNING("last i=%d  i=%d", i, i)

    return 0;
}

// =================================================================
