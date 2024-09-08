#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "ode_tools.h"
#include "lenses_lib.h"
#include "roots_lib.h"
#include "contour_lib.h"

// =====================================================================

int main(int argc, char *argv[])
{
    int i, n_points, n_ctrs;
    //~ int j;
    double y, sigmaf;
    double tau, I;
    double psi0;
    //~ double rc;
    double x10, x20;
    pNamedLens *pNLens;
    SolODE *sol;
    SolODE **sols;

    CritPoint *points;
    Center *ctrs;
    //~ pIntegral2d p;
    //~ Lens Psi;

    // Get impact parameter
    if(argc==2)
        y = atof(argv[1]);
    else
    {
        printf("Usage: %s y\n", argv[0]);
        return 1;
    }

    // Create the lens
    //~ pNLens = create_pLens_offcenterCIS(1, 0.05, 0.1, 0.1);
    pNLens = create_pLens_offcenterSIS(1, 0., 0.);
    //~ pNLens = create_pLens_eSIS(1, 1.2, 0.01);

    // benchmark lens
    //~ rc = 0.05;
    //~ psi0 = 1./4.;
    //~ pNLens = create_pLens_CombinedLens(4);
    //~ add_lens_CombinedLens(create_pLens_offcenterCIS(psi0, rc,  0.3,    0), pNLens);
    //~ add_lens_CombinedLens(create_pLens_offcenterCIS(psi0, rc, -0.6,  0.3), pNLens);
    //~ add_lens_CombinedLens(create_pLens_offcenterCIS(psi0, rc,  0.3, -0.3), pNLens);
    //~ add_lens_CombinedLens(create_pLens_offcenterCIS(psi0, rc,    0,    0), pNLens);

    // benchmark lens 3
    psi0 = 1./4.;
    pNLens = create_pLens_CombinedLens(4);
    add_lens_CombinedLens(create_pLens_offcenterBall(psi0, 1.0,  0.3,    0), pNLens);
    add_lens_CombinedLens(create_pLens_offcenterBall(psi0, 1.0, -0.6,  0.3), pNLens);
    add_lens_CombinedLens(create_pLens_offcenterBall(psi0, 1.0,  0.3, -0.3), pNLens);
    add_lens_CombinedLens(create_pLens_offcenterBall(psi0, 1.0,    0,    0), pNLens);

    // test finding cusps and sing points, even in recursive settings
    if(_FALSE_)
    //~ if(_TRUE_)
    {
        int i, n;
        //~ double *xvec;
        //~ pNamedLens *pNLens2;
        //~ pNamedLens *pNLens1;

        CritPoint *pp;
        //~ Lens Psi;

        // try valgrind with this to see if allocation is allright
        //~ psi0 = 0.25;
        //~ pNLens1 = create_pLens_CombinedLens(2);
        //~ add_lens_CombinedLens(create_pLens_offcenterSIS(psi0, 0,    0), pNLens1);
        //~ add_lens_CombinedLens(create_pLens_offcenterSIS(psi0, 0,    0), pNLens1);

        //~ pNLens2 = create_pLens_CombinedLens(2);
        //~ add_lens_CombinedLens(create_pLens_offcenterSIS(psi0, 0,    0), pNLens2);
        //~ add_lens_CombinedLens(create_pLens_offcenterSIS(psi0, 0,    0), pNLens2);

        //~ pNLens = create_pLens_CombinedLens(2);
        //~ add_lens_CombinedLens(pNLens1, pNLens);
        //~ add_lens_CombinedLens(pNLens2, pNLens);

        //~ n = 0;
        //~ xvec = get_cusp_sing(&n, pNLens);
        //~ printf("n = %d\n", n);
        //~ for(i=0;i<n;i+=2)
            //~ printf("x = (%e, %e)\n", xvec[i], xvec[i+1]);
        //~ free(xvec);

        //~ Psi = init_lens(pNLens);
        //~ n = 0;
        //~ pp = init_cusp_sing(&n, y, &Psi, pNLens);
        //~ printf("n = %d\n", n);
        //~ for(i=0;i<n;i++)
            //~ display_CritPoint(pp+i);
        //~ free(pp);

        pp = driver_all_CritPoints_2D(&n, y, pNLens);
        ctrs = init_all_Center(&n, pp, y, pNLens);

        printf("\n");
        for(i=0;i<n;i++)
        {
            printf("i = %d\n", i);
            display_Center(ctrs+i);
            printf("\n");
        }

        free(pp);
        free_all_Center(ctrs);
        free_pLens(pNLens);

        exit(1);
    }

    // Test computation of contours
    // -----------------------------------------------------------------
    x10 = 3.2;
    x20 = 2.;
    n_points = 100;
    sigmaf = 1e2;
    sol = init_SolODE(N_dims, n_points);
    driver_get_contour2d_x1x2(x10, x20, y, sigmaf, n_points, pNLens, sol);

    //~ for(i=0;i<sol->n_points;i++)
        //~ printf("sigma=%7.3f\t x1=%7.3f\t x2=%7.3f\n", sol->t[i], sol->y[i_x1][i], sol->y[i_x2][i]);

    free_SolODE(sol);
    // -----------------------------------------------------------------

    // Test centers
    // -----------------------------------------------------------------

    // find and display critical points
    points = driver_all_CritPoints_2D(&n_points, y, pNLens);

    //~ for(i=0;i<n_points;i++)
        //~ display_CritPoint(points+i);

    // find and display centers
    n_ctrs = n_points;
    ctrs = init_all_Center(&n_ctrs, points, y, pNLens);

    printf("\n");
    for(i=0;i<n_ctrs;i++)
    {
        printf("i = %d\n", i);
        display_Center(ctrs+i);
        printf("\n");
    }

    // compute integral
    for(i=1;i<10;i++)
    {
        tau = i*3e-1/10;
        I = driver_contour2d(tau, n_ctrs, ctrs, y, pNLens)/M_2PI;

        printf("tau=%f    I/2pi=%f\n", tau, I);
    }

    //~ printf("\nI=%f\n", driver_contour2d(0.045, n_points, ctrs, y, pNLens)/M_2PI);
    // -----------------------------------------------------------------

    // Test contours (w points)
    // -----------------------------------------------------------------
    tau = 0.07;
    n_points = 0;
    sols = driver_get_contour2d(tau, n_points, n_ctrs, ctrs, y, pNLens);

    //~ for(i=0;i<n_ctrs;i++)
    //~ {
        //~ if(sols[i] != NULL)
        //~ {
            //~ printf("i=%d\n", i);
            //~ for(j=0;j<sols[i]->n_points;j++)
                //~ printf("sigma=%f   alpha/pi=%f    R=%f    x1=%f     x2=%f\n",
                       //~ sols[i]->t[j], sols[i]->y[0][j]/M_PI, sols[i]->y[1][j], sols[i]->y[2][j], sols[i]->y[3][j]);
            //~ printf("\n");
        //~ }
    //~ }

    free_SolODE_contour2d(n_ctrs, sols);
    // -----------------------------------------------------------------

    free(points);
    free_all_Center(ctrs);
    free_pLens(pNLens);

    return 0;
}
