#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"
#include "single_integral_lib.h"

// =================================================================

int main(int argc, char *argv[])
{
    int ntaus;
    double psi0, y, tau;
    double tau_min, tau_max, tmin, I;
    pNamedLens *p;
    Lens Psi;
    //~ Contours *cnt;

    int i, n_points;
    //~ int j;
    CritPoint *points;
    pSIntegral pI;

    handle_GSL_errors();
    pprec.no_gslerrors = 1;

    if(argc==3)
    {
        y = atof(argv[1]);
        tau = atof(argv[2]);
    }
    else
    {
        printf("Usage: %s y tau\n", argv[0]);
        return 1;
    }

    psi0 = 1;

    // Different lenses only appear here
    //~ p = create_pLens_SIS(psi0);
    p = create_pLens_PointLens(psi0);
    //~ p = create_pLens_CIS(psi0, 0.1);
    //~ p = create_pLens_NFW(psi0, 0.01);
    Psi = init_lens(p);

    printf("\n");

    // -----------------------------
    // ----- TEST BRACKETS
    // -----------------------------

    points = find_all_CritPoints_1D(&n_points, y, &Psi);

    for(i=0;i<n_points;i++)
        display_CritPoint(points+i);

    tmin = points[0].t;

    pI.y = y;
    pI.tau = tau;
    pI.t = tau + tmin;
    pI.n_points = n_points;
    pI.points = points;
    pI.Psi = &Psi;

    sort_x_CritPoint(n_points, points);

    printf(" -- Brackets found:\n");
    find_Brackets(&pI);
    for(i=0;i<pI.n_brackets;i++)
        printf("br[%d] = (%g, %g)\n", i, pI.brackets[i].a, pI.brackets[i].b);
    free_Brackets(&pI);

    //~ free(points);
    //~ free_pLens(p);
    //~ exit(1);

    ntaus = 1000;
    tau_min = 1e-3;
    tau_max = 10;
    for(i=0;i<ntaus;i++)
    {
        tau = exp(log(tau_min) + i/(ntaus-1.)*(log(tau_max) - log(tau_min)));
        I = driver_SingleIntegral(tau, y, tmin, n_points, points, p, m_integral_qng);

        printf("tau=%e   I=%e\n", tau, I/M_2PI);
    }

    //~ cnt = driver_get_contour_SingleIntegral(tau, 100, y, n_points, points, p);
    //~ for(i=0;i<cnt->n_contours;i++)
    //~ {
        //~ printf("Contour %d\n", i);
        //~ for(j=0;j<cnt->n_points;j++)
        //~ {
            //~ printf("x1 = %g     x2 = %g\n", cnt->x1[i][j], cnt->x2[i][j]);
        //~ }
        //~ printf("\n");
    //~ }
    //~ free_Contours(cnt);

    free(points);
    free_pLens(p);

    return 0;
}

// =================================================================
