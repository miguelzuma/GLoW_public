#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "single_contour_lib.h"

#define N 1000

// =================================================================

int main(int argc, char *argv[])
{
    int i;
    double psi0, I;
    double x1_min, x2_min;
    double y;
    pNamedLens *p;

    double tau_i, tau_f;
    double R_grid[N];
    double tau_grid[N];

    SolODE *sol;
    //~ int method;
    double tau = 0.1;

    if(argc==2)
        y = atof(argv[1]);
    else
    {
        printf("Usage: %s y\n", argv[0]);
        return 1;
    }

    psi0 = 1;
    x1_min = psi0 + y;
    x2_min = 0;

    // Different lenses only appear here
    p = create_pLens_SIS(psi0);
    //~ p = create_pLens_offcenterSIS(psi0, 0., 0.);
    //~ p = create_pLens_CIS(psi0, 0.05);

    //~ p = create_pLens_CombinedLens(4);
    //~ add_lens_CombinedLens(create_pLens_SIS(0.25*psi0), p);
    //~ add_lens_CombinedLens(create_pLens_SIS(0.25*psi0), p);
    //~ add_lens_CombinedLens(create_pLens_SIS(0.25*psi0), p);
    //~ add_lens_CombinedLens(create_pLens_SIS(0.25*psi0), p);

    //~ p = create_pLens_Grid1d("external/tmp", 10000);

    // create tau_grid
    tau_i = 1e-4;
    tau_f = 100;
    for(i=0; i<N; i++)
    {
        //~ logtau = logtau_i + i/(N-1.)*(logtau_f-logtau_i);
        tau_grid[i] = exp(log(tau_i) + i/(N-1.)*log(tau_f/tau_i));
        //~ printf("%e\n", tau_grid[i]);
    }
    if(N == 1)
        tau_grid[0] = tau_f;

    for(i=0; i<N; i++)
    {
        I = driver_contour(tau_grid[i], x1_min, x2_min, y, p, 0);

        printf("tau=%e   I=%e\n", tau_grid[i], I);
    }
    printf("---------------------------------------------------------\n");

    driver_dR_dtau(N, R_grid, tau_grid, x1_min, x2_min, y, p);
    for(i=0; i<N; i++)
        printf("tau=%e   R=%e\n", tau_grid[i], R_grid[i]);
    printf("---------------------------------------------------------\n");

    driver_R_of_tau(N, R_grid, tau_grid, x1_min, x2_min, y, p);
    for(i=0; i<N; i++)
        printf("tau=%e   R=%e\n", tau_grid[i], R_grid[i]);
    printf("---------------------------------------------------------\n");

    driver_R_of_tau_bracket(N, R_grid, tau_grid, x1_min, x2_min, y, p);
    for(i=0; i<N; i++)
        printf("tau=%e   R=%e\n", tau_grid[i], R_grid[i]);
    printf("---------------------------------------------------------\n");

    // ----------------------------------------------

    // test computation of contours
    sol = init_SolODE(3, 20);
    driver_get_contour(tau, 0, x1_min, x2_min, y,
                       p, m_contour_std,
                       sol);

    for(i=0;i<sol->n_points;i++)
        printf("th=%7.3e\t R=%7.3e\t x1=%7.3e\t x2=%7.3e\n", sol->t[i]/2/M_PI -1,
                                                        sol->y[0][i],
                                                        sol->y[1][i],
                                                        sol->y[2][i]);
    free_SolODE(sol);
    printf("---------------------------------------------------------\n");


    sol = init_SolODE(3, 20);
    driver_get_contour(tau, 0, x1_min, x2_min, y,
                       p, m_contour_robust,
                       sol);

    for(i=0;i<sol->n_points;i++)
        printf("th=%7.3e\t R=%7.3e\t x1=%7.3e\t x2=%7.3e\n", sol->t[i]/2./M_PI -1,
                                                        sol->y[0][i],
                                                        sol->y[1][i],
                                                        sol->y[2][i]);
    free_SolODE(sol);
    // ------------------------------

    free_pLens(p);

    return 0;
}
