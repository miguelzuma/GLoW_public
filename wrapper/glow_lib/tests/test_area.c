#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"
#include "area_lib.h"

#define N_t 2000

// =================================================================

int main(int argc, char *argv[])
{
    double psi0, y;
    pNamedLens *p;
    Lens Psi;
    
    int i;
    double R, tmin, tau_max;
    double tau_grid[N_t], It_grid[N_t];
    pAreaIntegral parea;
    
    // ------------------------------------------------------
    if(argc==2)
        y = atof(argv[1]);
    else
    {
        printf("Usage: %s y\n", argv[0]);
        return 1;
    }
    
    psi0 = 1;
    
    // Different lenses only appear here
    p = create_pLens_SIS(psi0);
    //~ p = create_pLens_CIS(psi0, 0.05);
    //~ p = create_pLens_NFW(psi0, 0.01);
    Psi = init_lens(p);
    
    // ------------------------------------------------------
    printf("\n");
    
    tmin = 0.5*y*y - 0.5*(psi0+y)*(psi0+y);
    tau_max = 2000;
    
    R = find_Rmax_AreaIntegral(y, tau_max, tmin, &Psi);
    
    printf("R=%e    dR=%e\n", R, R-y-sqrt(2*tau_max));
    
    // =================================================================
    
    parea.n_rho = 20000;
    parea.n_theta = 2000;
    parea.y = y;
    parea.tau_max = 100;
    parea.pNLens = p;
    
    integrate_AreaIntegral(&tmin, tau_grid, It_grid, N_t, &parea);
    
    for(i=0;i<N_t;i++)
        printf("tau=%e    It=%e\n", tau_grid[i], It_grid[i]);
    
    return 0;
}
