#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "special_lib.h"

// =================================================================

int main(int argc, char *argv[])
{
    
    int i, n;
    double M, nu;
    double z_min, z_max, dz, z;
    
    n = 500;
    z_min = 1;
    z_max = 15;
    dz = (z_max-z_min)/n;
    
    nu = -2.1;
    
    for(i=0;i<n;i++)
    {
        z = z_min+i*dz;
        
        M = Mtilde_Struve(z, nu, 1e-10);
        printf("   M = %e     z = %e    (series)\n", M, z);
        
        M = Mtilde_Struve_Asymptotic(z, nu, 1e-10);
        printf("   M = %e     z = %e    (asymp)\n", M, z);
        printf("\n");
    }
    
    return 0;
}
