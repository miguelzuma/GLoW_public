#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

int main(int argc, char *argv[])
{
    double complex resultII;

    resultII = I*I;
    printf("i*i = %f\n", creal(resultII));

    return 0;
}
