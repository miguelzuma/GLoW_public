#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_version.h>
#include <gsl/gsl_errno.h>

int main(int argc, char *argv[])
{
    printf("GSL version = %s\n", gsl_version);
    return 0;
}
