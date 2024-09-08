#include <stdio.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int i;
    int n = omp_get_max_threads();

	printf("OMP test (num threads = %d)\n", n);
	omp_set_num_threads(n);

    #pragma omp parallel for
	for(i=0;i<6;i++)
    {
        n = omp_get_thread_num();
		//printf("i=%d    thread %d\n", i, n);
    }

    return 0;
}
