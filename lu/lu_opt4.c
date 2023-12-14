#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
// #define POLYBENCH_DUMP_ARRAYS
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"

#pragma omp declare target
#define SM 64

#define NTHRDS7 (1 << 0x7) /* 2^{7}  */
#define NTHRDS8 (1 << 0x8) /* 2^{8}  */
#define NTHRDS9 (1 << 0x9) /* 2^{9}  */

#define LTEAMSD (1 << 0xD) /* 2^{13} */
#define LTEAMSE (1 << 0xE) /* 2^{14} */
#define LTEAMSF (1 << 0xF) /* 2^{15} */
#define LTEAMSG (1 << 020) /* 2^{16} */
#pragma omp end declare target

/* Array initialization. */
static void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
            if ((i * n + j) % 20 == 0)
                fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j, k;
    DATA_TYPE c1, c2;

    #pragma omp target map(to: _PB_N) map(tofrom: A[:_PB_N][:_PB_N])
    for (k = 0; k < _PB_N - 1; k++)
    {
        c1 = A[k][k];

	#pragma omp teams shared(A) firstprivate(_PB_N, k, c1)
	{

            #pragma omp distribute parallel for private(i, j)
            for (j = k + 1; j < _PB_N; j++)
                A[k][j] /= c1;

            #pragma omp distribute parallel for private(i, j, c2)
            for (i = k + 1; i < _PB_N; i++)
            {
                c2 = A[i][k];

                for (j = k + 1; j < _PB_N; j++)
                    A[i][j] -= (c2 * A[k][j]);
            }
	}
    }
}

int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

    /* Initialize array(s). */
    init_array(n, POLYBENCH_ARRAY(A));

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_lu(n, POLYBENCH_ARRAY(A));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A);

    return 0;
}
