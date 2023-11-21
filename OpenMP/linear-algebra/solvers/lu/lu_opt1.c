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

    for (k = 0; k < _PB_N; k++)
    {
        c1 = A[k][k];

#pragma omp parallel for schedule(static) private(i, j)
        for (j = k + 1; j < _PB_N; j++)
            A[k][j] /= c1;

#pragma omp parallel for schedule(static) private(i, j)
        for (i = k + 1; i < _PB_N; i++)
        {
            c2 = A[i][k];
            for (j = k + 1; j < _PB_N; j++)
                A[i][j] -= (c2 * A[k][j]);
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
    // #pragma omp target data map(tofrom:A[:_PB_N][:_PB_N])
    kernel_lu(n, POLYBENCH_ARRAY(A));
    // #pragma omp target data exit mapfrom:A[:_PB_N][:_PB_N])

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