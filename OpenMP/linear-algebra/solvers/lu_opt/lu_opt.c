#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j;

 #pragma omp parallel for private(i, j) shared(A) collapse(2)
 //usato collapse per portare i due cicli su un unico livello di parallelismo e definito lo scope delle variabili
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j;

    #pragma omp parallel for private(i, j) shared(A) collapse(2)
  //stessa roba del ciclo sopra
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
  //  #pragma omp target data map(A[:_PB_N][:_PB_N])
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j, k;

    //#pragma omp parallel for
     // #pragma omp target teams distribute \
parallel for
    for (k = 0; k < _PB_N; k++)
    {
        #pragma omp distribute
        for (j = k + 1; j < _PB_N; j++)
            A[k][j] /= A[k][k];
        #pragma omp distribute //simd collapse(2)
        for (i = k + 1; i < _PB_N; i++)
            for (j = k + 1; j < _PB_N; j++)
                A[i][j] -= A[i][k] * A[k][j];
    }
}


int main(int argc, char **argv)
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
     #pragma omp target data map(A[:_PB_N][:_PB_N])

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
