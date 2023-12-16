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

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

static inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/* Array initialization. */
static void init_array(int n, DATA_TYPE* A)
{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[(i * n) + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE* A)
{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            fprintf(stderr, DATA_PRINTF_MODIFIER, A[(i * n) + j]);
            if ((i * n + j) % 20 == 0)
                fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

__global__ void gpuFirstLoop(int n, DATA_TYPE* A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + k + 1;

    if (j < n)
    {
        DATA_TYPE A_k_k = A[(k * n) + k];

        A[(k * n) + j] /= A_k_k;
    }
}

__global__ void gpuSecondLoop(int n, DATA_TYPE* A, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + k + 1;

    if (i < n && j < n)
    {
        DATA_TYPE A_i_k = A[(i * n) + k];
        DATA_TYPE A_k_j = A[(k * n) + j];

        A[(i * n) + j] -= (A_i_k * A_k_j);
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#ifdef CUDA_TIME
static float kernel_lu(int n, DATA_TYPE* A)
#else
static void kernel_lu(int n, DATA_TYPE* A)
#endif
{
    DATA_TYPE* d_A;
    dim3 secondLoopBlocKsize(BLOCK_SIZE, BLOCK_SIZE);

#ifdef CUDA_TIME
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));
#endif

    gpuErrchk(cudaMalloc(&d_A, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMemcpy(d_A, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));

    for (int k = 0; k < _PB_N - 1; k++)
    {
        dim3 firstLoopGridSize((n - k - 2 + BLOCK_SIZE) / BLOCK_SIZE);
        gpuFirstLoop<<<firstLoopGridSize, BLOCK_SIZE>>>(n, d_A, k);
        gpuErrchk(cudaPeekAtLastError());

        dim3 secondLoopGridSize((n - k - 2 + BLOCK_SIZE) / BLOCK_SIZE, (n - k - 2 + BLOCK_SIZE) / BLOCK_SIZE);
        gpuSecondLoop<<<secondLoopGridSize, secondLoopBlocKsize>>>(n, d_A, k);
        gpuErrchk(cudaPeekAtLastError());
    }

    gpuErrchk(cudaMemcpy(A, d_A, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_A));

#ifdef CUDA_TIME
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;

    gpuErrchk(cudaEventElapsedTime(&elapsed, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return (elapsed / 1000);
#endif
}

int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE* A = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * n * n);

    /* Initialize array(s). */
    init_array(n, A);

    /* Start timer. */
    polybench_start_instruments;

/* Run kernel. */
#ifdef CUDA_TIME
    float elapsed = kernel_lu(n, A);
#else
    kernel_lu(n, A);
#endif

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    polybench_prevent_dce(print_array(n, A));

    /* Be clean. */
    free(A);

#ifdef CUDA_TIME
    printf("%.6f\n", elapsed);
#endif

    return 0;
}
