#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "lu.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

__global__ void lu_kernel(int n, DATA_TYPE *A, int k)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    extern __shared__ DATA_TYPE shared_A[];

    if (row < n && col < n)
    {
        DATA_TYPE c1, c2;

        // Copy data from global memory to shared memory
        shared_A[row * n + col] = A[row * n + col];

        __syncthreads();

        if (col == k)
        {
            c1 = shared_A[k * n + k];

            // Normalizzazione della colonna corrente
            for (int j = k + 1; j < n; j++)
                shared_A[k * n + j] /= c1;
        }

        __syncthreads();

        if (row > k && col > k)
        {
            c2 = shared_A[row * n + k];

            // Eliminazione gaussiana nella riga corrente
            for (int j = k + 1; j < n; j++)
                shared_A[row * n + j] -= (c2 * shared_A[k * n + j]);
        }

        __syncthreads();

        // Copy data back to global memory
        A[row * n + col] = shared_A[row * n + col];
    }
}

static void kernel_lu(int n, DATA_TYPE *A)
{
    DATA_TYPE *d_A;
    cudaMalloc((void **)&d_A, sizeof(DATA_TYPE) * n * n);
    cudaMemcpy(d_A, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks((n + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (n + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    size_t shared_size = sizeof(DATA_TYPE) * BLOCK_SIZE_Y * n;
    lu_kernel<<<numBlocks, threadsPerBlock, shared_size>>>(n, d_A, 0);

    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    cudaMemcpy(A, d_A, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
}

static void init_array(int n, DATA_TYPE *A)
{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[i * n + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
}

static void print_array(int n, DATA_TYPE *A)
{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            fprintf(stderr, "%f ", A[i * n + j]);
            if ((i * n + j) % 20 == 0)
                fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

int main()
{
    int n = N; 
    DATA_TYPE *A = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * n * n);

    init_array(n, A);

    // Start timer.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run kernel.
    kernel_lu(n, A);

    // Stop and print timer.
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f ms\n", milliseconds);

    // Print result for verification or analysis
    // Comment this out for large problem sizes as it might flood the console
    print_array(n, A);

    free(A);

    return 0;
}
