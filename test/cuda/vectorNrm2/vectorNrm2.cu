#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void nrm2_kernel(const double* A, double* partial_sum, int N) {
    __shared__ double cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    double temp = 0.0;
    while (tid < N) {
        temp += A[tid] * A[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (cacheIdx < stride) {
            cache[cacheIdx] += cache[cacheIdx + stride];
        }
        __syncthreads();
    }

    if (cacheIdx == 0)
        partial_sum[blockIdx.x] = cache[0];
}

int main() {
    const int N = 1 << 20;
    const int BLOCKS = 128;
    const int THREADS = 256;
    const int trials = 10;

    size_t size = N * sizeof(double);
    double *h_A = (double*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)rand() / RAND_MAX; 
    }

    double *d_A, *d_partial;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_partial, BLOCKS * sizeof(double));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    double total_time = 0.0;

    for (int i = 0; i < trials; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        nrm2_kernel<<<BLOCKS, THREADS>>>(d_A, d_partial, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms / 1000.0;

        double h_partial[BLOCKS];
        cudaMemcpy(h_partial, d_partial, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int j = 0; j < BLOCKS; ++j)
        {
            // printf("Partial sum %d: %.6f\n", j, h_partial[j]);
            sum += h_partial[j];
        }

        double nrm2 = sqrt(sum);
        printf("Trial %d: ||A||_2 = %.6f\n", i + 1, nrm2);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("Average Time: %.6f s\n", total_time / trials);

    cudaFree(d_A);
    cudaFree(d_partial);
    free(h_A);

    return 0;
}
