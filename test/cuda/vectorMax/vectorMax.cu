#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void idamax_kernel(const double* A, int* partial_idx, int N) {
    __shared__ double cache_val[256];
    __shared__ int cache_idx[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    double max_val = -1.0;
    int max_idx = -1;

    while (tid < N) {
        double val = fabs(A[tid]);
        if (val > max_val) {
            max_val = val;
            max_idx = tid;
        }
        tid += blockDim.x * gridDim.x;
    }

    cache_val[cacheIdx] = max_val;
    cache_idx[cacheIdx] = max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (cacheIdx < stride) {
            if (cache_val[cacheIdx] < cache_val[cacheIdx + stride]) {
                cache_val[cacheIdx] = cache_val[cacheIdx + stride];
                cache_idx[cacheIdx] = cache_idx[cacheIdx + stride];
            }
        }
        __syncthreads();
    }

    if (cacheIdx == 0)
        partial_idx[blockIdx.x] = cache_idx[0];
}

int main() {
    const int N = 1 << 20;
    const int BLOCKS = 128;
    const int THREADS = 256;
    const int trials = 10;

    size_t size = N * sizeof(double);
    double *h_A = (double*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)(rand() % 1000 - 500);
    }
    h_A[123456] = 9999.0;

    double *d_A;
    int *d_partial_idx;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_partial_idx, BLOCKS * sizeof(int));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    double total_time = 0.0;

    for (int i = 0; i < trials; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        idamax_kernel<<<BLOCKS, THREADS>>>(d_A, d_partial_idx, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms / 1000.0;

        int h_partial_idx[BLOCKS];
        cudaMemcpy(h_partial_idx, d_partial_idx, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

        double max_val = -1.0;
        int max_idx = -1;
        for (int j = 0; j < BLOCKS; ++j) {
            double val = fabs(h_A[h_partial_idx[j]]);
            if (val > max_val) {
                max_val = val;
                max_idx = h_partial_idx[j];
            }
        }

        printf("Trial %d: Max abs value = %.1f at index %d\n", i + 1, max_val, max_idx);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("Average Time: %.6f s\n", total_time / trials);

    cudaFree(d_A);
    cudaFree(d_partial_idx);
    free(h_A);

    return 0;
}
