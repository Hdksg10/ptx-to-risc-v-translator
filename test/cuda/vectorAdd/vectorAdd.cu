
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const double* A, const double* B, double* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 20;  // 1M
    const int trials = 10;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t size = N * sizeof(double);
    double *h_A = (double*)malloc(size);
    double *h_B = (double*)malloc(size);
    double *h_C = (double*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    float total_time = 0.0;

    for (int i = 0; i < trials; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms / 1000.0;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // printf("Sample Result C[0] = %.2f\n", h_C[0]);
    printf("Average Time: %.6f s\n", total_time / trials);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
