#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// SYRK Kernel: C = alpha * A * A^T + beta * C (only upper triangle)
__global__ void syrk_kernel(float* A, float* C, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col of C

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * A[col * K + i];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void init_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int N = 5 * 4 * 32;  // output matrix C size: N x N
    int K = 5 * 2 * 32;  // A: N x K
    int num_runs = 10;

    size_t size_A = N * K * sizeof(float);
    size_t size_C = N * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_C = (float*)malloc(size_C);

    init_matrix(h_A, N * K);
    init_matrix(h_C, N * N);

    float *d_A, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    float alpha = 1.0f;
    float beta  = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_time = 0.0f;
    float best_time = 1e10;

    for (int run = 0; run < num_runs; ++run) {
        CHECK_CUDA(cudaEventRecord(start));
        syrk_kernel<<<grid, block>>>(d_A, d_C, N, K, alpha, beta);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_time += ms;
        if (ms < best_time) best_time = ms;
        // printf("Run %2d: %.3f ms\n", run + 1, ms);
    }

    double avg_time = total_time / num_runs;
    double ops = 2.0 * N * N * K / 2.0;
    double avg_gflops = ops / (avg_time / 1000.0) / 1e9;
    double best_gflops = ops / (best_time / 1000.0) / 1e9;

    printf("\n[Summary for %dx%d SYRK (AAᵗ) update]\n", N, K);
    printf("Average Time: %.3f ms | Average GFLOPS: %.2f\n", avg_time, avg_gflops);
    printf("Best Time: %.3f ms | Best GFLOPS: %.2f\n", best_time, best_gflops);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_C);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
