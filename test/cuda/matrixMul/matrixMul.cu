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

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Col of C

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void init_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int M = 5 * 4 * 32, N = 5 * 2 * 32, K = 5 * 2 * 32; 
    int num_runs = 10; 

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_time = 0.0f;
    float best_time = 1e10;

    for (int run = 0; run < num_runs; ++run) {
        CHECK_CUDA(cudaEventRecord(start));
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_time += ms;
        if (ms < best_time) best_time = ms;
        // printf("Run %2d: %.3f ms\n", run + 1, ms);
    }

    double avg_time = total_time / num_runs;
    double ops = 2.0 * M * N * K;
    double avg_gflops = ops / (avg_time / 1000.0) / 1e9;
    double best_gflops = ops / (best_time / 1000.0) / 1e9;

    printf("\n[Summary for %dx%dx%d matrix multiplication]\n", M, K, N);
    printf("Average Time: %.3f ms | Average GFLOPS: %.2f\n", avg_time, avg_gflops);

    // clean-up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}