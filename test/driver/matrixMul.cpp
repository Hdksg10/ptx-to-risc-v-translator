#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        CUresult err = call;                                                 \
        if (err != CUDA_SUCCESS) {                                           \
            const char* errStr;                                              \
            cuGetErrorString(err, &errStr);                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)


void init_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

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

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size_A));
    CHECK_CUDA(cuMemAlloc(&d_B, size_B));
    CHECK_CUDA(cuMemAlloc(&d_C, size_C));

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size_A));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size_B));

    CUmodule module;
    CUfunction kernel;
    CHECK_CUDA(cuModuleLoad(&module, "ptx/matrix_mul.ptx"));
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "_Z13matmul_kernelPfS_S_iii"));

    int block_x = 16, block_y = 16;
    int grid_x = (N + block_x - 1) / block_x;
    int grid_y = (M + block_y - 1) / block_y;

    void* args[] = { &d_A, &d_B, &d_C, &M, &N, &K };

    double total_time = 0.0;
    double best_time = 1e10;

    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        CHECK_CUDA(cuLaunchKernel(kernel, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, args, 0));
        CHECK_CUDA(cuCtxSynchronize());
        auto stop = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(stop - start).count();
        total_time += ms;
        if (ms < best_time) best_time = ms;
    }

    double avg_time = total_time / num_runs;
    double ops = 2.0 * M * N * K;
    double avg_gflops = ops / (avg_time / 1000.0) / 1e9;
    double best_gflops = ops / (best_time / 1000.0) / 1e9;

    printf("\n[Summary for %dx%dx%d matrix multiplication]\n", M, K, N);
    printf("Average Time: %.3f ms | Average GFLOPS: %.2f\n", avg_time, avg_gflops);

    CHECK_CUDA(cuMemFree(d_A));
    CHECK_CUDA(cuMemFree(d_B));
    CHECK_CUDA(cuMemFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(ctx));

    return 0;
}
