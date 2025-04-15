#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define CUDA_CHECK(f, msg) \
    if ((r = f) != CUDA_SUCCESS) { \
        fprintf(stderr, "%s: %d\n", msg, r); \
        return -1; \
    }

int main() {
    CUresult r;
    const int N = 1 << 20;
    const int trials = 10;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t size = N * sizeof(double);

    // Host memory
    double *h_A = (double*)malloc(size);
    double *h_B = (double*)malloc(size);
    double *h_C = (double*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }

    cuInit(0);

    CUdevice device;
    cuDeviceGet(&device, 0);

    CUcontext context;
    cuCtxCreate(&context, 0, device);

    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_C, size);

    cuMemcpyHtoD(d_A, h_A, size);
    cuMemcpyHtoD(d_B, h_B, size);

    CUmodule module;
    CUDA_CHECK(cuModuleLoad(&module, "ptx/vector_add.ptx"), "Failed to load module: ");
    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "_Z17vector_add_kernelPKdS0_Pdi"), "Failed to get function: ");
    double total_time = 0.0;

    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::high_resolution_clock::now();

        void* args[] = { &d_A, &d_B, &d_C, (void*)&N };

        cuLaunchKernel(
            kernel,
            blocksPerGrid, 1, 1,
            threadsPerBlock, 1, 1,
            0, 0, args, 0
        );

        cuCtxSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }

    cuMemcpyDtoH(h_C, d_C, size);

    // printf("Sample Result C[0] = %.2f\n", h_C[0]);
    printf("Average Time: %.6f s\n", total_time / trials);

    free(h_A);
    free(h_B);
    free(h_C);
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}