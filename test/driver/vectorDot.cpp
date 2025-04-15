#include <stdio.h>
#include <cuda.h>
#include <chrono>

int main() {
    const int N = 1 << 20;
    const int BLOCKS = 128;
    const int THREADS = 256;
    const int trials = 10;

    size_t size = N * sizeof(double);

    double* h_A = (double*)malloc(size);
    double* h_B = (double*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }
    cuInit(0);

    CUdevice device;
    cuDeviceGet(&device, 0);

    CUcontext context;
    cuCtxCreate(&context, 0, device);

    CUdeviceptr d_A, d_B, d_partial;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_B, size);
    cuMemAlloc(&d_partial, BLOCKS * sizeof(double));

    cuMemcpyHtoD(d_A, h_A, size);
    cuMemcpyHtoD(d_B, h_B, size);

    CUmodule module;
    cuModuleLoad(&module, "ptx/vector_dot.ptx");
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "_Z18dot_product_kernelPKdS0_Pdi");

    double total_time = 0.0;

    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::high_resolution_clock::now();

        void* args[] = {
            (void*)&d_A,
            (void*)&d_B,
            (void*)&d_partial,
            (void*)&N
        };

        cuLaunchKernel(
            kernel,
            BLOCKS, 1, 1,
            THREADS, 1, 1,
            0, 0, args, 0
        );

        cuCtxSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();

        double h_partial[BLOCKS];
        cuMemcpyDtoH(h_partial, d_partial, BLOCKS * sizeof(double));

        double result = 0.0;
        for (int j = 0; j < BLOCKS; ++j)
            result += h_partial[j];

        // printf("Trial %d: Time = %.6f s, Result = %.2f\n", t + 1, elapsed.count(), result);
    }

    printf("Average Time: %.6f s\n", total_time / trials);

    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_partial);
    free(h_A);
    free(h_B);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
