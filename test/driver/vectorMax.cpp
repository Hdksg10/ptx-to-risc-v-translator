#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <chrono>
#include <cstdlib>

int main() {
    const int N = 1 << 20;
    const int BLOCKS = 128;
    const int THREADS = 256;
    const int trials = 10;

    size_t size = N * sizeof(double);
    double* h_A = (double*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)(rand() % 1000 - 500);
    }
    h_A[123456] = 9999.0;

    cuInit(0);

    CUdevice device;
    cuDeviceGet(&device, 0);

    CUcontext context;
    cuCtxCreate(&context, 0, device);

    CUdeviceptr d_A, d_partial_idx;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_partial_idx, BLOCKS * sizeof(int));

    cuMemcpyHtoD(d_A, h_A, size);

    CUmodule module;
    cuModuleLoad(&module, "ptx/vector_max.ptx");
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "_Z13idamax_kernelPKdPii");

    double total_time = 0.0;

    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::high_resolution_clock::now();

        void* args[] = {
            (void*)&d_A,
            (void*)&d_partial_idx,
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

        int h_partial_idx[BLOCKS];
        cuMemcpyDtoH(h_partial_idx, d_partial_idx, BLOCKS * sizeof(int));

        double max_val = -1.0;
        int max_idx = -1;
        for (int j = 0; j < BLOCKS; ++j) {
            double val = fabs(h_A[h_partial_idx[j]]);
            if (val > max_val) {
                max_val = val;
                max_idx = h_partial_idx[j];
            }
        }

        printf("Trial %d: Max abs value = %.1f at index %d\n", t + 1, max_val, max_idx);
    }

    printf("Average Time: %.6f s\n", total_time / trials);

    cuMemFree(d_A);
    cuMemFree(d_partial_idx);
    free(h_A);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
