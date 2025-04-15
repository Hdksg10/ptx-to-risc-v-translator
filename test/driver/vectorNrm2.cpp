#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <chrono>

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
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);
    CUdeviceptr d_A, d_partial;
    cuMemAlloc(&d_A, size);
    cuMemAlloc(&d_partial, BLOCKS * sizeof(double));
    cuMemcpyHtoD(d_A, h_A, size);
    cuModuleLoad(&cuModule, "ptx/vector_nrm2.ptx");
    cuModuleGetFunction(&cuFunction, cuModule, "_Z11nrm2_kernelPKdPdi");
    double total_time = 0.0;
    void* args[] = { &d_A, &d_partial, (void*)&N };
    for (int i = 0; i < trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cuLaunchKernel(cuFunction, BLOCKS, 1, 1, THREADS, 1, 1, 0, 0, args, 0);
        cuCtxSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = stop - start;
        total_time += elapsed.count();
        double h_partial[BLOCKS];
        cuMemcpyDtoH(h_partial, d_partial, BLOCKS * sizeof(double));
        double sum = 0.0;
        for (int j = 0; j < BLOCKS; ++j)
            sum += h_partial[j];
        double nrm2 = sqrt(sum);
        printf("Trial %d: ||A||_2 = %.6f\n", i + 1, nrm2);
    }
    printf("Average Time: %.6f s\n", total_time / trials);
    cuMemFree(d_A);
    cuMemFree(d_partial);
    cuCtxDestroy(cuContext);
    free(h_A);
    return 0;
}
