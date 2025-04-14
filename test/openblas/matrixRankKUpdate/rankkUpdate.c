// perform symmetric rank-k update test with OpenBLAS cblas_dsyrk function
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#define N (5 * 4 * 32)  // Size of square matrix C
#define K (5 * 2 * 32)  // Number of columns in A (A is NxK)
#define REPEAT 500

double get_time_in_seconds() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

void fill_random(double* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    printf("Symmetric rank-k update: C = alpha*A*A' + beta*C\n");
    printf("Dimensions: A(%dx%d), C(%dx%d)\n", N, K, N, N);

    double* A = (double*)malloc(sizeof(double) * N * K);
    double* C = (double*)malloc(sizeof(double) * N * N);

    if (!A || !C) {
        fprintf(stderr, "allocation error\n");
        return 1;
    }

    srand(time(NULL));
    fill_random(A, N * K);
    fill_random(C, N * N);  // Initialize C with random values

    double alpha = 1.0, beta = 1.0;  // beta=1 to accumulate results

    double start = get_time_in_seconds();

    for (int i = 0; i < REPEAT; ++i) {
        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                    N, K, alpha,
                    A, K,
                    beta,
                    C, N);
    }

    double end = get_time_in_seconds();
    double total_time = end - start;
    double avg_time = total_time / REPEAT;

    // FLOP count for syrk: N*N*K (since it's roughly half of gemm)
    double flops = N * N * K;
    double gflops = (flops * REPEAT) / (total_time * 1e9);

    printf("Average time: %f seconds\n", avg_time);
    printf("Total time: %f seconds\n", total_time);
    printf("GFLOPS: %f\n", gflops);

    free(A);
    free(C);
    return 0;
}