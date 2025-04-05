#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#define M (5 * 4 * 32) 
#define K (5 * 2 * 32) 
#define N (5 * 2 * 32) 
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
    printf("Dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);

    double* A = (double*)malloc(sizeof(double) * M * K);
    double* B = (double*)malloc(sizeof(double) * K * N);
    double* C = (double*)malloc(sizeof(double) * M * N);

    if (!A || !B || !C) {
        fprintf(stderr, "allocation error\n");
        return 1;
    }

    srand(time(NULL));
    fill_random(A, M * K);
    fill_random(B, K * N);

    double alpha = 1.0, beta = 0.0;

    double start = get_time_in_seconds();

    for (int i = 0; i < REPEAT; ++i) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    A, K,
                    B, N,
                    beta,
                    C, N);
    }

    double end = get_time_in_seconds();
    double total_time = end - start;
    double avg_time = total_time / REPEAT;

    double flops = 2.0 * M * N * K;
    double gflops = (flops * REPEAT) / (total_time * 1e9);

    printf("Average time: %f seconds\n", avg_time);
    printf("Total time: %f seconds\n", total_time);
    printf("GFLOPS: %f\n", gflops);

    free(A);
    free(B);
    free(C);
    return 0;
}
