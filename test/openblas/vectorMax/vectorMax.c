// perfom max idx using OpenBLAS cblas_idamax function
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

double get_time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    const int N = 1 << 20;  
    const int trials = 10;
    double *A = (double*)malloc(N * sizeof(double));
    if (A == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        A[i] = (double)rand() / RAND_MAX;
    }
    double total_time = 0.0;
    int max_idx = 0;
    for (int i = 0; i < trials; ++i) {
        double start = get_time_in_seconds();
        max_idx = cblas_idamax(N, A, 1) - 1; // cblas_idamax returns 1-based index
        double end = get_time_in_seconds();
        double elapsed = end - start;
        total_time += elapsed;
        // printf("Trial %d: Time = %.6f s, Max Index = %d\n", i + 1, elapsed, max_idx);
        }
    printf("Average Time: %.6f s\n", total_time / trials);
    printf("Max Index: %d, Max Value: %.2f\n", max_idx, A[max_idx]);
    free(A);
    return 0;
}