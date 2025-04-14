// perform vector sum using OpenBLAS
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
    for (int i = 0; i < N; ++i) {
        A[i] = (double)rand() / RAND_MAX;
    }
    double total_time = 0.0;
    double sum = 0.0;
    for (int i = 0; i < trials; ++i) {
        double start = get_time_in_seconds();
        sum = cblas_dasum(N, A, 1);
        double end = get_time_in_seconds();
        double elapsed = end - start;
        total_time += elapsed;
        // printf("Trial %d: Time = %.6f s, Norm = %.2f\n", i + 1, elapsed, norm);
    }
    printf("Average Time: %.6f s\n", total_time / trials);
    printf("Sum: %.2f\n", sum);
    free(A);
    return 0;
}