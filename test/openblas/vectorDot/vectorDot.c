// dot_openblas.c
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>

double get_time_in_seconds() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main() {
    const int N = 1 << 20;  
    const int trials = 10;

    double *A = (double*)malloc(N * sizeof(double));
    double *B = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double total_time = 0.0;
    double result = 0.0;

    for (int i = 0; i < trials; ++i) {
        double start = get_time_in_seconds();
        result = cblas_ddot(N, A, 1, B, 1);
        double end = get_time_in_seconds();

        double elapsed = end - start;
        total_time += elapsed;

        // printf("Trial %d: Time = %.6f s, Result = %.2f\n", i + 1, elapsed, result);
    }

    printf("Average Time: %.6f s\n", total_time / trials);

    free(A);
    free(B);
    return 0;
}
