
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time_in_seconds() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

void vector_add(const double* A, const double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const int trials = 10;

    double *A = (double*)malloc(N * sizeof(double));
    double *B = (double*)malloc(N * sizeof(double));
    double *C = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double total_time = 0.0;

    for (int i = 0; i < trials; ++i) {
        double start = get_time_in_seconds();

        vector_add(A, B, C, N);

        double end = get_time_in_seconds();
        double elapsed = end - start;
        total_time += elapsed;

        // printf("Trial %d: Time = %.6f s, C[0] = %.2f\n", i + 1, elapsed, C[0]);
    }

    printf("Average Time: %.6f s\n", total_time / trials);

    free(A);
    free(B);
    free(C);
    return 0;
}
