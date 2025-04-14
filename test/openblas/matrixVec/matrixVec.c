#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>

double get_time_in_seconds() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

void matrix_vector_multiply(double* A, double* x, double* y, int N) {
    // y = A * x
    // A is NxN matrix (column-major order)
    // x is N vector
    // y is N vector
    cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1.0, A, N, x, 1, 0.0, y, 1);
}

int main() {
    const int N = 1 << 10; 
    const int trials = 10;

    double *A = (double*)malloc(N * N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    double *y = (double*)malloc(N * sizeof(double));

    // Initialize matrix A and vector x
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0;  // Fill matrix with 1s
    }
    for (int i = 0; i < N; ++i) {
        x[i] = 2.0;  // Fill vector with 2s
    }

    double total_time = 0.0;

    for (int i = 0; i < trials; ++i) {
        double start = get_time_in_seconds();

        matrix_vector_multiply(A, x, y, N);

        double end = get_time_in_seconds();
        double elapsed = end - start;
        total_time += elapsed;

        // printf("Trial %d: Time = %.6f s, y[0] = %.2f\n", i + 1, elapsed, y[0]);
    }

    printf("Average Time: %.6f s\n", total_time / trials);

    free(A);
    free(x);
    free(y);
    return 0;
}