#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024
#define BS 64

void multiply_blocked(double **A, double **B, double **C, int n) {
    for (int ii = 0; ii < n; ii += BS) {
        int i_max = (ii + BS < n) ? ii + BS : n;
        for (int jj = 0; jj < n; jj += BS) {
            int j_max = (jj + BS < n) ? jj + BS : n;
            for (int kk = 0; kk < n; kk += BS) {
                int k_max = (kk + BS < n) ? kk + BS : n;
                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        double sum = C[i][j];
                        for (int k = kk; k < k_max; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

double **alloc_matrix(int n) {
    double **m = (double **) malloc(n * sizeof(double *));

    if (!m) { 
        fprintf(stderr, "malloc failed\n"); 
        exit(1); 
    }

    for (int i = 0; i < n; i++) {
        m[i] = (double *) malloc(n * sizeof(double));
        if (!m[i]) { 
            fprintf(stderr, "malloc failed\n"); 
            exit(1); 
        }
    }

    return m;
}

void free_matrix(double **m, int n) {
    for (int i = 0; i < n; i++) {
        free(m[i]);
    }
    free(m);
}

void fill(double **m, int n, double val) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = val;
        }
    }
}

int check_result(double **C, int n) {
    int ok = 1;

    for (int i = 0; i < n && ok; i++) {
        for (int j = 0; j < n && ok; j++) {
            if (C[i][j] != (double)n) {
                ok = 0;
            }
        }
    }

    return ok;
}

int main(void) {
    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int count = (int) (sizeof(sizes) / sizeof(sizes[0]));

    printf("Block size: %d x %d\n\n", BS, BS);
    printf("%6s %16s %10s\n", "n", "time (s)", "check");
    printf("------------------------------------\n");

    for (int s = 0; s < count; s++) {
        int n = sizes[s];

        double **A = alloc_matrix(n);
        double **B = alloc_matrix(n);
        double **C = alloc_matrix(n);

        fill(A, n, 1.0);
        fill(B, n, 1.0);
        fill(C, n, 0.0);

        clock_t start = clock();
        multiply_blocked(A, B, C, n);
        clock_t end = clock();

        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        int ok = check_result(C, n);

        printf("%6d %16.6f %10s\n", n, elapsed, ok ? "OK" : "ERROR");

        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(C, n);
    }

    return 0;
}