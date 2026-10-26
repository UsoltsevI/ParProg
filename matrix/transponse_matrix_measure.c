#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

void transpose(double **src, double **dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[j][i] = src[i][j];
        }
    }
}

void multiply_transposed(double **A, double **BT, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * BT[j][k];
            }
            C[i][j] = sum;
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

    printf("%6s %16s %10s\n", "n", "time (s)", "check");
    printf("------------------------------------\n");

    for (int s = 0; s < count; s++) {
        int n = sizes[s];

        double **A  = alloc_matrix(n);
        double **B  = alloc_matrix(n);
        double **BT = alloc_matrix(n);
        double **C  = alloc_matrix(n);

        fill(A, n, 1.0);
        fill(B, n, 1.0);
        fill(BT, n, 0.0);
        fill(C, n, 0.0);

        clock_t start = clock();
        transpose(B, BT, n);
        multiply_transposed(A, BT, C, n);
        clock_t end = clock();

        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        int ok = check_result(C, n);

        printf("%6d %16.6f %10s\n", n, elapsed, ok ? "OK" : "ERROR");

        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(BT, n);
        free_matrix(C, n);
    }

    return 0;
}