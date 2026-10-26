#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 512
#define BASE 64

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

void multiply_naive(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void add_sub(double **A, double **B, double **C, int h, int sign) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            C[i][j] = A[i][j] + sign * B[i][j];
        }
    }
}

void strassen(double **A, double **B, double **C, int n) {
    if (n <= BASE) {
        multiply_naive(A, B, C, n);
        return;
    }

    int h = n / 2;

    double **A11 = alloc_matrix(h);
    double **A12 = alloc_matrix(h);
    double **A21 = alloc_matrix(h);
    double **A22 = alloc_matrix(h);
    double **B11 = alloc_matrix(h);
    double **B12 = alloc_matrix(h);
    double **B21 = alloc_matrix(h);
    double **B22 = alloc_matrix(h);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + h];
            A21[i][j] = A[i + h][j];
            A22[i][j] = A[i + h][j + h];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + h];
            B21[i][j] = B[i + h][j];
            B22[i][j] = B[i + h][j + h];
        }
    }

    double **M1 = alloc_matrix(h);
    double **M2 = alloc_matrix(h);
    double **M3 = alloc_matrix(h);
    double **M4 = alloc_matrix(h);
    double **M5 = alloc_matrix(h);
    double **M6 = alloc_matrix(h);
    double **M7 = alloc_matrix(h);
    double **T1 = alloc_matrix(h);
    double **T2 = alloc_matrix(h);

    add_sub(A11, A22, T1, h, +1);
    add_sub(B11, B22, T2, h, +1);
    strassen(T1, T2, M1, h);

    add_sub(A21, A22, T1, h, +1);
    strassen(T1, B11, M2, h);

    add_sub(B12, B22, T2, h, -1);
    strassen(A11, T2, M3, h);

    add_sub(B21, B11, T2, h, -1);
    strassen(A22, T2, M4, h);

    add_sub(A11, A12, T1, h, +1);
    strassen(T1, B22, M5, h);

    add_sub(A21, A11, T1, h, -1);
    add_sub(B11, B12, T2, h, +1);
    strassen(T1, T2, M6, h);

    add_sub(A12, A22, T1, h, -1);
    add_sub(B21, B22, T2, h, +1);
    strassen(T1, T2, M7, h);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            C[i][j]         = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + h]     = M3[i][j] + M5[i][j];
            C[i + h][j]     = M2[i][j] + M4[i][j];
            C[i + h][j + h] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }

    free_matrix(A11, h); free_matrix(A12, h);
    free_matrix(A21, h); free_matrix(A22, h);
    free_matrix(B11, h); free_matrix(B12, h);
    free_matrix(B21, h); free_matrix(B22, h);
    free_matrix(M1, h); free_matrix(M2, h); free_matrix(M3, h);
    free_matrix(M4, h); free_matrix(M5, h); free_matrix(M6, h);
    free_matrix(M7, h);
    free_matrix(T1, h); free_matrix(T2, h);
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

    printf("Strassen (base = %d)\n\n", BASE);
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
        strassen(A, B, C, n);
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