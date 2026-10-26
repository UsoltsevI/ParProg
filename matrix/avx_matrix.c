#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define N 512

double **alloc_matrix(int n) {
    double **m = (double **) malloc(n * sizeof(double *));

    if (!m) { 
        fprintf(stderr, "malloc failed\n"); 
        exit(1); 
    }

    for (int i = 0; i < n; i++) {
        m[i] = (double *) aligned_alloc(32, n * sizeof(double));
        if (!m[i]) { 
            fprintf(stderr, "aligned_alloc failed\n"); 
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

void transpose(double **src, double **dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[j][i] = src[i][j];
        }
    }
}

void multiply_avx(double **A, double **BT, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __m256d acc = _mm256_setzero_pd();
            int k = 0;

            for (; k + 4 <= n; k += 4) {
                __m256d a = _mm256_load_pd(&A[i][k]);
                __m256d b = _mm256_load_pd(&BT[j][k]);
                acc = _mm256_fmadd_pd(a, b, acc);
            }

            double tmp[4];
            _mm256_storeu_pd(tmp, acc);
            double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

            for (; k < n; k++) {
                sum += A[i][k] * BT[j][k];
            }

            C[i][j] = sum;
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

    printf("AVX (4 x double per instruction)\n\n");
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
        multiply_avx(A, BT, C, n);
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