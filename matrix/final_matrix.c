#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define N 512
#define BS 64


double **alloc_matrix(int n) {
    double **m = (double **) malloc(n * sizeof(double *));

    if (!m) { 
        fprintf(stderr, "malloc failed\n"); 
        exit(1); 
    }

    for (int i = 0; i < n; i++) {
        m[i] = (double *) aligned_alloc(64, ((n * sizeof(double) + 63) / 64) * 64);
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

void transpose_blocked(double **src, double **dst, int n) {
    #pragma omp parallel for schedule(static)
    for (int bi = 0; bi < n; bi += BS) {
        int i_max = (bi + BS < n) ? bi + BS : n;
        for (int bj = 0; bj < n; bj += BS) {
            int j_max = (bj + BS < n) ? bj + BS : n;
            for (int i = bi; i < i_max; i++) {
                for (int j = bj; j < j_max; j++) {
                    dst[j][i] = src[i][j];
                }
            }
        }
    }
}

void multiply_avx_blocked(double **A, double **BT, double **C, int n) {
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < n; ii += BS) {
        int i_max = (ii + BS < n) ? ii + BS : n;
        for (int jj = 0; jj < n; jj += BS) {
            int j_max = (jj + BS < n) ? jj + BS : n;
            for (int i = ii; i < i_max; i++) {
                for (int j = jj; j < j_max; j++) {
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

    printf("Final: blocked + transpose + AVX/FMA + OpenMP\n");
    printf("BS = %d, threads = %d\n\n", BS, omp_get_max_threads());
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

        double start = omp_get_wtime();
        transpose_blocked(B, BT, n);
        multiply_avx_blocked(A, BT, C, n);
        double end = omp_get_wtime();

        int ok = check_result(C, n);

        printf("%6d %16.6f %10s\n", n, end - start, ok ? "OK" : "ERROR");

        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(BT, n);
        free_matrix(C, n);
    }

    return 0;
}