#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double **alloc(int n) {
    double **m = malloc(n * sizeof *m);
    for (int i = 0; i < n; i++) {
        m[i] = malloc(n * sizeof **m);
    }
    return m;
}

void freem(double **m, int n) { 
    for (int i = 0; i < n; i++) {
        free(m[i]); 
    }
    free(m); 
}

void naive(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int k = 0; k < n; k++) {
                s += A[i][k]*B[k][j];
            }
            C[i][j] = s;
        }
    }
}

int main(void) {
    int sizes[] = {32, 64, 128, 256, 512, 1024};
    int ns = sizeof(sizes) / sizeof(sizes[0]);

    printf("%6s %14s %10s\n", "n", "time (s)", "t(n)/t(n/2)");
    double prev = 0;

    for (int s = 0; s < ns; s++) {
        int n = sizes[s];

        double **A = alloc(n);
        double **B = alloc(n);
        double **C = alloc(n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) { 
                A[i][j]=1; 
                B[i][j]=1; 
                C[i][j]=0; 
            }
        }

        double t0 = now_sec();
        naive(A, B, C, n);
        double dt = now_sec() - t0;

        double ratio = prev > 0 ? dt/prev : 0;

        printf("%6d %14.6f %10s\n", n, dt, prev > 0 ? (char[16]){0} : "-");

        if (prev > 0) {
            printf(" "); 
        }

        prev = dt;

        freem(A,n); 
        freem(B,n); 
        freem(C,n);
    }
    return 0;
}