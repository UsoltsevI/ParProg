#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define K_COEFF 0.001
#define M_COEFF 0.001

void free_fu(double** fu, int k) {
    for (int i = 0; i < k; i++) {
        free(fu[i]);
    }
    free(fu);
}

void out_like_csv(FILE* out, double** u, int K, int M) {
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M - 1; m++) {
            fprintf(out, "%.15f,", u[k][m]);
        }
        fprintf(out, "%.15f", u[k][M - 1]);
        fprintf(out, "\n");
    }
}

double** create_array(int K, int M) {
    double** ans = (double**) malloc(sizeof(double*) * K);

    for (int k = 0; k < K; k++) {
        ans[k] = (double*) malloc(sizeof(double) * M);
    }

    return ans;
}

double** get_exact_cosexp(int K, int M) {
    // решение при 
    // u(0, x) = exp(-x)
    // u(t, 0) = cos(pi * t)
    // f(t, x) = x + t
    double** ans = create_array(K, M);

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < k; m++) {
            ans[k][m] = cos(M_PI * (k * K_COEFF - m * M_COEFF)) + m * M_COEFF * k * K_COEFF;
        }
        for (int m = k; m < M; m++) {
            ans[k][m] = exp(- m * M_COEFF + k * K_COEFF) + m * M_COEFF * k * K_COEFF;
        }
    }

    return ans;
}

double** get_exact_sinsin(int K, int M) {
    // решение при 
    // u(0, x) = sin(2 * pi * x)
    // u(t, 0) = - sin(2 * pi * t)
    // f(t, x) = 0

    double** ans = create_array(K, M);

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            ans[k][m] = sin(2 * M_PI * (m * M_COEFF - k * K_COEFF));
        }
    }

    return ans;
}


double** get_exact_x2t2(int K, int M) {
    // решение при 
    // u(0, x) = x^2
    // u(t, 0) = t^2
    // f(t, x) = 4x + 4t
    double** ans = create_array(K, M);

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            ans[k][m] = (m * M_COEFF + k * K_COEFF) * (m * M_COEFF + k * K_COEFF);
        }
    }

    return ans;
}

int main(int argc, char** argv) {
    int K = 1000;
    int M = 1000;

    double** exact;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <variant>\n", argv[0]);
        return -1;
    }

    if (!strcmp(argv[1], "COS_EXP")) {
        exact = get_exact_cosexp(K, M);
    } else if (!strcmp(argv[1], "SIN_SIN")) {
        exact = get_exact_sinsin(K, M);
    } else if (!strcmp(argv[1], "X2_T2")) {
        exact = get_exact_x2t2(K, M);
    } else {
        printf("Usage: %s <variant>\n", argv[0]);
        printf("Allowed options: COS_EXP, SIN_SIN, X2_T2");
        return -1;
    }

    out_like_csv(stdout, exact, K, M);
    free_fu(exact, K);

    return 0;
}