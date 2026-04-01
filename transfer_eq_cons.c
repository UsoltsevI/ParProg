// 
// Решение задачи переноса последовательным алгоритмом
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

/**
 * Получить значения u(0, x) на сетке [0, M]
 */
double* get_phi(int M) {
    double* phi_values = (double *) calloc(M, sizeof(double));
    if (phi_values == NULL) {
        fprintf(stderr, "Ошибка выделения памяти: %s\n", strerror(errno));
        return NULL;
    }
    for (int m = 0; m < M; m++) {
        phi_values[m] = sin(m);
    }
    return phi_values;
}

/**
 * Получить значения u(t, 0) на сетке [0, K]
 */
double* get_psi(int K) {
    double* psi_values = (double *) calloc(K, sizeof(double));
    if (psi_values == NULL) {
        fprintf(stderr, "Ошибка выделения памяти: %s\n", strerror(errno));
        return NULL;
    }
    for (int k = 0; k < K; k++) {
        psi_values[k] = sin(k);
    }
    return psi_values;
}

/**
 * Освободить память f или u до строки k не включая
 */
void free_fu(double** fu, int k) {
    for (int i = 0; i < k; i++) {
        free(fu[i]);
    }
    free(fu);
}

/**
 * Получить значения f(t, x) на сетке [0, K][0, M]
 */
double** get_f(int K, int M) {
    double** f_values = (double **) calloc(K, sizeof(double *));

    if (f_values == NULL) {
        fprintf(stderr, "Ошибка выделения памяти: %s\n", strerror(errno));
        return NULL;
    }

    for (int k = 0; k < K; k++) {
        f_values[k] = (double *) calloc(M, sizeof(double));
        if (f_values[k] == NULL) {
            fprintf(stderr, "Ошибка выделения памяти: %s\n", strerror(errno));
            free_fu(f_values, k);
            return NULL;
        }
    }

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            f_values[k][m] = sin(k * m);
        }
    }

    return f_values;
}

/**
 * Вернуть начальную матрицу u[k][m]
 * Учитывая условия u(t, 0) и u(0, x)
 */
double** get_initial_u(double* psi, double* phi, int K, int M) {
    double** u = (double **) calloc(K, sizeof(double *));

    if (u == NULL) {
        fprintf(stderr, "Ошибка выделения памяти: %s\n", strerror(errno));
        return NULL;
    }

    for (int k = 0; k < K; k++) {
        u[k] = (double *) calloc(M, sizeof(double));
        if (u[k] == NULL) {
            fprintf(stderr, "Ошибка выделения памяти: %s\n", strerror(errno));
            free_fu(u, k);
            return NULL;
        }
    }

    // Заполняем краевые условия
    for (int k = 0; k < K; k++) {
        u[k][0] = psi[k];
    }

    for (int m = 0; m < M; m++) {
        u[0][m] = phi[m];
    }

    return u;
}

/**
 * Вычислить значение в точке [k + 1][m] по схеме "явный левый уголок"
 * без присвоения значения узлу u[k + 1][m]
 */
double calc_corner(double** u, double** f, int k, int m, double tau, double h) {
    return (f[k][m] - (u[k][m] - u[k][m - 1]) * h) * tau + u[k][m];
}

/**
 * Вычислить значение в точке [k + 1][m] по схеме "явная центральная 
 * трехточечная" без присвоения значения узлу u[k + 1][m]
 */
double calc_center_three(double** u, double** f, int k, int m, double tau, double h) {
    return (f[k][m] - (u[k][m + 1] - u[k][m - 1]) / (2 * h)) * tau + (u[k][m + 1] + u[k][m - 1]) / 2;
}

/**
 * Вычислить значение в точке [k + 1][m] по схеме "крест" 
 * без присвоения значения узлу u[k + 1][m]
 */
double calc_cross(double** u, double** f, int k, int m, double tau, double h) {
    return (f[k][m] - (u[k][m + 1] - u[k][m - 1]) / (2 * h)) * (2 * tau) + u[k - 1][m];
}

/**
 * Вывести наддые в файл в формате csv
 */
void out_like_csv(FILE* out, double** u, int K, int M) {
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M - 1; m++) {
            fprintf(out, "%.15f,", u[k][m]);
        }
        fprintf(out, "%.15f", u[k][M - 1]);
        fprintf(out, "\n");
    }
}

int main() {
    int K = 100;
    int M = 100;
    double tau = 0.01;
    double h = 0.01;

    double* psi = get_psi(K);
    double* phi = get_phi(M);
    double** f = get_f(K, M);
    double** u = get_initial_u(psi, phi, K, M);

    // Сначала заполняем 1й слой, используя 
    // явную центральную трехточечную схему
    for (int m = 1; m < M - 1; m++) {
        u[1][m] = calc_center_three(u, f, 0, m, tau, h);
    }
    // в конце схемой уголок
    u[1][M - 1] = calc_corner(u, f, 0, M - 1, tau, h);

    // Теперь все остальные слои по схеме крест
    for (int k = 1; k < K - 1; k++) {
        for (int m = 1; m < M - 1; m++) {
            u[k + 1][m] = calc_cross(u, f, k, m, tau, h);
        }
        // последний элемент схемой уголок
        u[k + 1][M - 1] = calc_corner(u, f, k, M - 1, tau, h);
    }

    // Вывод данных
    out_like_csv(stdout, u, K, M);

    free(psi);
    free(phi);
    free_fu(f, K);
    free_fu(u, K);

    return 0;
}