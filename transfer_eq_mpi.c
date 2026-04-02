// 
// Решение задачи переноса параллельным алгоритмом
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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
 * Получить значение u(k, 0)
 * 
 * Отличается от функции последовательного плгоритма!!!
 */
double get_psi(int k) {
    return sin(k);
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
 * Получить значения f(k, x)
 */
void get_fk(double* fk, int k, int M) {
    for (int m = 0; m < M; m++) {
        fk[m] = sin(k * m);
    }
}

/**
 * Вернуть начальную матрицу u[k][m]
 * Учитывая условия u(t, 0) и u(0, x)
 * 
 * Применяется только на root процессе
 */
double** get_empty_u(int K, int M) {
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

    return u;
}

/**
 * Дать массив шагов (мест, когда делать пересылку)
 * [0, chunk, 2*chunk, ... , M] - это size + 1 элемент
 */
int* get_steps(int M, int size) {
    int chunk_size = M / size;
    // int remainder = M % size;
    int* steps = (int *) calloc(size + 1, sizeof(int));
    steps[0] = 0;
    for (int i = 1; i < size; i++) {
        steps[i] = steps[i - 1] + chunk_size;
    }
    steps[size] = M;
    return steps;
}

/**
 * Вычислить значение в точке [k + 1][m] по схеме "явный левый уголок"
 * без присвоения значения узлу u[k + 1][m]
 */
double calc_corner(double* uk, double* fk, int m, double tau, double h) {
    return (fk[m] - (uk[m] - uk[m - 1]) * h) * tau + uk[m];
}

/**
 * Вычислить значение в точке [k + 1][m] по схеме "явная центральная 
 * трехточечная" без присвоения значения узлу u[k + 1][m]
 */
double calc_center_three(double* uk, double* fk, int m, double tau, double h) {
    return (fk[m] - (uk[m + 1] - uk[m - 1]) / (2 * h)) * tau + (uk[m + 1] + uk[m - 1]) / 2;
}

/**
 * Вычислить значение в точке [k + 1][m] по схеме "крест" 
 * без присвоения значения узлу u[k + 1][m]
 * 
 * uk1 = u[k - 1]
 */
double calc_cross(double* uk, double* uk1, double* fk, int m, double tau, double h) {
    return (fk[m] - (uk[m + 1] - uk[m - 1]) / (2 * h)) * (2 * tau) + uk1[m];
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

int calc_tag_uk(int k, int part, int size) {
    return k + part;
}

int get_prev_rank(int rank, int size) {
    return (rank + size - 1) % size;
}

int get_next_rank(int rank, int size) {
    return (rank + 1) % size;
}

/**
 * tag для отправки всего ряда нулевому процессу для сборки
 */
int calc_root_tag_uk(int k) {
    return 32000 - k; // 32767
}

// DEBUG
void print_double_arr(double* arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.4lf ", arr[i]);
    }
}

/**
 * Отправить данные о части процессу
 */
void send_uk_data(double* uk, double* uk1, int start, int end, int to_rank, int tag, int M) {
    // отправляем с следующем формате:
    // из tag сразу можно дстать k и part
    // поэтому отпрвляем чисто uk и uk1 подряд
    int sz = end - start;
    double* ukuk = (double *) calloc(sz * 2, sizeof(double));
    memcpy(ukuk, uk + start, sz * sizeof(double));
    memcpy(ukuk + sz, uk1 + start, sz * sizeof(double));

    // printf("sending to process %d\n", to_rank);
    // printf("uk");
    // print_double_arr(uk + start, sz);
    // printf("\n uk1 \n");
    // print_double_arr(uk1 + start, sz);
    // printf("\n");
    // print_double_arr(ukuk, sz * 2);
    // printf("\n");

    MPI_Request request;
    
    // printf("Sending TAG %d\n", tag);
    MPI_Isend(ukuk, sz * 2, MPI_DOUBLE, to_rank, tag, MPI_COMM_WORLD, &request);
    // free(ukuk); пока с утечкой
}

/**
 * Принять данные от предыдущего процесса
 */
void recv_uk_data(double* uk, double* uk1, int start, int end, int from_rank, int tag, int M) {
    int sz = end - start;
    double* buf = (double *) malloc(sz * 2 * sizeof(double));
    MPI_Status status;
    // printf("Waiting (from %d) for TAG %d\n", from_rank, tag);
    MPI_Recv(buf, sz * 2, MPI_DOUBLE, from_rank, tag, MPI_COMM_WORLD, &status);
    memcpy(uk + start, buf, sz * sizeof(double));
    memcpy(uk1 + start, buf + sz, sz * sizeof(double));

    // printf("recieved from process %d\n", from_rank);
    // print_double_arr(buf, sz * 2);
    // printf("\n");
    // print_double_arr(uk + start, sz);
    // printf("\n");
    // print_double_arr(uk1 + start, sz);
    // printf("\n");

    free(buf);
}

/**
 * Отправить результаты на 0 процесс
 */
void send_uk_to_root(double* uk, int k, int M) {
    double* copy = (double *) malloc(M * sizeof(double));
    memcpy(copy, uk, M * sizeof(double));
    // printf("sending to root, k = %d\n", k);
    // print_double_arr(uk, M);
    // printf("\n");
    MPI_Request request;
    // printf("Sending ROOTTAG %d\n", calc_root_tag_uk(k));
    MPI_Isend(copy, M, MPI_DOUBLE, 0, calc_root_tag_uk(k), MPI_COMM_WORLD, &request);
    // free(copy); пока с утечкой
}

/**
 * Принять результаты на 0 процессе
 */
void recv_uk_root(double** u, int k, int M) {
    // printf("Waiting for ROOTTAG %d\n", calc_root_tag_uk(k));
    MPI_Recv(u[k], M, MPI_DOUBLE, MPI_ANY_SOURCE, calc_root_tag_uk(k), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // printf("recieved at root, k = %d\n", k);
    // print_double_arr(u[k], M);
    // printf("\n");
}

int main(int argc, char** argv) {
    int rank, size;
    int tag = 0;
    MPI_Status status;

    int K = 100;
    int M = 100;
    double tau = 0.01;
    double h = 0.01;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "Size should be greater than 1\n");
        MPI_Finalize();
        return 1;
    }

    int* steps = get_steps(M, size);

    double* fk = (double*) calloc(M, sizeof(double));
    double* uk = (double*) calloc(M, sizeof(double));
    double* uk1 = (double*) calloc(M, sizeof(double));
    double* uk_next = (double *) calloc(M, sizeof(double));
    double* phi = NULL;

    if (rank == 0) {
        phi = get_phi(M);
        get_fk(fk, 0, M);

        uk_next[0] = get_psi(1);
        uk_next[M - 1] = calc_corner(phi, fk, M - 1, tau, h);

        // считаем первый слой
        // size(steps) == size + 1
        for (int part = 0; part < size; part++) {
            // Считаем до m + 1 элемента, потому что следующему процессу он 
            // понадобится
            int start = steps[part];
            int end = steps[part + 1];
            for (int m = MAX(start, 1); m < MIN(end, M - 1); m++) {
                uk_next[m] = calc_center_three(phi, fk, m, tau, h);
            }
            send_uk_data(uk_next, phi, start, end, get_next_rank(rank, size), calc_tag_uk(1, part, size), M);
        }

        send_uk_to_root(uk_next, 1, M);
    }

    // Мы считаем для k + 1 
    for (int k = rank; k < K; k += size) {
        if (k == 0) {
            continue; // Первый шаг уже сделали
        }
        assert(k != 0);
        get_fk(fk, k, M);
        uk_next[0] = get_psi(k + 1);

        for (int part = 0; part < size; part++) {
            // Получаем данные от k - 1 процесса
            int start = steps[part];
            int end = steps[part + 1];
            recv_uk_data(uk, uk1, start, end, get_prev_rank(rank, size), calc_tag_uk(k, part, size), M);

            if (start > 0) {
                // читаем первый элемент
                uk_next[start - 1] = calc_cross(uk, uk1, fk, start - 1, tau, h);
                if (k < K - 1) {
                    int prev_start = steps[part - 1];
                    int prev_end = steps[part];
                    assert(prev_end == start);
                    // отправка (part - 1) !!!
                    send_uk_data(uk_next, uk, prev_start, prev_end,  get_next_rank(rank, size), calc_tag_uk(k + 1, part - 1, size), M);
                }
            }

            for (int m = MAX(start, 1); m < end - 1; m++) {
                // следующий элемент посчитать не можем, так как не достаточно данных
                uk_next[m] = calc_cross(uk, uk1, fk, m, tau, h);
            }
            if (part == size - 1) {
                assert(end - 1 == M - 1);
                uk_next[M - 1] = calc_corner(uk, fk, M - 1, tau, h);
            }
            if (part == size - 1) {
                assert(end == M);
            }
            if (end == M && k < K - 1) {
                assert(part == size - 1);
                send_uk_data(uk_next, uk, start, end, get_next_rank(rank, size), calc_tag_uk(k + 1, part, size), M);
            }
        }

        send_uk_to_root(uk_next, k + 1, M);
    }

    free(fk);
    free(uk);
    free(uk1);
    free(uk_next);
    free(steps);

    if (rank == 0) {
        double** u = get_empty_u(K, M);

        for (int m = 0; m < M; m++) {
            u[0][m] = phi[m];
        }

        for (int k = 1; k < K; k++) {
            recv_uk_root(u, k, M);
        }

        // Вывод данных
        out_like_csv(stdout, u, K, M);

        free(u);
        free(phi);
    }

    MPI_Finalize();

    return 0;
}