#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpfr.h>
#include <math.h>
#include <string.h>

int find_N_for_precision(int K) {
    if (K <= 0) {
        return 1;
    }
    
    int N = 1;
    double log_fact = 0.0;
    double log10 = log(10);
    
    while (log_fact <= K * log10) {
        N++;
        log_fact += log(N);  // log(N!) = sum(log(i))
    }
    
    return N;
}


int mpfr_to_buf(mpfr_t x, unsigned char **buf, size_t *buf_size, int base) {
    char *str_buf;
    mpfr_exp_t exponent;
    size_t str_len;
    unsigned int str_len_uint;
    int exp_int;
    unsigned char *tmp_buf;
    size_t total_size;
    int header_size = 8; // 4 байта для длины строки + 4 байта для экспоненты
    
    if (buf == NULL || buf_size == NULL) {
        fprintf(stderr, "mpfr_to_buf: передан NULL указатель\n");
        return -1;
    }
    
    str_buf = mpfr_get_str(NULL, &exponent, base, 0, x, MPFR_RNDN);
    if (str_buf == NULL) {
        fprintf(stderr, "mpfr_to_buf: не удалось преобразовать mpfr в строку\n");
        return -2;
    }
    
    str_len = strlen(str_buf);
    str_len_uint = (unsigned int)str_len;
    exp_int = (int)exponent;
    
    total_size = header_size + str_len + 1;
    
    tmp_buf = (unsigned char*)malloc(total_size);
    if (tmp_buf == NULL) {
        fprintf(stderr, "mpfr_to_buf: не удалось выделить %zu байт\n", total_size);
        mpfr_free_str(str_buf);
        return -4;
    }
    
    memcpy(tmp_buf, &str_len_uint, 4);
    memcpy(tmp_buf + 4, &exp_int, 4);
    memcpy(tmp_buf + 8, str_buf, str_len + 1);
    
    *buf = tmp_buf;
    *buf_size = total_size;
    
    mpfr_free_str(str_buf);
    return 0;
}

int buf_to_mpfr(mpfr_t x, const unsigned char *buf, size_t buf_size, int base) {
    unsigned int str_len;
    int exp_int;
    int header_size = 8;
    int result;
    char *full_str;
    
    if (buf == NULL || buf_size < header_size + 1) {
        return -1;
    }
    
    memcpy(&str_len, buf, 4);
    memcpy(&exp_int, buf + 4, 4);
    
    if (header_size + str_len + 1 > buf_size) {
        return -3;
    }
    
    // Создаем строку с десятичной точкой для mpfr_set_str
    // Формат: цифры с точкой после первой цифры + "e" + экспонента-1
    full_str = malloc(str_len + 32); // Достаточно места
    if (full_str == NULL) {
        return -4;
    }
    
    // Копируем первую цифру
    full_str[0] = buf[header_size];
    full_str[1] = '.';
    
    // Копируем остальные цифры
    memcpy(full_str + 2, buf + header_size + 1, str_len - 1);
    
    // Добавляем экспоненту
    // exp_int из mpfr_get_str означает: число = цифры * base^exp_int
    // Для mpfr_set_str с точкой после первой цифры, экспонента = exp_int - 1
    if (exp_int - 1 != 0) {
        sprintf(full_str + str_len + 1, "e%d", exp_int - 1);
    } else {
        full_str[str_len + 1] = '\0';
    }
    
    result = mpfr_set_str(x, full_str, base, MPFR_RNDN);
    
    free(full_str);
    
    if (result != 0) {
        fprintf(stderr, "buf_to_mpfr: ошибка преобразования '%s'\n", full_str);
        return -5;
    }
    
    return 0;
}

int send_mpfr(mpfr_t x, int dest, int tag) {
    int base = 10;
    unsigned char *buf = NULL;
    size_t buf_size;
    int result;
    
    result = mpfr_to_buf(x, &buf, &buf_size, base);
    if (result != 0) {
        return result;
    }
    
    MPI_Send(buf, buf_size, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
    free(buf);
    
    return 0;
}

int recv_mpfr(mpfr_t x, int src, int tag) {
    int base = 10;
    MPI_Status status;
    int count;
    unsigned char *buf = NULL;
    int result;
    
    MPI_Probe(src, tag, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_BYTE, &count);
    
    if (count <= 0) {
        fprintf(stderr, "recv_mpfr: получено пустое сообщение\n");
        return -1;
    }
    
    buf = (unsigned char*)malloc(count);
    if (buf == NULL) {
        fprintf(stderr, "recv_mpfr: не удалось выделить %d байт\n", count);
        return -2;
    }
    
    MPI_Recv(buf, count, MPI_BYTE, src, tag, MPI_COMM_WORLD, &status);
    
    result = buf_to_mpfr(x, buf, count, base);
    free(buf);
    
    return result;
}

int test_mpfr() {
    mpfr_t x, y;
    int K = 100;
    mpfr_init2(x, K);
    mpfr_init2(y, K);

    mpfr_set_d(x, 1.23431233, MPFR_RNDN);
    unsigned char *buf;
    size_t buf_size;
    mpfr_to_buf(x, &buf, &buf_size, 10);

    buf_to_mpfr(y, buf, buf_size, 10);

    printf("Test buf x = ");
    mpfr_out_str(stdout, 10, 10, x, MPFR_RNDN);
    printf("\n");

    printf("Test buf y = ");
    mpfr_out_str(stdout, 10, 10, y, MPFR_RNDN);
    printf("\n");
}

int main(int argc, char** argv) {
    int rank, size;
    int tag = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "Size should be greater than 1\n");
        MPI_Finalize();
        return 1;
    }

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <K>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int K = atoi(argv[1]);
    if (K <= 0) {
        fprintf(stderr, "K must be greater than 0\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = find_N_for_precision(K);
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    mpfr_prec_t prec = K * 4;  // Запас прочности
    mpfr_t n_fact, sum, prev_fact, temp, term, n_mp;
    
    mpfr_init2(n_fact, prec);
    mpfr_init2(prev_fact, prec);
    mpfr_init2(sum, prec);
    mpfr_init2(temp, prec);
    mpfr_init2(term, prec);
    mpfr_init2(n_mp, prec);

    mpfr_set_zero(sum, 0);
    mpfr_set_ui(n_fact, 1, MPFR_RNDN);

    int chunk_size = N / size;
    int remainder = N % size;
    
    int start, end;
    if (rank < remainder) {
        start = rank * (chunk_size + 1);
        end = start + chunk_size + 1;
    } else {
        start = rank * chunk_size + remainder;
        end = start + chunk_size;
    }
    
    if (end > N) {
        end = N;
    }
    
    // printf("Process %d: computing terms %d to %d\n", rank, start, end-1);

    mpfr_set_ui(n_fact, 1, MPFR_RNDN);
    
    for (int n = start; n < end; n++) {
        if (n > 0) {
            mpfr_div_ui(n_fact, n_fact, n, MPFR_RNDN);
        }
        mpfr_add(sum, sum, n_fact, MPFR_RNDN);
    }

    // printf("rank = %d, sum = ", rank);
    // mpfr_out_str(stdout, 10, 10, sum, MPFR_RNDN);
    // printf("\n");

    // printf("rank = %d, n_fact = ", rank);
    // mpfr_out_str(stdout, 10, 10, n_fact, MPFR_RNDN);
    // printf("\n");


    if (rank == 0) {
        // Процесс 0 отправляет свой конечный факториал процессу 1
        if (size > 1 && end > 0) {
            send_mpfr(n_fact, rank + 1, tag);
        }
    } else {
        // Получаем факториал от предыдущего процесса
        recv_mpfr(prev_fact, rank - 1, tag);

        // printf("rank = %d, prev_fact = ", rank);
        // mpfr_out_str(stdout, 10, 10, prev_fact, MPFR_RNDN);
        // printf("\n");

        // Домножаем до нужного значения
        mpfr_mul(n_fact, n_fact, prev_fact, MPFR_RNDN);
        mpfr_mul(sum, sum, prev_fact, MPFR_RNDN);
        
        if (rank < size - 1) {
            send_mpfr(n_fact, rank + 1, tag);
        }
    }

    // printf("rank = %d, sum_a_d = ", rank);
    // mpfr_out_str(stdout, 10, 10, sum, MPFR_RNDN);
    // printf("\n");

    unsigned char *buf = NULL;
    size_t buf_size;
    
    int result = mpfr_to_buf(sum, &buf, &buf_size, 10);
    if (result != 0) {
        fprintf(stderr, "Process %d: не получилось преобразовать sum в buf\n", rank);
        MPI_Finalize();
        return 1;
    }

    // Собираем размеры буферов
    int *recvcounts = NULL;
    int *displs = NULL;
    unsigned char *recvbuf = NULL;
    int total_size = 0;
    
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
    }
    
    int send_size = (int)buf_size;
    MPI_Gather(&send_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
        total_size = displs[size-1] + recvcounts[size-1];
        recvbuf = (unsigned char*)malloc(total_size);
    }
    
    // Собираем данные
    MPI_Gatherv(buf, send_size, MPI_BYTE, 
                recvbuf, recvcounts, displs, MPI_BYTE, 
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        mpfr_t total_sum;
        mpfr_init2(total_sum, prec);
        mpfr_set_zero(total_sum, 0);
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            mpfr_t partial_sum;
            mpfr_init2(partial_sum, prec);
            
            result = buf_to_mpfr(partial_sum, recvbuf + displs[i], recvcounts[i], 10);
            if (result == 0) {
                mpfr_add(total_sum, total_sum, partial_sum, MPFR_RNDN);
                // printf("total_sum = ");
                // mpfr_out_str(stdout, 10, 10, total_sum, MPFR_RNDN);
                // printf("\n");
                // printf("partial_sum = ");
                // mpfr_out_str(stdout, 10, 10, partial_sum, MPFR_RNDN);
                // printf("\n");
            } else {
                fprintf(stderr, "Ошибка преобразования буфера %d\n", i);
            }
            mpfr_clear(partial_sum);
        }
        
        // printf("e = ");
        mpfr_out_str(stdout, 10, K, total_sum, MPFR_RNDN);
        printf("\n");
        
        mpfr_clear(total_sum);
        free(recvbuf);
        free(recvcounts);
        free(displs);
    }
    
    free(buf);
    
    // Очистка
    mpfr_clear(n_fact);
    mpfr_clear(prev_fact);
    mpfr_clear(sum);
    mpfr_clear(temp);
    mpfr_clear(term);
    mpfr_clear(n_mp);
    
    MPI_Finalize();
    return 0;
}