#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size != 2) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 2 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int num_iterations = 1000;      // количество измерений
    const int warmup_iterations = 100;    // разогрев (не учитываем)
    
    int message_sizes[] = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304
    };
    
    int num_sizes = sizeof(message_sizes) / sizeof(message_sizes[0]);

    if (world_rank == 0) {
        printf("%-20s %-20s\n", "Message size (bytes)", "Latency (microseconds)");
        printf("--------------------------------------------\n");
    }

    // Синхронизация перед началом
    MPI_Barrier(MPI_COMM_WORLD);

    for (int s = 0; s < num_sizes; s++) {
        int size_bytes = message_sizes[s];
        int count = size_bytes / sizeof(char);
        if (count == 0) count = 1;

        char* send_buffer = (char*)malloc(count * sizeof(char));
        char* recv_buffer = (char*)malloc(count * sizeof(char));
        
        if (send_buffer == NULL || recv_buffer == NULL) {
            fprintf(stderr, "Process %d: Memory allocation failed for size %d\n", 
                    world_rank, size_bytes);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Заполняем буфер тестовыми данными
        memset(send_buffer, 'A', count * sizeof(char));

        // Разогрев (warm-up) - исключаем первоначальные накладные расходы
        for (int i = 0; i < warmup_iterations; i++) {
            if (world_rank == 0) {
                MPI_Send(send_buffer, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (world_rank == 1) {
                MPI_Recv(recv_buffer, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buffer, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        // Синхронизация перед измерениями
        MPI_Barrier(MPI_COMM_WORLD);

        double total_rtt = 0.0; // суммарное время "туда-обратно"

        for (int i = 0; i < num_iterations; i++) {
            double start_time = 0.0, end_time = 0.0;

            if (world_rank == 0) {
                start_time = MPI_Wtime();
                MPI_Send(send_buffer, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                end_time = MPI_Wtime();
                total_rtt += (end_time - start_time);
            } else if (world_rank == 1) {
                MPI_Recv(recv_buffer, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buffer, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        if (world_rank == 0) {
            double avg_rtt = total_rtt / num_iterations;
            double avg_one_way_latency = avg_rtt / 2.0;
            double latency_us = avg_one_way_latency * 1e6;
            
            printf("%-20d %-20.3f\n", size_bytes, latency_us);
            fflush(stdout);
        }

        free(send_buffer);
        free(recv_buffer);
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        printf("--------------------------------------------\n");
        printf("Measurement completed.\n");
    }

    MPI_Finalize();
    return 0;
}