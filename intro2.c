#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int N;
    int rank, size;
    double local_sum = 0.0;
    double global_sum = 0.0;
    int start, end;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Usage: %s <N>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N = atoi(argv[1]);
        if (N <= 0){
            fprintf(stderr, "N must be greater than 0");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Calculation: N = %d, size = %d\n", N, size);
        start_time = MPI_Wtime();
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = N / size;
    int rem = N % size;

    if (rank < rem) {
        start = rank * (chunk_size + 1) + 1;
        end = start + chunk_size;
    } else {
        start = rank * chunk_size + rem + 1;
        end = start + chunk_size - 1;
    }

    if (start > N) {
        start = N + 1;
        end = N;
    }
    
    local_sum = 0.0;
    for (int i = start; i <= end; i++) {
        local_sum += 1.0 / i;
    }

    printf("Rank %d, sum from %d to %d = %f\n", rank, start, end, local_sum);

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0 ) {
        end_time = MPI_Wtime();
        printf("Total sum from 1 to %d: %f\n", N, global_sum);
        printf("Execution time: %f sec\n", end_time - start_time);
    }

    MPI_Finalize();

    return 0;
}