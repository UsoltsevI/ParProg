#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int message;
    int tag = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Size must be greater than 1\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        message = 1;
        printf("Process %d: created a message = %d\n", rank, message);
        
        MPI_Send(&message, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
        printf("Process %d: sent message = %d to process %d\n", rank, message, 1);
    }

    if (rank > 0 && rank < size - 1) {
        MPI_Recv(&message, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD, &status);
        printf("Process %d: received message = %d from process %d\n", 
               rank, message, rank - 1);

        message++;
        printf("Process %d: after increment = %d\n", rank, message);
        
        MPI_Send(&message, 1, MPI_INT, rank + 1, tag, MPI_COMM_WORLD);
        printf("Process %d: sent message = %d to process %d\n", 
               rank, message, rank + 1);
    }

    if (rank == size - 1) {
        MPI_Recv(&message, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD, &status);
        printf("Process %d: received message = %d from process %d\n", 
               rank, message, rank - 1);
        
        message++;
        printf("Process %d: after increment = %d\n", rank, message);
        
        MPI_Send(&message, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
        printf("Process %d: sent message = %d to process 0\n", rank, message);
    }

    if (rank == 0) {
        MPI_Recv(&message, 1, MPI_INT, size - 1, tag, MPI_COMM_WORLD, &status);
        printf("Process %d: received FINAL message = %d from process %d\n", 
               rank, message, size - 1);
    }

    MPI_Finalize();
    return 0;
}