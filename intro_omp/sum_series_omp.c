#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <N> <num_threads>\n", argv[0]);
        printf("Example: %s 1000000 4\n", argv[0]);
        return 1;
    }
    
    long long N = atoll(argv[1]);
    int num_threads = atoi(argv[2]);
    
    if (N <= 0) {
        printf("N must be positive\n");
        return 1;
    }
    
    if (num_threads <= 0) {
        printf("Number of threads must be positive\n");
        return 1;
    }
    
    omp_set_num_threads(num_threads);
    
    double sum = 0.0;
    double start_time, end_time;
    
    start_time = omp_get_wtime();
    
    #pragma omp parallel reduction(+:sum)
    {
        int thread_id = omp_get_thread_num();
        int actual_threads = omp_get_num_threads();
        
        #pragma omp single
        {
            printf("Requested threads: %d\n", num_threads);
            printf("Actual threads: %d\n", actual_threads);
            printf("Calculating harmonic sum for N = %lld\n\n", N);
        }
        
        #pragma omp for
        for (long long n = 1; n <= N; n++) {
            sum += 1.0 / n;
        }
        
        #pragma omp critical
        {
            printf("Thread %d finished its part\n", thread_id);
        }
    }
    
    end_time = omp_get_wtime();
    
    printf("\nResult: %.15f\n", sum);
    printf("Time: %.6f seconds\n", end_time - start_time);
    
    return 0;
}