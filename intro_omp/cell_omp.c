#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
    int num_threads = 4;  
    
    if (argc == 2) {
        num_threads = atoi(argv[1]);
    }
    
    omp_set_num_threads(num_threads);
    
    int shared_counter = 0;
    
    printf("Starting sequential access simulation with %d threads\n", num_threads);
    printf("Each thread will increment the shared counter in order\n");
    printf("--------------------------------------------------------\n");
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        
        for (int turn = 0; turn < total_threads; turn++) {
            if (thread_id == turn) {
                #pragma omp critical
                {
                    shared_counter++;
                    
                    printf("Thread %d: Incremented counter to %d\n", 
                           thread_id, shared_counter);
                }
            }

            #pragma omp barrier
        }
    }
    
    printf("--------------------------------------------------------\n");
    printf("Final counter value: %d (should equal %d)\n", 
           shared_counter, num_threads);
    
    return 0;
}