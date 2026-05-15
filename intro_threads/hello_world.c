#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define NUM_THREADS 4 // default value

void* print_hello(void* args) {
    int* data = (int*) args;
    printf("Hello from thread %d (total %d)\n", data[0], data[1]);
    return NULL;
}

int main(int argc, char** argv) {
    pthread_t threads[NUM_THREADS];
    int thread_data[NUM_THREADS][2];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i][0] = i;
        thread_data[i][1] = NUM_THREADS;
        pthread_create(&threads[i], NULL, print_hello, thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}