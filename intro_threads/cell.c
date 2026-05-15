#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int turn = 0;

typedef struct {
    int id;
    int num_threads;
} thread_data_t;

void* worker(void* arg) {
    thread_data_t* data = (thread_data_t*) arg;

    pthread_mutex_lock(&mutex);

    while (turn != data->id) {
        pthread_cond_wait(&cond, &mutex);
    }

    counter++;

    printf("Thread %d: counter = %d\n", data->id, counter);

    turn++;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char** argv) {
    int num_threads = 4;

    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0) {
            printf("num_threads must be > 0\n");
            return 1;
        }
    }

    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].id = i;
        thread_data[i].num_threads = num_threads;
        pthread_create(&threads[i], NULL, worker, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("\nFinal counter value: %d\n", counter);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
}