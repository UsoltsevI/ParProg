#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

double global_sum = 0.0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    long long start;
    long long end;
} range_t;

void* calculate(void* arg) {
    range_t* r = (range_t*) arg;
    double local_sum = 0.0;

    for (long long n = r->start; n <= r->end; n++) {
        local_sum += 1.0 / n;
    }

    pthread_mutex_lock(&mutex);
    global_sum += local_sum;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <N> <threads_num>\n", argv[0]);
        return 1;
    }

    long long N = atoll(argv[1]);
    int num_threads = atoi(argv[2]);

    pthread_t threads[num_threads];
    range_t ranges[num_threads];

    long long chunk = N / num_threads;

    for (int i = 0; i < num_threads; i++) {
        ranges[i].start = i * chunk + 1;
        ranges[i].end = (i == num_threads - 1) ? N : (i + 1) * chunk;
        pthread_create(&threads[i], NULL, calculate, &ranges[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("%.15f\n", global_sum);

    return 0;
}