// функция: sin(1/x) [1/1000, 1]

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

double func(double x) {
    return sin(1.0 / x);
}

typedef struct {
    double a;
    double b;
    double tolerance;
} Task;

typedef struct {
    Task *tasks;
    int capacity;
    volatile int head;  
    volatile int tail;
    omp_lock_t lock;
} TaskQueue;

TaskQueue* queue_create(size_t capacity) {
    TaskQueue *q = (TaskQueue*) malloc(sizeof(TaskQueue));
    q->tasks = (Task*) malloc(sizeof(Task) * capacity);
    q->capacity = capacity;
    q->head = 0;
    q->tail = 0;
    omp_init_lock(&q->lock);
    return q;
}

void queue_destroy(TaskQueue *q) {
    omp_destroy_lock(&q->lock);
    free(q->tasks);
    free(q);
}

int queue_push(TaskQueue *q, Task t) {
    omp_set_lock(&q->lock);
    if (q->tail - q->head >= q->capacity) {
        omp_unset_lock(&q->lock);
        return 0; 
    }
    q->tasks[q->tail % q->capacity] = t;
    q->tail++;
    omp_unset_lock(&q->lock);
    return 1;
}

int queue_pop(TaskQueue *q, Task *t) {
    omp_set_lock(&q->lock);
    if (q->head >= q->tail) {
        omp_unset_lock(&q->lock);
        return 0;
    }
    *t = q->tasks[q->head % q->capacity];
    q->head++;
    omp_unset_lock(&q->lock);
    return 1;
}

double simpson_basic(double a, double b) {
    double mid = (a + b) * 0.5;
    double h = (b - a) * 0.5;
    return (h / 3.0) * (func(a) + 4.0 * func(mid) + func(b));
}

static void integrate_with_error(double a, double b,
                                  double *value, double *error) {
    double mid = (a + b) * 0.5;
    
    double S1 = simpson_basic(a, b);
    
    double S2_left  = simpson_basic(a, mid);
    double S2_right = simpson_basic(mid, b);
    double S2 = S2_left + S2_right;
    
    *value = S2;
    *error = fabs(S2 - S1) / 15.0;
}

double parallel_adaptive(double a, double b, double tolerance,
                         int num_threads) {
    size_t capacity = 1000000;
    TaskQueue *q = queue_create(capacity);
    
    double global_sum = 0.0;
    long long total_evals = 0;
    
    Task initial;
    initial.a = a;
    initial.b = b;
    initial.tolerance = tolerance;
    queue_push(q, initial);
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        Task t;
        double value, error;
        long long local_evals = 0;
        
        while (1) {
            // Пытаемся извлечь задачу
            if (!queue_pop(q, &t)) {
                // Очередь пуста, но могли остаться задачи у других потоков
                // Ждём немного и пробуем снова
                #pragma omp flush
                if (!queue_pop(q, &t)) {
                    // Две попытки — если всё ещё пусто, всё готово
                    // Проверим, что никто не добавит (barrier)
                    #pragma omp barrier
                    if (!queue_pop(q, &t))
                        break;
                }
            }
            
            // Считаем интеграл и ошибку
            integrate_with_error(t.a, t.b, &value, &error);
            local_evals += 5;  // 5 вызовов func на один интервал
            
            // Длина интервала для пропорционального распределения ошибки
            double width = t.b - t.a;
            
            if (error <= t.tolerance * width / (b - a) || width < 1e-14) {
                // Принимаем результат
                #pragma omp atomic
                global_sum += value;
            } else {
                // Дробим пополам, ошибка делится пропорционально
                double mid = (t.a + t.b) * 0.5;
                Task left  = {t.a, mid, t.tolerance * 0.5};
                Task right = {mid, t.b, t.tolerance * 0.5};
                
                // Кладём в очередь (с повтором при переполнении)
                while (!queue_push(q, right)) { /* spin */ }
                while (!queue_push(q, left))  { /* spin */ }
            }
        }
        
        #pragma omp atomic
        total_evals += local_evals;
    }
    
    queue_destroy(q);
    
    printf("Всего вычислений функции: %lld\n", total_evals);
    return global_sum;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <threads_number> <tolerance>\n", argv[0]);
        return 1;
    }
    
    int num_threads = atoi(argv[1]);
    double tolerance = atof(argv[2]);
    
    double a = 1.0 / 1000.0;
    double b = 1.0;      
    
    printf("Интегрирование sin(1/x) на [%.6f, %.6f]\n", a, b);
    printf("Потоков: %d, Погрешность: %.2e\n", num_threads, tolerance);
    
    // Разогрев (первый запуск может быть медленнее)
    parallel_adaptive(a, b, tolerance, num_threads);
    
    // Замер времени
    double start = omp_get_wtime();
    double result = parallel_adaptive(a, b, tolerance, num_threads);
    double end = omp_get_wtime();
    
    double elapsed = end - start;
    
    printf("Результат: %.15f\n", result);
    printf("Время: %.6f сек\n", elapsed);
    
    return 0;
}