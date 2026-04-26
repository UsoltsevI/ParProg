#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include <omp.h>

#define PREC 128

static void func(mpfr_t res, const mpfr_t x) {
    mpfr_t one, inv;
    mpfr_inits2(PREC, one, inv, NULL);
    mpfr_set_si(one, 1, MPFR_RNDN);
    mpfr_div(inv, one, x, MPFR_RNDN);
    mpfr_sin(res, inv, MPFR_RNDN);
    mpfr_clears(one, inv, NULL);
}

static void simpson(mpfr_t result, const mpfr_t a, const mpfr_t b, int n) {
    if (n % 2 != 0) {
        n++;
    }

    mpfr_t h, x, f, sum, temp;

    mpfr_inits2(PREC, h, x, f, sum, temp, NULL);

    mpfr_sub(h, b, a, MPFR_RNDN);
    mpfr_div_si(h, h, n, MPFR_RNDN);
    mpfr_set_si(sum, 0, MPFR_RNDN);

    func(f, a); 
    mpfr_add(sum, sum, f, MPFR_RNDN);
    func(f, b); 
    mpfr_add(sum, sum, f, MPFR_RNDN);

    for (int i = 1; i < n; i += 2) {
        mpfr_mul_si(x, h, i, MPFR_RNDN); 
        mpfr_add(x, a, x, MPFR_RNDN);
        func(f, x); 
        mpfr_mul_si(temp, f, 4, MPFR_RNDN); 
        mpfr_add(sum, sum, temp, MPFR_RNDN);
    }

    for (int i = 2; i < n; i += 2) {
        mpfr_mul_si(x, h, i, MPFR_RNDN); 
        mpfr_add(x, a, x, MPFR_RNDN);
        func(f, x); 
        mpfr_mul_si(temp, f, 2, MPFR_RNDN); 
        mpfr_add(sum, sum, temp, MPFR_RNDN);
    }

    mpfr_mul(result, sum, h, MPFR_RNDN);
    mpfr_div_si(result, result, 3, MPFR_RNDN);
    mpfr_clears(h, x, f, sum, temp, NULL);
}

static void integrate_with_error(mpfr_t value, mpfr_t error, const mpfr_t a, const mpfr_t b) {
    int n = 4;
    mpfr_t S1, S2, diff;

    mpfr_inits2(PREC, S1, S2, diff, NULL);

    simpson(S1, a, b, n);
    simpson(S2, a, b, 2 * n);

    mpfr_set(value, S2, MPFR_RNDN);
    mpfr_sub(diff, S2, S1, MPFR_RNDN);
    mpfr_abs(diff, diff, MPFR_RNDN);
    mpfr_div_si(error, diff, 15, MPFR_RNDN);
    mpfr_clears(S1, S2, diff, NULL);
}

typedef struct TaskNode {
    mpfr_t a, b, tol;
    struct TaskNode *next;
} TaskNode;

typedef struct {
    TaskNode *head;
    TaskNode *tail;
    omp_lock_t lock;
} TaskQueue;

static void queue_init(TaskQueue *q) {
    q->head = NULL;
    q->tail = NULL;
    omp_init_lock(&q->lock);
}

static void queue_push(TaskQueue *q, const mpfr_t a, const mpfr_t b, const mpfr_t tol) {
    TaskNode *node = malloc(sizeof(TaskNode));

    mpfr_inits2(PREC, node->a, node->b, node->tol, NULL);
    mpfr_set(node->a, a, MPFR_RNDN);
    mpfr_set(node->b, b, MPFR_RNDN);
    mpfr_set(node->tol, tol, MPFR_RNDN);

    node->next = NULL;

    omp_set_lock(&q->lock);

    if (q->tail) {
        q->tail->next = node;
    } else {
        q->head = node;
    }

    q->tail = node;
    omp_unset_lock(&q->lock);
}

static int queue_pop(TaskQueue *q, mpfr_t a, mpfr_t b, mpfr_t tol) {
    omp_set_lock(&q->lock);

    if (q->head == NULL) {
        omp_unset_lock(&q->lock);
        return 0;
    }

    TaskNode *node = q->head;
    q->head = node->next;

    if (q->head == NULL) {
        q->tail = NULL;
    }

    omp_unset_lock(&q->lock);

    mpfr_set(a, node->a, MPFR_RNDN);
    mpfr_set(b, node->b, MPFR_RNDN);
    mpfr_set(tol, node->tol, MPFR_RNDN);
    mpfr_clears(node->a, node->b, node->tol, NULL);

    free(node);

    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <threads> <tolerance>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);

    mpfr_t a_start, b_start, tol_start, total_width, global_tol, global_sum;
    mpfr_inits2(PREC, a_start, b_start, tol_start, total_width, global_tol, global_sum, NULL);

    mpfr_set_d(a_start, 1.0 / 1000.0, MPFR_RNDN);
    mpfr_set_d(b_start, 1.0, MPFR_RNDN);
    mpfr_set_str(global_tol, argv[2], 10, MPFR_RNDN);
    mpfr_set(tol_start, global_tol, MPFR_RNDN);
    mpfr_set_si(global_sum, 0, MPFR_RNDN);
    mpfr_sub(total_width, b_start, a_start, MPFR_RNDN);

    TaskQueue queue;
    queue_init(&queue);
    queue_push(&queue, a_start, b_start, tol_start);

    printf("Integrating sin(1/x) on [%.6f, %.6f]\n", 1.0/1000.0, 1.0);
    printf("Threads: %d, tolerance: ", num_threads);
    mpfr_out_str(stdout, 10, 0, global_tol, MPFR_RNDN);
    printf("\n");

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(num_threads)
    {
        mpfr_t a, b, tol, value, error, width, mid, scaled_tol, half_tol;
        mpfr_inits2(PREC, a, b, tol, value, error, width, mid, scaled_tol, half_tol, NULL);

        while (queue_pop(&queue, a, b, tol)) {
            integrate_with_error(value, error, a, b);

            mpfr_sub(width, b, a, MPFR_RNDN);
            mpfr_div(scaled_tol, width, total_width, MPFR_RNDN);
            mpfr_mul(scaled_tol, scaled_tol, global_tol, MPFR_RNDN);

            if (mpfr_cmp(error, scaled_tol) <= 0 || mpfr_cmp_d(width, 1e-15) < 0) {
                #pragma omp critical
                mpfr_add(global_sum, global_sum, value, MPFR_RNDN);
            } else {
                mpfr_add(mid, a, b, MPFR_RNDN);
                mpfr_div_si(mid, mid, 2, MPFR_RNDN);
                mpfr_div_si(half_tol, tol, 2, MPFR_RNDN);

                queue_push(&queue, a, mid, half_tol);
                queue_push(&queue, mid, b, half_tol);
            }
        }

        mpfr_clears(a, b, tol, value, error, width, mid, scaled_tol, half_tol, NULL);
    }

    double end = omp_get_wtime();

    printf("Result: ");
    mpfr_out_str(stdout, 10, 60, global_sum, MPFR_RNDN);
    printf("\n");
    printf("Time: %.6f sec\n", end - start);

    omp_destroy_lock(&queue.lock);
    mpfr_clears(a_start, b_start, tol_start, total_width, global_tol, global_sum, NULL);

    return 0;
}