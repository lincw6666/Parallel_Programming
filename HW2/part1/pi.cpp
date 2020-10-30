#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <stdlib.h>
#include <pthread.h>    // For Pthread
#include "shishua.h"
using namespace std;


#define BUFSIZE 128
#define MAX_N_THREAD 1024


typedef unsigned long long ull;


typedef struct thread_arg {
    ull thread_id;
    ull n_tosses;
} thread_arg_t;

typedef union shishua_buf {
    uint8_t buf8[BUFSIZE];
    uint32_t buf32[BUFSIZE>>2];
} shishua_buf_u;


ull g_seed = time(NULL);
ull *g_shishua_seed;
ull g_n_circle = 0;
pthread_mutex_t mutex_n_circle;


void *monte_carlo_pi_estimate(void *arg) {
    thread_arg_t *args = (thread_arg_t *)arg;
    ull seed = g_seed + args->thread_id;
    ull n_circle = 0;
    double x, y, distance_squared;

    for (int i = 0; i < args->n_tosses; ++i) {
        seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
        x = (double)seed / RAND_MAX;
        seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
        y = (double)seed / RAND_MAX;
        distance_squared = x*x + y*y;
        if (distance_squared <= 1)
            n_circle += 1;
    }

    pthread_mutex_lock(&mutex_n_circle);
    g_n_circle += n_circle;
    pthread_mutex_unlock(&mutex_n_circle);

    pthread_exit(NULL);
}

void *calculate_n_circle(void *arg) {
    thread_arg_t *args = (thread_arg_t *)arg;
    ull n_circle = 0;
    int toss;
    double x, y, distance_squared;
    // For ShiShua
    SEEDTYPE shishua_seed[4] = {g_shishua_seed[args->thread_id], 0, 0, 0};
    prng_state s = prng_init(shishua_seed);
    shishua_buf_u buf __attribute__ ((aligned (64)));

    for (toss = 0; toss < args->n_tosses; ++toss) {
        prng_gen(&s, buf.buf8, sizeof(buf));
        for (int i = 0; i < (BUFSIZE>>2); i += 2) {
            x = (double)buf.buf32[i] / 0xFFFFFFFF;
            y = (double)buf.buf32[i+1] / 0xFFFFFFFF;
            distance_squared = x*x + y*y;
            if (distance_squared <= 1)
                n_circle += 1;
        }
    }

    pthread_mutex_lock(&mutex_n_circle);
    g_n_circle += n_circle;
    pthread_mutex_unlock(&mutex_n_circle);

    pthread_exit(NULL);
}


int main(int argc, char **argv) {
    ull n_circle = 0; // # of darts in circle.
    int n_thread = atoi(argv[1]) - 1;
    ull n_tosses = atoi(argv[2]);    // # of tosses
    ull thread_tosses = n_tosses / (n_thread+1);
    ull seed;
    double x, y, distance_squared;
    double pi_estimate = 0.0;
    pthread_t *thread = NULL;
    thread_arg_t *args = NULL;
    // For ShiShua
    SEEDTYPE shishua_seed[4] = {static_cast<SEEDTYPE>(g_seed), 0, 0, 0};
    shishua_buf_u buf __attribute__ ((aligned (64)));
    prng_state s = prng_init(shishua_seed);

    // ShiShua test
    if (true) {
        int step = BUFSIZE >> 3, toss;
        
        // Prevent using large memory space for @buf.
        if (n_thread > MAX_N_THREAD) n_thread = MAX_N_THREAD;
        thread_tosses = (n_tosses/step) / (n_thread+1);

        // Allocate thread structure
        thread = (pthread_t *)malloc(n_thread * sizeof(pthread_t));
        args = (thread_arg_t *)malloc(n_thread * sizeof(thread_arg_t));
        // Initialize mutex.
        pthread_mutex_init(&mutex_n_circle, NULL);

        // Initialize global seed.
        g_shishua_seed = (ull *)malloc(n_thread * sizeof(ull));
        for (int i = 0; i < n_thread; ++i) {
            g_seed = ((g_seed * 1103515245U) + 12345U) & 0x7fffffff;
            g_shishua_seed[i] = g_seed;
        }

        // Create threads
        for (ull id = 0; id < n_thread; ++id) {
            args[id].thread_id = id;
            args[id].n_tosses = thread_tosses;
            pthread_create(&thread[id],
                           NULL,
                           calculate_n_circle,
                           (void *)&args[id]);
        }

        for (toss = n_thread*thread_tosses*step; toss+step < n_tosses; toss+= step) {
            prng_gen(&s, buf.buf8, sizeof(buf));
            for (int i = 0; i < (BUFSIZE>>2); i += 2) {
                x = (double)buf.buf32[i] / 0xFFFFFFFF;
                y = (double)buf.buf32[i+1] / 0xFFFFFFFF;
                distance_squared = x*x + y*y;
                if (distance_squared <= 1)
                    n_circle += 1;
            }
        }
        prng_gen(&s, buf.buf8, sizeof(buf));
        for (int i = 0; i < (n_tosses-toss)*2; i += 2) {
            x = (double)buf.buf32[i] / 0xFFFFFFFF;
            y = (double)buf.buf32[i+1] / 0xFFFFFFFF;
            distance_squared = x*x + y*y;
            if (distance_squared <= 1)
                n_circle += 1;
        }

        // Join threads
        for (int id = 0; id < n_thread; ++id) {
            pthread_join(thread[id], NULL);
        }

        pi_estimate = 4 * (g_n_circle+n_circle) / ((double)n_tosses);
        printf("%.9lf\n", pi_estimate);

        // Destroy mutex.
        pthread_mutex_destroy(&mutex_n_circle);

        free(thread);
        free(args);
        free(g_shishua_seed);

        return 0;
    }

    if (n_thread < 1) {
        seed = (ull)time(NULL);
        for (int toss = 0; toss < n_tosses; ++toss) {
            seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
            x = (double)seed / RAND_MAX;
            seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
            y = (double)seed / RAND_MAX;
            distance_squared = x*x + y*y;
            if (distance_squared <= 1)
                n_circle += 1;
        }
        pi_estimate = 4 * n_circle /((double)n_tosses);
        printf("%.9lf\n", pi_estimate);
    
        return 0;
    }

    // Allocate thread structure.
    thread = (pthread_t *)malloc(n_thread * sizeof(pthread_t));
    args = (thread_arg_t *)malloc(n_thread * sizeof(thread_arg_t));
    // Initialize mutex.
    pthread_mutex_init(&mutex_n_circle, NULL);

    g_seed = (ull)time(NULL);
    for (ull id = 0; id < n_thread; ++id) {
        args[id].thread_id = id;
        args[id].n_tosses = thread_tosses;
        pthread_create(&thread[id],
                       NULL,
                       monte_carlo_pi_estimate,
                       (void *)&args[id]);
    }

    seed = g_seed + n_thread;
    for (int toss = n_thread*thread_tosses; toss < n_tosses; ++toss) {
        seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
        x = (double)seed / RAND_MAX;
        seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
        y = (double)seed / RAND_MAX;
        distance_squared = x*x + y*y;
        if (distance_squared <= 1)
            n_circle += 1;
    }

    // Join threads.
    for (int id = 0; id < n_thread; ++id) {
        pthread_join(thread[id], NULL);
    }

    free(thread);
    free(args);

    // Destroy mutex.
    pthread_mutex_destroy(&mutex_n_circle);
    
    pi_estimate = 4 * (g_n_circle+n_circle) /((double)n_tosses);
    printf("%.9lf\n", pi_estimate);

    return 0;
}

