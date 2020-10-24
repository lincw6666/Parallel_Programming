#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <pthread.h>    // For Pthread
using namespace std;


typedef unsigned long long ull;


typedef struct thread_arg {
    ull thread_id;
    ull n_tosses;
} thread_arg_t;


const ull g_seed = time(NULL);
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


int main(int argc, char **argv) {
    clock_t start;
    ull n_circle = 0; // # of darts in circle.
    int n_thread = atoi(argv[1]);
    ull n_tosses = atoi(argv[2]);    // # of tosses
    ull thread_tosses = n_tosses / (n_thread+1);
    ull seed;
    double x, y, distance_squared;
    double pi_estimate = 0.0;
    pthread_t thread[n_thread];
    thread_arg_t args[n_thread];

    start = clock();
    
    // Initialize mutex.
    pthread_mutex_init(&mutex_n_circle, NULL);
    // Initialize thread attribute.
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int id = 0; id < n_thread; ++id) {
        args[id].thread_id = id;
        args[id].n_tosses = thread_tosses;
        pthread_create(&thread[id],
                       &attr,
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
    // Destroy attribute.
    pthread_attr_destroy(&attr);

    // Join threads.
    for (int id = 0; id < n_thread; ++id) {
        pthread_join(thread[id], NULL);
    }

    // Destroy mutex.
    pthread_mutex_destroy(&mutex_n_circle);
    
    printf("Time: %lf.\n", (clock()-start)/(double)CLOCKS_PER_SEC);

    pi_estimate = 4 * (g_n_circle+n_circle) /((double)n_tosses);
    printf("%.9lf\n", pi_estimate);

    return 0;
}

