#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <climits>
using namespace std;

#define SIZE 1024
typedef unsigned long long ull;


int main(int argc, char **argv) {
    clock_t start;
    ull n_circle = 0; // # of darts in circle.
    ull n_tosses = atoi(argv[2]);    // # of tosses
    float *__restrict x, *__restrict y;
    double pi_estimate;
    ull g_seed = time(NULL);
    ull cnt, now_tosses;

    start = clock();
    __builtin_assume(SIZE == 1024);
    x = (float *)malloc(SIZE * sizeof(float));
    y = (float *)malloc(SIZE * sizeof(float));
    x = (float *)__builtin_assume_aligned(x, 32);
    y = (float *)__builtin_assume_aligned(y, 32);
    for (cnt = 0; cnt+SIZE <= n_tosses; cnt += SIZE) {
        for (ull toss = 0, seed = g_seed + cnt; toss < SIZE; ++toss) {
            seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
            x[toss] = (float)seed / RAND_MAX;
            seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
            y[toss] = (float)seed / RAND_MAX;
        }
        for (ull toss = 0; toss < SIZE; ++toss) {
            float tmp1 = x[toss], tmp2 = y[toss];
            float distance_squared = tmp1*tmp1 + tmp2*tmp2;
            if (distance_squared <= 1)
                n_circle += 1;
        }
    }

    now_tosses = n_tosses - cnt;
    for (ull toss = 0, seed = g_seed + cnt; toss < now_tosses; ++toss) {
        seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
        x[toss] = (float)seed / RAND_MAX;
        seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
        y[toss] = (float)seed / RAND_MAX;
    }
    for (ull toss = 0; toss < now_tosses; ++toss) {
        float tmp1 = x[toss], tmp2 = y[toss];
        float distance_squared = tmp1*tmp1 + tmp2*tmp2;
        if (distance_squared <= 1)
            n_circle += 1;
    }
    free(x);
    free(y);
    printf("Time: %lf.\n", (clock()-start)/(double)CLOCKS_PER_SEC);

    pi_estimate = 4 * n_circle /((double)n_tosses);
    printf("%.9lf\n", pi_estimate);

    return 0;
}

