#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <climits>
using namespace std;


typedef unsigned long long ull;


inline double get_rand() {
    // return (double)( (((ull)rand()<<31) | rand()) & 0x1FFFFFFFFFFFFF) / 0x1FFFFFFFFFFFFF;
    // return (double) ((ull)rand()<<22 | rand()) / 0x1FFFFFFFFFFFFF;
    return (double) rand() / RAND_MAX;
}


int main(int argc, char **argv) {
    clock_t start;
    ull n_circle = 0; // # of darts in circle.
    ull n_tosses = atoi(argv[2]);    // # of tosses
    double x, y, distance_squared;
    double pi_estimate = 0.0;

    // Initialize random seeds.
        
    start = clock();
    for (ull toss = 0, seed = time(NULL); toss < n_tosses; ++toss) {
        seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
        x = (double)seed / RAND_MAX;
        seed = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
        y = (double)seed / RAND_MAX;
        distance_squared = x*x + y*y;
        n_circle += (distance_squared <= 1);
    }
    printf("Time: %lf.\n", (clock()-start)/(double)CLOCKS_PER_SEC);

    pi_estimate = 4 * n_circle /((double)n_tosses);
    printf("%.9lf\n", pi_estimate);

    return 0;
}

