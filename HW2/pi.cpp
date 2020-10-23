#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <climits>
using namespace std;


typedef unsigned long long ull;


inline double get_rand() {
    return (double)( (((ull)rand()<<31) | rand()) & 0x1FFFFFFFFFFFFF) / 0x1FFFFFFFFFFFFF;
}


int main(int argc, char **argv) {
    clock_t start;
    ull n_circle = 0; // # of darts in circle.
    ull n_tosses = atoi(argv[2]);    // # of tosses
    double x, y, distance_squared;
    double pi_estimate = 0.0;

    // Initialize random seeds.
    srand(time(NULL));
    
    start = clock();
    for (ull toss = 0; toss < n_tosses; ++toss) {
        x = get_rand();
        y = get_rand();
        distance_squared = x*x + y*y;
        n_circle += (distance_squared <= 1);
    }
    printf("Time: %lf.\n", (clock()-start)/(double)CLOCKS_PER_SEC);

    pi_estimate = 4 * n_circle /((double)n_tosses);
    cout << setprecision(9) << pi_estimate << endl;

    return 0;
}

