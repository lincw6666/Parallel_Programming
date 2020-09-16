#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <climits>
using namespace std;


typedef unsigned long long ull;


inline double get_rand() {
    return ((double)( (((ull)rand()<<22)|(rand()&0x3FFFFE)) ) / 0xFFFFFFFFFFFFF) - 1;
}


int main() {
    clock_t start;
    ull n_circle = 0; // # of darts in circle.
    ull n_tosses = 2e9;    // # of tosses
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

