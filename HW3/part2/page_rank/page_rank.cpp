#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{
    int num_node = num_nodes(g);
    double equal_prob = 1.0 / num_node;

    double *inter_score = (double *)malloc(sizeof(double) * num_node);
    double lonely_sum = 0.0;
    double global_diff = 0.0;

    // For OpenMP
    int max_thread = omp_get_max_threads();
    double thread_lonely_sum[max_thread][4];
    double thread_global_diff[max_thread][4];

    // Initialize vertex weights to uniform probability.
#pragma omp parallel for
    for (int i = 0; i < num_node; ++i) {
        solution[i] = equal_prob;
    }

    do {
        lonely_sum = 0.0;
        global_diff = 0.0;
        
        // Initialize @thread_lonely_sum, @thread_global_diff
        for (int i = 0; i < max_thread; ++i) {
            thread_lonely_sum[i][0] = 0.0;
            thread_global_diff[i][0] = 0.0;
        }

        // Calculate the intermeidate score.
#pragma omp parallel
{
        int id = omp_get_thread_num();
        int n_thrds = omp_get_num_threads();

        for (int i = id; i < num_node; i += n_thrds) {
            int div = outgoing_size(g, i);

            if (div == 0) {
                thread_lonely_sum[id][0] += solution[i];
            }
            else inter_score[i] = solution[i] / div;
        }
 }

        // Get @lonely_sum
        for (int i = 0; i < max_thread; ++i) {
            lonely_sum += thread_lonely_sum[i][0];
        }
        lonely_sum *= damping / num_node;

        // Compute new score for all nodes.
#pragma omp parallel
{
        int id = omp_get_thread_num();
        int n_thrds = omp_get_num_threads();

        for (int i = id; i < num_node; i += n_thrds) {
            const Vertex *start = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);
            double sum = 0.0;

            for (const Vertex *v = start; v != end; ++v) {
                sum += inter_score[*v];
            }
            sum = (damping * sum) + (1.0-damping) / num_node + lonely_sum;
            
            thread_global_diff[id][0] += abs(solution[i] - sum);
            solution[i] = sum;
        }
}

        for (int i = 0; i < max_thread; ++i) {
            global_diff += thread_global_diff[i][0];
        }
    } while (global_diff >= convergence);

    free(inter_score);
}
