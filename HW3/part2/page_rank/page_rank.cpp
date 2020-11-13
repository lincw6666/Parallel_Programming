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

    double *score_old = (double *)malloc(sizeof(double) * num_node);
    int *lonely_nodes = (int *)malloc(sizeof(int) * num_node);
    int lonely_nodes_size = 0;
    double lonely_sum = 0.0;
    double global_diff = 0.0;

    // Initialize vertex weights to uniform probability.
    for (int i = 0; i < num_node; ++i) {
        solution[i] = equal_prob;
        
        // Find nodes with no incoming nodes
        if (outgoing_size(g, i) == 0) {
            lonely_nodes[lonely_nodes_size++] = i;
        }
    }

    do {
        lonely_sum = 0.0;
        global_diff = 0.0;
        
        // Update @score_old.
        for (int i = 0; i < num_node; ++i) {
            score_old[i] = solution[i];
        }

        // Sum up the score of nodes without outgoing edges.
        for (int i = 0; i < lonely_nodes_size; ++i) {
            lonely_sum += damping * score_old[lonely_nodes[i]] / num_node;
        }

        // Compute new score for all nodes.
        for (int i = 0; i < num_node; ++i) {
            const Vertex *start = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);
            double sum = 0.0;

            for (const Vertex *v = start; v != end; ++v) {
                sum += score_old[*v] / outgoing_size(g, *v);
            }
            sum = (damping * sum) + (1.0-damping) / num_node;
            
            solution[i] = sum + lonely_sum;
        }

        for (int i = 0; i < num_node; ++i) {
            global_diff += abs(solution[i] - score_old[i]);
        }
    } while (global_diff >= convergence);

    free(score_old);
    free(lonely_nodes);
}
