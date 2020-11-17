#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <stdint.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    uint64_t *visit,
    uint64_t *out,
    int *distances,
    const int &level,
    const uint32_t &bitmask_size)
{

#pragma omp parallel for schedule(guided, 128)
    for (int i = 0; i < frontier->count; i++) {
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if ((~visit[outgoing>>6]) & (1UL<<(outgoing&0x3F))) {
            #pragma omp atomic
                visit[outgoing>>6] |= 1UL << (outgoing&0x3F);
            #pragma omp atomic
                out[outgoing>>6] |= 1UL << (outgoing&0x3F);
            }
        }
    }

    frontier->count = 0;
    for (int i = 0; i < bitmask_size-1; i++) {
        if (out[i]) {
            for (int j = 0; j < 64; j++) {
                if (out[i] & (1UL<<j)) {
                    frontier->vertices[frontier->count++] = (i<<6) + j;
                    distances[(i<<6) + j] = level;
                }
            }
            out[i] = 0;
        }
    }

    uint32_t mask = bitmask_size - 1;

    if (out[mask]) {
        for (int j = 0; j < 64; j++) {
            if (out[mask] & (1UL<<j)) {
                frontier->vertices[frontier->count++] = (mask<<6) + j;
                distances[(mask<<6) + j] = level;
            }
        }
        out[mask] = 0;
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    int level = 1;
    const uint32_t bitmask_size = (graph->num_nodes + 63) / 64;
    uint64_t *visit = (uint64_t *)malloc(sizeof(uint64_t) * bitmask_size);
    uint64_t *out = (uint64_t *)malloc(sizeof(uint64_t) * bitmask_size);

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set *frontier = &list1;

    // initialize all nodes to NOT_VISITED
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
#pragma omp parallel for
    for (int i = 0; i < bitmask_size; i++) {
        visit[i] = out[i] = 0;
    }

    // Set the distance of ROOT_NODE_ID
    sol->distances[ROOT_NODE_ID] = 0;
    visit[ROOT_NODE_ID >> 6] |= 1UL << (ROOT_NODE_ID&0x3F);
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;

    while (frontier->count != 0) {
        top_down_step(graph, frontier, visit, out, sol->distances, level, bitmask_size);
        ++level;
    }

    free(visit);
    free(out);
}

void bottom_up_step(
    graph *g,
    int &frontier_count,
    const uint64_t *visit,
    uint64_t *out,
    int *distances,
    const int &level,
    const uint32_t &bitmask_size)
{
    int local_frontier_count = 0;

#pragma omp parallel
{
    #pragma omp for reduction(+: frontier_count) schedule(guided, 1024) nowait
    for (uint32_t mask = 0; mask < bitmask_size-1; mask++) {
        // If there exists nodes not visited
        if (~visit[mask]) {
            for (uint32_t i = 0; i < 64; i++) {
                // Check whether such node is not visited
                if ((~visit[mask]) & (1UL<<i)) {
                    uint32_t node = (mask<<6) + i;
                    int start_edge = g->incoming_starts[node];
                    int end_edge = (node == g->num_nodes - 1)
                                    ? g->num_edges
                                    : g->incoming_starts[node + 1];

                    // To see whether its parent is in the frontier
                    for(int parent = start_edge; parent < end_edge; parent++) {
                        uint32_t incoming = g->incoming_edges[parent];

                        // Add the node into new frontier if its parent is already in
                        // the frontier
                        if(visit[incoming>>6] & (1UL<<(incoming&0x3F))) {
                            distances[node] = level;
                            out[node>>6] |= 1UL << (node&0x3F);
                            ++local_frontier_count;
                            break;
                        }
                    }
                }
            }
        }
    }

#pragma omp master
{
    uint32_t mask = bitmask_size - 1;

    if (~visit[mask]) {
        for (uint32_t i = 0; i < 64; i++) {
            uint32_t node = (mask<<6) + i;
            if (node >= g->num_nodes) break;

            if ((~visit[mask]) & (1UL<<i)) {
                int start_edge = g->incoming_starts[node];
                int end_edge = (node == g->num_nodes - 1)
                                ? g->num_edges
                                : g->incoming_starts[node + 1];

                // To see whether its parent is in the frontier
                for(int parent = start_edge; parent < end_edge; parent++) {
                    uint32_t incoming = g->incoming_edges[parent];

                    // Add the node into new frontier if its parent is already in
                    // the frontier
                    if(visit[incoming>>6] & (1UL<<(incoming&0x3F))) {
                        distances[node] = level;
                        out[node>>6] |= 1UL << (node&0x3F);
                        ++local_frontier_count;
                        break;
                    }
                }
            }
        }
    }
}

} // OpenMP End

    frontier_count = local_frontier_count;
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    int frontier_count = 1, level = 1;
    const uint32_t bitmask_size = (graph->num_nodes + 63) / 64;
    uint64_t *visit = (uint64_t *)malloc(sizeof(uint64_t) * bitmask_size);
    uint64_t *out = (uint64_t *)malloc(sizeof(uint64_t) * bitmask_size);

    // initialize all nodes to NOT_VISITED
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
#pragma omp parallel for
    for (int i = 0; i < bitmask_size; i++) {
        visit[i] = out[i] = 0;
    }

    // Set the distance of ROOT_NODE_ID
    sol->distances[ROOT_NODE_ID] = 0;
    visit[ROOT_NODE_ID >> 6] |= 1UL << (ROOT_NODE_ID&0x3F);

    while (frontier_count != 0) {
        frontier_count = 0;
        bottom_up_step(graph, frontier_count, visit, out, sol->distances, level, bitmask_size);
    #pragma omp parallel for
        for (int i = 0; i < bitmask_size; i++) {
            visit[i] |= out[i];
        }
        ++level;
    }

    free(visit);
    free(out);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
