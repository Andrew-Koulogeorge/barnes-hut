/*
Testing K1: build_tree_kernel
Compares GPU octree structure against CPU octree structure.

Strategy:
  - Build tree on GPU, copy children array back to host
  - Build tree on CPU using reference implementation
  - For each body, trace its octant path in both trees and compare
  - Verify every body appears exactly once as a leaf in the GPU tree
*/

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <utility>
#include "bhut_cpu.h"
#include "kernels.cuh"
using namespace std;

// ======== GPU tree verification helpers ========

/*
  Walk the GPU children array from root to find which leaf slot body_id lands in.
  Records the octant path taken (sequence of 0-7 choices).
  Returns true if body was found in the tree.
  NOTE: ensure that traversal logic in gpu and cpu is the same
*/
bool trace_body_gpu(int body_id, const int *children, int max_nodes, int N,
                    const float *h_x, const float *h_y, const float *h_z,
                    float root_half, vector<int> &path) {
    path.clear();
    int node = max_nodes - 1; // root
    float3 box_center = {0.0f, 0.0f, 0.0f};
    float half = root_half;

    for (int depth = 0; depth < DEPTH_LIMIT; depth++) {
        // compute which octant this body belongs in
        int oct = 0;
        if (box_center.x <= h_x[body_id]) oct |= 1;
        if (box_center.y <= h_y[body_id]) oct |= 2;
        if (box_center.z <= h_z[body_id]) oct |= 4;
        path.push_back(oct);    
        int child = children[node * OCT_CHILDREN + oct];

        // found the body as a leaf
        if (child == body_id) return true;

        // null or different body = not found on this path
        if (child == NULL_VAL_INT) return false;
        if (child < N && child != body_id) return false;

        // internal node: descend
        if (child >= N) {
            half *= 0.5f;
            box_center.x += (oct & 1 ? half : -half);
            box_center.y += (oct & 2 ? half : -half);
            box_center.z += (oct & 4 ? half : -half);
            node = child;
        }
    }
    return false;
}

/*
  Walk the CPU tree from root to find which leaf slot a body lands in.
  Records the octant path taken.
  Returns true if body was found.
*/
bool trace_body_cpu(const Body &body, OctTreeNode *node, vector<int> &path) {
    path.clear();
    while (node != nullptr) {
        if (node->type == LEAF) {
            // check if this leaf holds our body (by position match)
            float dx = fabsf(node->cntr_mass.pt.x - body.pt.x);
            float dy = fabsf(node->cntr_mass.pt.y - body.pt.y);
            float dz = fabsf(node->cntr_mass.pt.z - body.pt.z);
            return (dx < 1e-2f && dy < 1e-2f && dz < 1e-2f);
        }
        if (node->type == EMPTY) return false;

        // internal node: figure out which octant
        Body b_copy = body;  // get_idx takes non-const ref
        int oct = get_idx(b_copy, node->loc);
        path.push_back(oct);
        node = node->children[oct];
    }
    return false;
}

int max_depth_gpu(const int *children, int max_nodes, int N) {
    int max_d = 0;
    // stack stores (node_index, depth) pairs
    vector<pair<int,int>> stack;
    stack.push_back({max_nodes - 1, 0}); // root

    while (!stack.empty()) {
        auto [node, depth] = stack.back();
        stack.pop_back();
        max_d = max(max_d, depth);

        for (int c = 0; c < OCT_CHILDREN; c++) {
            int child = children[node * OCT_CHILDREN + c];
            // only descend into internal nodes
            if (child >= N) {
                stack.push_back({child, depth + 1});
            }
        }
    }
    return max_d+1;
}

// CPU: recursive max depth
int max_depth_cpu(OctTreeNode *node, int depth) {
    if (node == nullptr || node->type == EMPTY) return depth;
    if (node->type == LEAF) return depth;
    int max_d = depth;
    for (int i = 0; i < NUM_CHILDREN; i++) {
        max_d = max(max_d, max_depth_cpu(node->children[i], depth + 1));
    }
    return max_d;
}

int cpu_internal_nodes(OctTreeNode *node){
    int num_int = 1; // fn is only called on internal
    for (int i = 0; i < NUM_CHILDREN; i++) {
        if (node->children[i] != nullptr && node->children[i]->type == INTERNAL)
            num_int += cpu_internal_nodes(node->children[i]);
    }    
    return num_int;
}


/*
  Count how many times each body index appears as a leaf in the GPU tree.
  Walks the entire children array.
*/
void count_leaves(const int *children, int max_nodes, int N, int first_cell, vector<int> &leaf_count) {
    leaf_count.assign(N, 0);
    // scan all allocated nodes (from the lowest cell up to root)
    for (int node = first_cell; node < max_nodes; node++) {
        for (int i = 0; i < OCT_CHILDREN; i++) {
            int child = children[node * OCT_CHILDREN + i];
            if (child >= 0 && child < N) {
                leaf_count[child]++;
            }
        }
    }
}

void print_cpu_tree(OctTreeNode *node, int id, int N) {
    if (node == nullptr) return;
    if (node->type == EMPTY) return;

    if (node->type == INTERNAL) {
        cout << "\n--- Node " << id << " ---\n";
        for (int c = 0; c < NUM_CHILDREN; c++) {
            cout << "  child[" << c << "] = ";
            if (node->children[c] == nullptr || node->children[c]->type == EMPTY) {
                cout << "-1 (null)";
            } else if (node->children[c]->type == LEAF) {
                // print position so you can match to body index
                cout << "LEAF (body @ "
                     << node->children[c]->cntr_mass.pt.x << ", "
                     << node->children[c]->cntr_mass.pt.y << ", "
                     << node->children[c]->cntr_mass.pt.z << ")";
            } else {
                cout << "INTERNAL";
            }
            cout << "\n";
        }
        // recurse into internal children
        for (int c = 0; c < NUM_CHILDREN; c++) {
            if (node->children[c] != nullptr && node->children[c]->type == INTERNAL) {
                print_cpu_tree(node->children[c], id * 10 + c, N);
            }
        }
    }
}

// ======== main ========
int main(int argc, char **argv) {
    // 1. Read input
    const char *fname = (argc > 1) ? argv[1] : "test_traces/test_500000.txt";
    ifstream file(fname);
    if (!file) { cerr << "Cannot open " << fname << "\n"; return 1; }

    vector<Body> bodys;
    float bx, by, bz, bm;
    while (file >> bx >> by >> bz >> bm)
        bodys.emplace_back(Float3{bx, by, bz}, bm);
    int N = bodys.size();
    int max_nodes = N * MULT;
    cout << "Loaded " << N << " bodies, max_nodes = " << max_nodes << "\n";

    // 2. Prepare host arrays
    float *h_x = new float[max_nodes];
    float *h_y = new float[max_nodes];
    float *h_z = new float[max_nodes];
    float *h_mass = new float[max_nodes];
    for (int i = 0; i < N; i++) {
        h_x[i] = bodys[i].pt.x;
        h_y[i] = bodys[i].pt.y;
        h_z[i] = bodys[i].pt.z;
        h_mass[i] = bodys[i].mass;
    }

    // =============================================
    // GPU: compute bounding box then build tree
    // =============================================
    float *d_x, *d_y, *d_z, *d_root_half;
    int *d_blk_counter;
    float *d_minx, *d_miny, *d_minz, *d_maxx, *d_maxy, *d_maxz;
    int *d_children, *d_next_cell;

    cudaMalloc(&d_x, max_nodes * sizeof(float));
    cudaMalloc(&d_y, max_nodes * sizeof(float));
    cudaMalloc(&d_z, max_nodes * sizeof(float));
    cudaMalloc(&d_root_half, sizeof(float));
    cudaMalloc(&d_blk_counter, sizeof(int));
    cudaMalloc(&d_minx, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_miny, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_minz, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_maxx, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_maxy, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_maxz, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_children, OCT_CHILDREN * max_nodes * sizeof(int));
    cudaMalloc(&d_next_cell, sizeof(int));

    // copy body positions to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);

    // init bounding box kernel inputs
    cudaMemset(d_root_half, 0, sizeof(float));
    cudaMemset(d_blk_counter, 0, sizeof(int));

    // init children to NULL_VAL_INT (-1)
    cudaMemset(d_children, 0xFF, OCT_CHILDREN * max_nodes * sizeof(int));

    // init next_cell to max_nodes - 1 (root is at top, cells allocated downward)
    // root is at max_nodes-1, first new cell will be max_nodes-2
    int h_next_cell = max_nodes - 2;
    cudaMemcpy(d_next_cell, &h_next_cell, sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_SIZE);

    // bounding box (already tested to be correct)
    body_reduce_kernel<<<grid, block>>>(d_x, d_y, d_z, N, d_root_half, d_blk_counter,
        d_minx, d_miny, d_minz, d_maxx, d_maxy, d_maxz);
    cudaDeviceSynchronize();

    float gpu_root_half;
    cudaMemcpy(&gpu_root_half, d_root_half, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "GPU root_half = " << gpu_root_half << "\n";

    // K1: build tree
    build_tree_kernel<<<grid, block>>>(d_x, d_y, d_z, d_children, d_next_cell,
        N, max_nodes, gpu_root_half, DEPTH_LIMIT);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // copy children array back to host
    int *h_children = new int[OCT_CHILDREN * max_nodes];
    cudaMemcpy(h_children, d_children, OCT_CHILDREN * max_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    int gpu_next_cell;
    cudaMemcpy(&gpu_next_cell, d_next_cell, sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_num_internal_nodes = max_nodes - (gpu_next_cell+1);
    cout << "root index: " << max_nodes - 1 << " \n";
    cout << "next cell to allocate : " << gpu_next_cell << " \n";

    // print children of each internal node, root first (backward)
    // for (int node = max_nodes - 1; node > gpu_next_cell; node--) {
    //     cout << "\n--- Node " << node << " ---\n";
    //     for (int c = 0; c < OCT_CHILDREN; c++) {
    //         int child = h_children[node * OCT_CHILDREN + c];
    //         cout << "  child[" << c << "] = " << child;
    //         if (child == NULL_VAL_INT) cout << " (null)";
    //         else if (child >= 0 && child < N) cout << " (body): (" << h_x[child] << ", " << h_y[child] << ", " << h_z[child] << " )";
    //         else if (child >= N) cout << " (internal)";
    //         else if (child == LOCK_VAL) cout << " (LOCK!)";
    //         cout << "\n";
    //     }
    // }


    // =============================================
    // CPU: build tree
    // =============================================
    float cpu_box = compute_box(bodys);
    cout << "CPU box length = " << cpu_box << "\n";
    OctTreeNode *cpu_root = build_tree(bodys, cpu_box);
    
    // cout << "\n=== CPU Tree ===\n";
    // print_cpu_tree(cpu_root, 0, N);

    // =============================================
    // TEST 1: every body appears exactly once as a leaf
    // =============================================
    cout << "\n=== TEST 1: Leaf count check ===\n";
    vector<int> leaf_count;
    count_leaves(h_children, max_nodes, N, gpu_next_cell+1, leaf_count);

    int missing = 0, duplicates = 0;
    for (int i = 0; i < N; i++) {
        if (leaf_count[i] == 0) missing++;
        if (leaf_count[i] > 1) duplicates++;
    }
    assert(leaf_count.size() == N);
    cout << " Number of leafs : " << leaf_count.size() << "\n";
    cout << "  Missing bodies : " << missing << "\n";
    cout << "  Duplicate bodies: " << duplicates << "\n";
    if (missing == 0 && duplicates == 0)
        cout << "  PASS\n";
    else
        cout << "  FAIL\n";


    // TEST 1.5: ensure depths match
    int gpu_depth = max_depth_gpu(h_children, max_nodes, N);
    int cpu_depth = max_depth_cpu(cpu_root, 0);

    cout << "\n=== TEST 1.5: Tree Depth ===\n";
    cout << "GPU max depth: " << gpu_depth << "\n";
    cout << "CPU max depth: " << cpu_depth << "\n";

    // TEST 1.75: ensure the number of internal nodes is the same
    int cpu_internal_count = cpu_internal_nodes(cpu_root);
    cout << "\n=== TEST 1.75: Internal Node Count ===\n";
    cout << "GPU: " << gpu_num_internal_nodes << "\n";
    cout << "CPU: " << cpu_internal_count << "\n";
    
    // // =============================================
    // // TEST 2: octant paths match between GPU and CPU
    // // =============================================
    cout << "\n=== TEST 2: Octant path comparison ===\n";
    int path_matches = 0;
    int path_mismatches = 0;
    int gpu_not_found = 0;
    int cpu_not_found = 0;
    for (int i = 0; i < N; i++) {
        vector<int> gpu_path, cpu_path;

        bool found_gpu = trace_body_gpu(i, h_children, max_nodes, N,
            h_x, h_y, h_z, gpu_root_half, gpu_path);
        
        bool found_cpu = trace_body_cpu(bodys[i], cpu_root, cpu_path);
        if (!found_gpu) { gpu_not_found++; continue; }
        if (!found_cpu) { cpu_not_found++; continue; }

        if (gpu_path == cpu_path)
            path_matches++;
        else
            path_mismatches++;
    }

    cout << "  Path matches   : " << path_matches << " / " << N << "\n";
    cout << "  Path mismatches: " << path_mismatches << "\n";
    cout << "  GPU not found  : " << gpu_not_found << "\n";
    cout << "  CPU not found  : " << cpu_not_found << "\n";

    if (path_mismatches == 0 && gpu_not_found == 0 && cpu_not_found == 0)
        cout << "  PASS\n";
    else
        cout << "  FAIL\n";

    // =============================================
    // TEST 3: no invalid indices in children array
    // =============================================
    cout << "\n=== TEST 3: Children array validity ===\n";
    int invalid_count = 0;
    int lock_count = 0;
    // only check allocated nodes (gpu_next_cell+1 up to max_nodes-1)
    for (int node = gpu_next_cell + 1; node < max_nodes; node++) {
        for (int c = 0; c < OCT_CHILDREN; c++) {
            int child = h_children[node * OCT_CHILDREN + c];
            if (child == LOCK_VAL) {
                lock_count++;
            } else if (child != NULL_VAL_INT) {
                // valid child should be a body [0, N) or an internal node [N, max_nodes)
                if (child < 0 || child >= max_nodes) {
                    invalid_count++;
                }
            }
        }
    }
    cout << "  Invalid indices: " << invalid_count << "\n";
    cout << "  Stale locks    : " << lock_count << "\n";
    if (invalid_count == 0 && lock_count == 0)
        cout << "  PASS\n";
    else
        cout << "  FAIL\n";

    // =============================================
    // Summary
    // =============================================
    cout << "\n=== Summary ===\n";
    bool all_pass = (missing == 0 && duplicates == 0 &&
                     path_mismatches == 0 && gpu_not_found == 0 && cpu_not_found == 0 &&
                     invalid_count == 0 && lock_count == 0);
    cout << (all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";

    // cleanup
    free_tree(cpu_root, 0);
    delete[] h_x; delete[] h_y; delete[] h_z; delete[] h_mass;
    delete[] h_children;
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_root_half); cudaFree(d_blk_counter);
    cudaFree(d_minx); cudaFree(d_miny); cudaFree(d_minz);
    cudaFree(d_maxx); cudaFree(d_maxy); cudaFree(d_maxz);
    cudaFree(d_children); cudaFree(d_next_cell);
    return 0;
}
