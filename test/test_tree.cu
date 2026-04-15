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

void print_gpu_tree(int *h_children, float *h_x, float *h_y, float *h_z, int gpu_next_cell, int N, int max_nodes){
    for (int node = max_nodes - 1; node > gpu_next_cell; node--) 
    {
        cout << "\n--- Node " << node << " ---\n";
        for (int c = 0; c < OCT_CHILDREN; c++) {
            int child = h_children[node * OCT_CHILDREN + c];
            cout << "  child[" << c << "] = " << child;
            if (child == NULL_VAL_INT) cout << " (null)";
            else if (child >= 0 && child < N) cout << " (body): (" << h_x[child] << ", " << h_y[child] << ", " << h_z[child] << " )";
            else if (child >= N) cout << " (internal)";
            else if (child == LOCK_VAL) cout << " (LOCK!)";
            cout << "\n";
        }
    }    
}


// ======== main ========
int main(int argc, char **argv) {
    // 1. Read input
    // const char *fname = (argc > 1) ? argv[1] : "test_traces/random.txt";
    const char *fname = (argc > 1) ? argv[1] : "test_traces/test_50000.txt";
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
    float *d_x, *d_y, *d_z, *d_mass, *d_root_half;
    int *d_blk_counter;
    float *d_minx, *d_miny, *d_minz, *d_maxx, *d_maxy, *d_maxz;
    int *d_children, *d_next_cell;

    cudaMalloc(&d_x, max_nodes * sizeof(float));
    cudaMalloc(&d_y, max_nodes * sizeof(float));
    cudaMalloc(&d_z, max_nodes * sizeof(float));
    cudaMalloc(&d_mass, max_nodes * sizeof(float));
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
    
    // ================= TEST BOUNDING BOX ================= 
    body_reduce_kernel<<<grid, block>>>(d_x, d_y, d_z, N, d_root_half, d_blk_counter,
        d_minx, d_miny, d_minz, d_maxx, d_maxy, d_maxz);
    cudaDeviceSynchronize();

    float gpu_root_half;
    cudaMemcpy(&gpu_root_half, d_root_half, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "GPU root_half = " << gpu_root_half << "\n";

    float cpu_root_half = compute_box(bodys);
    cout << "CPU box length = " << cpu_root_half << "\n";
    assert (std::abs(gpu_root_half - cpu_root_half) < 1e-3);


    // ================= TEST BUILD TREE ================= 
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


    // CPU: build tree
    OctTreeNode *cpu_root = build_tree(bodys, cpu_root_half);

    // TEST 1: every body appears exactly once as a leaf
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


    // ================= TEST 4: Center of Mass =================
    cout << "\n=== TESTING Center of Mass ===\n";

    // need to upload mass array (bodies + init internal to -1)
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = N; i < max_nodes; i++) h_mass[i] = NULL_VAL_FLOAT;
    cudaMemcpy(d_mass, h_mass, max_nodes * sizeof(float), cudaMemcpyHostToDevice);

    compute_cmass_kernelv1<<<grid, block>>>(d_x, d_y, d_z, d_mass, d_children,
        gpu_next_cell + 1, max_nodes - 1, N);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "cmass kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // copy back
    cudaMemcpy(h_x, d_x, max_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, max_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, max_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mass, d_mass, max_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_root_cx = h_x[max_nodes - 1];
    float gpu_root_cy = h_y[max_nodes - 1];
    float gpu_root_cz = h_z[max_nodes - 1];
    float gpu_root_mass = h_mass[max_nodes - 1];

    float cpu_root_cx = cpu_root->cntr_mass.pt.x;
    float cpu_root_cy = cpu_root->cntr_mass.pt.y;
    float cpu_root_cz = cpu_root->cntr_mass.pt.z;
    float cpu_root_mass = cpu_root->cntr_mass.mass;

    cout << "\n=== TEST 5: Root center of mass ===\n";
    cout << "  GPU root: cmass=(" << gpu_root_cx << ", " << gpu_root_cy << ", " << gpu_root_cz
        << ")  mass=" << gpu_root_mass << "\n";
    cout << "  CPU root: cmass=(" << cpu_root_cx << ", " << cpu_root_cy << ", " << cpu_root_cz
        << ")  mass=" << cpu_root_mass << "\n";

    float rel_tol = 1e-3f; // 0.1%
    bool mass_ok = fabsf(gpu_root_mass - cpu_root_mass) / cpu_root_mass < rel_tol;
    bool cx_ok = fabsf(gpu_root_cx - cpu_root_cx) / (fabsf(cpu_root_cx) + 1e-6f) < rel_tol;
    bool cy_ok = fabsf(gpu_root_cy - cpu_root_cy) / (fabsf(cpu_root_cy) + 1e-6f) < rel_tol;
    bool cz_ok = fabsf(gpu_root_cz - cpu_root_cz) / (fabsf(cpu_root_cz) + 1e-6f) < rel_tol;

    if (mass_ok && cx_ok && cy_ok && cz_ok)
        cout << "  PASS\n";
    else {
        if (!mass_ok) cout << "  FAIL: mass diff = " << fabsf(gpu_root_mass - cpu_root_mass) << "\n";
        if (!cx_ok)   cout << "  FAIL: cx diff = " << fabsf(gpu_root_cx - cpu_root_cx) << "\n";
        if (!cy_ok)   cout << "  FAIL: cy diff = " << fabsf(gpu_root_cy - cpu_root_cy) << "\n";
        if (!cz_ok)   cout << "  FAIL: cz diff = " << fabsf(gpu_root_cz - cpu_root_cz) << "\n";
    }    


// ================= TEST 6: Force Computation =================
    cout << "\n=== TEST 6: Force Computation ===\n";
    assert ((G == G_GPU) && (EPS == EPS_GPU));
    // allocate force arrays
    float *d_Fx, *d_Fy, *d_Fz;
    cudaMalloc(&d_Fx, N * sizeof(float));
    cudaMalloc(&d_Fy, N * sizeof(float));
    cudaMalloc(&d_Fz, N * sizeof(float));
    cudaMemset(d_Fx, 0, N * sizeof(float));
    cudaMemset(d_Fy, 0, N * sizeof(float));
    cudaMemset(d_Fz, 0, N * sizeof(float));

    compute_forces_kernelv1<<<grid, block>>>(d_x, d_y, d_z, d_mass, d_children,
        N, max_nodes, gpu_root_half, d_Fx, d_Fy, d_Fz, 0.5);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "forces kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // copy back forces
    float *h_Fx = new float[N];
    float *h_Fy = new float[N];
    float *h_Fz = new float[N];
    cudaMemcpy(h_Fx, d_Fx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fy, d_Fy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fz, d_Fz, N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU: compute forces via tree traversal
    vector<Float3> cpu_forces(N, {0.0f, 0.0f, 0.0f});
    traverse_tree(cpu_root, bodys, cpu_forces, 0.5);

    // compare first 5 bodies
    int num_check = min(5, N);
    float force_rel_tol = 0.05f; // 5% relative tolerance
    int force_pass = 0;
    int force_fail = 0;

    for (int i = 0; i < num_check; i++) {
        float gpu_fx = h_Fx[i], gpu_fy = h_Fy[i], gpu_fz = h_Fz[i];
        float cpu_fx = cpu_forces[i].x, cpu_fy = cpu_forces[i].y, cpu_fz = cpu_forces[i].z;

        float rel_fx = fabsf(gpu_fx - cpu_fx) / (fabsf(cpu_fx) + 1e-6f);
        float rel_fy = fabsf(gpu_fy - cpu_fy) / (fabsf(cpu_fy) + 1e-6f);
        float rel_fz = fabsf(gpu_fz - cpu_fz) / (fabsf(cpu_fz) + 1e-6f);

        cout << "  Body " << i << ":\n";
        cout << "    GPU: Fx=" << gpu_fx << "  Fy=" << gpu_fy << "  Fz=" << gpu_fz << "\n";
        cout << "    CPU: Fx=" << cpu_fx << "  Fy=" << cpu_fy << "  Fz=" << cpu_fz << "\n";
        cout << "    Rel err: Fx=" << rel_fx << "  Fy=" << rel_fy << "  Fz=" << rel_fz << "\n";

        if (rel_fx < force_rel_tol && rel_fy < force_rel_tol && rel_fz < force_rel_tol)
            force_pass++;
        else {
            force_fail++;
            cout << "    ** ABOVE TOLERANCE **\n";
        }
    }

    cout << "\n  Pass: " << force_pass << " / " << num_check << "\n";
    cout << "  Fail: " << force_fail << "\n";
    if (force_fail == 0)
        cout << "  PASS\n";
    else
        cout << "  FAIL\n";


// ================= TEST 7: Apply Forces (Position Update) =================
    cout << "\n=== TEST 7: Position Update ===\n";

    float dt = 0.1f;

    // GPU: allocate and init velocities to 0
    float *d_Vx, *d_Vy, *d_Vz;
    cudaMalloc(&d_Vx, N * sizeof(float));
    cudaMalloc(&d_Vy, N * sizeof(float));
    cudaMalloc(&d_Vz, N * sizeof(float));
    cudaMemset(d_Vx, 0, N * sizeof(float));
    cudaMemset(d_Vy, 0, N * sizeof(float));
    cudaMemset(d_Vz, 0, N * sizeof(float));

    // forces already on device from TEST 6
    apply_forces_kernel<<<grid, block>>>(d_x, d_y, d_z, d_mass,
        d_Vx, d_Vy, d_Vz, d_Fx, d_Fy, d_Fz, N, dt);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "apply forces kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // copy back updated positions
    float *h_gpu_x = new float[N];
    float *h_gpu_y = new float[N];
    float *h_gpu_z = new float[N];
    cudaMemcpy(h_gpu_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU: apply forces with same dt, starting from original positions
    vector<Float3> cpu_vels(N, {0.0f, 0.0f, 0.0f});
    vector<Body> cpu_bodys_copy = bodys;
    update_points(cpu_bodys_copy, cpu_vels, cpu_forces, dt);

    // compare first 5 bodies
    int pos_check = min(5, N);
    float pos_rel_tol = 0.05f;
    int pos_pass = 0;

    for (int i = 0; i < pos_check; i++) {
        float gx = h_gpu_x[i], gy = h_gpu_y[i], gz = h_gpu_z[i];
        float cx = cpu_bodys_copy[i].pt.x, cy = cpu_bodys_copy[i].pt.y, cz = cpu_bodys_copy[i].pt.z;

        assert(!isnan(gx) && !isnan(gy) && !isnan(gz) && "GPU position is NaN");
        assert(!isinf(gx) && !isinf(gy) && !isinf(gz) && "GPU position is Inf");

        float rel_x = fabsf(gx - cx) / (fabsf(cx) + 1e-6f);
        float rel_y = fabsf(gy - cy) / (fabsf(cy) + 1e-6f);
        float rel_z = fabsf(gz - cz) / (fabsf(cz) + 1e-6f);

        cout << "  Body " << i << ":\n";
        cout << "    GPU: (" << gx << ", " << gy << ", " << gz << ")\n";
        cout << "    CPU: (" << cx << ", " << cy << ", " << cz << ")\n";
        cout << "    Rel err: x=" << rel_x << "  y=" << rel_y << "  z=" << rel_z << "\n";

        assert(rel_x < pos_rel_tol && "x position relative error too large");
        assert(rel_y < pos_rel_tol && "y position relative error too large");
        assert(rel_z < pos_rel_tol && "z position relative error too large");

        pos_pass++;
    }

    cout << "\n  Pass: " << pos_pass << " / " << pos_check << "\n";
    cout << "  PASS\n";

    delete[] h_gpu_x; delete[] h_gpu_y; delete[] h_gpu_z;
    cudaFree(d_Vx); cudaFree(d_Vy); cudaFree(d_Vz);
        
    delete[] h_Fx; delete[] h_Fy; delete[] h_Fz;
    cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);    

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
