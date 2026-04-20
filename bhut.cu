/*
Barnes-Hut implementation in CUDA
@author: Andrew Koulogeorge
*/

#include <iostream>
#include <fstream>
#include <limits>
#include <cfloat>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels.cuh"
using namespace std;

// Per-kernel GPU timing results (milliseconds) for one call to barnes_hut_cuda.
struct KernelTimes {
    float body_reduce_ms    = 0.f;
    float build_tree_ms     = 0.f;
    float compute_cmass_ms  = 0.f;
    float compute_forces_ms = 0.f;
    float apply_forces_ms   = 0.f;
};

// Helper: synchronize stop event then return elapsed GPU time in ms.
static inline float event_ms(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}


__global__ void slow_compute_force_bf(float *x, float *y, float *z, float *mass,
                                  float *Fx, float *Fy, float *Fz, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride) {
        float t_x = x[i], t_y = y[i], t_z = z[i], t_m = mass[i];
        float fx = 0, fy = 0, fz = 0;
        // loop over all other bodies (no approx)
        for (int j = 0; j < N; j++) {
            float dx = x[j] - t_x;
            float dy = y[j] - t_y;
            float dz = z[j] - t_z;
            float r2 = dx * dx + dy * dy + dz * dz;
            float d = sqrtf(r2 + EPS_GPU);
            float F = G_GPU * mass[j] * t_m / (r2 + EPS_GPU);
            fx += F * dx / d;
            fy += F * dy / d;
            fz += F * dz / d;
        }
        Fx[i] = fx;
        Fy[i] = fy;
        Fz[i] = fz;
    }
}


__global__ void compute_force_bf(float *x, float *y, float *z, float *mass,
                                  float *Fx, float *Fy, float *Fz, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    int tid = threadIdx.x;

    // shared memory tiles for a block's worth of bodies
    __shared__ float s_x[BLOCK_SIZE];
    __shared__ float s_y[BLOCK_SIZE];
    __shared__ float s_z[BLOCK_SIZE];
    __shared__ float s_mass[BLOCK_SIZE];

    for (int i = idx; i < N; i += stride) {
        float t_x = x[i], t_y = y[i], t_z = z[i], t_m = mass[i];
        float fx = 0, fy = 0, fz = 0;

        // walk through all bodies one tile at a time
        for (int tile_start = 0; tile_start < N; tile_start += BLOCK_SIZE) {
            // cooperatively loads assigned body into shared memory
            int load_idx = tile_start + tid;
            if (load_idx < N) {
                s_x[tid] = x[load_idx];
                s_y[tid] = y[load_idx];
                s_z[tid] = z[load_idx];
                s_mass[tid] = mass[load_idx];
            }
            __syncthreads();  // wait for all threads to finish loading

            //  compute interactions against this tile using shared memory
            int tile_end = min(BLOCK_SIZE, N - tile_start);
            for (int k = 0; k < tile_end; k++) {
                float dx = s_x[k] - t_x;
                float dy = s_y[k] - t_y;
                float dz = s_z[k] - t_z;
                float r2 = dx * dx + dy * dy + dz * dz;
                float d = sqrtf(r2 + EPS_GPU);
                float F = G_GPU * s_mass[k] * t_m / (r2 + EPS_GPU);
                fx += F * dx / d;
                fy += F * dy / d;
                fz += F * dz / d;
            }
            __syncthreads();  // wait to ensure no overwriting shared memory with next tile
        }

        Fx[i] = fx;
        Fy[i] = fy;
        Fz[i] = fz;
    }
}

/*
Brute force N^2 on GPU. Returns forces in host arrays.
*/
void brute_force_cuda(vector<float4> &bodys, vector<float3> &velocitys, float dt,
                      float *h_Fx, float *h_Fy, float *h_Fz) {
    int N = bodys.size();

    float *h_x = new float[N], *h_y = new float[N], *h_z = new float[N], *h_mass = new float[N];
    float *h_vx = new float[N], *h_vy = new float[N], *h_vz = new float[N];
    for (int i = 0; i < N; i++) {
        h_x[i] = bodys[i].x; h_y[i] = bodys[i].y;
        h_z[i] = bodys[i].z; h_mass[i] = bodys[i].w;
        h_vx[i] = velocitys[i].x; h_vy[i] = velocitys[i].y; h_vz[i] = velocitys[i].z;
    }

    float *d_x, *d_y, *d_z, *d_mass, *d_Fx, *d_Fy, *d_Fz, *d_Vx, *d_Vy, *d_Vz;
    cudaMalloc(&d_x, N * sizeof(float)); cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float)); cudaMalloc(&d_mass, N * sizeof(float));
    cudaMalloc(&d_Fx, N * sizeof(float)); cudaMalloc(&d_Fy, N * sizeof(float));
    cudaMalloc(&d_Fz, N * sizeof(float));
    cudaMalloc(&d_Vx, N * sizeof(float)); cudaMalloc(&d_Vy, N * sizeof(float));
    cudaMalloc(&d_Vz, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_Fx, 0, N * sizeof(float));
    cudaMemset(d_Fy, 0, N * sizeof(float));
    cudaMemset(d_Fz, 0, N * sizeof(float));

    compute_force_bf<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_x, d_y, d_z, d_mass, d_Fx, d_Fy, d_Fz, N);
    cudaDeviceSynchronize();

    // copy forces before apply_forces overwrites positions
    cudaMemcpy(h_Fx, d_Fx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fy, d_Fy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fz, d_Fz, N * sizeof(float), cudaMemcpyDeviceToHost);

    apply_forces_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_x, d_y, d_z, d_mass,
        d_Vx, d_Vy, d_Vz, d_Fx, d_Fy, d_Fz, N, dt);
    cudaDeviceSynchronize();

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_mass);
    cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);
    cudaFree(d_Vx); cudaFree(d_Vy); cudaFree(d_Vz);
    delete[] h_x; delete[] h_y; delete[] h_z; delete[] h_mass;
    delete[] h_vx; delete[] h_vy; delete[] h_vz;
}

/*
wrapper around GPU kernels for barnes-hut computation.
If kt != nullptr, per-kernel GPU times (ms) are written into *kt.
*/
void barnes_hut_cudav1(std::vector<float4> &bodys, std::vector<float3> &velocitys, float dt, float theta,
    float *h_Fx, float *h_Fy, float *h_Fz, KernelTimes *kt = nullptr){
    //////////// START DATA INIT ////////////

    // init data on host
    int N = bodys.size();
    int max_nodes = N*MULT;
    float *h_x = new float[max_nodes];
    float *h_y = new float[max_nodes];
    float *h_z = new float[max_nodes];
    float *h_mass = new float[max_nodes];
    float *h_vx = new float[max_nodes];
    float *h_vy = new float[max_nodes];
    float *h_vz = new float[max_nodes];

    for (int i = 0; i < N; ++i){
        h_x[i] = bodys[i].x;
        h_y[i] = bodys[i].y;
        h_z[i] = bodys[i].z;
        h_mass[i] = bodys[i].w;
        h_vx[i] = velocitys[i].x;
        h_vy[i] = velocitys[i].y;
        h_vz[i] = velocitys[i].z;
    }
    // to init internal node masses
    float *h_neg1 = new float[max_nodes-N];
    for (int i = 0; i < max_nodes-N; ++i){
        h_neg1[i] = -1;
    }

    // alloc data on device
    float *d_x, *d_y, *d_z, *d_mass;
    float *d_Vx, *d_Vy, *d_Vz;
    float *d_Fx, *d_Fy, *d_Fz;
    float *d_root_half;
    float *d_dt;
    int *d_children;
    int *d_next_cell;
    int *d_N, *d_max_nodes;

    cudaMalloc(&d_x, max_nodes * sizeof(float));
    cudaMalloc(&d_y, max_nodes * sizeof(float));
    cudaMalloc(&d_z, max_nodes * sizeof(float));
    cudaMalloc(&d_mass, max_nodes * sizeof(float));

    cudaMalloc(&d_Vx, max_nodes * sizeof(float));
    cudaMalloc(&d_Vy, max_nodes * sizeof(float));
    cudaMalloc(&d_Vz, max_nodes * sizeof(float));

    cudaMalloc(&d_Fx, max_nodes * sizeof(float));
    cudaMalloc(&d_Fy, max_nodes * sizeof(float));
    cudaMalloc(&d_Fz, max_nodes * sizeof(float));

    cudaMalloc(&d_children, OCT_CHILDREN * max_nodes * sizeof(int));
    cudaMalloc(&d_next_cell, sizeof(int));

    cudaMalloc(&d_N, sizeof(int));
    cudaMalloc(&d_max_nodes, sizeof(int));
    cudaMalloc(&d_root_half, sizeof(float));
    cudaMalloc(&d_dt, sizeof(float));


    // copy data to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dt, &dt, sizeof(float), cudaMemcpyHostToDevice);

    // NOTE: cudaMemset sets raw bytes, not ints
    // forces init 0
    cudaMemset(d_Fx, 0, max_nodes * sizeof(float));
    cudaMemset(d_Fy, 0, max_nodes * sizeof(float));
    cudaMemset(d_Fz, 0, max_nodes * sizeof(float));

    // bounding box radius init to 0
    cudaMemset(d_root_half, 0, sizeof(float));

    // set children pointers to be NULL_VAL
    cudaMemset(d_children, NULL_VAL_INT, OCT_CHILDREN * max_nodes * sizeof(int));

    // copy mass of bodies (first N)
    // init rest of values to -1 with mem copy
    cudaMemcpy(d_mass, h_mass, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass + N, h_neg1, (max_nodes-N) * sizeof(float), cudaMemcpyHostToDevice);

    // init node counts
    cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_nodes, &max_nodes, sizeof(int), cudaMemcpyHostToDevice);

    // set next_cell to be the last index -1
    // NOTE: we do this because atomicAdd / atomicSub return the OLD value
    // and we want to ensure we dont over-write the root node
    int h_next_cell = max_nodes-2;
    cudaMemcpy(d_next_cell, &h_next_cell, sizeof(int), cudaMemcpyHostToDevice);

    //////////// END DATA INIT ////////////

    //// START CORE KERNELS ////

    dim3 grid_dim(NUM_BLOCKS, 1, 1);
    dim3 block_dim(BLOCK_SIZE, 1, 1);

    // One event pair reused for each kernel.
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // 0) compute bounding box
    int *d_block_counter;
    cudaMalloc(&d_block_counter, sizeof(int));
    cudaMemset(d_block_counter, 0, sizeof(int));

    // allocate block-size min/max float arrays (6 total)
    float *d_minx, *d_miny, *d_minz;
    float *d_maxx, *d_maxy, *d_maxz;
    cudaMalloc(&d_minx, NUM_BLOCKS * sizeof(float)); cudaMalloc(&d_maxx, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_miny, NUM_BLOCKS * sizeof(float)); cudaMalloc(&d_maxy, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_minz, NUM_BLOCKS * sizeof(float)); cudaMalloc(&d_maxz, NUM_BLOCKS * sizeof(float));

    cudaEventRecord(ev_start);
    body_reduce_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, N, d_root_half, d_block_counter,
        d_minx, d_miny, d_minz, d_maxx, d_maxy, d_maxz);
    cudaEventRecord(ev_stop);
    if (kt) kt->body_reduce_ms = event_ms(ev_start, ev_stop);

    // copy back L / 2 of bounding box
    float root_half;
    cudaMemcpy(&root_half, d_root_half, sizeof(float), cudaMemcpyDeviceToHost);

    // 1) construct oct tree in parallel
    cudaEventRecord(ev_start);
    
    
    // V1
    // build_tree_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_children, d_next_cell, N, max_nodes, root_half, DEPTH_LIMIT);
    // V1

    // V2
    build_tree_kernelv2<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_children, d_next_cell, N, max_nodes, root_half, DEPTH_LIMIT);
    // V1


    cudaEventRecord(ev_stop);
    if (kt) kt->build_tree_ms = event_ms(ev_start, ev_stop);

    // copy back first index to internal node
    int node_start_idx;
    cudaMemcpy(&node_start_idx, d_next_cell, sizeof(int), cudaMemcpyDeviceToHost);

    // 2) compute center of mass for each node
    cudaEventRecord(ev_start);
    compute_cmass_kernelv1<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_mass, d_children, node_start_idx, max_nodes-1, N);
    cudaEventRecord(ev_stop);
    if (kt) kt->compute_cmass_ms = event_ms(ev_start, ev_stop);

    // 3) compute forces acted on each body (1 thread per body, traverse the tree)
    cudaEventRecord(ev_start);
    compute_forces_kernelv1<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_mass, d_children, N, max_nodes, root_half, d_Fx, d_Fy, d_Fz, theta);
    cudaEventRecord(ev_stop);
    if (kt) kt->compute_forces_ms = event_ms(ev_start, ev_stop);

    // 4) update position of bodies based on computed net forces (very parallel; fixed computation per body)
    cudaEventRecord(ev_start);
    apply_forces_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_mass, d_Vx, d_Vy, d_Vz, d_Fx, d_Fy, d_Fz, N, dt);
    cudaEventRecord(ev_stop);
    if (kt) kt->apply_forces_ms = event_ms(ev_start, ev_stop);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    //// END CORE KERNELS ////

    //// DATA COPY BACK CLEAN UP ////

    // copy back coords of bodys
    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_Fx, d_Fx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fy, d_Fy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fz, d_Fz, N * sizeof(float), cudaMemcpyDeviceToHost);

    //// DATA COPY BACK ////

    //// CLEAN UP ////

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_mass);
    cudaFree(d_Vx); cudaFree(d_Vy); cudaFree(d_Vz);
    cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);
    cudaFree(d_root_half); cudaFree(d_dt); cudaFree(d_children);
    cudaFree(d_next_cell); cudaFree(d_N); cudaFree(d_max_nodes);
    cudaFree(d_block_counter);
    cudaFree(d_minx); cudaFree(d_maxx);
    cudaFree(d_miny); cudaFree(d_maxy);
    cudaFree(d_minz); cudaFree(d_maxz);
    delete[] h_x; delete[] h_y; delete[] h_z;
    delete[] h_mass; delete[] h_vx; delete[] h_vy; delete[] h_vz;
    delete[] h_neg1;
    //// CLEAN UP ////
}



void barnes_hut_cudav2(std::vector<float4> &bodys, std::vector<float3> &velocitys, float dt, float theta,
    float *h_Fx, float *h_Fy, float *h_Fz, KernelTimes *kt = nullptr){
    //////////// START DATA INIT ////////////

    // init data on host
    int N = bodys.size();
    int max_nodes = N*MULT;
    float *h_x = new float[max_nodes];
    float *h_y = new float[max_nodes];
    float *h_z = new float[max_nodes];
    float *h_mass = new float[max_nodes];
    float *h_vx = new float[max_nodes];
    float *h_vy = new float[max_nodes];
    float *h_vz = new float[max_nodes];

    for (int i = 0; i < N; ++i){
        h_x[i] = bodys[i].x;
        h_y[i] = bodys[i].y;
        h_z[i] = bodys[i].z;
        h_mass[i] = bodys[i].w;
        h_vx[i] = velocitys[i].x;
        h_vy[i] = velocitys[i].y;
        h_vz[i] = velocitys[i].z;
    }
    // to init internal node masses
    float *h_neg1 = new float[max_nodes-N];
    for (int i = 0; i < max_nodes-N; ++i){
        h_neg1[i] = -1;
    }

    // alloc data on device
    float *d_x, *d_y, *d_z, *d_mass;
    float *d_Vx, *d_Vy, *d_Vz;
    float *d_Fx, *d_Fy, *d_Fz;
    float *d_root_half;
    float *d_dt;
    int *d_children, *d_subtree_body_size, *d_sorted_bodys;
    int *d_next_cell;
    int *d_N, *d_max_nodes;

    cudaMalloc(&d_x, max_nodes * sizeof(float));
    cudaMalloc(&d_y, max_nodes * sizeof(float));
    cudaMalloc(&d_z, max_nodes * sizeof(float));
    cudaMalloc(&d_mass, max_nodes * sizeof(float));
    cudaMalloc(&d_subtree_body_size, max_nodes * sizeof(int));
    cudaMalloc(&d_sorted_bodys, N * sizeof(int));

    cudaMalloc(&d_Vx, max_nodes * sizeof(float));
    cudaMalloc(&d_Vy, max_nodes * sizeof(float));
    cudaMalloc(&d_Vz, max_nodes * sizeof(float));

    cudaMalloc(&d_Fx, max_nodes * sizeof(float));
    cudaMalloc(&d_Fy, max_nodes * sizeof(float));
    cudaMalloc(&d_Fz, max_nodes * sizeof(float));

    cudaMalloc(&d_children, OCT_CHILDREN * max_nodes * sizeof(int));
    cudaMalloc(&d_next_cell, sizeof(int));

    cudaMalloc(&d_N, sizeof(int));
    cudaMalloc(&d_max_nodes, sizeof(int));
    cudaMalloc(&d_root_half, sizeof(float));
    cudaMalloc(&d_dt, sizeof(float));


    // copy data to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dt, &dt, sizeof(float), cudaMemcpyHostToDevice);

    // NOTE: cudaMemset sets raw bytes, not ints
    // forces init 0
    cudaMemset(d_Fx, 0, max_nodes * sizeof(float));
    cudaMemset(d_Fy, 0, max_nodes * sizeof(float));
    cudaMemset(d_Fz, 0, max_nodes * sizeof(float));

    // bounding box radius init to 0
    cudaMemset(d_root_half, 0, sizeof(float));

    // set children pointers to be NULL_VAL
    cudaMemset(d_children, NULL_VAL_INT, OCT_CHILDREN * max_nodes * sizeof(int));

    // copy mass of bodies (first N)
    // init rest of values to -1 with mem copy
    cudaMemcpy(d_mass, h_mass, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass + N, h_neg1, (max_nodes-N) * sizeof(float), cudaMemcpyHostToDevice);

    // init node counts
    cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_nodes, &max_nodes, sizeof(int), cudaMemcpyHostToDevice);

    // set next_cell to be the last index -1
    // NOTE: we do this because atomicAdd / atomicSub return the OLD value
    // and we want to ensure we dont over-write the root node
    int h_next_cell = max_nodes-2;
    cudaMemcpy(d_next_cell, &h_next_cell, sizeof(int), cudaMemcpyHostToDevice);

    // TODO: add extra fields that are needed for cmass, sorting, v2
    // sorted int array
    // array to hold number of bodies in a subtree

    //////////// END DATA INIT ////////////

    //// START CORE KERNELS ////

    dim3 grid_dim(NUM_BLOCKS, 1, 1);
    dim3 block_dim(BLOCK_SIZE, 1, 1);

    // One event pair reused for each kernel.
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // 0) compute bounding box
    int *d_block_counter;
    cudaMalloc(&d_block_counter, sizeof(int));
    cudaMemset(d_block_counter, 0, sizeof(int));

    // allocate block-size min/max float arrays (6 total)
    float *d_minx, *d_miny, *d_minz;
    float *d_maxx, *d_maxy, *d_maxz;
    cudaMalloc(&d_minx, NUM_BLOCKS * sizeof(float)); cudaMalloc(&d_maxx, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_miny, NUM_BLOCKS * sizeof(float)); cudaMalloc(&d_maxy, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_minz, NUM_BLOCKS * sizeof(float)); cudaMalloc(&d_maxz, NUM_BLOCKS * sizeof(float));

    cudaEventRecord(ev_start);
    body_reduce_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, N, d_root_half, d_block_counter,
        d_minx, d_miny, d_minz, d_maxx, d_maxy, d_maxz);
    cudaEventRecord(ev_stop);
    if (kt) kt->body_reduce_ms = event_ms(ev_start, ev_stop);

    // copy back L / 2 of bounding box
    float root_half;
    cudaMemcpy(&root_half, d_root_half, sizeof(float), cudaMemcpyDeviceToHost);

    // 1) construct oct tree in parallel
    cudaEventRecord(ev_start);

    build_tree_kernelv2<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_children, d_next_cell, N, max_nodes, root_half, DEPTH_LIMIT);

    cudaEventRecord(ev_stop);
    if (kt) kt->build_tree_ms = event_ms(ev_start, ev_stop);

    // copy back first index to internal node
    int node_start_idx;
    cudaMemcpy(&node_start_idx, d_next_cell, sizeof(int), cudaMemcpyDeviceToHost);

    // 2) compute center of mass for each node
    cudaEventRecord(ev_start);
    compute_cmass_kernelv2<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_mass, d_children, d_subtree_body_size, node_start_idx+1, max_nodes-1, N);
    cudaEventRecord(ev_stop);
    if (kt) kt->compute_cmass_ms = event_ms(ev_start, ev_stop);

    // 2.5: sort bodies based on in-order body traversal
    // ensure that subtree_body_size(root) = -1
    top_down_body_sort_kernel<<<grid_dim, block_dim>>>(d_children, d_sorted_bodys, d_subtree_body_size, node_start_idx+1, max_nodes-1, N);            


    // 3) compute forces acted on each body (1 thread per body, traverse the tree)
    cudaEventRecord(ev_start);
    // compute_forces_kernelv1<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_mass, d_children, N, max_nodes, root_half, d_Fx, d_Fy, d_Fz, theta);
    compute_forces_kernelv2<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_mass, d_sorted_bodys, d_children, N, max_nodes, root_half, d_Fx, d_Fy, d_Fz, theta);
    cudaEventRecord(ev_stop);
    if (kt) kt->compute_forces_ms = event_ms(ev_start, ev_stop);

    // 4) update position of bodies based on computed net forces (very parallel; fixed computation per body)
    cudaEventRecord(ev_start);
    apply_forces_kernel<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_mass, d_Vx, d_Vy, d_Vz, d_Fx, d_Fy, d_Fz, N, dt);
    cudaEventRecord(ev_stop);
    if (kt) kt->apply_forces_ms = event_ms(ev_start, ev_stop);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    //// END CORE KERNELS ////

    //// DATA COPY BACK CLEAN UP ////

    // copy back coords of bodys
    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_Fx, d_Fx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fy, d_Fy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fz, d_Fz, N * sizeof(float), cudaMemcpyDeviceToHost);

    //// DATA COPY BACK ////

    //// CLEAN UP ////

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_mass);
    cudaFree(d_Vx); cudaFree(d_Vy); cudaFree(d_Vz);
    cudaFree(d_Fx); cudaFree(d_Fy); cudaFree(d_Fz);
    cudaFree(d_root_half); cudaFree(d_dt); cudaFree(d_children);
    cudaFree(d_next_cell); cudaFree(d_N); cudaFree(d_max_nodes);
    cudaFree(d_block_counter);
    cudaFree(d_minx); cudaFree(d_maxx);
    cudaFree(d_miny); cudaFree(d_maxy);
    cudaFree(d_minz); cudaFree(d_maxz);
    delete[] h_x; delete[] h_y; delete[] h_z;
    delete[] h_mass; delete[] h_vx; delete[] h_vy; delete[] h_vz;
    delete[] h_neg1;
    //// CLEAN UP ////

}





int main() {
    // warm up GPU before benchmarking
    cudaFree(0);
    bool v2 = true; 
    float dt = 0.1f;
    vector<float> thetas = {0.25f, 0.5f, 1.0f, 5.0f};
    // vector<string> file_names = {
    //     "test/test_traces/test_5000.txt", "test/test_traces/test_10000.txt",
    //     "test/test_traces/test_25000.txt", "test/test_traces/test_50000.txt",
    //     "test/test_traces/test_500000.txt", "test/test_traces/test_1000000.txt"};
    // vector<string> file_names = {"test/test_traces/test_50000.txt"};       
    // vector<string> file_names = {
        // "test/test_traces/test_25000.txt", "test/test_traces/test_50000.txt",
        // "test/test_traces/test_500000.txt", "test/test_traces/test_1000000.txt"};    
    vector<string> file_names = {"test/test_traces/test_2000000.txt", "test/test_traces/test_5000000.txt"};

    ofstream csv("cuda_benchmark_resultsv3_big.csv");
    csv << "N,theta,brute_force_ms,barnes_hut_ms,speedup,avg_rel_error_pct\n";

    ofstream kcsv("cuda_kernel_timesv3_big.csv");
    kcsv << "N,theta,body_reduce_ms,build_tree_ms,compute_cmass_ms,compute_forces_ms,apply_forces_ms,barnes_hut_ms\n";

    for (auto &file_name : file_names) {
        vector<float4> bodys;
        vector<float3> velo;
        ifstream file(file_name);
        if (!file) { cerr << "Cannot open " << file_name << "\n"; continue; }

        float x, y, z, mass;
        while (file >> x >> y >> z >> mass){
            bodys.push_back(make_float4(x, y, z, mass));
            velo.push_back(make_float3(0,0,0));
        }
        int N = bodys.size();
        cout << "N=" << N << "\n";

        // brute force (once per input, independent of theta)
        float *bf_Fx = new float[N], *bf_Fy = new float[N], *bf_Fz = new float[N];

        auto bf_start = chrono::high_resolution_clock::now();
        brute_force_cuda(bodys, velo, dt, bf_Fx, bf_Fy, bf_Fz);
        auto bf_end = chrono::high_resolution_clock::now();
        auto bf_ms = chrono::duration_cast<chrono::milliseconds>(bf_end - bf_start).count();
        cout << "  Brute Force: " << bf_ms << "ms\n";

        // TODO: when you make THETA a runtime parameter, loop here
        // for now using the compile-time THETA
        for (auto& theta: thetas)
        {
            float *bh_Fx = new float[N], *bh_Fy = new float[N], *bh_Fz = new float[N];
            float bh_ms;
            KernelTimes kt;
            if (!v2){
            auto bh_start = chrono::high_resolution_clock::now();
            barnes_hut_cudav1(bodys, velo, dt, theta, bh_Fx, bh_Fy, bh_Fz, &kt);
            auto bh_end = chrono::high_resolution_clock::now();
            bh_ms = chrono::duration_cast<chrono::milliseconds>(bh_end - bh_start).count();
            }
            else{
            auto bh_start = chrono::high_resolution_clock::now();
            barnes_hut_cudav2(bodys, velo, dt, theta, bh_Fx, bh_Fy, bh_Fz, &kt);
            auto bh_end = chrono::high_resolution_clock::now();
            bh_ms = chrono::duration_cast<chrono::milliseconds>(bh_end - bh_start).count();                
            }

            // compute average relative error
            float total_rel_err = 0.0f;
            for (int i = 0; i < N; i++) {
                float dx = fabsf(bh_Fx[i] - bf_Fx[i]);
                float dy = fabsf(bh_Fy[i] - bf_Fy[i]);
                float dz = fabsf(bh_Fz[i] - bf_Fz[i]);
                float mag = sqrtf(bf_Fx[i]*bf_Fx[i] + bf_Fy[i]*bf_Fy[i] + bf_Fz[i]*bf_Fz[i]);
                float err = sqrtf(dx*dx + dy*dy + dz*dz) / (mag + 1e-10f);
                total_rel_err += err;
            }
            float avg_rel_err = total_rel_err / N;
            float speedup = (bh_ms > 0) ? (float)bf_ms / bh_ms : 0.0f;

            cout << "  theta=" << theta
                 << "  BH: " << bh_ms << "ms"
                 << "  speedup: " << speedup << "x"
                 << "  error: " << avg_rel_err * 100.0f << "%\n"
                 << "    body_reduce=" << kt.body_reduce_ms << "ms"
                 << "  build_tree=" << kt.build_tree_ms << "ms"
                 << "  cmass=" << kt.compute_cmass_ms << "ms"
                 << "  forces=" << kt.compute_forces_ms << "ms"
                 << "  apply=" << kt.apply_forces_ms << "ms\n\n";

            csv << N << "," << theta << "," << bf_ms << "," << bh_ms << ","
                << speedup << "," << avg_rel_err * 100.0f <<  "\n";

            kcsv << N << "," << theta << ","
                 << kt.body_reduce_ms << "," << kt.build_tree_ms << ","
                 << kt.compute_cmass_ms << "," << kt.compute_forces_ms << ","
                 << kt.apply_forces_ms << "," << bh_ms << "\n";

            delete[] bh_Fx; delete[] bh_Fy; delete[] bh_Fz;
        }

        delete[] bf_Fx; delete[] bf_Fy; delete[] bf_Fz;
    }

    csv.close();
    kcsv.close();
    cout << "Results written to cuda_benchmark_resultsv2.csv and cuda_kernel_timesv2.csv\n";
}
