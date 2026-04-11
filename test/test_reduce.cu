/*
Testing K0 reduction kernel that computes bounding box over bodies
*/

#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include "bhut_cpu.h"
#include "kernels.cuh"
using namespace std;

// ---- main ----
int main(int argc, char **argv){
    // read input
    const char *fname = (argc > 1) ? argv[1] : "test_traces/random.txt";
    ifstream file(fname);
    if (!file){ cerr << "Cannot open " << fname << "\n"; return 1; }
    
    vector<Body> bodys;
    float bx, by, bz, bm;
    while (file >> bx >> by >> bz >> bm)
        bodys.emplace_back(Float3{bx,by,bz}, bm);
    int N = bodys.size();
    cout << "Loaded " << N << " bodies\n";

    // malloc host mem
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_z = new float[N];
    for (int i = 0; i < N; i++){
        h_x[i] = bodys[i].pt.x;
        h_y[i] = bodys[i].pt.y;
        h_z[i] = bodys[i].pt.z;
    }

    // malloc device mem
    float *d_x, *d_y, *d_z, *d_root_half;
    int   *d_blk_counter;
    float *d_minx, *d_miny, *d_minz, *d_maxx, *d_maxy, *d_maxz;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_root_half, sizeof(float));
    cudaMalloc(&d_blk_counter, sizeof(int));
    cudaMalloc(&d_minx, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_miny, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_minz, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_maxx, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_maxy, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_maxz, NUM_BLOCKS * sizeof(float));

    // copy to device & init
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_root_half, 0, sizeof(float));
    cudaMemset(d_blk_counter, 0, sizeof(int));

    // launch kernel
    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_SIZE);
    body_reduce_kernel<<<grid, block>>>(d_x, d_y, d_z, N, d_root_half, d_blk_counter,
        d_minx, d_miny, d_minz, d_maxx, d_maxy, d_maxz);
    
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // copy result back
    float gpu_root_half;
    cudaMemcpy(&gpu_root_half, d_root_half, sizeof(float), cudaMemcpyDeviceToHost);

    // cpu references
    float cpu_ref_original = compute_box(bodys);

    // compare
    cout << "\n=== Results ===\n";
    cout << "GPU body_reduce_kernel           : " << gpu_root_half << "\n";
    cout << "CPU                              :" << cpu_ref_original << " \n";

    float diff = fabsf(gpu_root_half - cpu_ref_original);
    cout << "\nGPU vs cpu diff  : " << diff << "\n";
    if (diff < 1e-3f)
        cout << "PASS\n";
    else
        cout << "FAIL\n";

    // cleanup
    delete[] h_x; delete[] h_y; delete[] h_z;
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_root_half); cudaFree(d_blk_counter);
    cudaFree(d_minx); cudaFree(d_miny); cudaFree(d_minz);
    cudaFree(d_maxx); cudaFree(d_maxy); cudaFree(d_maxz);
    return 0;
}