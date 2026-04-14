/*
Kernels for barnes hut 
@author: Andrew Koulogeorge
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <cfloat>
using namespace std; 

// for prealloc total number of nodes
constexpr int MULT = 3; 
constexpr int OCT_CHILDREN = 8; 
constexpr int NULL_VAL_INT = -1;
constexpr float NULL_VAL_FLOAT = -1;
constexpr int LOCK_VAL = -2; 

constexpr int DEPTH_LIMIT = 30; 
constexpr int STACK_SIZE = 64; 


constexpr float THETA_GPU = 0.5; 
constexpr float EPS_GPU = 1e-2;
constexpr float G_GPU = 1;

// thread block params // 
constexpr int SM_COUNT = 46;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4; 

constexpr int BLOCK_SIZE = 128; // warp_size * num_warps
const int NUM_BLOCKS = SM_COUNT;

/* return index of child based on spatial loc */
__device__ int get_oct_idx(float3 box_center, float3 body_pos);

/* create new bounding box based on parent box and oct idx */
__device__ float3 update_box(float3 box_center, float half, int oct_idx);

///////////// BEGIN CORE KERNEL SECTION  /////////////

/* 
Perform reduction over bodies to compute radius of cube points reside inside
Grid / Block are both 1d
Reduction typically memory bound! have to read in all the body information from gmem
*/
__global__ void body_reduce_kernel(float *x, float *y, float *z, int N, float *root_half, int *blk_counter,
    float *blk_minx, float *blk_miny, float *blk_minz, float *blk_maxx, float *blk_maxy, float *blk_maxz);

/*
populate child_pointers array (form OctTree structure)
thread block structure: 1d grid and 1d blocks
assume box is centered at (0,0,0) with half length L
*/
__global__ void build_tree_kernel(float *x, float *y, float *z, int *children, int *next_cell,
    int N, int max_nodes, float root_half, int depth_limit);

__global__ void build_tree_kernelv2(float *x, float *y, float *z, int *children, int *next_cell,
    int N, int max_nodes, float root_half, int depth_limit);


/* bottom up traversal to compute center of mass for each node in OctTree*/
__global__ void compute_cmass_kernel(float *x, float *y, float *z, float *mass, int *children,
    int first_cell, int last_cell, int N);

/* 
Traverse OctTree to approximate forces on each body 
This kernel takes up most of the running time of barnes-hut
This kernel is highly memory bound
Warp divergence is a major challange: each body will traverse down some prefix of the tree
the greater variance in that prefix among threads in the same warp, the greater the divergence

for a v0 implementation, we store a separate iteration stack for each thread
if the stack size is too large to be kept in registers, it will spin to gmem and harm performance

Their optimized implementation does a form of warp specialization
They leverage the fact that they sort in step 4 to place bodies near each other
And they remove divergent warps all together

*/
__global__ void compute_forces_kernel(float *x, float *y, float *z, float *mass, int *children, int N, int max_nodes, 
    float root_half, float *Fx, float *Fy, float *Fz, float theta);

/* 
Apply accumulated forces on bodys to update pos and vel 
Streaming kernel; no use of sharmed mem
*/
__global__ void apply_forces_kernel(float *x, float *y, float *z, float *mass, 
    float *Vx, float *Vy, float *Vz, float *Fx, float *Fy, float *Fz, int N, float dt);