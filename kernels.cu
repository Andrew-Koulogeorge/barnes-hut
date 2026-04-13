/*
Kernels for barnes hut 
@author: Andrew Koulogeorge
*/

#include <limits>
#include <cfloat>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels.cuh"
#include <cstdint>
using namespace std; 

/* return index of child based on spatial loc */
__device__ int get_oct_idx(float3 box_center, float3 body_pos){
    int oct_idx = 0; 
    if (box_center.x <= body_pos.x) oct_idx |= 1;
    if (box_center.y <= body_pos.y) oct_idx |= 2;
    if (box_center.z <= body_pos.z) oct_idx |= 4;
    return oct_idx; 
}
/* create new bounding box based on parent box and oct idx */
__device__ float3 update_box(float3 box_center, float half, int oct_idx){
    float3 new_box_center = box_center; 
    new_box_center.x += (oct_idx & 1 ? half : -half);
    new_box_center.y += (oct_idx & 2 ? half : -half);
    new_box_center.z += (oct_idx & 4 ? half : -half);;
    return new_box_center; 
}

///////////// END DEVICE HELPER SECTION  /////////////

///////////// BEGIN CORE KERNEL SECTION  /////////////

/* 
Perform reduction over bodies to compute radius of cube points reside inside
Grid / Block are both 1d
Reduction typically memory bound! have to read in all the body information from gmem
*/
__global__ void body_reduce_kernel(float *x, float *y, float *z, int N, float *root_half, int *blk_counter,
    float *blk_minx, float *blk_miny, float *blk_minz, float *blk_maxx, float *blk_maxy, float *blk_maxz){

    // 0: declare shared memory for within block reduction
    __shared__ float s_minx[BLOCK_SIZE]; __shared__ float s_maxx[BLOCK_SIZE]; 
    __shared__ float s_miny[BLOCK_SIZE]; __shared__ float s_maxy[BLOCK_SIZE];     
    __shared__ float s_minz[BLOCK_SIZE]; __shared__ float s_maxz[BLOCK_SIZE];     
    
    // thread index witihn grid of blocks
    // stride = total kernel thread count 
    int global_tidx = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = gridDim.x*blockDim.x; 
    
    // 1: each thread computes local min / max for x y and z (GMEM -> registers) 
    float t_minx, t_maxx, t_miny, t_maxy, t_minz, t_maxz;  
    t_minx = t_miny = t_minz = FLT_MAX;
    t_maxx = t_maxy = t_maxz = -FLT_MAX;
    for (int i = global_tidx; i < N; i += stride){
        t_minx = fminf(t_minx, x[i]); t_maxx = fmaxf(t_maxx, x[i]); 
        t_miny = fminf(t_miny, y[i]); t_maxy = fmaxf(t_maxy, y[i]); 
        t_minz = fminf(t_minz, z[i]); t_maxz = fmaxf(t_maxz, z[i]); 
    }
    // 2: load local min / max x y z into shared memory (register -> SMEM)
    s_minx[threadIdx.x] = t_minx; s_maxx[threadIdx.x] = t_maxx;
    s_miny[threadIdx.x] = t_miny; s_maxy[threadIdx.x] = t_maxy;
    s_minz[threadIdx.x] = t_minz; s_maxz[threadIdx.x] = t_maxz; 
    
    // prevent some warps from starting the reduction before all the data has been written to SMSM
    __syncthreads(); 

    // 3: each block performs tree style reduction leveraging shared memory (SMEM)
    // top down approach prevents race conditions on updates
    // assume block size is a power of 2
    for (int gap = BLOCK_SIZE/2; gap >= 1; gap >>= 1){
        if (threadIdx.x < gap){
            s_minx[threadIdx.x] = fminf(s_minx[threadIdx.x], s_minx[threadIdx.x+gap]);
            s_maxx[threadIdx.x] = fmaxf(s_maxx[threadIdx.x], s_maxx[threadIdx.x+gap]);
            s_miny[threadIdx.x] = fminf(s_miny[threadIdx.x], s_miny[threadIdx.x+gap]);
            s_maxy[threadIdx.x] = fmaxf(s_maxy[threadIdx.x], s_maxy[threadIdx.x+gap]);
            s_minz[threadIdx.x] = fminf(s_minz[threadIdx.x], s_minz[threadIdx.x+gap]);
            s_maxz[threadIdx.x] = fmaxf(s_maxz[threadIdx.x], s_maxz[threadIdx.x+gap]);
        }
        // prevent some warps from jumping to next level before others have finished
        __syncthreads(); 
    }

    // 4: have elected thread per block write reduced value to global memory (SMEM -> GMEM) 
    bool final_block = false;
    if (threadIdx.x == 0){
        blk_minx[blockIdx.x] = s_minx[0]; blk_maxx[blockIdx.x] = s_maxx[0];
        blk_miny[blockIdx.x] = s_miny[0]; blk_maxy[blockIdx.x] = s_maxy[0];
        blk_minz[blockIdx.x] = s_minz[0]; blk_maxz[blockIdx.x] = s_maxz[0]; 
        
        // ensure that block-wise values have been stored before we perform reduction across blocks
        // without fence, may read value before they are written tp
        __threadfence(); 
        // leverage atomic increment to elect block to perform reduction over values written to GMEM; 
        int old = atomicAdd(blk_counter, 1); 
        final_block = (old == NUM_BLOCKS-1);
    }
    // 5: elected thread in final block has true flag; performs reduction across blocks from GMEM
    // NOTE: GMEM needed because we need a way to agg block-level solutions together
    if (final_block){
        for (int i = 0; i < NUM_BLOCKS; i++){
            t_minx = fminf(t_minx, blk_minx[i]); t_maxx = fmaxf(t_maxx, blk_maxx[i]); 
            t_miny = fminf(t_miny, blk_miny[i]); t_maxy = fmaxf(t_maxy, blk_maxy[i]); 
            t_minz = fminf(t_minz, blk_minz[i]); t_maxz = fmaxf(t_maxz, blk_maxz[i]);                
        }
        // 6: compute max length and write to output in gmem
        float dx = t_maxx - t_minx; 
        float val = fmaxf(t_maxy - t_miny, dx); 
        *root_half = fmaxf(t_maxz - t_minz, val)/2;
    }    
}

/*
populate child_pointers array (form OctTree structure)
thread block structure: 1d grid and 1d blocks
assume box is centered at (0,0,0) with half length L
*/
__global__ void build_tree_kernel(float *x, float *y, float *z, int *children, int *next_cell,
    int N, int max_nodes, float root_half, int depth_limit){
    
    // thread index within grid of blocks
    int tx = threadIdx.x + blockDim.x*blockIdx.x;
    // each thread in the block may get several bodies if N > total_threads
    int grid_stride = gridDim.x * blockDim.x; 
    // local body we are inserting 
    float3 l_body_pos; 
    // size of bounding box as we traverse
    float l_half; 
    // nodex = current node
    // child_octx = {0,1,...,7}
    // childx = child node
    int nodex, child_octx, childx;
    // prevent inf region splitting
    int depth; 
    //// thread inserting body_i into tree ////   
    for (int i = tx; i < N; i += grid_stride){      
        // begin traversal at root
        nodex = max_nodes-1; 
        l_half = root_half;
        depth = 0; 
        // read body pos from global memory (memory fetch here is coal)
        l_body_pos.x = x[i]; l_body_pos.y = y[i]; l_body_pos.z = z[i];
        // init center at origin 
        float3 l_box_center = make_float3(0,0,0);

        // keep traversing until node is inserted
        bool inserted = false;     
        while (!inserted) {
            if (depth > depth_limit){
                // skip this star
                inserted = true; 
                break;
            }
            // get child node region 
            child_octx = get_oct_idx(l_box_center, l_body_pos);
            // get index of child: 
            // 0:N-1 = body
            // N < = internal 
            // -1 = nullptr            
            // -2 = LOCKED
            childx = children[nodex*OCT_CHILDREN + child_octx];
            if (childx == LOCK_VAL){
                // someone else has the lock; almost want to put this thread to sleep
                continue;
            }
            // case 0: child is internal node; keep traversing
            if (childx >= N){
                // move down a level; update bounding box and curr node
                l_half *= 0.5;
                l_box_center = update_box(l_box_center, l_half, child_octx);                
                nodex = childx; 
                depth++;
                continue;
            }
            else{
                // if the node is not internal, we will attemtpt to insert a node here
                // to prevent losing insertions, we need a lock! 
                // if nobody else has inserted a node at this edge since we read the child, we will have the lock
                int lock = atomicCAS(&children[nodex*OCT_CHILDREN + child_octx], childx, LOCK_VAL);
                if (lock != childx){
                    // someone else was faster; need to retry 
                    continue; 
                }

                // LOCK OBTAINED AT THIS POINT // 
                // KEY IDEA: once we update "children[nodex*OCT_CHILDREN + child_octx]", the lock is released
                // case 1: child is null pointer; simply insert body as child 
                if (childx == NULL_VAL_INT){
                    children[nodex*OCT_CHILDREN + child_octx] = i; 
                    inserted = true;
                }
                // case 2: child is body; subdiv region at least once
                else {
                    // new_cell = internal node alloc (always triggered)
                    // need to add new internal node to OctTree --> decrement counter 
                    int new_cell = atomicSub(next_cell, 1);
                    // keep track of first cell we alloc 
                    int first_new_cell = new_cell;
                    // new_new_cell = internal node alloc when both bodies in same region
                    int new_new_cell;
                    // even if we subdiv region several times, 2 bodies stay same;
                    float3 l_body_pos2 = make_float3(x[childx], y[childx], z[childx]);

                    // need to div again! otherwise will always find these 2 nodes to be in same
                    l_half *= 0.5;
                    l_box_center = update_box(l_box_center, l_half, child_octx);                
                    depth++;

                    // need to loop indef since could split region many times
                    // TODO: add logic where depth of tree should never exceed threshold
                    while (true){
                        // compute region for both bodies after split
                        int child1_octx = get_oct_idx(l_box_center, l_body_pos);
                        int child2_octx = get_oct_idx(l_box_center, l_body_pos2);

                        // easier subcase: 2 nodes belong in different regions
                        if (child1_octx != child2_octx){
                            // since we hold the lock for this parent, no other node can be touching this memory
                            // write body indexs into these 2 slots
                            children[new_cell*OCT_CHILDREN + child1_octx] = i;
                            children[new_cell*OCT_CHILDREN + child2_octx] = childx;

                            // must ensure that bodys are written to the leaf nodes before we release the lock
                            // without this fense, could release the lock before children are written to
                            __threadfence();
                            // RELEASING LOCK
                            children[nodex*OCT_CHILDREN + child_octx] = first_new_cell;
                            // RELEASING LOCK
                            inserted = true; 
                            break; 
                        }
                        else{
                            // 2 nodes lie in same region again; 
                            // alloc another internal node and set its parent to internal node
                            new_new_cell = atomicSub(next_cell, 1);
                            children[new_cell*OCT_CHILDREN + child1_octx] = new_new_cell;
                            new_cell = new_new_cell; 
                            // divide region to prepare for next level
                            l_half *= 0.5;
                            depth++;
                            l_box_center = update_box(l_box_center, l_half, child1_octx);
                        }
                        if (depth > depth_limit){
                            // skip this star
                            inserted = true; 
                            break;
                        }
                    }
                }
            }
        }
    }
}

/* 
bottom up traversal to compute center of mass for each node in OctTree
LEARNING: vol keyword
*/
__global__ void compute_cmass_kernel(float *x, float *y, float *z, float *mass, int *children,
    int first_cell, int last_cell, int N){
    
    // threads mapped to internal nodes that have been allocated in the tree construction phase
    int stride = blockDim.x*gridDim.x; 
    int global_tidx = threadIdx.x + blockIdx.x*blockDim.x; 
    for (int cell_id = first_cell+global_tidx; cell_id <= last_cell; cell_id += stride){
        //// computing center of gravity, total mass for internal node at cell_id ////
        float cell_x, cell_y, cell_z, cell_m; 
        cell_x = cell_y = cell_z = cell_m = 0; 
        // com accumulation weight
        float wt; 
        // vars to mark children we are waiting on
        int missing = 0; 
        int missing_children[OCT_CHILDREN];
        // checking if a node has finished
        int old_missing;
        bool finished = false;
        while (!finished){
            if (missing == 0){
                // accum gravity in bodys/ready internal nodes, mark missing node
                for (int i = 0; i < OCT_CHILDREN; ++i){
                    int child = children[cell_id*OCT_CHILDREN + i];
                    // if no child, skip
                    if (child == NULL_VAL_INT) continue;  
                    // if child is a body or a ready internal cell, accumulate
                    if (child < N || ((volatile float*)mass)[child] != NULL_VAL_FLOAT){
                        wt = (((volatile float*)mass)[child] / (cell_m + ((volatile float*)mass)[child]));
                        cell_x += (((volatile float*)x)[child]-cell_x) * wt; 
                        cell_y += (((volatile float*)y)[child]-cell_y) * wt;
                        cell_z += (((volatile float*)z)[child]-cell_z) * wt;
                        cell_m += ((volatile float*)mass)[child];
                    }
                    // otherwise, internal node is not ready
                    else {
                        missing++;
                        missing_children[missing-1] = child;
                    }
                }
            }
            // wait unil child nodes are finished
            // NOTE: this only runs a single time if the mass isnt ready
            if (missing != 0){
                do {
                    // cache prev missing value
                    old_missing = missing; 
                    // check if node has been updated
                    int cached_child = missing_children[missing-1];
                    // LOAD to GMEM
                    if (((volatile float*)mass)[cached_child] != NULL_VAL_FLOAT){ 
                        // if node ready, accum
                        wt = (((volatile float*)mass)[cached_child] / (cell_m + ((volatile float*)mass)[cached_child]));
                        cell_x += (((volatile float*)x)[cached_child]-cell_x) * wt; 
                        cell_y += (((volatile float*)y)[cached_child]-cell_y) * wt;
                        cell_z += (((volatile float*)z)[cached_child]-cell_z) * wt;
                        cell_m += ((volatile float*)mass)[cached_child];
                        missing--; 
                    }
                } while ((old_missing != missing) && missing != 0);
            }
            // trick: using synchronization primatives to reduce interconnect traffic
            // takes load off interconnect to main memory; if a thread has to wait, prevent it from querying GMEM
            // __syncthreads();

            if (missing == 0){
                // write center of gravity
                x[cell_id] = cell_x; y[cell_id] = cell_y; z[cell_id] = cell_z;
                // memory fence to ensure center of gravity updated before anyone reads it
                __threadfence();
                // mark this node as ready by writing total mass
                mass[cell_id] = cell_m;
                finished=true; 
            }
        }
    }
}

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
    float root_half, float *Fx, float *Fy, float *Fz, float theta){
    int stride = blockDim.x*gridDim.x; 
    int global_tidx = threadIdx.x + blockIdx.x*blockDim.x;
    // local agg forces
    float t_Fx, t_Fy, t_Fz; 
    // body stats we are computing forces onto
    float t_x, t_y, t_z, t_mass; 
    // traversal vars
    int parent;
    uint8_t parent_depth;
    int child; 
    // per thread stack for dfs
    int stack[STACK_SIZE];
    uint8_t depth_stack[STACK_SIZE];
    int top;
    
    //// accumulating grav forces on body_i ////
    for (int idx = global_tidx; idx < N; idx += stride){
        // init data for body_i    
        t_Fx = t_Fy = t_Fz = 0;
        t_x = x[idx]; t_y = y[idx]; t_z = z[idx]; t_mass = mass[idx];
        depth_stack[0] = 0;
        stack[0] = max_nodes-1;  
        top = 1; 
        // each thread performs a tree traversal; while stack not empty
        while (top > 0){
            // pop node and node_depth from the stack
            top--; 
            parent = stack[top]; parent_depth = depth_stack[top];
            // loop over children of this node             
            for (int i = 0; i < OCT_CHILDREN; ++i){
                child = children[parent*OCT_CHILDREN + i];
                // case 1: null = skip
                if (child == NULL_VAL_INT) continue;
                // compute distance from parent to child center of mass
                float dx = x[child] - t_x; 
                float dy = y[child] - t_y;
                float dz = z[child] - t_z;
                float r2 = dx*dx + dy*dy + dz*dz;
                float d = sqrtf(r2 + EPS_GPU);
                // case 2: leaf
                if (child < N){
                    // compute gravity between parent and child node 
                    float F = G_GPU*mass[child]*t_mass / (r2 + EPS_GPU);
                    // agg forces of leaf on body_i
                    t_Fx += F*dx / d;
                    t_Fy += F*dy / d;
                    t_Fz += F*dz / d;
                }
                // case 3: internal node. 
                else {
                    // compute s (width of child cell) 
                    float s = 2 * root_half / (1 << (parent_depth+1));
                    float threshold = s / d;
                    // case A: far enough away, we can approx
                    if (threshold < theta){
                        float F = G_GPU*mass[child]*t_mass / (r2 + EPS_GPU);
                        t_Fx += F*dx / d;
                        t_Fy += F*dy / d;
                        t_Fz += F*dz / d;                        
                    }
                    // case B; not far enough way, push child onto stack
                    else{                                 
                        stack[top] = child;
                        depth_stack[top] = parent_depth+1;
                        top++;   
                        }
                    }
                }
            }
        // write local foces to GMEM
        Fx[idx] = t_Fx; Fy[idx] = t_Fy; Fz[idx] = t_Fz;        
        }
    }

/* 
Apply accumulated forces on bodys to update pos and vel 
Streaming kernel; no use of sharmed mem
*/
__global__ void apply_forces_kernel(float *x, float *y, float *z, float *mass, 
    float *Vx, float *Vy, float *Vz, float *Fx, float *Fy, float *Fz, int N, float dt){
    int global_tidx = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = gridDim.x*blockDim.x;

    for (int i = global_tidx; i < N; i+=stride){
        float ax = Fx[i] / mass[i];
        float ay = Fy[i] / mass[i];
        float az = Fz[i] / mass[i];

        Vx[i] += dt*ax;
        Vy[i] += dt*ay;
        Vz[i] += dt*az;
        
        // update position of each body; using velocity at t+dt to update position at time t
        x[i] += dt*Vx[i];
        y[i] += dt*Vy[i];
        z[i] += dt*Vz[i];
    }
}