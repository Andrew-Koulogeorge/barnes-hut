/*
Barnes-Hut implementation in CUDA
@author: Andrew Koulogeorge
*/

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// for prealloc total number of nodes
int const MULT = 3; 
int const OCT_CHILDREN = 8; 
int const NULL_VAL = -1
int const LOCK_VAL = -2; 

///////////// BEGIN DEVICE HELPER SECTION  /////////////
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

/* reduction over bodys */
__global__ void box_kernel(){
    return;
}

/** 
populate child_pointers array (form OctTree structure)
thread block structure: 1d grid and 1d blocks
// assume box is centered at (0,0,0) with half length L
*/
__global__ void build_tree(float *x, float *y, float *z, int *children, int *next_cell,
    int N, int max_nodes, float root_half){
    
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
    
    //// thread inserting body_i into tree ////   
    for (int i = tx; i < N; i += grid_stride){      
        // begin traversal at root
        nodex = max_nodes-1; 
        l_half = root_half

        // read body pos from global memory
        l_body_pos.x = x[i]; l_body_pos.y = y[i]; l_body_pos.z = z[i];
        
        // init center at origin 
        l_box_center = float3({0.0f, 0.0f, 0.0f});

        // keep traversing until node is inserted
        bool inserted = false;     
        while (!inserted) {
            // get child node region 
            child_octx = get_oct_idx(l_box_center, l_body_pos);
            // get index of child: 
            // 0:N-1 = body
            // N < = internal 
            // -1 = nullptr            
            childx = children[nodex*OCT_CHILDREN + child_octx];

            // case 0: child is internal node; keep traversing
            if (childx > N){
                // move down a level; update bounding box and curr node
                half *= 0.5;
                l_box_center = update_box(l_box_center, half, child_octx);                
                nodex = childx; 
                continue
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
                // Q: what is preventing a thread from reading LOCK_val on its retry and getting back to this atomic
                // CAS operation and finding that lock == childx == LOCK_VAL ?

                // LOCK OBTAINED AT THIS POINT // 
                // case 1: child is null pointer; simply insert body as child 
                if (childx == NULL_VAL){
                    children[nodex*OCT_CHILDREN + child_octx] = i; 
                    inserted = true;
                }
                // case 2: child is body; subdiv region at least once
                else {
                    // save original child_octx
                    int og_child_octx = child_octx;
                    // split region
                    half *= 0.5;
                    l_box_center = update_box(l_box_center, half, child_octx);
                    
                    // need to add new internal node to OctTree --> decrement counter 
                    int new_cell = atomicSub(&next_cell, 1);
                    
                    // child is body --> we have access to 2 body coords;
                    float3 l_body_pos2 = make_float3({x[childx], y[childx], z[childx]})

                    // compute region for both bodies after split
                    child_octx = get_oct_idx(l_box_center, l_body_pos);
                    int child2_octx = get_oct_idx(l_box_center, l_body_pos2);

                    // easier subcase: 2 nodes belong in different regions
                    if (child_octx != child2_octx){
                        // since we hold the lock for this parent, no other node can be touching this memory
                        children[childx*OCT_CHILDREN + child_octx] = i;
                        children[childx*OCT_CHILDREN + child2_octx] = childx;

                        // THREAD FENSE GOES HERE //

                        // have parent point to this new internal node
                        // Q: why cant this go earlier?
                        children[nodex*OCT_CHILDREN + og_child_octx] = new_cell;
                        inserted = true; 
                    }
                    else{
                        // need to divide further
                    }
                }
            }
        }

        // reached null_ptr or leaf node; try and insert node here
        // this will require attempting to grab a lock (-2 value = got lock)

        // QUESTION: what if we fail to grab the lock and someone else adds an internal node? 
        // do we have to resume the above while loop?
    }
    return;
}

/* bottom up traversal to compute center of mass for each node in OctTree*/
__global__ void compute_cmass(){
    return;
}

/* traverse OctTree to approximate forces on each body */
__global__ void compute_forces(){
    return;
}

/* apply accumulated forces on bodys to update pos and vel */
__global__ void apply_forces(){
    return; 
}

///////////// END CORE KERNEL SECTION  /////////////

/*
wrapper around GPU kernels for barnes-hut computation
*/
void barnes_hut_cudav0(vector<float4> &bodys, vector<float3> &velocitys, int total_iters, float dt){
    //// START DATA INIT ////

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

    // alloc data on device
    float *d_x, *d_y, *d_z, *d_mass; 
    float *d_vx, *d_vy, *d_vz;
    float *d_Fx, *d_Fy, *d_Fz;
    int *d_children; 
    int *d_next_cell; 
    float3 *d_box_min, *d_box_max;
    
    cudaMalloc(&d_x, max_nodes * sizeof(float));
    cudaMalloc(&d_y, max_nodes * sizeof(float));
    cudaMalloc(&d_z, max_nodes * sizeof(float));
    cudaMalloc(&d_mass, max_nodes * sizeof(float));
    
    cudaMalloc(&d_vx, max_nodes * sizeof(float));
    cudaMalloc(&d_vy, max_nodes * sizeof(float));
    cudaMalloc(&d_vz, max_nodes * sizeof(float));
    
    cudaMalloc(&d_Fx, max_nodes * sizeof(float));
    cudaMalloc(&d_Fy, max_nodes * sizeof(float));
    cudaMalloc(&d_Fz, max_nodes * sizeof(float));

    cudaMalloc(&d_children, OCT_CHILDREN * max_nodes * sizeof(int));
    cudaMalloc(&d_next_cell, sizeof(int));

    cudaMalloc(&d_box_min, sizeof(float3));
    cudaMalloc(&d_box_max, sizeof(float3));    
    
    // copy data to device 
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice);

    // forces init 0
    cudaMemSet(d_Fx, 0, max_nodes * sizeof(float));
    cudaMemSet(d_Fy, 0, max_nodes * sizeof(float));
    cudaMemSet(d_Fz, 0, max_nodes * sizeof(float));

    // set children pointers to be NULL_VAL
    cudaMemSet(d_children, NULL_VAL, OCT_CHILDREN * max_nodes * sizeof(int));
    
    // set next_cell to be the last index 
    int h_next_cell = max_nodes-1;
    cudaMemcpy(d_next_cell, &h_next_cell, sizeof(int), cudaMemcpyHostToDevice)
    
    //// END DATA INIT ////
    
    //// START CORE KERNELS ////

    // [Easy; but need to review reduction kernels]
    //  0) compute bounding box (can be done simply by manipulating body)

    // 1) construct oct tree in parallel 



    // 2) compute center of mass for each boy

    // 3) compute forces acted on each body (naivly, 1 thread per body, traverse the tree)

    // 4) update position of bodies based on computed net forces (very parallel; fixed computation per body)

    //// START CORE KERNELS ////

    //// DATA COPY BACK AND CLEAN UP ////

    //// DATA COPY BACK AND CLEAN UP ////
    
}


int main(){
    int total_iters = 1;
    float dt = 0.1; 
    vector<float4> bodys;
    vector<float3> velo;    
    std::ifstream file("tests/test_5000.txt");    
    // read in bodies from file in same manner as we did in cpp
    float x,y,z,mass;
    while (file >> x >> y >> z >> mass){
        bodys.emplace_back(x,y,z,mass);
        velo.push_back(0.0f,0.0f,0.0f);
    }
    std::cout << "N=" << bodys.size() << "\n";

    barnes_hut_cudav0(bodys, velo, total_iters, dt)
}