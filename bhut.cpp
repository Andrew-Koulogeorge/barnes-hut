/**
Serial Implementation of bhut in cpp

new = allocating memory on the heap and returning pointer
can also directly call constructor to get access to object

TODO: 
1) change logic to ensure that center of mass of TreeNode when not leaf is actual center of mass
include logic to update center of mass in traversal
2) compute radius based on input points
3) physics update formulas
4) read in command line args
*/

#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
using namespace std;

const int EMPTY = 0; 
const int LEAF = 1; 
const int INTERNAL = 2; 
const int NUM_CHILDREN = 8; 
const int MAX_DEPTH = 30; 
const float THETA = 0.5; 
const float EPS = 0.001; // avoiding instability when 2 bodies get near

// constexpr double G = 6.67430e-11;
// constexpr double G = 1; // test
constexpr double G = 10; // test
// constexpr double G = 100; // test

struct Float3 {
    float x; 
    float y; 
    float z; 
    Float3(): x(0.0f), y(0.0f), z(0.0f) {}
    Float3(float x, float y, float z): x(x), y(y), z(z) {}
};

/* Body we are simulating; spatial loc and mass */
struct Body {
    Float3 pt; 
    float mass; 
    Body() : pt(), mass(0.0f) {}
    Body(Float3 pt, float m) : pt(pt), mass(m) {}
};

/* Each internal node in OctTree represents sqaure of total space; space defined by center and radius */
struct GridLoc {
    Float3 center;
    float length; 
    GridLoc(Float3 c, float l) : center(c), length(l) {}
};


// Node Struct
// when type = LEAF, cntr_mass = Body (points, mass)
// when type - Internal, cntr_mass cords store m1*p1x + m2*p2x and mass = total mass; enables quick insert & center of mass comp
struct OctTreeNode {
    Body cntr_mass; 
    GridLoc loc; 
    int type = EMPTY;               // empty, internal, leaf 
    vector<OctTreeNode*> children;  // array of pointers to children nodes? 

    // root constructor
    OctTreeNode(float l):
        cntr_mass(Float3{0.0f,0.0f,0.0f}, -1),
        loc(Float3{0.0f,0.0f,0.0f}, l), 
        type(EMPTY), 
        children(NUM_CHILDREN, nullptr) {}
    // node constructor
    OctTreeNode(Body bdy, GridLoc loc, int type) : 
        cntr_mass(bdy), loc(loc), type(type), children(NUM_CHILDREN, nullptr) {}
};

void free_tree(OctTreeNode *node, int d){
    // loop over all children that are not null and call free free on them 
    // delete node after visiting children
    if (d > 1000){
        std::cerr << "tree depth: " << d << "\n";
    }
    for(auto& child : node->children){
        if (child == nullptr) continue;
        free_tree(child, d+1);
    }
    delete node; 
}

/* computing which region of space to place Body */
int get_idx(Body &base, GridLoc &loc){
    int idx = 0; 
    if (base.pt.x >= loc.center.x) idx |= 1;
    if (base.pt.y >= loc.center.y) idx |= 2;
    if (base.pt.z >= loc.center.z) idx |= 4;
    return idx; 
}

// compute updated center of node based on reff point
GridLoc get_loc(int octrant, GridLoc &loc){
    GridLoc new_loc(Float3{loc.center.x, loc.center.y, loc.center.z}, loc.length / 2.0);
    // for each coord, we move center half way in either direction
    if (octrant & 4) new_loc.center.z += loc.length/4;
    else new_loc.center.z -= loc.length/4;
    if (octrant & 2) new_loc.center.y += loc.length/4;
    else new_loc.center.y -= loc.length/4;
    if (octrant & 1) new_loc.center.x += loc.length/4;
    else new_loc.center.x -= loc.length/4;
    return new_loc;
}

// merge center of mass 
void merge_bodys(Body &body, OctTreeNode *node){
    float wt = (body.mass / (node->cntr_mass.mass + body.mass));
    node->cntr_mass.pt.x += wt * (body.pt.x - node->cntr_mass.pt.x);
    node->cntr_mass.pt.y += wt * (body.pt.y - node->cntr_mass.pt.y);
    node->cntr_mass.pt.z += wt * (body.pt.z - node->cntr_mass.pt.z);
    node->cntr_mass.mass += body.mass;    
}

/**
- prev node was a leaf node and now we need to place these points in children of the root
- if point we are inserting and previous point belong in same region, need to recursivly call div_regions
*/
void divide_regions(Body &b1, OctTreeNode *node, int depth){
    if (depth > MAX_DEPTH){
        merge_bodys(b1, node);
        node->type = LEAF; 
        return;
    }
    // when node is leaf, center of mass is just the point 
    Body b2 = node->cntr_mass; 
    
    // update node to be internal
    float total_mass = b1.mass + b2.mass;
    node->cntr_mass.pt.x = (b1.pt.x*b1.mass + b2.pt.x*b2.mass) / total_mass;
    node->cntr_mass.pt.y = (b1.pt.y*b1.mass + b2.pt.y*b2.mass) / total_mass;
    node->cntr_mass.pt.z = (b1.pt.z*b1.mass + b2.pt.z*b2.mass) / total_mass;
    node->cntr_mass.mass = total_mass; 
    node->type = INTERNAL;
    
    // construct new Nodes and add them as children into existing parent
    int b1x = get_idx(b1, node->loc);
    GridLoc b1loc = get_loc(b1x, node->loc);

    int b2x = get_idx(b2, node->loc);
    GridLoc b2loc = get_loc(b2x, node->loc);

    // at this point, the grid b1 and b2 should be half the width of node

    // case 1: 2 points dont belong to same region, create new Tree Nodes and return
    if (b1x != b2x){
        node->children[b1x] = new OctTreeNode(b1, b1loc, LEAF);
        node->children[b2x] = new OctTreeNode(b2, b2loc, LEAF);
    }
    // case 2: 2 points do belong in same region and we have to divide more
    else{
        // create new OctTreeNode and recursivly divide space
        node->children[b2x] = new OctTreeNode(b2, b2loc, LEAF);
        divide_regions(b1,node->children[b2x], depth+1);
    }
}

/* recursivly insert point into root */
void insert(Body &body, OctTreeNode *node, int depth){
    // if depth is larger than threshold, we will merge these 2 nodes into 1; node will become a leaf
    if (depth > MAX_DEPTH){
        merge_bodys(body, node);
        node->type = LEAF; 
        return;
    }
    // if current node does not have any points in it (empty node) make it a leaf node, and return
    if (node->type == EMPTY){
        // no changes needed to center, established in init
        node->cntr_mass = body;
        node->type = LEAF;
    }
    else if (node->type == LEAF){
        // recursivley sub-divide this leaf node into internal node with children
        divide_regions(body, node, depth+1);
    }
    else if (node->type == INTERNAL){
        // update center of mass using running weighted sum
        merge_bodys(body, node);

        // get region to place body into
        int bx = get_idx(body, node->loc);
        if (node->children[bx] == nullptr){
            // create new grid loc for this region 
            GridLoc bloc = get_loc(bx, node->loc);
            node->children[bx] = new OctTreeNode(body, bloc, EMPTY);
        }
        insert(body, node->children[bx], depth+1);
    }
    else{
        assert(false && "node can only be 3 types");
    }
}

OctTreeNode* build_tree(vector<Body> &bodys, float length){
    OctTreeNode *root = new OctTreeNode(length); 
    int N = bodys.size();
    for (int i = 0; i < N; ++i){
        insert(bodys[i], root, 0);
    }
    return root; 

}

/* accumulate net forces acting on body1 */
void agg_forces(Body &body1, Body &body2, Float3 &net_forces){
    float dx = body2.pt.x - body1.pt.x;
    float dy = body2.pt.y - body1.pt.y;
    float dz = body2.pt.z - body1.pt.z;
    float r2 = dx*dx + dy*dy + dz*dz;
    // F = (Gm1m2) / r^2
    float F = G*body1.mass*body2.mass / (r2+EPS);
    net_forces.x += F*dx / std::sqrt(r2+EPS);
    net_forces.y += F*dy / std::sqrt(r2+EPS);
    net_forces.z += F*dz / std::sqrt(r2+EPS);
}

/* traverse oct tree, accumulating net force acted on the body */
void traverse(OctTreeNode *node, Body &body, Float3 &net_force){
    // case 1: leaf node. Compute force between node center of mass (just a body)
    if (node->type == LEAF){
        agg_forces(body, node->cntr_mass, net_force);
    }
    else if (node->type == INTERNAL){
        // case 2: internal node. compute s / d where s = width of region, d = distance between body and center of mass
        float dx = node->cntr_mass.pt.x - body.pt.x;
        float dy = node->cntr_mass.pt.y - body.pt.y;
        float dz = node->cntr_mass.pt.z - body.pt.z;
        float d = dx*dx + dy*dy + dz*dz;
        float threshold = node->loc.length / std::sqrt(d + EPS);
        // if s / d < theta, accumulate net force of center of mass
        if (threshold < THETA){
            agg_forces(body, node->cntr_mass, net_force);
        }
        else{
            for (int i = 0; i < NUM_CHILDREN; ++i){
                if(node->children[i] == nullptr || node->children[i]->type == EMPTY) 
                    continue;  
                traverse(node->children[i], body, net_force);
            }
        }
    }
}

/* populate net forces for each point with tree traversal */
void traverse_tree(OctTreeNode *root, vector<Body> &bodys, vector<Float3> &net_forces){
    int N = bodys.size(); 
    for (int i = 0; i < N; ++i){
        traverse(root, bodys[i], net_forces[i]);
    }
}

void update_points(vector<Body> &bodys, vector<Float3> &velocitys, vector<Float3> &net_forces, float dt){
    int N = bodys.size();
    float ax, ay, az; 
    // loop over points and update velocity and position in place
    for (int i = 0; i < N; ++i){

        // TODO: COME BACK AND REVISIT PHYSICS EQUATIONS
        // compute acceleration with newtons law: F = ma -> a = F/m
        ax = net_forces[i].x / bodys[i].mass;
        ay = net_forces[i].y / bodys[i].mass;
        az = net_forces[i].z / bodys[i].mass;

        // update velocity of each body (acceleration * dt = change in velo)
        // assuming acceleration is constant over dt interval; so we are computing velocity at time t+dt
        velocitys[i].x += dt*ax;
        velocitys[i].y += dt*ay;
        velocitys[i].z += dt*az; 
        
        // update position of each body; using velocity at t+dt to update position at time t
        bodys[i].pt.x += dt*velocitys[i].x;
        bodys[i].pt.y += dt*velocitys[i].y;
        bodys[i].pt.z += dt*velocitys[i].z; 
    }
}

/* after each iteration, write point locations to output file */
void write_bodies(std::ofstream& file, const vector<Body>& bodys){
    int N = bodys.size();
    for (auto& b : bodys){
        file << b.pt.x << " " << b.pt.y << " " << b.pt.z << " " << b.mass << "\n";
    }
    file << "\n";
}

float compute_box(vector<Body> &bodys){
    int N = bodys.size();
    float L = 0; 
    const float margin = 1;
    // want to compute max coordinate x or y
    for (auto& b: bodys){
        L = std::max(L, std::abs(b.pt.x));
        L = std::max(L, std::abs(b.pt.y));
        L = std::max(L, std::abs(b.pt.z));
    }
    return 2*L + margin; 
}

void barnes_hut(vector<Body> &bodys, vector<Float3> &velocitys, int total_iters, 
    float dt, bool record){
    int N = bodys.size();
    std::ofstream outfile("outputs/output1.txt");
    if (record){
        write_bodies(outfile, bodys);
    }
    vector<Float3> net_forces(N, {0.0f, 0.0f, 0.0f});  // init net force to zero each time
    for (int i = 0; i < total_iters; ++i){
        // compute bounding box based on body locations
        float length = compute_box(bodys);
        // construct OctTree
        OctTreeNode *root = build_tree(bodys, length);

        // compute net Fx Fy Fz from all other stars via tree traversal
        traverse_tree(root, bodys, net_forces);
        
        // loop over each point and apply numerical method to update location of each point
        update_points(bodys, velocitys, net_forces, dt);
        if (record) write_bodies(outfile, bodys);
        free_tree(root, 0);
        root = nullptr;
        net_forces.assign(N, {0.0f, 0.0f, 0.0f});
    }
}

/// BRUTE FORCE CODE

// compute the N^2 approach
void brute_force(vector<Body> &bodys, vector<Float3> &velocitys, int total_iters, float dt, bool record){
    int N = bodys.size();
    std::ofstream outfile("outputs/output1.txt");
    if (record) write_bodies(outfile, bodys);
    
    // compute net Fx Fy Fz from all other stars via tree traversal
    vector<Float3> net_forces(N, {0.0f, 0.0f, 0.0f});  // init net force to zero each time
    for (int i = 0; i < total_iters; ++i){
        // std::cout << "iter i: " << i << "\n";
        // loop over each point and apply numerical method to update location of each point
        for (int i = 0; i < N; ++i){
            for (int j = 0; j < N; ++j){
                // compute force of i on j; include the self force because the bhut code does as well
                agg_forces(bodys[i], bodys[j], net_forces[i]);
            }
        }
        update_points(bodys, velocitys, net_forces, dt);
        if (record) write_bodies(outfile, bodys);
        net_forces.assign(N, {0.0f, 0.0f, 0.0f});
    }    
}
/// BRUTE FORCE CODE

int main(){
    // TODO: read in arguments for bhut program
    int total_iters = 1;
    float dt = 0.1; 
    bool record = false; 
    vector<string> file_names = {"tests/test_10.txt", "tests/test_100.txt", 
        "tests/test_500.txt", "tests/test_1000.txt", "tests/test_5000.txt", 
        "tests/test_10000.txt", "tests/test_25000.txt", "tests/test_50000.txt"};
    std::cout << "running tests with theta= " << THETA << "\n";
    for(auto& file_name: file_names){        
        vector<Body> bodys;
        vector<Float3> velo;    
        std::ifstream file(file_name);

        float x,y,z,mass;
        while (file >> x >> y >> z >> mass){
            bodys.emplace_back(Float3{x,y,z}, mass);
            velo.push_back(Float3{0.0f,0.0f,0.0f});
        }
        std::cout << "N=" << bodys.size() << "\n";

        auto bforce_start = chrono::high_resolution_clock::now();
        brute_force(bodys, velo, total_iters, dt, record);
        auto bforce_end = chrono::high_resolution_clock::now();
        
        auto bf_ms = chrono::duration_cast<chrono::milliseconds>(bforce_end-bforce_start);
        std::cout << "Brute Force speed: "<< bf_ms.count() << "ms \n";

        auto bhut_start = chrono::high_resolution_clock::now();
        barnes_hut(bodys, velo, total_iters, dt, record);
        auto bhut_end = chrono::high_resolution_clock::now();

        auto bh_ms = chrono::duration_cast<chrono::milliseconds>(bhut_end-bhut_start);
        std::cout << "Barnes-Hut speed: " << bh_ms.count() << "ms \n";
    }
}