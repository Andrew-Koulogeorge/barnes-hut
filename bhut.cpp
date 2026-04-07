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

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

const int EMPTY = 0; 
const int LEAF = 1; 
const int INTERNAL = 2; 
const int NUM_CHILDREN = 8; 

const float THETA = 1; 
constexpr double G = 6.67430e-11;

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
    // std::cout << "parent and child length of square: " << loc.length << " " << new_loc.length << "\n";
    // std::cout << "parent central location: " << loc.center.x << " " << loc.center.y << " " << loc.center.z << "\n";
    // std::cout << "child central location: " << new_loc.center.x << " " << new_loc.center.y << " " << new_loc.center.z << "\n";
    return new_loc;
}

/**
- prev node was a leaf node and now we need to place these points in children of the root
- if point we are inserting and previous point belong in same region, need to recursivly call div_regions
*/
void divide_regions(Body &b1, OctTreeNode *node){
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
    // std::cout << "width of parent node: " << node->loc.length << "\n";
    // std::cout << "width of b1: " << b1loc.length << "\n";
    // std::cout << "width of b2: " << b2loc.length << "\n";

    // case 1: 2 points dont belong to same region, create new Tree Nodes and return
    if (b1x != b2x){
        node->children[b1x] = new OctTreeNode(b1, b1loc, LEAF);
        node->children[b2x] = new OctTreeNode(b2, b2loc, LEAF);
    }
    // case 2: 2 points do belong in same region and we have to divide more
    else{
        // create new OctTreeNode and recursivly divide space
        // std::cout << "recursivly calling div regions \n";
        node->children[b2x] = new OctTreeNode(b2, b2loc, LEAF);
        divide_regions(b1,node->children[b2x]);
    }
}

/* recursivly insert point into root */
void insert(Body &body, OctTreeNode *node){

    // if current node does not have any points in it (empty node) make it a leaf node, and return
    if (node->type == EMPTY){
        // std::cout << "insert() found empty node \n";
        // no changes needed to center, established in init
        node->cntr_mass = body;
        node->type = LEAF;
    }
    else if (node->type == LEAF){
        // recursivley sub-divide this leaf node into internal node with children
        // std::cout << "insert() found divide_regions \n";
        divide_regions(body, node);
    }
    else if (node->type == INTERNAL){
        // std::cout << "insert() found an internal node \n";
        // update center of mass using running weighted sum
        float wt = (body.mass / node->cntr_mass.mass);
        node->cntr_mass.pt.x += wt * (body.pt.x - node->cntr_mass.pt.x);
        node->cntr_mass.pt.y += wt * (body.pt.y - node->cntr_mass.pt.y);
        node->cntr_mass.pt.z += wt * (body.pt.z - node->cntr_mass.pt.z);
        node->cntr_mass.mass += body.mass;
        // get region to place body into
        int bx = get_idx(body, node->loc);
        if (node->children[bx] == nullptr){
            // create new grid loc for this region 
            GridLoc bloc = get_loc(bx, node->loc);
            node->children[bx] = new OctTreeNode(body, bloc, EMPTY);
        }
        // std::cout << "insert() recursivly calling insert() \n";
        insert(body, node->children[bx]);
    }
    else{
        assert(false && "node can only be 3 types");
    }
}

OctTreeNode* build_tree(vector<Body> &bodys, float length){
    OctTreeNode *root = new OctTreeNode(length); 
    int N = bodys.size();
    // std::cout << "init loc of root should be L : " << root->loc.length << " \n"; 
    for (int i = 0; i < N; ++i){
        // std::cout << "inserting body " << i << " \n"; 
        insert(bodys[i], root);
    }
    return root; 

}

/* accumulate net forces acting on body1 */
void agg_forces(Body &body1, Body &body2, Float3 &net_forces){
    float dx = body2.pt.x - body1.pt.x;
    float dy = body2.pt.y - body1.pt.y;
    float dz = body2.pt.z - body1.pt.z;
    float r2 = dx*dx + dy*dy + dz*dz;
    float F = G*body1.mass*body2.mass / r2;
    net_forces.x += F*dx / std::sqrt(r2);
    net_forces.y += F*dy / std::sqrt(r2);
    net_forces.z += F*dz / std::sqrt(r2);
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
        float threshold = node->loc.length / d;
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
    Float3 net_force, velocity;
    Body body; 
    // loop over points and update velocity and position in place
    for (int i = 0; i < N; ++i){
        net_force = net_forces[i];
        body = bodys[i];
        velocity = velocitys[i];
        // TODO: COME BACK AND REVISIT PHYSICS EQUATIONS
        // compute acceleration with newtons law: F = ma -> a = F/m
        ax = net_force.x / body.mass;
        ay = net_force.y / body.mass;
        az = net_force.y / body.mass;

        // update velocity of each body (acceleration * dt = change in velo)
        // assuming acceleration is constant over dt interval; so we are computing velocity at time t+dt
        velocity.x = velocity.x + dt*ax;
        velocity.y = velocity.y + dt*ay;
        velocity.z = velocity.z + dt*az; 
        
        // update position of each body; using velocity at t+dt to update position at time t
        body.pt.x = body.pt.x + dt*velocity.x;
        body.pt.y = body.pt.y + dt*velocity.y;
        body.pt.z = body.pt.z + dt*velocity.z; 
    }
}

void barnes_hut(vector<Body> &bodys, vector<Float3> &velocitys, int total_iters, float dt, float length){
    int N = bodys.size();
    std::cout << "number of bodies: " << N << "\n";
    vector<Float3> net_forces(N, {0.0f, 0.0f, 0.0f}); 
    for (int i = 0; i < total_iters; ++i){
        // construct OctTree
        std::cout << "building tree \n";
        OctTreeNode *root = build_tree(bodys, length);

        // compute net Fx Fy Fz from all other stars via tree traversal
        std::cout << "traversing tree \n";
        traverse_tree(root, bodys, net_forces);
        
        // loop over each point and apply leap frog numerical method to update location of each point
        std::cout << "updating points \n";
        update_points(bodys, velocitys, net_forces, dt);
    }
}

int main(){
    // TODO: read in arguments for bhut program
    int total_iters = 3;
    float dt = 1; 

    vector<Body> bodys;
    vector<Float3> velo;    
    std::ifstream file("tests/test0.txt");

    float x,y,z,mass;
    int length;
    file >> length; // first line is bounding box length for points
    while (file >> x >> y >> z >> mass){
        bodys.emplace_back(Float3{x,y,z}, mass);
        velo.push_back(Float3{0.0f,0.0f,0.0f});
    }

    // ensure that we have read in the bodys correctly

    barnes_hut(bodys, velo, total_iters, dt, length);
}