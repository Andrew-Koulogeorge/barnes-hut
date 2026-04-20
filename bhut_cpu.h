using namespace std;

const int EMPTY = 0; 
const int LEAF = 1; 
const int INTERNAL = 2; 
const int NUM_CHILDREN = 8; 
const int MAX_DEPTH = 30; 
const float EPS = 1e-2; // avoiding instability when 2 bodies get near
constexpr double G = 1; // test

// constexpr double G = 6.67430e-11;
// constexpr double G = 10; // test
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
    float half; // this is half in gpu
    GridLoc(Float3 c, float l) : center(c), half(l) {}
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

// struct to store time of each kernel
struct BHPhaseTimes {
    long long compute_box_us;
    long long build_tree_us;
    long long traverse_tree_us;
    long long update_points_us;
};

void free_tree(OctTreeNode *node, int d);

/* computing which region of space to place Body */
int get_idx(Body &base, GridLoc &loc);

// compute updated center of node based on reff point
GridLoc get_loc(int octrant, GridLoc &loc);

// merge center of mass 
void merge_bodys(Body &body, OctTreeNode *node);

/**
- prev node was a leaf node and now we need to place these points in children of the root
- if point we are inserting and previous point belong in same region, need to recursivly call div_regions
*/
void divide_regions(Body &b1, OctTreeNode *node, int depth);

/* recursivly insert point into root */
void insert(Body &body, OctTreeNode *node, int depth);

OctTreeNode* build_tree(vector<Body> &bodys, float length);

/* accumulate net forces acting on body1 */
void agg_forces(Body &body1, Body &body2, Float3 &net_forces);

/* traverse oct tree, accumulating net force acted on the body */
void traverse(OctTreeNode *node, Body &body, Float3 &net_force);

/* populate net forces for each point with tree traversal */
void traverse_tree(OctTreeNode *root, vector<Body> &bodys, vector<Float3> &net_forces, float theta);

void update_points(vector<Body> &bodys, vector<Float3> &velocitys, vector<Float3> &net_forces, float dt);

float compute_box(vector<Body> &bodys);

vector<Float3> barnes_hut(vector<Body> &bodys, vector<Float3> &velocitys, float dt, float theta, BHPhaseTimes &times);