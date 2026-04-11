#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include "bhut_cpu.h"
using namespace std;

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