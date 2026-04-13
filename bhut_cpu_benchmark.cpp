#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include "bhut_cpu.h"
using namespace std;


// compute the N^2 approach
vector<Float3> brute_force(vector<Body> &bodys, vector<Float3> &velocitys, float dt){
    int N = bodys.size();
    // compute net Fx Fy Fz from all other stars via tree traversal
    vector<Float3> net_forces(N, {0.0f, 0.0f, 0.0f});  // init net force to zero each time
    // loop over each point and apply numerical method to update location of each point
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            // compute force of i on j; include the self force because the bhut code does as well
            agg_forces(bodys[i], bodys[j], net_forces[i]);
        }
    }
    update_points(bodys, velocitys, net_forces, dt);
    return net_forces;
}

int main(){
    bool only_bh = true;
    float dt = 0.1f;
    // skip larger traces for T=1
    vector<string> file_names = {"test/test_traces/test_10.txt", "test/test_traces/test_100.txt",
        "test/test_traces/test_500.txt", "test/test_traces/test_1000.txt", "test/test_traces/test_5000.txt",
        "test/test_traces/test_10000.txt", "test/test_traces/test_25000.txt", "test/test_traces/test_50000.txt", "test/test_traces/test_500000.txt",
        "test/test_traces/test_1000000.txt"};

    vector<float> thetas = {0.25f, 0.5f, 1.0f};

    // write CSV header
    ofstream csv("cpp_benchmark_results.csv");
    csv << "N,theta,brute_force_ms,barnes_hut_ms,speedup,avg_rel_error_pct\n";

    for (auto& file_name : file_names) {
        // read bodies once per file
        vector<Body> bodys_orig;
        ifstream file(file_name);
        if (!file) { cerr << "Cannot open " << file_name << "\n"; continue; }

        float x, y, z, mass;
        while (file >> x >> y >> z >> mass) {
            bodys_orig.emplace_back(Float3{x, y, z}, mass);
        }
        int N = bodys_orig.size();

        // brute force only needs to run once per file (independent of theta)
        vector<Body> bf_bodys = bodys_orig;
        vector<Float3> bf_velo(N, {0.0f, 0.0f, 0.0f});
        auto bf_start = chrono::high_resolution_clock::now();
        // vector<Float3> reff_forces = brute_force(bf_bodys, bf_velo, dt);
        vector<Float3> reff_forces(N, {0.0f, 0.0f, 0.0f});
        auto bf_end = chrono::high_resolution_clock::now();
        auto bf_ms = chrono::duration_cast<chrono::milliseconds>(bf_end - bf_start).count();

        cout << "N=" << N << "  Brute Force: " << bf_ms << "ms\n";

        for (float theta : thetas) {
            // fresh copy for each theta
            vector<Body> bh_bodys = bodys_orig;
            vector<Float3> bh_velo(N, {0.0f, 0.0f, 0.0f});

            auto bh_start = chrono::high_resolution_clock::now();
            vector<Float3> bh_forces = barnes_hut(bh_bodys, bh_velo, dt, theta);
            auto bh_end = chrono::high_resolution_clock::now();
            auto bh_ms = chrono::duration_cast<chrono::milliseconds>(bh_end - bh_start).count();

            // compute average relative error
            float total_rel_err = 0.0f;
            for (int i = 0; i < N; i++) {
                float dx = fabsf(bh_forces[i].x - reff_forces[i].x);
                float dy = fabsf(bh_forces[i].y - reff_forces[i].y);
                float dz = fabsf(bh_forces[i].z - reff_forces[i].z);
                float mag = sqrtf(reff_forces[i].x * reff_forces[i].x +
                                  reff_forces[i].y * reff_forces[i].y +
                                  reff_forces[i].z * reff_forces[i].z);
                float err = sqrtf(dx*dx + dy*dy + dz*dz) / (mag + 1e-10f);
                total_rel_err += err;
            }
            float avg_rel_err = total_rel_err / N;
            float speedup = (bh_ms > 0) ? (float)bf_ms / bh_ms : 0.0f;

            cout << "  theta=" << theta
                 << "  BH: " << bh_ms << "ms"
                 << "  speedup: " << speedup << "x"
                 << "  error: " << avg_rel_err * 100.0f << "%\n";

            csv << N << "," << theta << "," << bf_ms << "," << bh_ms << ","
                << speedup << "," << avg_rel_err * 100.0f << "\n";
        }
        cout << "\n";
    }
    csv.close();
    cout << "Results written to benchmark_results.csv\n";
}