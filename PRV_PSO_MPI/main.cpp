#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include "tinyxml2.h"

using namespace std;
using namespace tinyxml2;

const int vehicle_capacity = 160; // capacidade do veículo

struct Client {
    double cx;
    double cy;
    double demand;
};

vector<double> best_global_position;
vector<double> best_global_route;
double best_global_val;

vector<Client> client_data;

double get_distance(int client_a, int client_b) {
    double d = 0;

    double xa = client_data[client_a].cx;
    double ya = client_data[client_a].cy;
    double xb = client_data[client_b].cx;
    double yb = client_data[client_b].cy;

    d = sqrt(pow(xa - xb, 2) + pow(ya - yb, 2));

    return d;
}

int read_client_file(const string& filename) {
    XMLDocument doc;
    if (doc.LoadFile(filename.c_str()) != XML_SUCCESS) {
        cout << "Failed to load file " << filename << endl;
        return 0;
    }

    XMLElement* root = doc.FirstChildElement("instance");
    if (!root) {
        cout << "Invalid format: Missing root element" << endl;
        return 0;
    }

    XMLElement* network = root->FirstChildElement("network");
    if (!network) {
        cout << "Invalid format: Missing network element" << endl;
        return 0;
    }

    XMLElement* nodes = network->FirstChildElement("nodes");
    if (!nodes) {
        cout << "Invalid format: Missing nodes element" << endl;
        return 0;
    }

    XMLElement* node = nodes->FirstChildElement("node");
    int max_id = 0;
    while (node) {
        int id;
        node->QueryIntAttribute("id", &id);
        if (id > max_id) max_id = id;
        node = node->NextSiblingElement("node");
    }

    client_data.resize(max_id + 1);

    node = nodes->FirstChildElement("node");
    while (node) {
        int id;
        node->QueryIntAttribute("id", &id);
        node->FirstChildElement("cx")->QueryDoubleText(&client_data[id].cx);
        node->FirstChildElement("cy")->QueryDoubleText(&client_data[id].cy);
        node = node->NextSiblingElement("node");
    }

    XMLElement* requests = root->FirstChildElement("requests");
    if (!requests) {
        cout << "Invalid format: Missing requests element" << endl;
        return 0;
    }

    XMLElement* request = requests->FirstChildElement("request");
    while (request) {
        int node_id;
        request->QueryIntAttribute("node", &node_id);
        request->FirstChildElement("quantity")->QueryDoubleText(&client_data[node_id].demand);
        request = request->NextSiblingElement("request");
    }

    return 1;
}

class particle {
    static mt19937 mt;
    static uniform_real_distribution<double> init_dist;
    static uniform_real_distribution<double> update_dist;

    vector<double> current_position;
    vector<double> current_velocity;
    vector<double> current_route;
    double current_value;
    vector<double> best_position;
    double best_value;

public:
    particle(int size_particle) {
        current_position.resize(size_particle);
        best_position.resize(size_particle);
        current_velocity.resize(size_particle);
        for (int i = 0; i < size_particle; i++) {
            current_position[i] = init_dist(mt);
            best_position[i] = current_position[i];
            current_velocity[i] = 0;
        }
        current_route.resize(size_particle + 2);
        map_route();
        current_value = best_value = evaluate_route();
    }
    void map_route() {
        vector<pair<double, int>> indexed_positions;
        for (int i = 0; i < current_position.size(); ++i) {
            indexed_positions.push_back({ current_position[i], i });
        }
        sort(indexed_positions.begin(), indexed_positions.end());

        for (int i = 0; i < current_position.size(); ++i) {
            current_route[i + 1] = indexed_positions[i].second;
            if (current_route[i + 1] >= client_data.size() - 1) {
                current_route[i + 1] = client_data.size() - 1;
            }
        }
        current_route.front() = client_data.size() - 1;
        current_route.back() = client_data.size() - 1;
    }
    double evaluate_route() {
        double total = 0;
        double current_load = 0;

        for (int i = 1; i < current_route.size(); i++) {
            if (current_route[i] == client_data.size() - 1) {
                current_load = 0;
            }
            else {
                current_load += client_data[current_route[i]].demand;
                if (current_load > vehicle_capacity) {
                    total += 10000000;
                    current_load = client_data[current_route[i]].demand;
                }
            }
            total += get_distance(current_route[i], current_route[i - 1]);
        }
        return total;
    }
    void update(double inertia_coeff, double cognitive_coeff, double social_coeff) {
        for (int i = 0; i < current_position.size(); i++) {
            double r = update_dist(mt);
            current_velocity[i] = (inertia_coeff * current_velocity[i]) +
                (cognitive_coeff * r * (best_position[i] - current_position[i])) +
                (social_coeff * r * (best_global_position[i] - current_position[i]));
            current_position[i] += current_velocity[i];
        }
        map_route();
        current_value = evaluate_route();
        if (current_value < best_value) {
            best_position = current_position;
            best_value = current_value;
        }
    }

    vector<double> get_current_position() {
        return current_position;
    }
    vector<double> get_current_velocity() {
        return current_velocity;
    }
    vector<double> get_current_route() {
        return current_route;
    }
    double get_current_value() {
        return current_value;
    }
    vector<double> get_best_position() {
        return best_position;
    }
    double get_best_value() {
        return best_value;
    }
};

mt19937 particle::mt(random_device{}());
uniform_real_distribution<double> particle::init_dist(0, 1000.0);
uniform_real_distribution<double> particle::update_dist(0, 1.0);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (!read_client_file("CMT01.xml")) {
        MPI_Finalize();
        return 0;
    }

    int num_vehicles = 10; // numero de veiculos
    int size_particle = client_data.size() - 1 + num_vehicles - 1; // ajusta o tamanho da partícula
    best_global_position.resize(size_particle);
    best_global_route.resize(size_particle + 2);

    int num_particles = 1200; // Número de partículas
    int num_iterations = 5000; // Número de iterações

    int particles_per_proc = num_particles / world_size;
    vector<particle> swarm;
    swarm.reserve(particles_per_proc);
    for (int i = 0; i < particles_per_proc; i++) {
        swarm.emplace_back(size_particle);
    }
    best_global_val = 99999999;

    double inertia_coeff = 0.9;
    double cognitive_coeff = 1.5;
    double social_coeff = 1.5;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        for (int j = 0; j < particles_per_proc; j++) {
            swarm[j].update(inertia_coeff, cognitive_coeff, social_coeff);
        }

        double local_best_val = best_global_val;
        vector<double> local_best_position = best_global_position;
        vector<double> local_best_route = best_global_route;

        for (int j = 0; j < particles_per_proc; j++) {
            if (swarm[j].get_best_value() < local_best_val) {
                local_best_position = swarm[j].get_best_position();
                local_best_route = swarm[j].get_current_route();
                local_best_val = swarm[j].get_best_value();
            }
        }

        MPI_Allreduce(&local_best_val, &best_global_val, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (local_best_val == best_global_val) {
            best_global_position = local_best_position;
            best_global_route = local_best_route;
        }

        if (world_rank == 0) {
            inertia_coeff -= (0.5 / num_iterations);
        }
        MPI_Bcast(&inertia_coeff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);

    if (world_rank == 0) {
		cout << duration.count() << " segundos" << " = " << best_global_val << endl;
        /*cout << "Melhor rota: ";
        for (int i = 0; i < best_global_route.size(); i++) {
            cout << best_global_route[i] << " ";
        }
        cout << endl;
        cout << "Valor da melhor rota: " << best_global_val << endl << endl;*/
    }

    MPI_Finalize();
    return 0;
}