#include <iostream>
#include <vector>
#include <chrono>

struct node {
    double val = 0.0;
    double bias = 0.0;
    struct Connection {
        node* node;
        double weight;
    };
    std::vector<Connection> next;

    double activate(double val) {
        return val;
    }

    void passValues() {
        val += bias;
        val = activate(val);
        for (auto& nextNode : next) {
            nextNode.node->val = val * nextNode.weight + nextNode.node->val;
        }
    }

    void connect(node* nextNode, double w = 1.0) {
        next.push_back({nextNode, w});
    }
};

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // vector of 100 nodes connected to each other
    std::vector<node> n1(1000);
    std::vector<node> n2(10000);
    for (int i = 0; i < n1.size(); i++) {
        for (int j = 0; j < n2.size(); j++) {
            n1[i].connect(&n2[j]);
        }
    }
    // pass values through the network
    for (int i = 0; i < n1.size(); i++) {
        n1[i].passValues();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";

    return 0;
}