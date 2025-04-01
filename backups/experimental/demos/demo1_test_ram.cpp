#include "../../experimental/utils/andreinet.h"

#pragma comment(linker, "/STACK:20000000") // Set stack size to 2 MB
#pragma comment(linker, "/HEAP:20000000") // Set heap size to 2 MB

size_t getNetworkSize(const net& n) {
        size_t totalSize = sizeof(n);

        // Size of expected and costs vectors
        totalSize += n.expected.capacity() * sizeof(double);
        totalSize += n.costs.capacity() * sizeof(double);

        // Size of layers vector
        totalSize += n.layers.capacity() * sizeof(Layer);

        for (const auto& layer : n.layers) {
            // Size of neurons vector in each layer
            totalSize += layer.nodes.capacity() * sizeof(Node);
            
            
            
            for (const auto& neuron : layer.nodes) {
                // Size of weights and deltaWeights vectors in each neuron
                totalSize += neuron.next.capacity() * sizeof(double);
                //totalSize += neuron.deltaWeights.capacity() * sizeof(double);
            }
        }

        return totalSize;
    }


int main()
{
    //system("cls");
    t.start();
    cout << "Starting..." << endl;
    net n({10000, 50000, 10000, 10000, 10000});//55.4803mil weights, 965.25 MB
    //nr of weights
    cout << "Counting params..." << endl;
    int p = 0;
    for (int i = 0; i < n.layers.size() - 1; i++) {
        p += n.layers[i].nodes.size() * n.layers[i + 1].nodes.size();
    }
    // Usage
    size_t networkSize = getNetworkSize(n);
    //in mb
    cout << "Size of the network: " << (double)networkSize / 1024 / 1024 << " MB" << endl;
    
    t.stop(0);
    //cin.get();
    //comp per sec
    //convert t.duration to double std::chrono::duration<double>(d).count()

    //cout << "Comp per msec: " << (double)n.calculationCount / t.duration.count()*1000 << endl;
    
    
}