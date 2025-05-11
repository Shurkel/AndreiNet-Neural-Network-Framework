#include "andreinet.h"
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed, std::setprecision

// XOR training data
TrainingSet getXORData() {
    TrainingSet data;
    data.push_back({{0.0, 0.0}, {0.0}});
    data.push_back({{0.0, 1.0}, {1.0}});
    data.push_back({{1.0, 0.0}, {1.0}});
    data.push_back({{1.0, 1.0}, {0.0}});
    return data;
}

// Simple progress callback for training
void trainingProgress(int epoch, double ssr, double ce, double duration, bool finished) {
    std::cout << "Epoch [" << epoch << "] "
              << "Avg SSR: " << std::fixed << std::setprecision(6) << ssr << " | "
              << "Avg CE: " << std::fixed << std::setprecision(6) << ce << " | "
              << "Time: " << std::fixed << std::setprecision(2) << duration << " ms";
    if (finished) {
        std::cout << " (Finished)";
    }
    std::cout << std::endl;
}
/*
    Activ fx:
    -1 = Linear
    0 = ReLU
    1 = Sigmoid
    2 = Softplus
    */

int main() {
    //DEMO ANDEINET
    // 1. Create Network
    // 2,1,1 -> network structure
    // Default activations: Input=Linear, Hidden=Sigmoid, Output=Sigmoid
    Net net({2, 3, 1});     
    std::cout << "Created Network." << std::endl;
    net.printNetworkStructure();

    


    // 2. Get XOR 
    TrainingSet xorData = getXORData();
    std::cout << "\nXOR Training Data (Input -> Target):" << std::endl;
    for(const auto& pair : xorData) {
        std::cout << "[" << pair.first[0] << ", " << pair.first[1] << "] -> [" << pair.second[0] << "]" << std::endl;
    }

    // 3. Test predictions BEFORE training
    std::cout << "\nPredictions BEFORE training:" << std::endl;
    for (const auto& sample : xorData) {
        const auto& input = sample.first;
        std::vector<double> output = net.predict(input);
        std::cout << "Input: [" << input[0] << ", " << input[1] << "] -> Output: [" 
                  << std::fixed << std::setprecision(4) << output[0] 
                  << "] (Target: " << sample.second[0] << ")" << std::endl;
    }

    // 4. Train the network
    int epochs = 100; // Increased epochs for better convergence
    double learningRate = 0.1; 
    std::cout << "\nStarting Training for " << epochs << " epochs with LR: " << learningRate << std::endl;
    net.train(xorData, epochs, learningRate, 1, true, trainingProgress); // batchSize 1 = SGD
    
    std::cout << "\nTraining Finished." << std::endl;

    // 5. Test predictions AFTER training
    std::cout << "\nPredictions AFTER training:" << std::endl;
    double totalSSR = 0.0;
    for (const auto& sample : xorData) {
        const auto& input = sample.first;
        std::vector<double> output = net.predict(input);
        totalSSR += net.calculateSSR(sample.second); // Calculate SSR for this sample
        std::cout << "Input: [" << input[0] << ", " << input[1] << "] -> Output: [" 
                  << std::fixed << std::setprecision(4) << output[0] 
                  << "] (Target: " << sample.second[0] << ")" << std::endl;
    }
    std::cout << "Final Average SSR over test set: " << std::fixed << std::setprecision(6) << (totalSSR / xorData.size()) << std::endl;
    std::cout << "Final Average Cross Entropy (from training): " << net.averageCrossEntropy << std::endl;


    // 6. Print some network details (optional)
    net.printAllLayers(true); // Print all layers in detail
    net.printLayerDetails(1, true); // Print hidden layer (ID 1) in detail
    net.printNodeDetails(2, 0); // Print output node (Layer 2, Node 0)

    std::cout << "\nDemo finished." << std::endl;

    return 0;
}