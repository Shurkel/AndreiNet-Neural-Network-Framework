#include "../utils/andreinet.h" // Include the main library header
#include <iostream>
#include <vector>
#include <iomanip>    // For std::setprecision, std::fixed
#include <numeric>    // For std::accumulate if needed elsewhere
#include <algorithm>  // For std::max_element if needed elsewhere
#include <map>        // For class names

// --- Helper functions (copy from above) ---
int get_class_index(const std::vector<double>& one_hot) {
    for (size_t i = 0; i < one_hot.size(); ++i) {
        if (one_hot[i] > 0.5) { return static_cast<int>(i); }
    }
    return -1;
}

int argmax(const std::vector<double>& vec) {
    if (vec.empty()) { return -1; }
    double max_val = vec[0];
    int max_idx = 0;
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > max_val) {
            max_val = vec[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

// --- Iris Data Loading (copy the full function from above) ---
TrainingSet get_iris_data() {
    TrainingSet irisData;
    // *** PASTE THE FULL 150 SAMPLES HERE ***
    // Example structure:
    // Iris Setosa (Class 0)
    irisData.push_back({{5.1, 3.5, 1.4, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.9, 3.0, 1.4, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.7, 3.2, 1.3, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.6, 3.1, 1.5, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.0, 3.6, 1.4, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.4, 3.9, 1.7, 0.4}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.6, 3.4, 1.4, 0.3}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.0, 3.4, 1.5, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.4, 2.9, 1.4, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.9, 3.1, 1.5, 0.1}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.4, 3.7, 1.5, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.8, 3.4, 1.6, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.8, 3.0, 1.4, 0.1}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.3, 3.0, 1.1, 0.1}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.8, 4.0, 1.2, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.7, 4.4, 1.5, 0.4}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.4, 3.9, 1.3, 0.4}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.1, 3.5, 1.4, 0.3}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.7, 3.8, 1.7, 0.3}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.1, 3.8, 1.5, 0.3}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.4, 3.4, 1.7, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.1, 3.7, 1.5, 0.4}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.6, 3.6, 1.0, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.1, 3.3, 1.7, 0.5}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.8, 3.4, 1.9, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.0, 3.0, 1.6, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.0, 3.4, 1.6, 0.4}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.2, 3.5, 1.5, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.2, 3.4, 1.4, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.7, 3.2, 1.6, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.8, 3.1, 1.6, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.4, 3.4, 1.5, 0.4}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.2, 4.1, 1.5, 0.1}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.5, 4.2, 1.4, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.9, 3.1, 1.5, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.0, 3.2, 1.2, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.5, 3.5, 1.3, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.9, 3.6, 1.4, 0.1}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.4, 3.0, 1.3, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.1, 3.4, 1.5, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.0, 3.5, 1.3, 0.3}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.5, 2.3, 1.3, 0.3}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.4, 3.2, 1.3, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.0, 3.5, 1.6, 0.6}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.1, 3.8, 1.9, 0.4}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.8, 3.0, 1.4, 0.3}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.1, 3.8, 1.6, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{4.6, 3.2, 1.4, 0.2}, {1.0, 0.0, 0.0}});
    irisData.push_back({{5.3, 3.7, 1.5, 0.2}, {1.0, 0.0, 0.0}}); // 50 Setosa

    // Iris Versicolor (Class 1)
    irisData.push_back({{7.0, 3.2, 4.7, 1.4}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.4, 3.2, 4.5, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.9, 3.1, 4.9, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.5, 2.3, 4.0, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.5, 2.8, 4.6, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.7, 2.8, 4.5, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.3, 3.3, 4.7, 1.6}, {0.0, 1.0, 0.0}});
    irisData.push_back({{4.9, 2.4, 3.3, 1.0}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.6, 2.9, 4.6, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.2, 2.7, 3.9, 1.4}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.0, 2.0, 3.5, 1.0}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.9, 3.0, 4.2, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.0, 2.2, 4.0, 1.0}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.1, 2.9, 4.7, 1.4}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.6, 2.9, 3.6, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.7, 3.1, 4.4, 1.4}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.6, 3.0, 4.5, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.8, 2.7, 4.1, 1.0}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.2, 2.2, 4.5, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.6, 2.5, 3.9, 1.1}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.9, 3.2, 4.8, 1.8}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.1, 2.8, 4.0, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.3, 2.5, 4.9, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.1, 2.8, 4.7, 1.2}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.4, 2.9, 4.3, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.6, 3.0, 4.4, 1.4}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.8, 2.8, 4.8, 1.4}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.7, 3.0, 5.0, 1.7}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.0, 2.9, 4.5, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.7, 2.6, 3.5, 1.0}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.5, 2.4, 3.8, 1.1}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.5, 2.4, 3.7, 1.0}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.8, 2.7, 3.9, 1.2}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.0, 2.7, 5.1, 1.6}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.4, 3.0, 4.5, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.0, 3.4, 4.5, 1.6}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.7, 3.1, 4.7, 1.5}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.3, 2.3, 4.4, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.6, 3.0, 4.1, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.5, 2.5, 4.0, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.5, 2.6, 4.4, 1.2}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.1, 3.0, 4.6, 1.4}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.8, 2.6, 4.0, 1.2}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.0, 2.3, 3.3, 1.0}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.6, 2.7, 4.2, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.7, 3.0, 4.2, 1.2}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.7, 2.9, 4.2, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{6.2, 2.9, 4.3, 1.3}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.1, 2.5, 3.0, 1.1}, {0.0, 1.0, 0.0}});
    irisData.push_back({{5.7, 2.8, 4.1, 1.3}, {0.0, 1.0, 0.0}}); // 50 Versicolor

    // Iris Virginica (Class 2)
    irisData.push_back({{6.3, 3.3, 6.0, 2.5}, {0.0, 0.0, 1.0}});
    irisData.push_back({{5.8, 2.7, 5.1, 1.9}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.1, 3.0, 5.9, 2.1}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.3, 2.9, 5.6, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.5, 3.0, 5.8, 2.2}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.6, 3.0, 6.6, 2.1}, {0.0, 0.0, 1.0}});
    irisData.push_back({{4.9, 2.5, 4.5, 1.7}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.3, 2.9, 6.3, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.7, 2.5, 5.8, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.2, 3.6, 6.1, 2.5}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.5, 3.2, 5.1, 2.0}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.4, 2.7, 5.3, 1.9}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.8, 3.0, 5.5, 2.1}, {0.0, 0.0, 1.0}});
    irisData.push_back({{5.7, 2.5, 5.0, 2.0}, {0.0, 0.0, 1.0}});
    irisData.push_back({{5.8, 2.8, 5.1, 2.4}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.4, 3.2, 5.3, 2.3}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.5, 3.0, 5.5, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.7, 3.8, 6.7, 2.2}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.7, 2.6, 6.9, 2.3}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.0, 2.2, 5.0, 1.5}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.9, 3.2, 5.7, 2.3}, {0.0, 0.0, 1.0}});
    irisData.push_back({{5.6, 2.8, 4.9, 2.0}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.7, 2.8, 6.7, 2.0}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.3, 2.7, 4.9, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.7, 3.3, 5.7, 2.1}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.2, 3.2, 6.0, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.2, 2.8, 4.8, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.1, 3.0, 4.9, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.4, 2.8, 5.6, 2.1}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.2, 3.0, 5.8, 1.6}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.4, 2.8, 6.1, 1.9}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.9, 3.8, 6.4, 2.0}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.4, 2.8, 5.6, 2.2}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.3, 2.8, 5.1, 1.5}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.1, 2.6, 5.6, 1.4}, {0.0, 0.0, 1.0}});
    irisData.push_back({{7.7, 3.0, 6.1, 2.3}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.3, 3.4, 5.6, 2.4}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.4, 3.1, 5.5, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.0, 3.0, 4.8, 1.8}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.9, 3.1, 5.4, 2.1}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.7, 3.1, 5.6, 2.4}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.9, 3.1, 5.1, 2.3}, {0.0, 0.0, 1.0}});
    irisData.push_back({{5.8, 2.7, 5.1, 1.9}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.8, 3.2, 5.9, 2.3}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.7, 3.3, 5.7, 2.5}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.7, 3.0, 5.2, 2.3}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.3, 2.5, 5.0, 1.9}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.5, 3.0, 5.2, 2.0}, {0.0, 0.0, 1.0}});
    irisData.push_back({{6.2, 3.4, 5.4, 2.3}, {0.0, 0.0, 1.0}});
    irisData.push_back({{5.9, 3.0, 5.1, 1.8}, {0.0, 0.0, 1.0}}); // 50 Virginica

    return irisData;
}

// --- Main Function ---
int main() {
    std::cout << "--- andreiNET Iris Classification Demo ---" << std::endl;

    // 1. Load Data
    TrainingSet irisTrainingData = get_iris_data();
    std::cout << "\n[+] Loaded Iris Training Data (" << irisTrainingData.size() << " examples)." << std::endl;
    std::cout << "[+] " << irisTrainingData.size() << " samples loaded.\n";
    if (irisTrainingData.empty() || irisTrainingData.size() != 149) {
         std::cerr << "Error: Full Iris dataset not loaded correctly!" << std::endl;
         return 1; // Ensure full dataset is pasted into get_iris_data()
    }


    // 2. Define Network Architecture
    // Input: 4 features
    // Hidden: 8 nodes (can be tuned)
    // Output: 3 classes
    std::vector<int> layerSizes = {4, 8, 3};
    Net irisNet(layerSizes);

    std::cout << "[+] Created Network with layer sizes: ";
    for (size_t i = 0; i < layerSizes.size(); ++i) {
        std::cout << layerSizes[i] << (i == layerSizes.size() - 1 ? "" : " -> ");
    }
    std::cout << std::endl;

    // 3. Configure Network Activations
    // Input layer (0) should be linear (default or set explicitly)
    irisNet.setLayerActivation(1, 1); // Hidden layer Sigmoid (ID 1)
    irisNet.setLayerActivation(2, 1); // Output layer Sigmoid (ID 1) - for independent class probabilities
                                      // Note: Softmax would be theoretically better for multi-class CE,
                                      // but Sigmoid fits the current backprop delta calculation (a-y)
                                      // if we assume it's optimizing a CE-like loss.

    irisNet.printNetworkStructure();

    // 4. Define Training Parameters
    int epochs = 1000;          // More epochs for a harder problem
    double learningRate = 0.05; // Potentially smaller learning rate
    int batchSize = 1;          // Stochastic Gradient Descent
    bool shuffleData = true;

    std::cout << "\n[+] Starting Training..." << std::endl;
    std::cout << "    Epochs: " << epochs << std::endl;
    std::cout << "    Learning Rate: " << learningRate << std::endl;
    std::cout << "    Batch Size: " << batchSize << std::endl;
    std::cout << "    Shuffle: " << (shuffleData ? "Yes" : "No") << std::endl;

    // 5. Train the Network and Time it
    Timer trainingTimer;
    trainingTimer.start();

    irisNet.train(irisTrainingData, epochs, learningRate, batchSize, shuffleData);

    trainingTimer.stop(false); // Stop timer, don't print yet

    std::cout << "\n[+] Training Complete." << std::endl;
    std::cout << "[+] Total Training Time: " << std::fixed << std::setprecision(2)
              << trainingTimer.getDurationMs() << " ms ("
              << trainingTimer.getDurationMs() / 1000.0 << " s)" << std::endl;


    // 6. Evaluate the Trained Network
    std::cout << "\n[+] Evaluating Trained Network on Iris Data:" << std::endl;

    int correctPredictions = 0;
    std::map<int, std::string> classNames = {{0, "Setosa"}, {1, "Versicolor"}, {2, "Virginica"}};

    for (const auto& pair : irisTrainingData) {
        const InputData& input = pair.first;
        const TargetData& expectedTarget = pair.second; // One-hot

        const std::vector<double>& predictionVec = irisNet.predict(input);

        int expectedClass = get_class_index(expectedTarget);
        int predictedClass = argmax(predictionVec); // Find index of highest output node

        if (predictedClass == expectedClass) {
            correctPredictions++;
        }

        // Optional: Print individual predictions (can be verbose for 150 samples)
        /*
        std::cout << "Input: [" << input[0] << "," << input[1] << "," << input[2] << "," << input[3] << "] | "
                  << "Expected: " << classNames[expectedClass] << " (" << expectedClass << ") | "
                  << "Predicted: " << classNames[predictedClass] << " (" << predictedClass << ") | "
                  << "Outputs: [" << std::fixed << std::setprecision(3) << predictionVec[0] << ","
                                   << predictionVec[1] << "," << predictionVec[2] << "]"
                  << (predictedClass == expectedClass ? "" : " <-- WRONG") << std::endl;
        */
    }

    double accuracy = (static_cast<double>(correctPredictions) / irisTrainingData.size()) * 100.0;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Evaluation Complete:" << std::endl;
    std::cout << "Correct Predictions: " << correctPredictions << " / " << irisTrainingData.size() << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << " %" << std::endl;
    std::cout << "------------------------------------------" << std::endl;


    return 0;
}