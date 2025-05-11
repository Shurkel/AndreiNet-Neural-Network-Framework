#ifndef NET_H
#define NET_H

#include "layer.h" 
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <numeric>   
#include <limits>    
#include <algorithm> 
#include <functional> 

using InputData = std::vector<double>;
using TargetData = std::vector<double>;
using TrainingPair = std::pair<InputData, TargetData>;
using TrainingSet = std::vector<TrainingPair>;

class Net
{ 
public:
    using ProgressCallback = std::function<void(int, double, double, double, bool)>;
    std::vector<Layer> layers;
    
    double averageSSR = 0.0;
    double averageCrossEntropy = 0.0;

    Net() {} 

    explicit Net(const std::vector<int> &layerSizes)
    { 
        if (layerSizes.empty())
        {
            std::cerr << "Error: Cannot create Net with empty layerSizes vector." << std::endl;
            return;
        }

        layers.reserve(layerSizes.size());
        for (size_t i = 0; i < layerSizes.size(); ++i)
        {
            layers.emplace_back(layerSizes[i], i); 
        }

        if (layers.size() > 1)
        {
            connectAllLayers(true); 
        }

        if (layers.size() > 1)
        {
            layers[0].noActivateAll(); 
            for (size_t i = 1; i < layers.size() - 1; ++i)
            {
                layers[i].setActivationFunctionAll(1); // Hidden layers sigmoid
            }
            layers.back().setActivationFunctionAll(1); // Output layer sigmoid
        }
    }

    void connectLayers(int layerIdx1, int layerIdx2, bool randomInit = true)
    {
        if (layerIdx1 >= 0 && layerIdx1 < layers.size() &&
            layerIdx2 >= 0 && layerIdx2 < layers.size() && layerIdx1 + 1 == layerIdx2)
        {
            layers[layerIdx1].connectTo(&layers[layerIdx2], -0.1, 0.1, randomInit);
        }
        else
        {
            std::cerr << "Error: Invalid layer indices for connection (" << layerIdx1 << " -> " << layerIdx2 << ")" << std::endl;
        }
    }

    void connectAllLayers(bool randomInit = true)
    {
        for (size_t i = 0; i < layers.size() - 1; ++i)
        {
            layers[i].connectTo(&layers[i + 1], -0.1, 0.1, randomInit);
        }
    }

    void disconnectAllLayers()
    {
        for (size_t i = 0; i < layers.size() - 1; ++i)
        {
            layers[i].disconnectFromNext();
        }
    }

    void randomizeAllWeights(double minVal = -0.1, double maxVal = 0.1)
    {
        for (Layer &layer : layers)
        {
            if (layer.next)
            {
                for (Node &node : layer.nodes)
                {
                    node.randomiseWeights(minVal, maxVal);
                }
            }
        }
    }
    void setAllBiases(double biasValue)
    {
        for (size_t i = 1; i < layers.size(); ++i) // Skip input layer bias
        {
            layers[i].setBiasAll(biasValue);
        }
    }
    void setAllWeights(double weightValue)
    {
        for (Layer &layer : layers)
        {
            if (layer.next)
            {
                for (Node &node : layer.nodes)
                {
                    node.setWeightAll(weightValue);
                }
            }
        }
    }

    void setLayerActivation(int layerId, int function)
    {
        if (layerId >= 0 && layerId < layers.size())
        {
            layers[layerId].setActivationFunctionAll(function);
        }
    }

    void setInput(const InputData &inputValues)
    { 
        if (layers.empty())
        {
            std::cerr << "Error: Cannot set input for an empty network." << std::endl;
            return;
        }
        layers[0].setValuesFromVector(inputValues);
    }

    const std::vector<double> predict(const InputData &inputValues)
    {
        setInput(inputValues);

        for (size_t i = 1; i < layers.size(); ++i)
        {
            layers[i].cleanNodesForForwardPass(); 
            layers[i].calculateActivations();
        }
        // If output layer is Softmax, user should call layers.back().applySoftMax() after predict,
        // or set output nodes to linear and call it.
        // This `predict` function returns values after Node-level activations.
        return layers.back().getValues();
    }

    const std::vector<double> getOutput() const
    {
        if (layers.empty())
        {
            static std::vector<double> empty_vec; // Return const ref to empty for safety
            return empty_vec;
        }
        // Returns current values of output layer. Call predict() first for up-to-date values.
        return layers.back().getValues();
    }


    double calculateSSR(const TargetData &expectedValues) const
    {
        if (layers.empty() || layers.back().nodes.size() != expectedValues.size())
        {
            std::cerr << "Error: SSR calculation size mismatch or empty network." << std::endl;
            return -1.0; 
        }
        double ssr = 0.0;
        const Layer &outputLayer = layers.back();
        for (size_t i = 0; i < outputLayer.nodes.size(); ++i)
        {
            double error = outputLayer.nodes[i].getValue() - expectedValues[i];
            ssr += error * error;
        }
        return 0.5 * ssr;
    }

    double calculateCrossEntropy(const TargetData &expectedValues) const
    {
        if (layers.empty() || layers.back().nodes.size() != expectedValues.size())
        {
            std::cerr << "Error: Cross-Entropy calculation size mismatch or empty network." << std::endl;
            return -1.0; 
        }
        double ce = 0.0;
        const Layer &outputLayer = layers.back();
        for (size_t i = 0; i < outputLayer.nodes.size(); ++i)
        {
            double predicted = outputLayer.nodes[i].getValue();
            double target = expectedValues[i];

            predicted = std::max(1e-15, std::min(1.0 - 1e-15, predicted)); // Clip for log stability

            if (target == 1.0)
            {
                ce -= std::log(predicted);
            }
            else if (target == 0.0)
            {
                ce -= std::log(1.0 - predicted);
            }
            else
            {
                // General binary cross-entropy form (handles probabilities if target is not 0 or 1)
                ce -= (target * std::log(predicted) + (1.0 - target) * std::log(1.0 - predicted));
            }
        }
        return ce;
    }

    void backPropagate(const TrainingPair &trainingExample, double learningRate)
    {
        const InputData &input = trainingExample.first;
        const TargetData &expected = trainingExample.second;

        predict(input); // Forward pass to set current activations and pre-activations

        if (layers.empty() || layers.back().nodes.size() != expected.size())
        {
            std::cerr << "Error: Backpropagation size mismatch or empty network." << std::endl;
            return;
        }

        Layer &outputLayer = layers.back();
        for (size_t i = 0; i < outputLayer.nodes.size(); ++i)
        {
            Node &node = outputLayer.nodes[i];
            double target = expected[i];
            double activatedValue = node.value;                
            double preActivationValue = node.unactivatedValue; 

            // Delta (dCost/dZ) for output layer nodes
            // This simplified delta calculation assumes specific loss/activation pairings:
            // - Sigmoid activation (type 1) implies Binary Cross-Entropy loss.
            // - Linear activation (type -1, potentially after a Layer::applySoftMax call) implies Categorical Cross-Entropy.
            // - Other activations imply Squared Error loss.
            if (node.activationFunction == 1) { // Sigmoid (implies BCE)
                node.delta = activatedValue - target;
            }
            // If Layer::applySoftMax() was used on a linear output layer, its nodes have activationFunction == -1
            else if (node.activationFunction == -1) { // Linear (implies CCE if Softmax was applied, or MSE if not)
                                                      // Assuming (a-target) for CE with Softmax.
                node.delta = activatedValue - target;
            }
            else { // General case: Assumes Mean Squared Error loss
                double dCost_dValue = activatedValue - target; // d(MSE)/da
                node.delta = dCost_dValue * node.activationDerivative(preActivationValue);
            }
        }

        for (int k = layers.size() - 2; k > 0; --k) // Hidden layers (from back to front, skipping input)
        { 
            Layer &hiddenLayer = layers[k];
            Layer &nextLayer = layers[k + 1]; 

            for (size_t j = 0; j < hiddenLayer.nodes.size(); ++j)
            { 
                Node &node = hiddenLayer.nodes[j];
                double sumWeightedDeltas = 0.0;

                for (size_t i = 0; i < nextLayer.nodes.size(); ++i)
                {
                    double weight_ji = node.getWeightTo(i); 
                    sumWeightedDeltas += nextLayer.nodes[i].delta * weight_ji;
                }
                node.delta = sumWeightedDeltas * node.activationDerivative(node.unactivatedValue);
            }
        }

        for (int k = 0; k < layers.size() - 1; ++k) // All layers with outgoing connections
        { 
            Layer &currentLayer = layers[k];
            Layer &nextLayer = layers[k + 1];

            for (size_t j = 0; j < currentLayer.nodes.size(); ++j)
            { 
                Node &node_j = currentLayer.nodes[j];
                double value_j = node_j.value; 

                for (size_t i = 0; i < nextLayer.nodes.size(); ++i)
                { 
                    Node &node_i = nextLayer.nodes[i];
                    double delta_i = node_i.delta; 
                    double weightUpdate = learningRate * delta_i * value_j;
                    node_j.next[i].weight -= weightUpdate; 
                }
            }
            // Update biases for the nodes in the *next* layer
            for (size_t i = 0; i < nextLayer.nodes.size(); ++i)
            {
                Node &node_i = nextLayer.nodes[i];
                double delta_i = node_i.delta;
                node_i.bias -= learningRate * delta_i; 
            }
        }
    } 

    // Note: batchSize > 1 currently performs SGD for batchSize iterations,
    // it does not accumulate gradients for a true mini-batch update.
    void train(const TrainingSet &trainingData, int epochs, double learningRate, 
               int batchSize = 1, bool shuffle = true, ProgressCallback progressCb = nullptr)
    {
        if (layers.empty()) {
            std::cerr << "Error: Cannot train an empty network." << std::endl;
            if (progressCb) progressCb(-1, -1, -1, -1, true);
            return;
        }
        size_t n_samples = trainingData.size();
        if (n_samples == 0) {
            std::cerr << "Warning: Training data is empty." << std::endl;
            if (progressCb) progressCb(-1, -1, -1, -1, true);
            return;
        }
        // batchSize is currently illustrative for SGD loop, true gradient accumulation not implemented.
        if (batchSize <= 0) batchSize = 1;


        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        Timer epochTimer;

        if (!progressCb) {
            std::cout << "Starting training...\n"
                      << "Epochs: " << epochs << ", Learning Rate: " << learningRate
                      << ", Samples: " << n_samples << std::endl;
        }

        for (int e = 0; e < epochs; ++e) {
            epochTimer.start();
            double epochSSR = 0.0;
            double epochCE = 0.0;

            if (shuffle) {
                std::shuffle(indices.begin(), indices.end(), u.getRng());
            }

            for (size_t i = 0; i < n_samples; ++i) { // Iterates through all samples (effectively batch size 1 for updates)
                const TrainingPair& currentSample = trainingData[indices[i]];
                backPropagate(currentSample, learningRate);
                // SSR and CE are calculated based on the state *after* the backPropagate call,
                // which includes the forward pass for that sample.
                epochSSR += calculateSSR(currentSample.second);
                epochCE += calculateCrossEntropy(currentSample.second);
            }

            averageSSR = epochSSR / n_samples;
            averageCrossEntropy = epochCE / n_samples;
            epochTimer.stop(false);

            if (progressCb) {
                progressCb(e + 1, averageSSR, averageCrossEntropy, epochTimer.getDurationMs(), (e == epochs - 1));
            } else {
                if ((e + 1) % 10 == 0 || e == 0 || e == epochs - 1) {
                    std::cout << "Epoch [" << (e + 1) << "/" << epochs << "] "
                              << "Avg SSR: " << std::fixed << std::setprecision(6) << averageSSR << " | "
                              << "Avg CE: " << std::fixed << std::setprecision(6) << averageCrossEntropy << " | "
                              << "Time: " << std::fixed << std::setprecision(2) << epochTimer.getDurationMs() << " ms"
                              << std::endl;
                }
            }
        }
        if (!progressCb) {
            std::cout << "Training finished.\n";
        }
    }

    std::string getNetworkStructureString() const {
        std::ostringstream oss;
        oss << "> Network Structure " << std::string(10, '-') << '\n';
        oss << "| Layers: " << layers.size() << '\n';
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            oss << "| L" << layer.layerId << ": " << layer.nodes.size() << " nodes";
            if (!layer.nodes.empty()) {
                int actFunc = layer.nodes[0].activationFunction; 
                std::string actStr = "Linear";
                if (actFunc == 0) actStr = "ReLU";
                else if (actFunc == 1) actStr = "Sigmoid";
                else if (actFunc == 2) actStr = "Softplus";
                oss << " (Act: " << actStr << ")";
            }
            if (layer.next) {
                oss << " --> L" << layer.next->layerId;
            }
            oss << '\n';
        }
        oss << std::string(28, '-') << '\n';
        return oss.str();
    }
    

    void printNetworkStructure() const
    {
        std::cout << (char)218 << " Network Structure " << std::string(20, '-') << '\n';
        std::cout << (char)195 << " Layer Count: " << layers.size() << '\n';
        std::cout << (char)195 << " Format: LayerID (Size) -> Activation [-Weights-> NextLayerID]" << '\n';
        for (size_t i = 0; i < layers.size(); ++i)
        {
            char connector = (i == layers.size() - 1) ? (char)192 : (char)195;
            std::cout << connector << " [" << layers[i].layerId << "] (" << layers[i].nodes.size() << ")";
            if (!layers[i].nodes.empty())
            {
                int actFunc = layers[i].nodes[0].activationFunction; 
                std::string actStr = "LIN";
                if (actFunc == 0) actStr = "ReLU";
                else if (actFunc == 1) actStr = "Sigmoid";
                else if (actFunc == 2) actStr = "Softplus";
                std::cout << " -> " << actStr;
            }

            if (layers[i].next)
            {
                std::cout << " -- W --> [" << layers[i].next->layerId << "]";
            }
            std::cout << '\n';
        }
        std::cout << std::string(38, '-') << '\n';
        std::cout.flush();
    }

    void printLayerDetails(int layerId, bool detailed = true) const
    {
        if (layerId >= 0 && layerId < layers.size())
        {
            layers[layerId].printLayer(detailed);
        }
        else
        {
            std::cerr << "Error: Invalid layer ID for printing: " << layerId << std::endl;
        }
    }

    void printAllLayers(bool detailed = false) const
    {
        std::cout << "\n--- Printing All Layer Details ---" << std::endl;
        for (const auto &layer : layers)
        {
            layer.printLayer(detailed);
        }
        std::cout << "--- End Layer Details ---\n"
                  << std::endl;
    }

    void printNodeDetails(int layerId, int nodeId) const
    {
        if (layerId >= 0 && layerId < layers.size())
        {
            if (nodeId >= 0 && nodeId < layers[layerId].nodes.size())
            {
                layers[layerId].nodes[nodeId].printDetails();
            }
            else
            {
                std::cerr << "Error: Invalid node ID (" << nodeId << ") for layer " << layerId << std::endl;
            }
        }
        else
        {
            std::cerr << "Error: Invalid layer ID for printing node: " << layerId << std::endl;
        }
    }
    const std::vector<Layer>& getLayers() const { return layers; }

}; 

#endif // NET_H