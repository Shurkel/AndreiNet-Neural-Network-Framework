#ifndef NET_H
#define NET_H

#include "layer.h" // Includes node, timer, util, includes
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <numeric>   // For std::accumulate
#include <limits>    // For numeric_limits
#include <algorithm> // For std::max, std::min

// Forward declaration (if needed, though Layer includes Node)
// class Layer;
// class Node;

// Define types for training data for clarity
using InputData = std::vector<double>;
using TargetData = std::vector<double>;
using TrainingPair = std::pair<InputData, TargetData>;
using TrainingSet = std::vector<TrainingPair>;

class Net
{ // Renamed class to follow convention
public:
    std::vector<Layer> layers;
    // std::vector<double> expected; // Store expected values per-example, not in the Net object
    // std::vector<double> costs;    // Calculate costs per-example

    // Performance Metrics (Calculated per epoch or batch)
    double averageSSR = 0.0;
    double averageCrossEntropy = 0.0;

    // --- Constructor ---
    Net() {} // Default constructor

    explicit Net(const std::vector<int> &layerSizes)
    { // Use const reference
        if (layerSizes.empty())
        {
            std::cerr << "Error: Cannot create Net with empty layerSizes vector." << std::endl;
            return;
        }

        layers.reserve(layerSizes.size());
        for (size_t i = 0; i < layerSizes.size(); ++i)
        {
            layers.emplace_back(layerSizes[i], i); // Pass layer ID
        }

        // Automatically connect layers and initialize weights (optional)
        if (layers.size() > 1)
        {
            connectAllLayers(true); // Connect with random weights
        }

        // Set default activations (example: sigmoid for hidden, linear for output)
        // You might want more control over this.
        if (layers.size() > 1)
        {
            layers[0].noActivateAll(); // Input layer linear
            for (size_t i = 1; i < layers.size() - 1; ++i)
            {
                layers[i].setActivationFunctionAll(1); // Hidden layers sigmoid (example)
            }
            layers.back().setActivationFunctionAll(1); // Output layer sigmoid (example for classification/regression 0-1)
                                                       // Or use linear for regression: layers.back().noActivateAll();
        }
    }

    // --- Network Setup ---

    // Connect layer i to layer i+1
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

    // Connect all adjacent layers
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
            // Weights are stored in the node connections *to* the next layer
            // So, iterate through all nodes in non-output layers
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
        // Skip input layer bias? Usually yes.
        for (size_t i = 1; i < layers.size(); ++i)
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

    // Set activation for a specific layer
    void setLayerActivation(int layerId, int function)
    {
        if (layerId >= 0 && layerId < layers.size())
        {
            layers[layerId].setActivationFunctionAll(function);
        }
    }

    // --- Forward Pass ---

    // Set input values for the network
    void setInput(const InputData &inputValues)
    { // Use const reference
        if (layers.empty())
        {
            std::cerr << "Error: Cannot set input for an empty network." << std::endl;
            return;
        }
        layers[0].setValuesFromVector(inputValues);
    }

    // Perform a full forward pass
    const std::vector<double> &predict(const InputData &inputValues)
    {
        setInput(inputValues);

        // Calculate activations layer by layer
        for (size_t i = 1; i < layers.size(); ++i)
        {
            layers[i].cleanNodesForForwardPass(); // Clear previous activations/sums
            layers[i].calculateActivations();
        }

        // Apply Softmax to output layer if needed (often done implicitly with loss)
        // if (/* check if output layer should be softmax */) {
        //     layers.back().applySoftMax();
        // }

        // Return the output layer's values
        // This is slightly inefficient as it copies. Consider returning const ref to layer/nodes.
        // Or provide a dedicated getOutput() method.
        // For now, let's add getOutput():
        // return layers.back().getValues();
        static std::vector<double> output_cache; // Static cache to return ref (use with care)
        output_cache = layers.back().getValues();
        return output_cache;
    }

    // Get the output of the last layer *after* predict() has been called
    const std::vector<double> &getOutput() const
    {
        if (layers.empty())
        {
            static std::vector<double> empty_vec;
            return empty_vec;
        }
        // This might return stale data if predict() hasn't been called recently.
        // Consider returning const ref directly from predict() or caching internally.
        // Re-using the static cache from predict:
        // return predict_output_cache; // Need a member variable or pass cache around
        // Safer: just call getValues() on demand
        static std::vector<double> output_cache;
        output_cache = layers.back().getValues();
        return output_cache;
    }

    // --- Cost Functions ---

    // Calculate Sum of Squared Residuals (MSE * N / 2) for the current network state
    double calculateSSR(const TargetData &expectedValues) const
    {
        if (layers.empty() || layers.back().nodes.size() != expectedValues.size())
        {
            std::cerr << "Error: SSR calculation size mismatch or empty network." << std::endl;
            return -1.0; // Indicate error
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

    // Calculate Cross-Entropy loss for the current network state
    double calculateCrossEntropy(const TargetData &expectedValues) const
    {
        if (layers.empty() || layers.back().nodes.size() != expectedValues.size())
        {
            std::cerr << "Error: Cross-Entropy calculation size mismatch or empty network." << std::endl;
            return -1.0; // Indicate error
        }
        double ce = 0.0;
        const Layer &outputLayer = layers.back();
        for (size_t i = 0; i < outputLayer.nodes.size(); ++i)
        {
            double predicted = outputLayer.nodes[i].getValue();
            double target = expectedValues[i];

            // Clip predicted values to avoid log(0) or log(1-0) -> log(1) issues
            predicted = std::max(1e-15, std::min(1.0 - 1e-15, predicted));

            // Binary Cross-Entropy formula: - [y*log(p) + (1-y)*log(1-p)]
            // If target is 0 or 1, simplifies:
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
                // Handle multi-class or non-binary targets if necessary
                // Using the general binary form here:
                ce -= (target * std::log(predicted) + (1.0 - target) * std::log(1.0 - predicted));
            }
        }
        return ce;
    }

    // --- Backpropagation ---

    // Performs one step of backpropagation and weight update for a single training example
    void backPropagate(const TrainingPair &trainingExample, double learningRate)
    {
        const InputData &input = trainingExample.first;
        const TargetData &expected = trainingExample.second;

        // 1. Forward pass (already done if predict was just called, but do it again for safety)
        predict(input); // Ensure network state matches input

        if (layers.empty() || layers.back().nodes.size() != expected.size())
        {
            std::cerr << "Error: Backpropagation size mismatch or empty network." << std::endl;
            return;
        }

        // 2. Calculate output layer delta (dC/dZ)
        //    Depends on the cost function and activation function
        Layer &outputLayer = layers.back();
        for (size_t i = 0; i < outputLayer.nodes.size(); ++i)
        {
            Node &node = outputLayer.nodes[i];
            double target = expected[i];
            double activatedValue = node.value;                // a
            double preActivationValue = node.unactivatedValue; // z

            // Example: Assuming MSE Loss
            // delta = (a - target) * activation_derivative(z)
            // node.delta = (activatedValue - target) * node.activationDerivative(preActivationValue);

            // Example: Assuming Cross-Entropy Loss AND Sigmoid activation
            // Simplifies nicely: delta = (a - target)
            if (node.activationFunction == 1)
            { // Sigmoid
                node.delta = activatedValue - target;
            }
            // Example: Assuming Cross-Entropy Loss AND Softmax output (handled together)
            else if (node.activationFunction == -1 /* check for specific softmax flag */)
            {
                // For Softmax + CE, the delta is also simply (a - target)
                node.delta = activatedValue - target;
            }
            else
            {                        // General case (e.g., MSE or other activations)
                double dCost_dValue; // Calculate dCost/dValue based on loss function
                // MSE: dCost/dValue = (activatedValue - target)
                dCost_dValue = activatedValue - target;
                // CE: dCost_dValue = -target / activatedValue + (1 - target) / (1 - activatedValue) // Handle clipping/zeros
                // dCost_dValue = ... (calculate carefully for CE)

                node.delta = dCost_dValue * node.activationDerivative(preActivationValue);
            }
        }

        // 3. Calculate hidden layer deltas (iterate backwards)
        for (int k = layers.size() - 2; k > 0; --k)
        { // Iterate hidden layers from back to front
            Layer &hiddenLayer = layers[k];
            Layer &nextLayer = layers[k + 1]; // The layer whose deltas we just computed

            for (size_t j = 0; j < hiddenLayer.nodes.size(); ++j)
            { // Iterate nodes in hidden layer (j)
                Node &node = hiddenLayer.nodes[j];
                double sumWeightedDeltas = 0.0;

                // Sum delta contributions from the next layer (i)
                for (size_t i = 0; i < nextLayer.nodes.size(); ++i)
                {
                    // Need weight FROM node j (current layer) TO node i (next layer)
                    double weight_ji = node.getWeightTo(i); // Assumes index access
                    sumWeightedDeltas += nextLayer.nodes[i].delta * weight_ji;
                }

                node.delta = sumWeightedDeltas * node.activationDerivative(node.unactivatedValue);
            }
        }

        // 4. Update weights and biases (iterate forward or backward)
        for (int k = 0; k < layers.size() - 1; ++k)
        { // Iterate layers with outgoing connections
            Layer &currentLayer = layers[k];
            Layer &nextLayer = layers[k + 1];

            for (size_t j = 0; j < currentLayer.nodes.size(); ++j)
            { // Node in current layer (j)
                Node &node_j = currentLayer.nodes[j];
                double value_j = node_j.value; // Activation of the source node

                for (size_t i = 0; i < nextLayer.nodes.size(); ++i)
                { // Node in next layer (i)
                    Node &node_i = nextLayer.nodes[i];
                    double delta_i = node_i.delta; // Delta of the destination node

                    // Calculate weight change: dW = learningRate * delta_i * value_j
                    double weightUpdate = learningRate * delta_i * value_j;

                    // Update weight (using index access for simplicity - requires careful structure)
                    // Weight from j to i is stored in node_j's connection list at index i
                    node_j.next[i].weight -= weightUpdate; // Gradient descent

                    // Update bias for the node in the *next* layer (only needs to happen once per node i)
                    // This is often done in the outer loop (i)
                }
            }
            // Update biases for the next layer
            for (size_t i = 0; i < nextLayer.nodes.size(); ++i)
            {
                Node &node_i = nextLayer.nodes[i];
                double delta_i = node_i.delta;
                node_i.bias -= learningRate * delta_i; // Gradient descent for bias
            }
        }
    } // End backPropagate

    // --- Training Loop ---
    void train(const TrainingSet &trainingData, int epochs, double learningRate, int batchSize = 1, bool shuffle = true)
    {
        if (layers.empty())
        {
            std::cerr << "Error: Cannot train an empty network." << std::endl;
            return;
        }

        size_t n_samples = trainingData.size();
        if (n_samples == 0)
        {
            std::cerr << "Warning: Training data is empty." << std::endl;
            return;
        }
        if (batchSize <= 0)
            batchSize = 1;
        if (batchSize > n_samples)
            batchSize = n_samples;

        size_t n_batches = (n_samples + batchSize - 1) / batchSize; // Ceiling division

        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n_samples-1

        std::ofstream loss_log("training_loss.txt"); // Log loss to file

        Timer epochTimer; // Time epochs

        std::cout << "Starting training...\n"
                  << "Epochs: " << epochs << ", Learning Rate: " << learningRate
                  << ", Batch Size: " << batchSize << ", Samples: " << n_samples
                  << ", Batches/Epoch: " << n_batches << std::endl;

        for (int e = 0; e < epochs; ++e)
        {
            epochTimer.start();
            double epochSSR = 0.0;
            double epochCE = 0.0;

            if (shuffle)
            {
                // std::shuffle(indices.begin(), indices.end(), u.rng); // Error: rng is inaccessible
                std::shuffle(indices.begin(), indices.end(), u.getRng()); // Correct: Use the public getter
            }

            // --- Batch processing would go here ---
            // For simplicity, this uses SGD (batchSize=1 implicitly if not modified)
            // To implement batches:
            // 1. Loop through batches.
            // 2. For each batch:
            //    a. Accumulate weight/bias gradients over samples in the batch.
            //    b. After processing the batch, update weights/biases using accumulated gradients / batchSize.
            // This example sticks to SGD (update after each sample) for now.

            for (size_t i = 0; i < n_samples; ++i)
            {
                const TrainingPair &currentSample = trainingData[indices[i]]; // Use shuffled index

                // Run backpropagation and update weights for this single sample
                backPropagate(currentSample, learningRate);

                // Accumulate loss for reporting (optional, can slow down if done every sample)
                epochSSR += calculateSSR(currentSample.second);
                epochCE += calculateCrossEntropy(currentSample.second);

            } // End loop through samples

            averageSSR = epochSSR / n_samples;
            averageCrossEntropy = epochCE / n_samples;
            epochTimer.stop(false); // Stop timer without printing yet

            // Print progress periodically
            if ((e + 1) % 10 == 0 || e == 0 || e == epochs - 1)
            {
                std::cout << "Epoch [" << (e + 1) << "/" << epochs << "] "
                          << "Avg SSR: " << std::fixed << std::setprecision(6) << averageSSR << " | "
                          << "Avg CE: " << std::fixed << std::setprecision(6) << averageCrossEntropy << " | "
                          << "Time: " << std::fixed << std::setprecision(2) << epochTimer.getDurationMs() << " ms"
                          << std::endl;
            }
            loss_log << (e + 1) << "\t" << averageSSR << "\t" << averageCrossEntropy << "\n";

        } // End loop through epochs

        std::cout << "Training finished.\n";
        loss_log.close();

    } // End train

    // --- Printing & Debugging ---

    void printNetworkStructure() const
    {
        std::cout << (char)218 << " Network Structure " << std::string(20, '-') << '\n';
        std::cout << (char)195 << " Layer Count: " << layers.size() << '\n';
        std::cout << (char)195 << " Format: LayerID (Size) -> Activation [-Weights-> NextLayerID]" << '\n';
        for (size_t i = 0; i < layers.size(); ++i)
        {
            char connector = (i == layers.size() - 1) ? (char)192 : (char)195;
            std::cout << connector << " [" << layers[i].layerId << "] (" << layers[i].nodes.size() << ")";
            // Show activation
            if (!layers[i].nodes.empty())
            {
                int actFunc = layers[i].nodes[0].activationFunction; // Assume uniform activation
                std::string actStr = "LIN";
                if (actFunc == 0)
                    actStr = "ReLU";
                else if (actFunc == 1)
                    actStr = "Sigmoid";
                else if (actFunc == 2)
                    actStr = "Softplus";
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

    // --- Deprecated / Old Functions (Keep or remove) ---
    /*
    void clean() { // Vague name, use cleanNodesForForwardPass
       for(Layer& layer : layers) {
           layer.cleanNodesForForwardPass();
       }
    }
    void passValues() { // Renamed to predict() for clarity
        // Logic moved to predict()
    }
     void getSSR() { // Renamed calculateSSR, takes target vector
        // Logic moved to calculateSSR()
    }
     void getCrossEntropy() { // Renamed calculateCrossEntropy, takes target vector
         // Logic moved to calculateCrossEntropy()
     }
      void backPropagate_new(...) { // Renamed backPropagate, simplified signature
         // Logic moved to backPropagate
      }
      void backPropagate_crossentropy(...) { // Merged logic into backPropagate based on cost/activation
          // Logic moved to backPropagate
      }
    */

}; // End class Net

#endif // NET_H