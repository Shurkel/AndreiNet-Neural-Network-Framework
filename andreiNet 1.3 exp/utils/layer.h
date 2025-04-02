#ifndef LAYER_H
#define LAYER_H

#include "includes.h" // Include headers
#include "util.h"     // Include Util class

#include <vector>     // Still use vector for layers container

// Forward declare Net if needed
class Net;

class Layer
{
public:
    int layerId = 0;
    int numNodes = 0;
    int activationFunction = -1; // -1: Linear, 0: ReLU, 1: Sigmoid, etc.

    // --- Eigen Data Structures ---
    Eigen::VectorXd activations;    // 'a': Output of this layer (size: numNodes)
    Eigen::VectorXd preActivations; // 'z': Weighted sum + bias before activation (size: numNodes)
    Eigen::VectorXd biases;         // 'b': Bias for each node (size: numNodes)
    Eigen::VectorXd deltas;         // 'd': Error signal (dCost/dZ) (size: numNodes)

    // Weights connecting *this* layer (j) to the *next* layer (i)
    // Dimensions: (nodes in next layer) x (nodes in this layer)
    Eigen::MatrixXd weights;

    // Gradients accumulated during backpropagation (for batching)
    Eigen::MatrixXd weightGradients; // dW
    Eigen::VectorXd biasGradients;   // db

    // Pointers to adjacent layers
    Layer *next = nullptr;
    Layer *prev = nullptr;

    // --- Constructor ---
    Layer(int nNodes, int id, int nextLayerNodes = 0) : layerId(id),
                                                        numNodes(nNodes)
    {
        // Resize Eigen vectors
        activations.resize(numNodes);
        preActivations.resize(numNodes);
        biases.resize(numNodes);
        deltas.resize(numNodes);
        biasGradients.resize(numNodes);

        // Initialize biases (e.g., to zero or small random values)
        biases.setZero();
        biasGradients.setZero(); // Initialize gradients

        // Initialize weights if this layer connects to a next layer
        if (nextLayerNodes > 0)
        {
            weights.resize(nextLayerNodes, numNodes); // Rows = next layer size, Cols = this layer size
            weightGradients.resize(nextLayerNodes, numNodes);

            // Initialize weights (e.g., Xavier/He initialization is common)
            double limit = std::sqrt(6.0 / (numNodes + nextLayerNodes)); // Xavier approx
            weights = Eigen::MatrixXd::Random(nextLayerNodes, numNodes) * limit;
            // Or simpler random: weights = Eigen::MatrixXd::Random(nextLayerNodes, numNodes) * 0.1;
            weightGradients.setZero(); // Initialize gradients
        }

        // Set default activation (e.g., linear)
        activationFunction = -1; // Example: Default to linear
    }

    // --- Core Methods ---

    void setActivationFunction(int funcType)
    {
        activationFunction = funcType;
    }

    // Reset accumulated gradients (call before each batch)
    void zeroGradients()
    {
        if (weightGradients.size() > 0)
            weightGradients.setZero();
        if (biasGradients.size() > 0)
            biasGradients.setZero();
    }

    // Apply accumulated gradients (call after processing a batch)
    void applyGradients(double learningRate, int batchSize)
    {
        if (batchSize <= 0)
            batchSize = 1;
        double effective_lr = learningRate / batchSize;

        // Update weights owned by the *previous* layer that connect TO this layer
        if (prev && prev->weights.size() > 0 && prev->weightGradients.size() == prev->weights.size())
        {
            prev->weights -= effective_lr * prev->weightGradients;
        }

        // Update biases owned by *this* layer
        if (biases.size() > 0 && biasGradients.size() == biases.size())
        {
            biases -= effective_lr * biasGradients;
        }
    }

    // Forward pass calculation for *this* layer
    void calculateActivations()
    {
        if (!prev)
        { // Input layer
            // Assume 'activations' are already set via Net::setInput
            // Input layer usually has no bias or preActivation calculation
            preActivations = activations; // z = a for input layer? Or just leave uninitialized?
            // Input layer might still have an activation function if needed (rare)
            // activations = getUtil().activateEigen(preActivations, activationFunction);
            return;
        }

        // Ensure previous layer's weights matrix is correctly sized
        if (prev->weights.rows() != numNodes || prev->weights.cols() != prev->numNodes)
        {
            std::cerr << "Error: Weight matrix dimension mismatch! Layer " << prev->layerId
                      << " weights (" << prev->weights.rows() << "x" << prev->weights.cols()
                      << ") cannot produce output for Layer " << layerId << " (" << numNodes << " nodes) "
                      << " from Layer " << prev->layerId << " (" << prev->numNodes << " nodes)." << std::endl;
            return; // Or throw
        }

        // z = W_prev * a_prev + b
        preActivations = prev->weights * prev->activations + biases;

        // a = activate(z)
        activations = getUtil().activateEigen(preActivations, activationFunction);
    }

    // --- Printing ---
    void printLayer(bool showMatrices = false) const
    {
        std::cout << "+----- Layer " << layerId << " (" << numNodes << " nodes) "
                  << " Act F: " << activationFunction << " -----+\n";
        if (showMatrices)
        {
            std::cout << "  Activations (a):\n"
                      << activations.transpose() << "\n";
            std::cout << "  Pre-Activations (z):\n"
                      << preActivations.transpose() << "\n";
            std::cout << "  Biases (b):\n"
                      << biases.transpose() << "\n";
            std::cout << "  Deltas (d):\n"
                      << deltas.transpose() << "\n";
            if (weights.size() > 0)
            {
                std::cout << "  Weights (W to next layer " << (next ? std::to_string(next->layerId) : "N/A") << ") ["
                          << weights.rows() << "x" << weights.cols() << "]:\n"
                          << weights << "\n";
            }
            if (weightGradients.size() > 0)
            {
                std::cout << "  Weight Gradients (dW) [" << weightGradients.rows() << "x" << weightGradients.cols() << "]:\n"
                          << weightGradients << "\n";
            }
            if (biasGradients.size() > 0)
            {
                std::cout << "  Bias Gradients (db):\n"
                          << biasGradients.transpose() << "\n";
            }
        }
        else
        {
            // Brief summary maybe?
            std::cout << "  (Use showMatrices=true for details)\n";
        }
        std::cout << "+----------------------------------------+\n";
    }
};

#endif // LAYER_H