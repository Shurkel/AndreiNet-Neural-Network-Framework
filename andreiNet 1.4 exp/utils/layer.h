// --- START OF FILE layer.h ---

#ifndef LAYER_H
#define LAYER_H

#include "includes.h"
#include "util.h"     // Include Util class

#include <vector>     // Still use vector for layers container

// Forward declare Net if needed
// class Net; // Probably not needed here anymore

// Define OptimizerType enum (can also be in net.h, but layer uses it)
enum class OptimizerType { SGD, ADAM };


class Layer
{
public:
    int layerId = 0;
    int numNodes = 0;
    int activationFunction = -1; // -1: Linear, 0: ReLU, 1: Sigmoid, 2: Softplus, 3: Tanh

    // --- Eigen Data Structures ---
    Eigen::VectorXd activations;    // 'a': Output of this layer (size: numNodes)
    Eigen::VectorXd preActivations; // 'z': Weighted sum + bias before activation (size: numNodes)
    Eigen::VectorXd biases;         // 'b': Bias for each node (size: numNodes)
    Eigen::VectorXd deltas;         // 'd': Error signal (dCost/dZ) (size: numNodes)

    // Weights connecting the *previous* layer to *this* layer (layer j)
    // Dimensions: (nodes in this layer j) x (nodes in previous layer k)
    Eigen::MatrixXd inputWeights;      // W_jk

    // Gradients accumulated during backpropagation (for batching)
    Eigen::MatrixXd inputWeightGradients; // dW_jk
    Eigen::VectorXd biasGradients;        // db_j

    // --- Adam Optimizer State (only used if Adam is selected) ---
    Eigen::MatrixXd m_weights; // 1st moment vector for weights
    Eigen::MatrixXd v_weights; // 2nd moment vector for weights
    Eigen::VectorXd m_biases;  // 1st moment vector for biases
    Eigen::VectorXd v_biases;  // 2nd moment vector for biases

    // Pointers to adjacent layers (still useful)
    Layer *next = nullptr;
    Layer *prev = nullptr;

    // --- Constructor ---
    // Takes the number of nodes in this layer and the previous layer
    Layer(int nNodes, int prevLayerNodes, int id) : layerId(id),
                                                    numNodes(nNodes)
    {
        // Resize Eigen vectors for this layer
        activations.resize(numNodes);
        preActivations.resize(numNodes);
        biases.resize(numNodes);
        deltas.resize(numNodes);
        biasGradients.resize(numNodes);

        // Initialize biases (e.g., to zero)
        biases.setZero();
        biasGradients.setZero(); // Initialize gradients
        m_biases.resize(numNodes); // Initialize Adam state
        v_biases.resize(numNodes);
        m_biases.setZero();
        v_biases.setZero();


        // Initialize weights and gradients if this is NOT the input layer
        if (prevLayerNodes > 0)
        {
            inputWeights.resize(numNodes, prevLayerNodes); // Rows = this layer size, Cols = prev layer size
            inputWeightGradients.resize(numNodes, prevLayerNodes);
            m_weights.resize(numNodes, prevLayerNodes); // Initialize Adam state
            v_weights.resize(numNodes, prevLayerNodes);


            // Initialize weights (e.g., Xavier/He initialization is common)
             // Use Glorot/Xavier initialization for Sigmoid/Tanh
            double limit = std::sqrt(6.0 / (prevLayerNodes + numNodes));
            inputWeights = Eigen::MatrixXd::Random(numNodes, prevLayerNodes) * limit;
             // Use He initialization for ReLU (comment out if using Xavier)
            // double limit_he = std::sqrt(2.0 / prevLayerNodes);
            // inputWeights = Eigen::MatrixXd::Random(numNodes, prevLayerNodes) * limit_he;

            inputWeightGradients.setZero(); // Initialize gradients
            m_weights.setZero();
            v_weights.setZero();
        } else {
            // Input layer has no input weights or corresponding state
            inputWeights.resize(0,0);
            inputWeightGradients.resize(0,0);
            m_weights.resize(0,0);
            v_weights.resize(0,0);
        }

        // Set default activation (e.g., linear) - This can be overridden by Net
        activationFunction = -1;
    }

    // --- Core Methods ---

    void setActivationFunction(int funcType)
    {
        activationFunction = funcType;
         // Re-initialize weights if activation changes? Maybe not needed if done at construction.
         // Consider if He vs Xavier should be reapplied based on funcType here.
    }

    // Reset accumulated gradients (call before each batch)
    void zeroGradients()
    {
        if (inputWeightGradients.size() > 0)
            inputWeightGradients.setZero();
        if (biasGradients.size() > 0)
            biasGradients.setZero();
    }

    // Apply accumulated gradients (call after processing a batch)
    // Now includes optimizer logic
    void applyGradients(double learningRate, int batchSize, OptimizerType optType,
                        double beta1, double beta2, double epsilon, int timeStep,
                        double L2Lambda = 0.0) // Added L2 lambda
    {
        if (batchSize <= 0) batchSize = 1;
        double alpha = learningRate; // Use alpha for learning rate in formulas

        // --- Apply gradients to input weights (if not input layer) ---
        if (inputWeights.size() > 0 && inputWeightGradients.size() == inputWeights.size())
        {
             // 1. Calculate average gradient over batch
             Eigen::MatrixXd grad_W = inputWeightGradients / batchSize;

             // 2. Add L2 regularization term (Weight Decay)
             // Note: Regularization is added BEFORE optimizer step
             if (L2Lambda > 0.0) {
                grad_W += (L2Lambda / batchSize) * inputWeights;
                // Alternative: Some implementations apply decay directly to weights after update:
                // inputWeights *= (1.0 - alpha * L2Lambda / batchSize); // If doing it post-SGD update
             }


             // 3. Apply optimizer step
             if (optType == OptimizerType::SGD) {
                 inputWeights -= alpha * grad_W;
             }
             else if (optType == OptimizerType::ADAM) {
                 if (timeStep <= 0) {
                     std::cerr << "Error: Adam timeStep must be positive." << std::endl; return;
                 }
                 // Update moments
                 m_weights = beta1 * m_weights + (1.0 - beta1) * grad_W;
                 v_weights = beta2 * v_weights.array() + (1.0 - beta2) * grad_W.array().square(); // Element-wise square

                 // Bias correction
                 double beta1_t = std::pow(beta1, timeStep);
                 double beta2_t = std::pow(beta2, timeStep);
                 Eigen::MatrixXd m_hat = m_weights / (1.0 - beta1_t);
                 Eigen::MatrixXd v_hat = v_weights / (1.0 - beta2_t);

                 // Update weights
                 inputWeights -= alpha * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
             }
        }


        // --- Apply gradients to biases ---
        if (biases.size() > 0 && biasGradients.size() == biases.size())
        {
             // 1. Calculate average gradient
             Eigen::VectorXd grad_B = biasGradients / batchSize;
             // Note: Biases are typically NOT regularized with L2

             // 2. Apply optimizer step
             if (optType == OptimizerType::SGD) {
                 biases -= alpha * grad_B;
             }
              else if (optType == OptimizerType::ADAM) {
                 if (timeStep <= 0) return; // Already printed error
                 // Update moments
                 m_biases = beta1 * m_biases + (1.0 - beta1) * grad_B;
                 v_biases = beta2 * v_biases.array() + (1.0 - beta2) * grad_B.array().square();

                 // Bias correction
                 double beta1_t = std::pow(beta1, timeStep);
                 double beta2_t = std::pow(beta2, timeStep);
                 Eigen::VectorXd m_hat = m_biases / (1.0 - beta1_t);
                 Eigen::VectorXd v_hat = v_biases / (1.0 - beta2_t);

                 // Update biases
                 biases -= alpha * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
             }
        }
    }


    // Forward pass calculation for *this* layer
    void calculateActivations()
    {
        if (!prev)
        { // Input layer
            // Assume 'activations' are already set via Net::setInput
            preActivations = activations; // No weights/bias for input layer
            // Input layer *could* have activation, but usually linear (-1)
             activations = getUtil().activateEigen(preActivations, activationFunction);
            return;
        }

        // Ensure previous layer's activations and this layer's weights are compatible
        if (inputWeights.cols() != prev->numNodes || inputWeights.rows() != numNodes )
        {
            std::cerr << "Error: Weight matrix dimension mismatch! Layer " << layerId
                      << " weights (" << inputWeights.rows() << "x" << inputWeights.cols()
                      << ") cannot process input from Layer " << prev->layerId << " (" << prev->numNodes << " nodes)." << std::endl;
             // Consider throwing an exception here for critical errors
             preActivations.setZero(); // Avoid using garbage data
             activations.setZero();
            return;
        }
         if (biases.size() != numNodes) {
            std::cerr << "Error: Bias vector size mismatch in Layer " << layerId << std::endl;
             preActivations.setZero();
             activations.setZero();
             return;
         }

        // z = W * a_prev + b
        preActivations = inputWeights * prev->activations + biases;

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
            std::cout << "  Activations (a):\n" << activations.transpose() << "\n";
            std::cout << "  Pre-Activations (z):\n" << preActivations.transpose() << "\n";
            std::cout << "  Biases (b):\n" << biases.transpose() << "\n";
            std::cout << "  Deltas (d):\n" << deltas.transpose() << "\n";

            if (inputWeights.size() > 0) {
                std::cout << "  Input Weights (W from layer " << (prev ? std::to_string(prev->layerId) : "N/A") << ") ["
                          << inputWeights.rows() << "x" << inputWeights.cols() << "]:\n" << inputWeights << "\n";
            }
            if (inputWeightGradients.size() > 0) {
                std::cout << "  Input Weight Gradients (dW) [" << inputWeightGradients.rows() << "x" << inputWeightGradients.cols() << "]:\n"
                          << inputWeightGradients << "\n";
            }
             if (biasGradients.size() > 0) {
                std::cout << "  Bias Gradients (db):\n" << biasGradients.transpose() << "\n";
            }
             // Optionally show Adam state if debugging
             /*
             if (m_weights.size() > 0) {
                 std::cout << "  Adam m_weights:\n" << m_weights << "\n";
                 std::cout << "  Adam v_weights:\n" << v_weights << "\n";
             }
             if (m_biases.size() > 0) {
                 std::cout << "  Adam m_biases:\n" << m_biases.transpose() << "\n";
                 std::cout << "  Adam v_biases:\n" << v_biases.transpose() << "\n";
             }
             */
        }
        else
        {
            std::cout << "  Nodes: " << numNodes;
            if (prev) std::cout << ", Input Weights: " << inputWeights.rows() << "x" << inputWeights.cols();
            std::cout << ", Biases: " << biases.size();
            std::cout << " (Use showMatrices=true for details)\n";
        }
        std::cout << "+----------------------------------------+\n";
    }
};

#endif // LAYER_H