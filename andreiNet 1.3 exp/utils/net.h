#ifndef NET_H
#define NET_H

#include "layer.h" // Includes Eigen, Util, etc.
#include <vector>
#include <string>
#include <numeric>
#include <algorithm> // For std::shuffle
#include <iostream>
#include <fstream>
#include <iomanip>  // For std::setprecision
#include <chrono>   // For timing
#include "timer.h"

// Define types for training data (can use Eigen types if preferred)
using InputDataEigen = Eigen::VectorXd;
using TargetDataEigen = Eigen::VectorXd;
using TrainingPairEigen = std::pair<InputDataEigen, TargetDataEigen>;
using TrainingSetEigen = std::vector<TrainingPairEigen>;


class Net {
public:
    std::vector<Layer> layers;
    // Metrics stored per epoch
    double averageSSR = 0.0;
    double averageCrossEntropy = 0.0;

    // Loss function type enum (example)
    enum class LossFunction { MSE, CROSS_ENTROPY };
    LossFunction lossType = LossFunction::MSE; // Default loss


    // --- Constructor ---
    explicit Net(const std::vector<int>& layerSizes) {
        if (layerSizes.size() < 2) {
            std::cerr << "Error: Network must have at least input and output layers." << std::endl;
            return;
        }
        layers.reserve(layerSizes.size());
        for (size_t i = 0; i < layerSizes.size(); ++i) {
            int nextLayerNodeCount = (i < layerSizes.size() - 1) ? layerSizes[i + 1] : 0;
            layers.emplace_back(layerSizes[i], i, nextLayerNodeCount);
            if (i > 0) {
                layers[i].prev = &layers[i - 1];
            }
            if (i < layerSizes.size() - 1) {
                layers[i].next = &layers[i + 1]; // This pointer is set in the next iteration's constructor
            }
        }
        // Fix next pointers for all but last layer
        for (size_t i = 0; i < layers.size() - 1; ++i) {
             layers[i].next = &layers[i+1];
        }


        // Example: Set default activations (customize as needed)
        layers[0].setActivationFunction(-1); // Input Linear
        for (size_t i = 1; i < layers.size() - 1; ++i) {
            layers[i].setActivationFunction(1); // Hidden Sigmoid
        }
        // Output layer depends on task
         layers.back().setActivationFunction(1); // Example: Sigmoid for classification
        // layers.back().setActivationFunction(-1); // Example: Linear for regression
    }

    void setLossFunction(LossFunction lf) {
        lossType = lf;
         // Adjust output layer activation if needed based on loss
         if (lf == LossFunction::CROSS_ENTROPY && layers.back().activationFunction != 1) {
              std::cout << "Warning: Cross-entropy loss usually used with Sigmoid/Softmax output. Setting output layer to Sigmoid.\n";
              layers.back().setActivationFunction(1);
         } else if (lf == LossFunction::MSE && layers.back().activationFunction == 1) {
              std::cout << "Warning: MSE loss often used with Linear output. Consider changing output activation.\n";
              // layers.back().setActivationFunction(-1); // Optionally change it
         }
    }

    // --- Forward Pass ---
    const Eigen::VectorXd& predict(const InputDataEigen& input) {
        if (layers.empty() || layers[0].numNodes != input.size()) {
            std::cerr << "Error: Input vector size (" << input.size()
                      << ") does not match input layer size ("
                      << (layers.empty() ? 0 : layers[0].numNodes) << ")." << std::endl;
            // Return a static empty vector or throw
             static Eigen::VectorXd error_vec; error_vec.resize(0);
             return error_vec;
        }

        // Set input layer activations
        layers[0].activations = input;

        // Propagate through hidden/output layers
        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i].calculateActivations();
        }

        // Return output layer activations
        return layers.back().activations;
    }

    const Eigen::VectorXd& getOutput() const {
         if (layers.empty()) {
             static Eigen::VectorXd error_vec; error_vec.resize(0);
             return error_vec;
         }
         return layers.back().activations;
    }

    // --- Cost Calculation ---
    double calculateCost(const TargetDataEigen& expected) {
         const Eigen::VectorXd& predicted = layers.back().activations;
         if (predicted.size() != expected.size()) return -1.0; // Error

         switch(lossType) {
             case LossFunction::MSE: {
                 // 0.5 * sum((a - y)^2)
                 return 0.5 * (predicted - expected).squaredNorm();
             }
             case LossFunction::CROSS_ENTROPY: {
                 // - sum(y*log(p) + (1-y)*log(1-p))
                 // Clamp predictions for numerical stability
                  Eigen::VectorXd p_clipped = predicted.array().max(1e-15).min(1.0 - 1e-15);
                  double ce = -( (expected.array() * p_clipped.array().log()) +
                                 ((1.0 - expected.array()) * (1.0 - p_clipped.array()).log()) ).sum();
                 return ce;
             }
             default: return -1.0; // Unknown loss
         }
    }


    // --- Backpropagation ---
    // Calculates gradients (dW, db) for a single example
    void calculateGradients(const TargetDataEigen& expected) {
        if (layers.empty() || layers.back().numNodes != expected.size()) {
            std::cerr << "Error: Target vector size mismatch during backprop." << std::endl;
            return;
        }

        const Layer& outputLayer = layers.back();
        const Eigen::VectorXd& activations_L = outputLayer.activations; // a[L]
        const Eigen::VectorXd& preActivations_L = outputLayer.preActivations; // z[L]

        // 1. Calculate output layer delta (d[L] = dC/dZ[L])
        Eigen::VectorXd dCost_dAct; // dC/dA[L]
        switch(lossType) {
            case LossFunction::MSE:
                 // dC/dA = (a[L] - y)
                 dCost_dAct = activations_L - expected;
                 // d[L] = dC/dA * activate'(Z[L])
                 layers.back().deltas = (dCost_dAct.array() * getUtil().activationDerivativeEigen(preActivations_L, outputLayer.activationFunction).array()).matrix();
                 break;

            case LossFunction::CROSS_ENTROPY:
                 // If output activation is Sigmoid, CE simplifies nicely: d[L] = a[L] - y
                 if (outputLayer.activationFunction == 1) { // Sigmoid
                     layers.back().deltas = activations_L - expected;
                 }
                  // If output is Softmax (not implemented here), CE also simplifies: d[L] = a[L] - y
                  // else if (outputLayer.activationFunction == SOFTMAX_ID) {
                  //    layers.back().deltas = activations_L - expected;
                 //}
                 else { // General case for CE (less common for output layer without sigmoid/softmax)
                      Eigen::VectorXd p_clipped = activations_L.array().max(1e-15).min(1.0 - 1e-15);
                      // dC/dA = -y/p + (1-y)/(1-p)
                      dCost_dAct = (-(expected.array() / p_clipped.array()) + (1.0 - expected.array()) / (1.0 - p_clipped.array())).matrix();
                      // d[L] = dC/dA * activate'(Z[L])
                      layers.back().deltas = (dCost_dAct.array() * getUtil().activationDerivativeEigen(preActivations_L, outputLayer.activationFunction).array()).matrix();
                 }
                 break;
        }


        // 2. Calculate hidden layer deltas (backwards)
        for (int k = layers.size() - 2; k > 0; --k) {
            Layer& hiddenLayer = layers[k];
            Layer& nextLayer = layers[k + 1]; // Layer k+1

            // d[k] = (W[k]^T * d[k+1]) .* activate'(Z[k])
            // Note: W[k] connects k -> k+1. Its dimensions are (nodes_k+1) x (nodes_k)
            hiddenLayer.deltas = ( (hiddenLayer.weights.transpose() * nextLayer.deltas).array() *
                                   getUtil().activationDerivativeEigen(hiddenLayer.preActivations, hiddenLayer.activationFunction).array()
                                 ).matrix();
        }

        // 3. Calculate gradients dW and db (can be done forwards or backwards)
        // Accumulate gradients into layer's gradient buffers
        for (size_t k = 0; k < layers.size() - 1; ++k) {
            Layer& currentLayer = layers[k]; // Layer k (e.g., input layer first)
            Layer& nextLayer = layers[k + 1]; // Layer k+1

            // db[k+1] = d[k+1] (Bias gradient for *next* layer)
            // Accumulate (+=) because we might be in a batch
            nextLayer.biasGradients += nextLayer.deltas;

            // dW[k] = d[k+1] * a[k]^T (Weight gradient for weights k -> k+1)
            // Accumulate (+=)
            currentLayer.weightGradients += nextLayer.deltas * currentLayer.activations.transpose();
        }
    }

    // --- Training Loop ---
    void train(const TrainingSetEigen& trainingData, int epochs, double learningRate, int batchSize = 1, bool shuffle = true) {

        if (layers.empty() || trainingData.empty()) {
            std::cerr << "Error: Cannot train empty network or with empty data." << std::endl;
            return;
        }

        size_t n_samples = trainingData.size();
        if (batchSize <= 0) batchSize = 1;
        if (batchSize > n_samples) batchSize = n_samples;
        size_t n_batches = (n_samples + batchSize - 1) / batchSize;

        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);

        std::ofstream loss_log("training_loss_eigen.txt");
        std::cout << "Starting training (Eigen)... Config: "
                  << epochs << " epochs, LR=" << learningRate << ", Batch=" << batchSize
                  << ", Loss=" << (lossType == LossFunction::MSE ? "MSE" : "CE")
                  << ", Samples=" << n_samples << std::endl;

        Timer totalTimer; totalTimer.start();

        for (int e = 0; e < epochs; ++e) {
            auto epoch_start_time = std::chrono::high_resolution_clock::now();
            double epochCost = 0.0;

            if (shuffle) {
                std::shuffle(indices.begin(), indices.end(), getUtil().getRng());
            }

            for (size_t b = 0; b < n_batches; ++b) {
                // Zero gradients before processing the batch
                for (Layer& layer : layers) {
                     layer.zeroGradients();
                }

                size_t batch_start_idx = b * batchSize;
                size_t batch_end_idx = std::min(batch_start_idx + batchSize, n_samples);
                int currentBatchSize = batch_end_idx - batch_start_idx;


                // Process batch: Calculate gradients for each sample and accumulate
                #pragma omp parallel for // Optional: Parallelize gradient calculation across samples
                for (size_t i = batch_start_idx; i < batch_end_idx; ++i) {
                     // Note: Need thread-local storage or careful atomic updates if parallelizing gradient accumulation
                     // For simplicity here, assuming serial accumulation within the batch loop first.
                     // --- If parallelizing, the following needs thread safety ---

                     const auto& sample = trainingData[indices[i]];
                     // 1. Forward pass
                     predict(sample.first);
                     // 2. Calculate and accumulate gradients (Requires modification for thread safety if OMP used)
                      calculateGradients(sample.second);

                     // --- End thread-unsafe section ---

                      // Optionally calculate cost per sample (can be done outside parallel region)
                      // #pragma omp critical
                      // epochCost += calculateCost(sample.second); // Cost calc needs latest prediction
                } // End loop over samples in batch


                 // --- After processing batch ---
                 // 3. Apply accumulated gradients (update weights/biases)
                 for (size_t k = 1; k < layers.size(); ++k) { // Start from first hidden layer
                     layers[k].applyGradients(learningRate, currentBatchSize);
                 }

                 // Calculate cost for the last sample of the batch (approximate batch cost)
                 // Need to re-run predict for the last sample if needed, or store cost during gradient calc.
                 // epochCost += calculateCost(trainingData[indices[batch_end_idx - 1]].second); // Rough cost tracking


            } // End loop over batches


             // --- After epoch ---
             // Calculate average cost more accurately (run predict on all samples?) - Expensive!
             // Or rely on the rough cost tracked per batch.
             // Let's do a proper cost evaluation on the whole dataset (or a validation set)
             double avgEpochCost = 0;
             for(size_t i = 0; i < n_samples; ++i) {
                 predict(trainingData[i].first);
                 avgEpochCost += calculateCost(trainingData[i].second);
             }
             avgEpochCost /= n_samples;


             auto epoch_end_time = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double, std::milli> epoch_duration = epoch_end_time - epoch_start_time;


             if ((e + 1) % 10 == 0 || e == 0 || e == epochs - 1) {
                 std::cout << "Epoch [" << std::setw(4) << (e + 1) << "/" << epochs << "] "
                           << "Avg Cost: " << std::fixed << std::setprecision(6) << avgEpochCost << " | "
                           << "Time: " << std::fixed << std::setprecision(2) << epoch_duration.count() << " ms"
                           << std::endl;
             }
             loss_log << (e + 1) << "\t" << avgEpochCost << "\n";


        } // End loop over epochs

        totalTimer.stop();
        loss_log.close();
        std::cout << "Training finished.\n";
    }

    // --- Printing & Debugging ---
    void printNetworkStructure(bool showMatrices = false) const {
        std::cout << "\n--- Network Structure (Eigen Based) ---\n";
        for (const auto& layer : layers) {
             layer.printLayer(showMatrices);
        }
        std::cout << "--- End Network Structure ---\n" << std::endl;
    }

}; // End class Net

#endif // NET_H