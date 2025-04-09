// --- START OF FILE net.h ---

#ifndef NET_H
#define NET_H

#include "layer.h" // Includes Eigen, Util, etc.
#include <vector>
#include <string>
#include <numeric>
#include <algorithm> // For std::shuffle
#include <iostream>
#include <fstream>   // Needed for save/load
#include <iomanip>   // For std::setprecision
#include <chrono>    // For timing in train
#include "timer.h"   // Your Timer class
#include <stdexcept> // For exceptions in save/load
#include <cstdint>   // For fixed-width integers (uint32_t, etc.)
#ifdef _OPENMP
#include <omp.h>     // Include OpenMP header if available
#endif

// Define types for training data
using InputDataEigen = Eigen::VectorXd;
using TargetDataEigen = Eigen::VectorXd;
using TrainingPairEigen = std::pair<InputDataEigen, TargetDataEigen>;
using TrainingSetEigen = std::vector<TrainingPairEigen>;


class Net {
public:
    std::vector<Layer> layers;
    double averageSSR = 0.0;
    double averageCrossEntropy = 0.0;

    // Loss function type enum
    enum class LossFunction { MSE, CROSS_ENTROPY };
    LossFunction lossType = LossFunction::MSE; // Default loss

    // Optimizer settings
    OptimizerType optimizerType = OptimizerType::SGD; // Default optimizer
    double adam_beta1 = 0.9;
    double adam_beta2 = 0.999;
    double adam_epsilon = 1e-8;

    // L2 Regularization setting
    double L2_lambda = 0.0; // Default: no regularization

     // Learning rate decay setting
     double learningRateDecay = 0.0; // Default: no decay. Example: 0.001

    // --- Constructor ---
    explicit Net(const std::vector<int>& layerSizes) {
        if (layerSizes.size() < 2) {
            throw std::runtime_error("Error: Network must have at least input and output layers.");
        }
        layers.reserve(layerSizes.size());
        int prevLayerNodeCount = 0;
        for (size_t i = 0; i < layerSizes.size(); ++i) {
             // Pass previous layer's node count to the Layer constructor
            layers.emplace_back(layerSizes[i], prevLayerNodeCount, i);
            if (i > 0) {
                layers[i].prev = &layers[i - 1];
                layers[i-1].next = &layers[i];
             }
            prevLayerNodeCount = layerSizes[i]; // Update for the next iteration
        }

        // Example: Set default activations (customize as needed)
        layers[0].setActivationFunction(-1); // Input Linear
        for (size_t i = 1; i < layers.size() - 1; ++i) {
            layers[i].setActivationFunction(3); // Hidden Tanh (or 1 for Sigmoid, 0 for ReLU)
        }
         // Output layer depends on task
         layers.back().setActivationFunction(1); // Example: Sigmoid for classification
         // layers.back().setActivationFunction(-1); // Example: Linear for regression
         // layers.back().setActivationFunction(3); // Example: Tanh for output in [-1, 1]
    }

    // --- Configuration Setters ---
    void setLossFunction(LossFunction lf) {
        lossType = lf;
         // Adjust output layer activation if needed based on loss
         if (lf == LossFunction::CROSS_ENTROPY && layers.back().activationFunction != 1 /*&& layers.back().activationFunction != SOFTMAX_ID*/) {
              std::cout << "Warning: Cross-entropy loss usually used with Sigmoid/Softmax output. Consider changing output activation.\n";
              // layers.back().setActivationFunction(1); // You might force this
         } else if (lf == LossFunction::MSE && (layers.back().activationFunction == 1 || layers.back().activationFunction == 3)) {
              std::cout << "Warning: MSE loss often used with Linear output (-1). Consider changing output activation.\n";
         }
    }

    void setOptimizer(OptimizerType opt, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8) {
        optimizerType = opt;
        if (opt == OptimizerType::ADAM) {
            adam_beta1 = b1;
            adam_beta2 = b2;
            adam_epsilon = eps;
             std::cout << "Optimizer set to ADAM (beta1=" << adam_beta1 << ", beta2=" << adam_beta2 << ", eps=" << adam_epsilon << ")" << std::endl;
             // Reset Adam state if optimizer changes during operation?
             // for(Layer& layer : layers) { layer.resetAdamState(); } // Need to implement resetAdamState if needed
        } else {
             std::cout << "Optimizer set to SGD." << std::endl;
        }
    }

    void setL2Regularization(double lambda) {
        if (lambda < 0.0) {
            std::cerr << "Warning: L2 lambda cannot be negative. Setting to 0." << std::endl;
            L2_lambda = 0.0;
        } else {
            L2_lambda = lambda;
             std::cout << "L2 Regularization Lambda set to: " << L2_lambda << std::endl;
        }
    }

     void setLearningRateDecay(double decay) {
         if (decay < 0.0) {
             std::cerr << "Warning: Learning rate decay cannot be negative. Setting to 0." << std::endl;
             learningRateDecay = 0.0;
         } else {
             learningRateDecay = decay;
             std::cout << "Learning Rate Decay set to: " << learningRateDecay << std::endl;
         }
     }


    // --- Forward Pass ---
    const Eigen::VectorXd& predict(const InputDataEigen& input) {
        if (layers.empty()) {
             throw std::runtime_error("Error: Cannot predict with an empty network.");
        }
         if (layers[0].numNodes != input.size()) {
             std::string errorMsg = "Error: Input vector size (" + std::to_string(input.size())
                       + ") does not match input layer size ("
                       + std::to_string(layers[0].numNodes) + ").";
             throw std::runtime_error(errorMsg);
         }

        // Set input layer activations
        layers[0].activations = input;
         // Apply activation if input layer has one (rare but possible)
         if (layers[0].activationFunction != -1) {
              layers[0].preActivations = layers[0].activations; // z=a for input?
              layers[0].activations = getUtil().activateEigen(layers[0].preActivations, layers[0].activationFunction);
         }


        // Propagate through hidden/output layers
        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i].calculateActivations(); // This now uses inputWeights
        }

        return layers.back().activations;
    }

    const Eigen::VectorXd& getOutput() const {
         // ... (no changes needed)
         if (layers.empty()) {
              static Eigen::VectorXd error_vec; error_vec.resize(0); // Return empty for empty net
              return error_vec;
         }
         // This returns the *current* activation state, assuming predict() was called appropriately
         return layers.back().activations;
    }


    // --- Cost Calculation ---
    double calculateCost(const TargetDataEigen& expected) {
        const Eigen::VectorXd& predicted = layers.back().activations;
        // ... (Check sizes) ...
         if (predicted.size() != expected.size()) { /* error handling */ return std::numeric_limits<double>::infinity(); }


         double dataLoss = 0.0;
         switch(lossType) {
             case LossFunction::MSE: {
                 dataLoss = 0.5 * (predicted - expected).squaredNorm();
                 break;
             }
             case LossFunction::CROSS_ENTROPY: {
                 Eigen::VectorXd p_clipped = predicted.array().max(1e-15).min(1.0 - 1e-15);
                 dataLoss = -( (expected.array() * p_clipped.array().log()) +
                               ((1.0 - expected.array()) * (1.0 - p_clipped.array()).log()) ).sum();
                  break;
             }
             default: /* error handling */ return std::numeric_limits<double>::infinity();
         }

         // Add L2 Regularization term to the cost (optional, mainly for monitoring)
         double L2_reg_cost = 0.0;
         if (L2_lambda > 0.0) {
             for(size_t i = 1; i < layers.size(); ++i) { // Skip input layer
                 if (layers[i].inputWeights.size() > 0) {
                      L2_reg_cost += layers[i].inputWeights.squaredNorm();
                 }
             }
             // Cost = DataLoss + (lambda / 2) * sum(W^2)
             // The division by batchSize 'm' is often implicitly handled in the gradient update step
             // Return Total Cost = Data Loss + Reg Loss Term / (2 * numOutputs maybe?) -> Let's just return data loss for simplicity now
             // The gradient update handles the regularization correctly.
             // We could add (L2_lambda / 2.0) * L2_reg_cost here if needed for reporting total loss.
         }


         // Check for NaN/Inf before returning
         if (!std::isfinite(dataLoss)) {
             std::cerr << "Warning: NaN or Inf encountered during Cost calculation." << std::endl;
             return std::numeric_limits<double>::infinity();
         }
         return dataLoss; // Return only the data loss component by default
    }


    // --- Backpropagation ---
    void calculateGradients(const TargetDataEigen& expected) {
        if (layers.empty()) { return; }
        // ... (Check sizes) ...
        if (layers.back().numNodes != expected.size()) { /* error handling */ return; }


        // 1. Calculate output layer delta (d[L] = dC/dZ[L])
        Layer& outputLayer = layers.back();
        const Eigen::VectorXd& activations_L = outputLayer.activations;
        const Eigen::VectorXd& preActivations_L = outputLayer.preActivations;

        // ... (Calculate output delta based on loss and activation as before) ...
         Eigen::VectorXd dCost_dAct;
         switch(lossType) {
            // ... MSE and CE delta calculations remain the same ...
             case LossFunction::MSE:
                  dCost_dAct = activations_L - expected;
                  outputLayer.deltas = (dCost_dAct.array() * getUtil().activationDerivativeEigen(preActivations_L, outputLayer.activationFunction).array()).matrix();
                  break;
             case LossFunction::CROSS_ENTROPY:
                  if (outputLayer.activationFunction == 1) { // Sigmoid + CE optimization
                      outputLayer.deltas = activations_L - expected;
                  } /* else if (outputLayer.activationFunction == SOFTMAX_ID) { // Softmax + CE optimization
                       outputLayer.deltas = activations_L - expected; // Same simplification
                   }*/
                   else { // General case
                       Eigen::VectorXd p_clipped = activations_L.array().max(1e-15).min(1.0 - 1e-15);
                       dCost_dAct = (-(expected.array() / p_clipped.array()) + (1.0 - expected.array()) / (1.0 - p_clipped.array())).matrix();
                       outputLayer.deltas = (dCost_dAct.array() * getUtil().activationDerivativeEigen(preActivations_L, outputLayer.activationFunction).array()).matrix();
                  }
                  break;
            default: throw std::runtime_error("Unsupported loss function in calculateGradients");
         }
         // ... (NaN/Inf check for outputLayer.deltas) ...
          if (!outputLayer.deltas.allFinite()) { /* handle non-finite deltas */ }


        // 2. Calculate hidden layer deltas (backwards)
        for (int k = layers.size() - 2; k >= 1; --k) { // Iterate down to the first hidden layer (index 1)
            Layer& currentLayer = layers[k];
            Layer& nextLayer = layers[k + 1]; // Layer k+1 used weights FROM layer k

             // d[k] = (W[k+1]^T * d[k+1]) .* activation_derivative(z[k])
             // W[k+1] are the inputWeights of the *next* layer
             if (nextLayer.inputWeights.rows() != nextLayer.numNodes || nextLayer.inputWeights.cols() != currentLayer.numNodes) {
                  std::cerr << "Error: Weight matrix dimension mismatch during delta backprop. Next layer " << nextLayer.layerId << " weights ("
                            << nextLayer.inputWeights.rows() << "x" << nextLayer.inputWeights.cols() << ") vs current layer " << currentLayer.layerId << " nodes ("
                            << currentLayer.numNodes << ")." << std::endl;
                  currentLayer.deltas.setZero(); continue;
             }
             if (nextLayer.deltas.size() != nextLayer.numNodes) {
                 std::cerr << "Error: Delta size mismatch in next layer " << nextLayer.layerId << std::endl;
                 currentLayer.deltas.setZero(); continue;
             }
              if (!nextLayer.deltas.allFinite()) {
                 std::cerr << "Warning: Non-finite deltas from layer " << nextLayer.layerId << ". Clamping preceding delta." << std::endl;
                 currentLayer.deltas.setZero(); continue; // Or clamp previous deltas? Propagating NaNs is bad.
             }


            Eigen::VectorXd weighted_deltas = nextLayer.inputWeights.transpose() * nextLayer.deltas;
            Eigen::VectorXd act_deriv = getUtil().activationDerivativeEigen(currentLayer.preActivations, currentLayer.activationFunction);

             if (weighted_deltas.size() != act_deriv.size() || weighted_deltas.size() != currentLayer.numNodes) {
                  std::cerr << "Error: Size mismatch during hidden delta calculation for layer " << k << std::endl;
                  currentLayer.deltas.setZero(); continue;
             }


            currentLayer.deltas = (weighted_deltas.array() * act_deriv.array()).matrix();

            if (!currentLayer.deltas.allFinite()) {
                 // std::cerr << "Warning: NaN/Inf detected in hidden layer " << k << " deltas. Clamping to zero." << std::endl;
                 currentLayer.deltas = currentLayer.deltas.unaryExpr([](double v){ return std::isfinite(v) ? v : 0.0; });
            }
        }
        // Step 3 (Gradient Accumulation) happens in the training loop.
    }

    // --- Save/Load Functionality ---
    // Needs adjustment for inputWeights
    void save(const std::string& filename) const {
         std::ofstream outFile(filename, std::ios::binary | std::ios::trunc);
         // ... (Error handling) ...

         // --- Header (Magic, Version, NumLayers) ---
         const char magic[9] = "ANDREI10"; // Updated Magic Number for new format
         const uint32_t version = 2;      // Updated File format version
         const uint32_t numLayers = static_cast<uint32_t>(layers.size());
         outFile.write(magic, 8);
         writeRaw(outFile, version);
         writeRaw(outFile, numLayers);

         // --- Layer Data ---
         for (const auto& layer : layers) {
             const uint32_t layerId = static_cast<uint32_t>(layer.layerId);
             const uint32_t numNodes = static_cast<uint32_t>(layer.numNodes);
             const int32_t activationId = static_cast<int32_t>(layer.activationFunction);
             const uint64_t biasSize = static_cast<uint64_t>(layer.biases.size());
             // Save INPUT weights dimensions
             const uint64_t weightRows = static_cast<uint64_t>(layer.inputWeights.rows());
             const uint64_t weightCols = static_cast<uint64_t>(layer.inputWeights.cols());

             writeRaw(outFile, layerId);
             writeRaw(outFile, numNodes);
             writeRaw(outFile, activationId);

             // Write Biases
             writeRaw(outFile, biasSize);
             if (biasSize > 0) {
                 outFile.write(reinterpret_cast<const char*>(layer.biases.data()), biasSize * sizeof(double));
             }

             // Write INPUT Weights
             writeRaw(outFile, weightRows);
             writeRaw(outFile, weightCols);
             if (weightRows > 0 && weightCols > 0) {
                 outFile.write(reinterpret_cast<const char*>(layer.inputWeights.data()), weightRows * weightCols * sizeof(double));
             }

             // Note: Adam state is NOT saved in this version. To save it, add m/v weights/biases here.

             if (!outFile) throw std::runtime_error("Error writing data for layer " + std::to_string(layerId));
         }
         outFile.close();
         // ... (Check close status) ...
         std::cout << "Network saved successfully to " << filename << " (Format v" << version << ")" << std::endl;
     }


     void load(const std::string& filename) {
         std::ifstream inFile(filename, std::ios::binary);
         // ... (Error handling) ...

         // --- Header ---
         char magicRead[9] = {0};
         uint32_t versionRead = 0;
         uint32_t numLayersRead = 0;
         inFile.read(magicRead, 8);
         // ... (Check read) ...
         magicRead[8] = '\0';
         readRaw(inFile, versionRead);
         readRaw(inFile, numLayersRead);

         const char expectedMagic[9] = "ANDREI10";
         if (std::string(magicRead) != expectedMagic) throw std::runtime_error("Invalid magic number");
         if (versionRead != 2) throw std::runtime_error("Incompatible file version");
         if (numLayersRead != layers.size()) throw std::runtime_error("Layer count mismatch");

         // --- Layer Data ---
         for (uint32_t i = 0; i < numLayersRead; ++i) {
             uint32_t layerIdRead, numNodesRead;
             int32_t activationIdRead;
             uint64_t biasSizeRead, weightRowsRead, weightColsRead;

             readRaw(inFile, layerIdRead);
             readRaw(inFile, numNodesRead);
             readRaw(inFile, activationIdRead);

             // Validate layer structure consistency
             if (layerIdRead != layers[i].layerId || numNodesRead != layers[i].numNodes) {
                 throw std::runtime_error("Layer structure mismatch in file for layer " + std::to_string(i));
             }
             layers[i].activationFunction = activationIdRead;

             // Read Biases
             readRaw(inFile, biasSizeRead);
             if (biasSizeRead != static_cast<uint64_t>(layers[i].biases.size())) {
                 throw std::runtime_error("Bias size mismatch for layer " + std::to_string(i));
             }
             if (biasSizeRead > 0) {
                 inFile.read(reinterpret_cast<char*>(layers[i].biases.data()), biasSizeRead * sizeof(double));
                 // ... (Check read) ...
             }

             // Read INPUT Weights
             readRaw(inFile, weightRowsRead);
             readRaw(inFile, weightColsRead);
             if (weightRowsRead != static_cast<uint64_t>(layers[i].inputWeights.rows()) ||
                 weightColsRead != static_cast<uint64_t>(layers[i].inputWeights.cols())) {
                  throw std::runtime_error("Weight dimensions mismatch for layer " + std::to_string(i));
             }
             if (weightRowsRead > 0 && weightColsRead > 0) {
                 inFile.read(reinterpret_cast<char*>(layers[i].inputWeights.data()), weightRowsRead * weightColsRead * sizeof(double));
                 // ... (Check read) ...
             }

             // IMPORTANT: Reset Adam state after loading weights, as it wasn't saved.
              if (layers[i].m_weights.size() > 0) layers[i].m_weights.setZero();
              if (layers[i].v_weights.size() > 0) layers[i].v_weights.setZero();
              if (layers[i].m_biases.size() > 0) layers[i].m_biases.setZero();
              if (layers[i].v_biases.size() > 0) layers[i].v_biases.setZero();

         }
         // ... (Check EOF, close file) ...
          inFile.peek(); // Try to read one more character
          if (!inFile.eof()) { std::cerr << "Warning: Extra data found at the end of file: " << filename << std::endl; }
          inFile.close();
         std::cout << "Network loaded successfully from " << filename << std::endl;
     }


    // --- Training Loop ---
     void train(const TrainingSetEigen& trainingData, int epochs, double initialLearningRate, int batchSize = 1, bool shuffle = true) {

         if (layers.empty() || trainingData.empty()) { /* error handling */ return; }

         size_t n_samples = trainingData.size();
         if (batchSize <= 0) batchSize = 1;
         if (batchSize > n_samples) batchSize = n_samples;
         size_t n_batches = (n_samples + batchSize - 1) / batchSize;

         std::vector<size_t> indices(n_samples);
         std::iota(indices.begin(), indices.end(), 0);

         std::ofstream loss_log("training_loss_eigen.txt"); // Maybe change filename based on optimizer?
         std::cout << "Starting training (Eigen)... Config: "
                   << epochs << " epochs, Initial LR=" << initialLearningRate
                   << ", LR Decay=" << learningRateDecay
                   << ", Batch=" << batchSize
                   << ", Optimizer=" << (optimizerType == OptimizerType::ADAM ? "ADAM" : "SGD")
                   << ", L2 Lambda=" << L2_lambda
                   << ", Loss=" << (lossType == LossFunction::MSE ? "MSE" : "CE")
                   << ", Samples=" << n_samples << std::endl;

         #ifdef _OPENMP
             int num_threads = omp_get_max_threads();
             std::cout << "OpenMP detected (" << num_threads << " threads max). Gradient accumulation loop parallelized." << std::endl;
         #else
              std::cout << "OpenMP not enabled/available. Running serially." << std::endl;
         #endif

         Timer totalTimer; totalTimer.start();
         int adamTimeStep = 0; // Adam time step counter, increments per BATCH update

         for (int e = 0; e < epochs; ++e) {
             auto epoch_start_time = std::chrono::high_resolution_clock::now();
             double epochCostSum = 0.0;
             double currentLearningRate = initialLearningRate / (1.0 + learningRateDecay * e); // Apply decay per epoch

             if (shuffle) {
                 std::shuffle(indices.begin(), indices.end(), getUtil().getRng());
             }

             for (size_t b = 0; b < n_batches; ++b) {
                 // --- Batch Start ---
                 // Zero gradients BEFORE processing the batch
                 for (Layer& layer : layers) {
                      layer.zeroGradients();
                 }

                 size_t batch_start_idx = b * batchSize;
                 size_t batch_end_idx = std::min(batch_start_idx + batchSize, n_samples);
                 int currentBatchSize = batch_end_idx - batch_start_idx;
                 if (currentBatchSize == 0) continue;

                 double batchCostSum = 0.0;

                 // --- Process Samples & Accumulate Gradients ---
                 // This loop can potentially be parallelized if gradient accumulation is thread-safe.
                 // We need temporary gradient storage per thread and then reduction, or atomic operations.
                 // For simplicity, let's keep it serial for now, matching the previous logic.
                 // If parallelizing: Each thread needs its own copy of gradients, summed up after the loop.

                 // #pragma omp parallel for reduction(+:batchCostSum) // Example if parallelizing sample processing
                 for (size_t i = batch_start_idx; i < batch_end_idx; ++i) {
                      const auto& sample = trainingData[indices[i]];

                      // Local Net state for this sample (Forward/Backward pass modifies internal Layer state)
                      // If running samples in parallel, each thread would need its own Net instance or careful state management.
                      // Current serial approach is safe.

                      // 1. Forward pass
                      try { predict(sample.first); }
                      catch (const std::exception& pred_err) { /* error handling */ continue; }

                      // 2. Calculate loss for this sample
                      batchCostSum += calculateCost(sample.second);

                      // 3. Calculate deltas based on current state
                      try { calculateGradients(sample.second); }
                      catch (const std::exception& grad_err) { /* error handling */ continue; }

                      // 4. Accumulate gradients (using calculated deltas and activations)
                      // This part updates the shared Layer gradient matrices. Must be atomic or locked if parallel.
                      // Keeping it serial within the sample loop is safest without complex changes.
                      for(size_t k=1; k < layers.size(); ++k) { // Start from first hidden layer
                          if (!layers[k].prev) continue; // Should not happen for k>=1

                           // db[k] += delta[k]
                          if(layers[k].deltas.size() == layers[k].biasGradients.size()) {
                              layers[k].biasGradients += layers[k].deltas;
                          } else { /* error */ }

                           // dW[k] += delta[k] * a[k-1]^T
                          if(layers[k].deltas.size() == layers[k].inputWeightGradients.rows() &&
                             layers[k].prev->activations.size() == layers[k].inputWeightGradients.cols())
                          {
                              // Outer product: (N_k x 1) * (1 x N_{k-1}) -> (N_k x N_{k-1})
                             layers[k].inputWeightGradients += layers[k].deltas * layers[k].prev->activations.transpose();
                          } else { /* error */ }
                      }
                 } // --- End loop over samples in batch ---


                 // --- After processing batch ---
                 adamTimeStep++; // Increment Adam time step AFTER processing batch, BEFORE applying gradients

                 // 5. Apply accumulated gradients (update weights/biases using selected optimizer)
                  #ifdef _OPENMP
                  #pragma omp parallel for // Parallelize gradient application across layers
                  #endif
                 for (size_t k = 1; k < layers.size(); ++k) { // Apply gradients layer by layer
                     layers[k].applyGradients(currentLearningRate, currentBatchSize, optimizerType,
                                             adam_beta1, adam_beta2, adam_epsilon, adamTimeStep,
                                             L2_lambda);
                 }

                 epochCostSum += batchCostSum; // Add batch cost to epoch cost

             } // --- End loop over batches ---


             // --- After epoch ---
             double avgEpochCost = (n_samples > 0) ? (epochCostSum / n_samples) : 0.0;
             auto epoch_end_time = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double, std::milli> epoch_duration = epoch_end_time - epoch_start_time;

             if ((e + 1) % 10 == 0 || e == 0 || e == epochs - 1) { // Print every 10 epochs
                 std::cout << "Epoch [" << std::setw(4) << (e + 1) << "/" << epochs << "] "
                           << "Avg Cost: " << std::fixed << std::setprecision(6) << avgEpochCost
                           << " LR: " << std::fixed << std::setprecision(6) << currentLearningRate
                           << " | Time: " << std::fixed << std::setprecision(2) << epoch_duration.count() << " ms"
                           << std::endl;
             }
             loss_log << (e + 1) << "\t" << avgEpochCost << "\t" << currentLearningRate << "\n";

             if (!std::isfinite(avgEpochCost)) { /* Handle NaN/Inf cost */ break; }

         } // End loop over epochs

         totalTimer.stop(false);
         loss_log.close();
         std::cout << "Training finished.\n";
         std::cout << "[+] Total Training Time: " << std::fixed << std::setprecision(2)
                   << totalTimer.getDurationMs() << " ms ("
                   << totalTimer.getDurationMs() / 1000.0 << " s)" << std::endl;
     } // End Train method


    // --- Printing & Debugging ---
     void printNetworkStructure(bool showMatrices = false) const {
        std::cout << "\n--- Network Structure (Eigen Based) ---\n";
        std::cout << "Optimizer: " << (optimizerType == OptimizerType::ADAM ? "ADAM" : "SGD")
                  << ", Loss: " << (lossType == LossFunction::MSE ? "MSE" : "CrossEntropy")
                  << ", L2 Lambda: " << L2_lambda << "\n";
        for (const auto& layer : layers) {
             layer.printLayer(showMatrices);
        }
        std::cout << "--- End Network Structure ---\n" << std::endl;
    }

    // Helper to write raw data (no change)
    template<typename T>
    static void writeRaw(std::ofstream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    // Helper to read raw data (no change)
    template<typename T>
    static void readRaw(std::ifstream& in, T& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in) throw std::runtime_error("Error reading data from file stream.");
    }

}; // End class Net

#endif // NET_H