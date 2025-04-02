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
             // Use throw instead of cerr for constructor failure
            throw std::runtime_error("Error: Network must have at least input and output layers.");
        }
        layers.reserve(layerSizes.size());
        for (size_t i = 0; i < layerSizes.size(); ++i) {
            int nextLayerNodeCount = (i < layerSizes.size() - 1) ? layerSizes[i + 1] : 0;
            layers.emplace_back(layerSizes[i], i, nextLayerNodeCount);
            if (i > 0) {
                layers[i].prev = &layers[i - 1];
                 // Set next pointer in the *previous* layer's iteration
                layers[i-1].next = &layers[i];
             }
        }
        // Note: The next pointer for the last layer remains nullptr, which is correct.


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
        if (layers.empty()) {
             // Returning a static empty vector is problematic if the caller expects a specific size
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

        // Propagate through hidden/output layers
        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i].calculateActivations();
        }

        // Return output layer activations
        return layers.back().activations;
    }

    const Eigen::VectorXd& getOutput() const {
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
          if (predicted.size() == 0 || expected.size() == 0) return 0.0; // No cost if empty
         if (predicted.size() != expected.size()) {
              std::cerr << "Warning: Cost calculation size mismatch. Pred: "
                        << predicted.size() << ", Exp: " << expected.size() << std::endl;
              return std::numeric_limits<double>::infinity(); // Return Inf to signal error
         }


         switch(lossType) {
             case LossFunction::MSE: {
                 // 0.5 * sum((a - y)^2)
                 double mse = 0.5 * (predicted - expected).squaredNorm();
                 return std::isfinite(mse) ? mse : std::numeric_limits<double>::infinity();
             }
             case LossFunction::CROSS_ENTROPY: {
                 // - sum(y*log(p) + (1-y)*log(1-p))
                 // Clamp predictions for numerical stability
                  Eigen::VectorXd p_clipped = predicted.array().max(1e-15).min(1.0 - 1e-15);
                  double ce = -( (expected.array() * p_clipped.array().log()) +
                                 ((1.0 - expected.array()) * (1.0 - p_clipped.array()).log()) ).sum();
                  if (!std::isfinite(ce)) {
                      std::cerr << "Warning: NaN or Inf encountered during Cross-Entropy calculation." << std::endl;
                      return std::numeric_limits<double>::infinity();
                  }
                 return ce;
             }
             default:
                 std::cerr << "Warning: Unknown loss function type in calculateCost." << std::endl;
                 return std::numeric_limits<double>::infinity(); // Indicate error
         }
    }


    // --- Backpropagation ---
    // Calculates gradients (dW, db) for a single example
    // NOTE: Modifies internal layer state (deltas). Not thread-safe for concurrent calls on same Net.
    void calculateGradients(const TargetDataEigen& expected) {
        if (layers.empty()) { return; }
        if (layers.back().numNodes == 0 || expected.size() == 0) return;

        if (layers.back().numNodes != expected.size()) {
            std::cerr << "Error: Target vector size (" << expected.size()
                      << ") mismatch during backprop with output layer size ("
                      << layers.back().numNodes << ")." << std::endl;
            return;
        }

        Layer& outputLayer = layers.back(); // Need non-const ref to modify delta
        const Eigen::VectorXd& activations_L = outputLayer.activations;
        const Eigen::VectorXd& preActivations_L = outputLayer.preActivations;

        if (activations_L.size() != expected.size() || preActivations_L.size() != expected.size()) {
             std::cerr << "Error: Internal state size mismatch in output layer during backprop." << std::endl;
             return;
        }

        // 1. Calculate output layer delta (d[L] = dC/dZ[L])
        Eigen::VectorXd dCost_dAct;
        switch(lossType) {
            case LossFunction::MSE:
                 dCost_dAct = activations_L - expected;
                 outputLayer.deltas = (dCost_dAct.array() * getUtil().activationDerivativeEigen(preActivations_L, outputLayer.activationFunction).array()).matrix();
                 break;

            case LossFunction::CROSS_ENTROPY:
                 if (outputLayer.activationFunction == 1) { // Sigmoid + CE optimization
                     outputLayer.deltas = activations_L - expected;
                 } else { // General case
                      Eigen::VectorXd p_clipped = activations_L.array().max(1e-15).min(1.0 - 1e-15);
                      dCost_dAct = (-(expected.array() / p_clipped.array()) + (1.0 - expected.array()) / (1.0 - p_clipped.array())).matrix();
                      outputLayer.deltas = (dCost_dAct.array() * getUtil().activationDerivativeEigen(preActivations_L, outputLayer.activationFunction).array()).matrix();
                 }
                 break;
            default:
                 throw std::runtime_error("Unsupported loss function in calculateGradients");
        }

        if (!outputLayer.deltas.allFinite()) {
             std::cerr << "Warning: NaN/Inf detected in output layer deltas. Clamping to zero." << std::endl;
             outputLayer.deltas = outputLayer.deltas.unaryExpr([](double v){ return std::isfinite(v) ? v : 0.0; });
        }

        // 2. Calculate hidden layer deltas (backwards)
        for (int k = layers.size() - 2; k > 0; --k) { // Iterate down to the first hidden layer (index 1)
            Layer& hiddenLayer = layers[k];
            Layer& nextLayer = layers[k + 1];

             // Check dimensions before proceeding
             if (hiddenLayer.weights.cols() != hiddenLayer.numNodes || hiddenLayer.weights.rows() != nextLayer.numNodes) {
                  std::cerr << "Error: Weight matrix dimension mismatch for layer " << k << " during delta backprop." << std::endl;
                  hiddenLayer.deltas.setZero(); // Avoid using potentially uninitialized delta
                  continue;
             }
              if (nextLayer.deltas.size() != nextLayer.numNodes) {
                  std::cerr << "Error: Delta vector size mismatch for layer " << (k+1) << " during delta backprop." << std::endl;
                   hiddenLayer.deltas.setZero();
                  continue;
             }
             if (!nextLayer.deltas.allFinite()) {
                 std::cerr << "Warning: Non-finite deltas encountered from layer " << (k+1) << ". Skipping backprop for layer " << k << std::endl;
                 hiddenLayer.deltas.setZero();
                 continue;
             }


             Eigen::VectorXd weighted_deltas = hiddenLayer.weights.transpose() * nextLayer.deltas;
             Eigen::VectorXd act_deriv = getUtil().activationDerivativeEigen(hiddenLayer.preActivations, hiddenLayer.activationFunction);

             if (weighted_deltas.size() != act_deriv.size()) {
                  std::cerr << "Error: Size mismatch between weighted deltas and activation derivative for layer " << k << std::endl;
                   hiddenLayer.deltas.setZero();
                  continue;
             }

            hiddenLayer.deltas = (weighted_deltas.array() * act_deriv.array()).matrix();

            if (!hiddenLayer.deltas.allFinite()) {
                 std::cerr << "Warning: NaN/Inf detected in hidden layer " << k << " deltas. Clamping to zero." << std::endl;
                 hiddenLayer.deltas = hiddenLayer.deltas.unaryExpr([](double v){ return std::isfinite(v) ? v : 0.0; });
            }
        }

        // Step 3 (Gradient Accumulation) happens OUTSIDE this function, in the training loop.
        // This function only calculates the necessary state (deltas).
    }

    // --- Save/Load Functionality ---

    // Helper to write raw data
    template<typename T>
    static void writeRaw(std::ofstream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    // Helper to read raw data
    template<typename T>
    static void readRaw(std::ifstream& in, T& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!in) { // Check read status
             throw std::runtime_error("Error reading data from file stream.");
        }
    }

    // Save network weights and biases to a binary file
    void save(const std::string& filename) const {
        std::ofstream outFile(filename, std::ios::binary | std::ios::trunc);
        if (!outFile) {
            throw std::runtime_error("Error: Cannot open file for saving: " + filename);
        }

        // --- Header ---
        const char magic[9] = "ANDREIN8"; // Magic number to identify file type
        const uint32_t version = 1;      // File format version
        const uint32_t numLayers = static_cast<uint32_t>(layers.size());

        outFile.write(magic, 8); // Write 8 bytes of magic number
        writeRaw(outFile, version);
        writeRaw(outFile, numLayers);

        // --- Layer Data ---
        for (const auto& layer : layers) {
            // Write layer metadata
            const uint32_t layerId = static_cast<uint32_t>(layer.layerId);
            const uint32_t numNodes = static_cast<uint32_t>(layer.numNodes);
            const int32_t activationId = static_cast<int32_t>(layer.activationFunction);
            const uint64_t biasSize = static_cast<uint64_t>(layer.biases.size()); // Use uint64_t for size
            const uint64_t weightRows = static_cast<uint64_t>(layer.weights.rows());
            const uint64_t weightCols = static_cast<uint64_t>(layer.weights.cols());

            writeRaw(outFile, layerId);
            writeRaw(outFile, numNodes);
            writeRaw(outFile, activationId);

            // Write Biases (size + data)
            writeRaw(outFile, biasSize);
            if (biasSize > 0) {
                // Directly write Eigen vector data
                outFile.write(reinterpret_cast<const char*>(layer.biases.data()), biasSize * sizeof(double));
            }

            // Write Weights (dimensions + data) - connecting this layer to next
            writeRaw(outFile, weightRows);
            writeRaw(outFile, weightCols);
            if (weightRows > 0 && weightCols > 0) {
                 // Directly write Eigen matrix data (column-major by default)
                outFile.write(reinterpret_cast<const char*>(layer.weights.data()), weightRows * weightCols * sizeof(double));
            }

            // Check stream status after each layer write
            if (!outFile) throw std::runtime_error("Error writing data for layer " + std::to_string(layerId) + " to file: " + filename);
        }

         outFile.close(); // Explicitly close
         if (!outFile) { // Check status after closing
              throw std::runtime_error("Error occurred during file closing after saving: " + filename);
         }
        std::cout << "Network saved successfully to " << filename << std::endl;
    }


    // Load network weights and biases from a binary file
    void load(const std::string& filename) {
        std::ifstream inFile(filename, std::ios::binary);
        if (!inFile) {
            throw std::runtime_error("Error: Cannot open file for loading: " + filename);
        }

        // --- Header ---
        char magicRead[9] = {0};
        uint32_t versionRead = 0;
        uint32_t numLayersRead = 0;

        inFile.read(magicRead, 8);
        if (!inFile || inFile.gcount() != 8) throw std::runtime_error("Error reading magic number from: " + filename);
        magicRead[8] = '\0'; // Null-terminate for comparison

        readRaw(inFile, versionRead);
        readRaw(inFile, numLayersRead);

        const char expectedMagic[9] = "ANDREIN8";
        if (std::string(magicRead) != expectedMagic) {
             throw std::runtime_error("Error: Invalid magic number in file: " + filename);
        }
         if (versionRead != 1) {
            throw std::runtime_error("Error: Incompatible file format version (" + std::to_string(versionRead) + ") in: " + filename);
        }

        if (numLayersRead != layers.size()) {
             throw std::runtime_error("Error: Layer count mismatch in file (" + std::to_string(numLayersRead)
                                      + ") vs network instance (" + std::to_string(layers.size()) + "). Cannot load.");
        }

        // --- Layer Data ---
        for (uint32_t i = 0; i < numLayersRead; ++i) {
            uint32_t layerIdRead = 0;
            uint32_t numNodesRead = 0;
            int32_t activationIdRead = 0;
            uint64_t biasSizeRead = 0;
            uint64_t weightRowsRead = 0;
            uint64_t weightColsRead = 0;

            // Read metadata for validation
            readRaw(inFile, layerIdRead);
            readRaw(inFile, numNodesRead);
            readRaw(inFile, activationIdRead);

            // Validate layer structure consistency
             if (layerIdRead != static_cast<uint32_t>(layers[i].layerId) || numNodesRead != static_cast<uint32_t>(layers[i].numNodes)) {
                  throw std::runtime_error("Error: Layer structure mismatch in file for layer " + std::to_string(i) +
                                           " (Expected ID " + std::to_string(layers[i].layerId) + ", Size " + std::to_string(layers[i].numNodes) +
                                           "; Got ID " + std::to_string(layerIdRead) + ", Size " + std::to_string(numNodesRead) + "). Cannot load.");
             }
             // Restore activation function
             layers[i].activationFunction = activationIdRead;


            // Read Biases (size + data)
            readRaw(inFile, biasSizeRead);
            if (biasSizeRead != static_cast<uint64_t>(layers[i].biases.size())) {
                 throw std::runtime_error("Error: Bias vector size mismatch for layer " + std::to_string(i) +
                                          " (Expected " + std::to_string(layers[i].biases.size()) +
                                          ", Got " + std::to_string(biasSizeRead) + "). Cannot load.");
            }
            if (biasSizeRead > 0) {
                inFile.read(reinterpret_cast<char*>(layers[i].biases.data()), biasSizeRead * sizeof(double));
                 if (!inFile) throw std::runtime_error("Error reading bias data for layer " + std::to_string(i) + " from file: " + filename);
            }

            // Read Weights (dimensions + data)
            readRaw(inFile, weightRowsRead);
            readRaw(inFile, weightColsRead);
            if (weightRowsRead != static_cast<uint64_t>(layers[i].weights.rows()) ||
                weightColsRead != static_cast<uint64_t>(layers[i].weights.cols()))
            {
                 throw std::runtime_error("Error: Weight matrix dimensions mismatch for layer " + std::to_string(i) +
                                           " (Expected " + std::to_string(layers[i].weights.rows()) + "x" + std::to_string(layers[i].weights.cols()) +
                                           ", Got " + std::to_string(weightRowsRead) + "x" + std::to_string(weightColsRead) + "). Cannot load.");
            }
             if (weightRowsRead > 0 && weightColsRead > 0) {
                inFile.read(reinterpret_cast<char*>(layers[i].weights.data()), weightRowsRead * weightColsRead * sizeof(double));
                 if (!inFile) throw std::runtime_error("Error reading weight data for layer " + std::to_string(i) + " from file: " + filename);
            }
        }

        // Check for extra data at the end
        inFile.peek(); // Try to read one more character
        if (!inFile.eof()) {
             std::cerr << "Warning: Extra data found at the end of file: " << filename << std::endl;
        }
        inFile.close();

        std::cout << "Network loaded successfully from " << filename << std::endl;
    }


    // --- Training Loop (Serial Sample Processing within Batch - CORRECT AND STABLE) ---
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

        #ifdef _OPENMP
            // Informational only, as sample processing loop is serial now.
            int num_threads = omp_get_max_threads();
            std::cout << "OpenMP is available (" << num_threads << " threads max). Sample processing within batch is serial for correctness." << std::endl;
        #else
             std::cout << "OpenMP not enabled/available. Running serially." << std::endl;
        #endif


        Timer totalTimer; totalTimer.start(); // Use your Timer class

        for (int e = 0; e < epochs; ++e) {
            auto epoch_start_time = std::chrono::high_resolution_clock::now();
            double epochCostSum = 0.0; // Accumulate cost over batches

            if (shuffle) {
                // Use getUtil().getRng() which returns a reference
                std::shuffle(indices.begin(), indices.end(), getUtil().getRng());
            }

            for (size_t b = 0; b < n_batches; ++b) {
                // --- Batch Start ---
                // Zero gradients before processing the batch
                for (Layer& layer : layers) {
                     layer.zeroGradients();
                }

                size_t batch_start_idx = b * batchSize;
                size_t batch_end_idx = std::min(batch_start_idx + batchSize, n_samples);
                int currentBatchSize = batch_end_idx - batch_start_idx;

                if (currentBatchSize == 0) continue;

                double batchCostSum = 0.0;

                // --- Process Samples Serially within Batch ---
                // NO #pragma omp parallel for here. This loop runs on a single thread.
                for (size_t i = batch_start_idx; i < batch_end_idx; ++i) {
                     const auto& sample = trainingData[indices[i]];

                     // 1. Forward pass (modifies internal state - safe now as it's serial)
                     // Use try-catch around predict in case of errors
                     try {
                         predict(sample.first);
                     } catch (const std::exception& pred_err) {
                          std::cerr << "Error during predict for sample " << indices[i] << ": " << pred_err.what() << std::endl;
                          // Skip this sample or handle error appropriately
                          continue;
                     }


                     // 2. Calculate loss for this sample
                     batchCostSum += calculateCost(sample.second);

                     // 3. Calculate gradients based on current state (modifies internal state: deltas - safe now)
                     try {
                         calculateGradients(sample.second);
                     } catch (const std::exception& grad_err) {
                          std::cerr << "Error during calculateGradients for sample " << indices[i] << ": " << grad_err.what() << std::endl;
                          // Skip gradient accumulation for this sample if calculation fails
                          continue;
                     }


                    // 4. Accumulate gradients (serially - safe now)
                    //    This uses the deltas and activations set by calculateGradients
                    for(size_t k=0; k < layers.size() -1; ++k) {
                         // Check if delta vector size matches bias gradient vector size
                         if(layers[k+1].deltas.size() == layers[k+1].biasGradients.size()) {
                             layers[k+1].biasGradients += layers[k+1].deltas; // db[k+1]
                         } else {
                              std::cerr << "Warning: Mismatch size between delta and bias gradient for layer " << (k+1) << std::endl;
                         }

                         // Check if matrix/vector dimensions are compatible for outer product
                         if(layers[k+1].deltas.size() == layers[k].weightGradients.rows() &&
                            layers[k].activations.size() == layers[k].weightGradients.cols())
                         {
                            layers[k].weightGradients += layers[k+1].deltas * layers[k].activations.transpose(); // dW[k]
                         } else {
                              std::cerr << "Warning: Dimension mismatch for weight gradient calculation for layer " << k << std::endl;
                         }
                    }

                } // --- End loop over samples in batch ---


                 // --- After processing batch ---
                 // 5. Apply accumulated gradients (update weights/biases)
                 for (size_t k = 1; k < layers.size(); ++k) { // Start from first hidden layer bias/weights
                     layers[k].applyGradients(learningRate, currentBatchSize);
                 }

                 epochCostSum += batchCostSum; // Add batch cost to epoch cost

            } // --- End loop over batches ---


             // --- After epoch ---
             double avgEpochCost = (n_samples > 0) ? (epochCostSum / n_samples) : 0.0;

             auto epoch_end_time = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double, std::milli> epoch_duration = epoch_end_time - epoch_start_time;

             // Print progress less frequently for long training runs
             if ((e + 1) % 50 == 0 || e == 0 || e == epochs - 1) { // Print every 50 epochs now
                 std::cout << "Epoch [" << std::setw(4) << (e + 1) << "/" << epochs << "] "
                           << "Avg Cost: " << std::fixed << std::setprecision(6) << avgEpochCost << " | "
                           << "Time: " << std::fixed << std::setprecision(2) << epoch_duration.count() << " ms"
                           << std::endl;
             }
             loss_log << (e + 1) << "\t" << avgEpochCost << "\n";

             // Check if cost is NaN or Inf and stop training if needed
             if (!std::isfinite(avgEpochCost)) {
                 std::cerr << "Error: Cost is NaN or Inf after epoch " << (e+1) << ". Stopping training." << std::endl;
                 break; // Exit epoch loop
             }


        } // End loop over epochs

        totalTimer.stop(false); // Stop the overall timer, don't print default message
        loss_log.close();
        std::cout << "Training finished.\n";

        // Print total time clearly
         std::cout << "[+] Total Training Time: " << std::fixed << std::setprecision(2)
                   << totalTimer.getDurationMs() << " ms ("
                   << totalTimer.getDurationMs() / 1000.0 << " s)" << std::endl;

    } // End Train method


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