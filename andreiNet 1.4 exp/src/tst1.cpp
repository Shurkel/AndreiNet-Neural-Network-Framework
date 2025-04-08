#include "../utils/andreinet.h"
#include <iostream>
#include <vector>
#include <iomanip> // For std::setprecision, std::fixed
#include <stdexcept> // For exception handling

// Include the main header for your library
// Note: andreinet.h should include net.h, which includes layer.h, etc.

int main() {
    std::cout << BOLDGREEN << "--- AndreiNET v1.3 Demo ---" << RESET << std::endl;
    std::cout << "Showcasing: Adam Optimizer, Tanh Activation, L2 Regularization, LR Decay" << std::endl;
    std::cout << "Problem: XOR Classification" << std::endl << std::endl;

    // --- 1. Prepare XOR Training Data ---
    // Inputs (Eigen Vectors)
    InputDataEigen x00(2); x00 << 0, 0;
    InputDataEigen x01(2); x01 << 0, 1;
    InputDataEigen x10(2); x10 << 1, 0;
    InputDataEigen x11(2); x11 << 1, 1;

    // Targets (Eigen Vectors) - for Cross-Entropy with Sigmoid output (0 or 1)
    TargetDataEigen y00(1); y00 << 0;
    TargetDataEigen y01(1); y01 << 1;
    TargetDataEigen y10(1); y10 << 1;
    TargetDataEigen y11(1); y11 << 0;

    // Create the training set (vector of pairs)
    TrainingSetEigen xorData = {
        {x00, y00},
        {x01, y01},
        {x10, y10},
        {x11, y11}
    };

    std::cout << "[+] Prepared XOR Training Data (" << xorData.size() << " samples)." << std::endl;

    // --- 2. Create and Configure the Network ---
    std::vector<int> topology = {2, 4, 1}; // Input(2) -> Hidden(4) -> Output(1)
    const std::string modelFilename = "xor_model_adam_tanh_reg.bin"; // Use the new version format

    try {
        Net xorNet(topology);
        std::cout << "[+] Created Network instance." << std::endl;

        // --- Configure NEW Features ---

        // a) Set Activation Functions (Explicitly set Tanh for hidden layer)
        // Layer IDs: 0=Input, 1=Hidden, 2=Output
        xorNet.layers[0].setActivationFunction(-1); // Input: Linear (-1)
        xorNet.layers[1].setActivationFunction(3);  // Hidden: Tanh (3)
        xorNet.layers[2].setActivationFunction(1);  // Output: Sigmoid (1) for Cross-Entropy
        std::cout << "    - Activation: Input=Linear, Hidden=Tanh, Output=Sigmoid" << std::endl;

        // b) Set Loss Function
        xorNet.setLossFunction(Net::LossFunction::CROSS_ENTROPY);
        std::cout << "    - Loss Function: Cross-Entropy" << std::endl;

        // c) Set Optimizer to Adam (with default parameters)
        xorNet.setOptimizer(OptimizerType::ADAM); // Or ADAM with custom params: OptimizerType::ADAM, 0.9, 0.999, 1e-7
        std::cout << "    - Optimizer: ADAM (beta1=" << xorNet.adam_beta1 << ", beta2=" << xorNet.adam_beta2 << ")" << std::endl;

        // d) Set L2 Regularization
        double l2Lambda = 0.001; // Small regularization value
        xorNet.setL2Regularization(l2Lambda);
        std::cout << "    - L2 Lambda: " << xorNet.L2_lambda << std::endl;

        // e) Set Learning Rate Decay
        double lrDecay = 0.0005; // Example decay factor
        xorNet.setLearningRateDecay(lrDecay);
        std::cout << "    - LR Decay: " << xorNet.learningRateDecay << std::endl;

        std::cout << "\n--- Initial Network State ---" << std::endl;
        xorNet.printNetworkStructure(false); // Print structure (set true for matrix details)

        // --- 3. Train the Network ---
        int epochs = 3000;
        double learningRate = 0.02; // Initial learning rate
        int batchSize = 4; // Train on full batch for XOR

        std::cout << "\n--- Starting Training ---" << std::endl;
        xorNet.train(xorData, epochs, learningRate, batchSize, false); // Shuffle=false for consistent XOR results
        std::cout << "--- Training Complete ---" << std::endl;

        std::cout << "\n--- Final Network State ---" << std::endl;
        xorNet.printNetworkStructure(false);

        // --- 4. Evaluate the Trained Network ---
        std::cout << "\n--- Network Predictions (after training) ---" << std::endl;
        std::cout << std::fixed << std::setprecision(5);
        for (const auto& sample : xorData) {
            const Eigen::VectorXd& input = sample.first;
            const Eigen::VectorXd& target = sample.second;
            const Eigen::VectorXd& prediction = xorNet.predict(input); // Get prediction

            std::cout << "Input: [" << input.transpose() << "]  "
                      << "Target: [" << target.transpose() << "]  "
                      << BOLDYELLOW << "Prediction: [" << prediction.transpose() << "]" << RESET
                      // Simple thresholding for binary classification view
                      << " -> Class: " << (prediction.coeff(0) > 0.5 ? 1 : 0)
                      << std::endl;
        }
        std::cout << std::endl;


        // --- 5. Save the Trained Network ---
        std::cout << "\n--- Saving Network ---" << std::endl;
        xorNet.save(modelFilename);


        // --- 6. Load the Network into a NEW Instance ---
        std::cout << "\n--- Loading Network into New Instance ---" << std::endl;
        Net loadedNet(topology); // Create new net with SAME topology

        // Important: Configure the loaded net the same way if settings affect structure/behavior
        // (Activations are saved/loaded, but optimizer/loss/lambda/decay are runtime settings)
        loadedNet.layers[0].setActivationFunction(-1);
        loadedNet.layers[1].setActivationFunction(3);
        loadedNet.layers[2].setActivationFunction(1);
        loadedNet.setLossFunction(Net::LossFunction::CROSS_ENTROPY); // Good practice, though not needed for predict
        // Optimizer state is reset on load, but type could be set if you were to continue training
        // loadedNet.setOptimizer(OptimizerType::ADAM);
        // loadedNet.setL2Regularization(l2Lambda);
        // loadedNet.setLearningRateDecay(lrDecay);

        loadedNet.load(modelFilename); // Load weights and biases

        std::cout << "\n--- Loaded Network State ---" << std::endl;
        loadedNet.printNetworkStructure(false);


        // --- 7. Evaluate the LOADED Network ---
        std::cout << "\n--- Loaded Network Predictions ---" << std::endl;
        std::cout << std::fixed << std::setprecision(5);
        bool load_success = true;
        for (const auto& sample : xorData) {
             const Eigen::VectorXd& input = sample.first;
             const Eigen::VectorXd& target = sample.second;
             const Eigen::VectorXd& prediction_orig = xorNet.predict(input); // Original prediction
             const Eigen::VectorXd& prediction_load = loadedNet.predict(input); // Loaded prediction

             std::cout << "Input: [" << input.transpose() << "]  "
                       << "Target: [" << target.transpose() << "]  "
                       << BOLDCYAN << "Prediction: [" << prediction_load.transpose() << "]" << RESET
                       << " -> Class: " << (prediction_load.coeff(0) > 0.5 ? 1 : 0)
                       << std::endl;

             // Check if loaded prediction matches original (within tolerance)
             if (!prediction_load.isApprox(prediction_orig, 1e-9)) {
                 load_success = false;
                 std::cerr << RED << "Mismatch detected between original and loaded prediction!" << RESET << std::endl;
             }
        }

        if (load_success) {
            std::cout << BOLDGREEN << "\n[+] Save/Load test successful. Predictions match." << RESET << std::endl;
        } else {
             std::cout << BOLDRED << "\n[-] Save/Load test FAILED. Predictions differ." << RESET << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << BOLDRED << "\n[!] An error occurred: " << e.what() << RESET << std::endl;
        return 1; // Indicate failure
    } catch (...) {
        std::cerr << BOLDRED << "\n[!] An unknown error occurred." << RESET << std::endl;
        return 1; // Indicate failure
    }


    std::cout << BOLDGREEN << "\n--- Demo Finished ---" << RESET << std::endl;
    return 0; // Indicate success
}
