#include "andreiNet 1.4 exp/utils/andreinet.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <iomanip>
#include <algorithm>

// Define M_PI if it's not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function to clear the console
void clearScreen() {
    #ifdef _WIN32
    std::system("cls");
    #else
    std::system("clear");
    #endif
}

// Helper function to get user input with validation
template <typename T>
T getInput(const std::string& prompt, T min_val, T max_val) {
    T value;
    while (true) {
        std::cout << prompt;
        if (std::cin >> value) {
            if (value >= min_val && value <= max_val) {
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                return value;
            }
            std::cout << "Value must be between " << min_val << " and " << max_val << ".\n";
        } else {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please try again.\n";
        }
    }
}

// Function to generate XOR dataset
TrainingSetEigen generateXORData(int samples = 1000) {
    TrainingSetEigen data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> noise(-0.1, 0.1);
    
    for (int i = 0; i < samples; i++) {
        // Generate random binary inputs with some noise
        double x1 = (std::rand() % 2) + noise(gen);
        double x2 = (std::rand() % 2) + noise(gen);
        
        // Calculate XOR output
        double y = (int(x1 + 0.5) != int(x2 + 0.5)) ? 1.0 : 0.0;
        
        // Create input and output vectors
        InputDataEigen input(2);
        input << x1, x2;
        
        TargetDataEigen target(1);
        target << y;
        
        data.push_back({input, target});
    }
    return data;
}

// Function to generate sine wave data
TrainingSetEigen generateSineData(int samples = 1000) {
    TrainingSetEigen data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0, 2 * M_PI);
    std::normal_distribution<double> noise(0, 0.05);
    
    for (int i = 0; i < samples; i++) {
        double x = dist(gen);
        double y = sin(x) + noise(gen);
        
        InputDataEigen input(1);
        input << x;
        
        TargetDataEigen target(1);
        target << y;
        
        data.push_back({input, target});
    }
    return data;
}

// Function to generate function approximation dataset (x^2 + 3x - 2)
TrainingSetEigen generatePolynomialData(int samples = 1000) {
    TrainingSetEigen data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-5, 5);
    std::normal_distribution<double> noise(0, 0.5);
    
    for (int i = 0; i < samples; i++) {
        double x = dist(gen);
        double y = x*x + 3*x - 2 + noise(gen);
        
        InputDataEigen input(1);
        input << x;
        
        TargetDataEigen target(1);
        target << y;
        
        data.push_back({input, target});
    }
    return data;
}

// Function to generate classification dataset (concentric circles)
TrainingSetEigen generateCircleData(int samples = 1000) {
    TrainingSetEigen data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> angle_dist(0, 2 * M_PI);
    std::normal_distribution<double> inner_radius_noise(0, 0.1);
    std::normal_distribution<double> outer_radius_noise(0, 0.1);
    
    int half_samples = samples / 2;
    
    // Inner circle (class 0)
    for (int i = 0; i < half_samples; i++) {
        double angle = angle_dist(gen);
        double radius = 1.0 + inner_radius_noise(gen);
        
        double x = radius * cos(angle);
        double y = radius * sin(angle);
        
        InputDataEigen input(2);
        input << x, y;
        
        TargetDataEigen target(1);
        target << 0.0;
        
        data.push_back({input, target});
    }
    
    // Outer circle (class 1)
    for (int i = 0; i < half_samples; i++) {
        double angle = angle_dist(gen);
        double radius = 3.0 + outer_radius_noise(gen);
        
        double x = radius * cos(angle);
        double y = radius * sin(angle);
        
        InputDataEigen input(2);
        input << x, y;
        
        TargetDataEigen target(1);
        target << 1.0;
        
        data.push_back({input, target});
    }
    
    return data;
}

// Function to test a trained network with custom input
void testNetwork(Net& network, int inputSize) {
    std::cout << "\n--- Test Your Trained Network ---\n";
    
    InputDataEigen input(inputSize);
    for (int i = 0; i < inputSize; i++) {
        input(i) = getInput<double>("Enter input value for feature " + std::to_string(i+1) + ": ", -100.0, 100.0);
    }
    
    const Eigen::VectorXd& output = network.predict(input);
    
    std::cout << "\nNetwork Output:\n";
    for (int i = 0; i < output.size(); i++) {
        std::cout << "Output " << (i+1) << ": " << output(i) << "\n";
    }
    
    std::cout << "\nPress Enter to continue...";
    std::cin.get();
}

// Function to visualize the training loss
void visualizeTrainingLoss() {
    std::string filename = "training_loss_eigen.txt";
    std::ifstream file(filename);
    
    if (!file) {
        std::cout << "Training loss file not found.\n";
        return;
    }
    
    std::vector<std::pair<int, double>> loss_data;
    int epoch;
    double loss;
    
    while (file >> epoch >> loss) {
        loss_data.push_back({epoch, loss});
    }
    
    if (loss_data.empty()) {
        std::cout << "No training data found in file.\n";
        return;
    }
    
    // Find min and max for scaling
    double min_loss = loss_data[0].second;
    double max_loss = loss_data[0].second;
    for (const auto& point : loss_data) {
        min_loss = std::min(min_loss, point.second);
        max_loss = std::max(max_loss, point.second);
    }
    
    // Simple ASCII visualization
    int width = 70;
    int height = 20;
    std::cout << "\n--- Training Loss Visualization ---\n\n";
    
    std::vector<std::vector<char>> graph(height, std::vector<char>(width, ' '));
    
    // Draw axes
    for (int i = 0; i < height; i++) {
        graph[i][0] = '|';
    }
    for (int j = 0; j < width; j++) {
        graph[height-1][j] = '-';
    }
    graph[height-1][0] = '+';
    
    // Sample points for plotting
    int step = std::max(1, (int)loss_data.size() / width);
    
    for (int j = 0; j < width-1; j++) {
        int data_idx = j * step;
        if (data_idx < loss_data.size()) {
            double normalized_loss = (loss_data[data_idx].second - min_loss) / (max_loss - min_loss);
            int y = height - 1 - (int)(normalized_loss * (height - 2));
            if (y >= 0 && y < height) {
                graph[y][j+1] = '*';
            }
        }
    }
    
    // Print the graph
    for (int i = 0; i < height; i++) {
        std::string line(graph[i].begin(), graph[i].end());
        std::cout << line << std::endl;
    }
    
    std::cout << "\nLoss range: " << min_loss << " to " << max_loss << "\n";
    std::cout << "Epochs: 1 to " << loss_data.back().first << "\n\n";
    
    std::cout << "Press Enter to continue...";
    std::cin.get();
}

// Main menu function
int showMainMenu() {
    clearScreen();
    std::cout << "==================================================\n";
    std::cout << "           ANDREINET EDUCATIONAL PROGRAM          \n";
    std::cout << "==================================================\n";
    std::cout << "1. Create and Train a Neural Network\n";
    std::cout << "2. Load a Pre-trained Network\n";
    std::cout << "3. About Neural Networks\n";
    std::cout << "4. Exit\n";
    std::cout << "==================================================\n";
    return getInput<int>("Enter your choice (1-4): ", 1, 4);
}

// Choose dataset menu
int chooseDatasetMenu() {
    clearScreen();
    std::cout << "==================================================\n";
    std::cout << "                 CHOOSE DATASET                   \n";
    std::cout << "==================================================\n";
    std::cout << "1. XOR Problem (Classification)\n";
    std::cout << "2. Sine Wave (Regression)\n";
    std::cout << "3. Polynomial Function (Regression)\n";
    std::cout << "4. Concentric Circles (Classification)\n";
    std::cout << "5. Return to Main Menu\n";
    std::cout << "==================================================\n";
    return getInput<int>("Enter your choice (1-5): ", 1, 5);
}

// Training options menu
void configureTraining(Net& network, TrainingSetEigen& trainingData, const std::string& datasetName) {
    clearScreen();
    std::cout << "==================================================\n";
    std::cout << "            CONFIGURE TRAINING OPTIONS            \n";
    std::cout << "==================================================\n";
    std::cout << "Dataset: " << datasetName << "\n";
    std::cout << "Network Structure: ";
    for (size_t i = 0; i < network.layers.size(); i++) {
        std::cout << network.layers[i].numNodes;
        if (i < network.layers.size() - 1) std::cout << "-";
    }
    std::cout << "\n";
    
    int epochs = getInput<int>("Enter number of epochs (1-10000): ", 1, 10000);
    double learningRate = getInput<double>("Enter learning rate (0.0001-1.0): ", 0.0001, 1.0);
    int batchSize = getInput<int>("Enter batch size (1-" + std::to_string(trainingData.size()) + "): ", 1, (int)trainingData.size());
    
    int lossChoice = getInput<int>("Choose loss function (1=MSE, 2=Cross Entropy): ", 1, 2);
    if (lossChoice == 1) {
        network.setLossFunction(Net::LossFunction::MSE);
    } else {
        network.setLossFunction(Net::LossFunction::CROSS_ENTROPY);
    }
    
    std::cout << "\nStarting training...\n";
    network.train(trainingData, epochs, learningRate, batchSize);
    
    std::cout << "\nTraining complete! Press Enter to continue...";
    std::cin.get();
}

// Define activation function type menu
void setActivationFunctions(Net& network) {
    clearScreen();
    std::cout << "==================================================\n";
    std::cout << "           SET ACTIVATION FUNCTIONS               \n";
    std::cout << "==================================================\n";
    std::cout << "Activation Function Types:\n";
    std::cout << "-1: Linear\n";
    std::cout << " 0: Step\n";
    std::cout << " 1: Sigmoid\n";
    std::cout << " 2: Tanh\n";
    std::cout << " 3: ReLU\n";
    std::cout << " 4: Leaky ReLU\n";
    
    // Input layer is always linear
    network.layers[0].setActivationFunction(-1);
    
    // Set hidden layers
    for (size_t i = 1; i < network.layers.size() - 1; i++) {
        int choice = getInput<int>("Activation for Hidden Layer " + std::to_string(i) + 
                                   " (" + std::to_string(network.layers[i].numNodes) + 
                                   " nodes) (-1 to 4): ", -1, 4);
        network.layers[i].setActivationFunction(choice);
    }
    
    // Set output layer
    int output_choice = getInput<int>("Activation for Output Layer (-1 to 4): ", -1, 4);
    network.layers.back().setActivationFunction(output_choice);
    
    std::cout << "\nActivation functions set successfully!\n";
    std::cout << "Press Enter to continue...";
    std::cin.get();
}

// Display info about neural networks
void showNeuralNetworkInfo() {
    clearScreen();
    std::cout << "==================================================\n";
    std::cout << "              NEURAL NETWORK BASICS               \n";
    std::cout << "==================================================\n";
    std::cout << "A neural network is a computational model inspired by the\n";
    std::cout << "human brain. It consists of layers of interconnected nodes\n";
    std::cout << "or 'neurons' that process information.\n\n";
    
    std::cout << "Key Components:\n";
    std::cout << "1. Input Layer: Receives the initial data\n";
    std::cout << "2. Hidden Layers: Process the data through weighted connections\n";
    std::cout << "3. Output Layer: Produces the final result\n";
    std::cout << "4. Weights: Connection strengths between neurons\n";
    std::cout << "5. Biases: Offset values for each neuron\n";
    std::cout << "6. Activation Functions: Introduce non-linearity\n\n";
    
    std::cout << "Learning Process:\n";
    std::cout << "1. Forward Pass: Data flows through the network\n";
    std::cout << "2. Error Calculation: Compare output to expected result\n";
    std::cout << "3. Backpropagation: Distribute error through the network\n";
    std::cout << "4. Weight Updates: Adjust connections to reduce error\n\n";
    
    std::cout << "Common Applications:\n";
    std::cout << "- Classification (e.g., XOR problem, image recognition)\n";
    std::cout << "- Regression (e.g., function approximation)\n";
    std::cout << "- Pattern Recognition\n";
    std::cout << "- Time Series Prediction\n\n";
    
    std::cout << "This program uses andreiNet, a C++ neural network library\n";
    std::cout << "developed by Roman Andrei Dan (2023-2024).\n\n";
    
    std::cout << "Press Enter to return to the main menu...";
    std::cin.get();
}

// Create and configure a neural network
Net configureNetwork(int inputSize, int outputSize) {
    clearScreen();
    std::cout << "==================================================\n";
    std::cout << "            NEURAL NETWORK ARCHITECTURE           \n";
    std::cout << "==================================================\n";
    std::cout << "Input size: " << inputSize << "\n";
    std::cout << "Output size: " << outputSize << "\n\n";
    
    int numHiddenLayers = getInput<int>("Enter number of hidden layers (0-5): ", 0, 5);
    
    std::vector<int> layerSizes;
    layerSizes.push_back(inputSize);
    
    for (int i = 0; i < numHiddenLayers; i++) {
        int nodes = getInput<int>("Enter number of neurons for hidden layer " + 
                                 std::to_string(i+1) + " (1-100): ", 1, 100);
        layerSizes.push_back(nodes);
    }
    
    layerSizes.push_back(outputSize);
    
    std::cout << "\nCreating network with architecture: ";
    for (size_t i = 0; i < layerSizes.size(); i++) {
        std::cout << layerSizes[i];
        if (i < layerSizes.size() - 1) std::cout << "-";
    }
    std::cout << "\n";
    
    Net network(layerSizes);
    
    // Ask if user wants to set activation functions
    std::cout << "\nDo you want to set custom activation functions? (1=Yes, 0=No): ";
    int choice;
    std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    if (choice == 1) {
        setActivationFunctions(network);
    } else {
        std::cout << "\nUsing default activation functions.\n";
        std::cout << "Hidden layers: Sigmoid, Output layer: ";
        if (outputSize == 1) {
            std::cout << "Sigmoid for classification tasks.\n";
        } else {
            std::cout << "Linear for regression tasks.\n";
        }
    }
    
    return network;
}

// Main program logic
int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    while (true) {
        int mainChoice = showMainMenu();
        
        // Create and train a network
        if (mainChoice == 1) {
            int datasetChoice;
            while ((datasetChoice = chooseDatasetMenu()) != 5) {
                TrainingSetEigen trainingData;
                std::string datasetName;
                int inputSize = 0;
                int outputSize = 0;
                
                switch (datasetChoice) {
                    case 1: // XOR
                        trainingData = generateXORData();
                        datasetName = "XOR Problem";
                        inputSize = 2;
                        outputSize = 1;
                        break;
                    case 2: // Sine
                        trainingData = generateSineData();
                        datasetName = "Sine Wave";
                        inputSize = 1;
                        outputSize = 1;
                        break;
                    case 3: // Polynomial
                        trainingData = generatePolynomialData();
                        datasetName = "Polynomial Function";
                        inputSize = 1;
                        outputSize = 1;
                        break;
                    case 4: // Circles
                        trainingData = generateCircleData();
                        datasetName = "Concentric Circles";
                        inputSize = 2;
                        outputSize = 1;
                        break;
                }
                
                Net network = configureNetwork(inputSize, outputSize);
                configureTraining(network, trainingData, datasetName);
                
                // Post-training menu
                bool postTrainingLoop = true;
                while (postTrainingLoop) {
                    clearScreen();
                    std::cout << "==================================================\n";
                    std::cout << "               POST-TRAINING OPTIONS              \n";
                    std::cout << "==================================================\n";
                    std::cout << "1. Test Network with Custom Input\n";
                    std::cout << "2. Visualize Training Loss\n";
                    std::cout << "3. Show Network Structure\n";
                    std::cout << "4. Save Trained Network\n";
                    std::cout << "5. Return to Dataset Selection\n";
                    std::cout << "==================================================\n";
                    
                    int postChoice = getInput<int>("Enter your choice (1-5): ", 1, 5);
                    
                    switch (postChoice) {
                        case 1:
                            testNetwork(network, inputSize);
                            break;
                        case 2:
                            visualizeTrainingLoss();
                            break;
                        case 3:
                            clearScreen();
                            network.printNetworkStructure();
                            std::cout << "\nPress Enter to continue...";
                            std::cin.get();
                            break;
                        case 4: {
                            std::string filename;
                            std::cout << "Enter filename to save network: ";
                            std::getline(std::cin, filename);
                            try {
                                network.save(filename);
                                std::cout << "Network saved successfully!\n";
                            } catch (const std::exception& e) {
                                std::cerr << "Error saving network: " << e.what() << std::endl;
                            }
                            std::cout << "\nPress Enter to continue...";
                            std::cin.get();
                            break;
                        }
                        case 5:
                            postTrainingLoop = false;
                            break;
                    }
                }
            }
        }
        // Load pre-trained network
        else if (mainChoice == 2) {
            clearScreen();
            std::cout << "==================================================\n";
            std::cout << "               LOAD TRAINED NETWORK               \n";
            std::cout << "==================================================\n";
            
            std::string filename;
            std::cout << "Enter filename of saved network: ";
            std::getline(std::cin, filename);
            
            // Need to create a network with the right structure first
            std::cout << "\nTo load a network, you need to specify its structure.\n";
            int inputSize = getInput<int>("Enter input layer size: ", 1, 100);
            int hiddenLayers = getInput<int>("Enter number of hidden layers: ", 0, 10);
            
            std::vector<int> architecture;
            architecture.push_back(inputSize);
            
            for (int i = 0; i < hiddenLayers; i++) {
                int hiddenSize = getInput<int>("Enter size of hidden layer " + std::to_string(i+1) + ": ", 1, 100);
                architecture.push_back(hiddenSize);
            }
            
            int outputSize = getInput<int>("Enter output layer size: ", 1, 100);
            architecture.push_back(outputSize);
            
            try {
                Net network(architecture);
                network.load(filename);
                
                clearScreen();
                std::cout << "Network loaded successfully!\n\n";
                network.printNetworkStructure();
                
                std::cout << "\nDo you want to test this network? (1=Yes, 0=No): ";
                int testChoice;
                std::cin >> testChoice;
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                
                if (testChoice == 1) {
                    testNetwork(network, inputSize);
                }
            } catch (const std::exception& e) {
                std::cerr << "\nError loading network: " << e.what() << std::endl;
                std::cout << "\nPress Enter to continue...";
                std::cin.get();
            }
        }
        // About neural networks
        else if (mainChoice == 3) {
            showNeuralNetworkInfo();
        }
        // Exit
        else if (mainChoice == 4) {
            clearScreen();
            std::cout << "Thank you for using the andreiNet Educational Program!\n";
            break;
        }
    }
    
    return 0;
}