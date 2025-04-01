#include "../../utils/andreinet.h"
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <ctime>

// Function to read text from file
std::string readTextFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::string content, line;
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return "";
    }
    
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    
    file.close();
    return content;
}

// Function to create character-based training data
// Our network only handles 2D input, so we'll encode characters as their ASCII values scaled to [0,1]
std::pair<std::vector<std::pair<double, double>>, std::vector<double>> 
createTrainingData(const std::string& text, int sequenceLength) {
    std::vector<std::pair<double, double>> inputs;
    std::vector<double> outputs;
    
    // We'll create pairs of (current char, next char) -> (char after next)
    for (size_t i = 0; i < text.length() - 2; i++) {
        // Scale ASCII values to [0,1]
        double char1 = static_cast<unsigned char>(text[i]) / 255.0;
        double char2 = static_cast<unsigned char>(text[i+1]) / 255.0;
        double nextChar = static_cast<unsigned char>(text[i+2]) / 255.0;
        
        inputs.push_back({char1, char2});
        outputs.push_back(nextChar);
    }
    
    return {inputs, outputs};
}

// Function to generate text using the trained network with temperature parameter
std::string generateText(net& network, char startChar1, char startChar2, int length, double temperature = 1.0) {
    std::string generatedText;
    generatedText += startChar1;
    generatedText += startChar2;
    
    char char1 = startChar1;
    char char2 = startChar2;
    
    // Initialize random engine for sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < length; i++) {
        // Prepare input (scaled ASCII values)
        double input1 = static_cast<unsigned char>(char1) / 255.0;
        double input2 = static_cast<unsigned char>(char2) / 255.0;
        
        // Feed the network
        network.clean();
        network.setInputFromVector({input1, input2});
        network.passValues();
        
        // Get the predicted value
        double outputValue = network.layers.back().nodes[0].value;
        
        // Apply temperature to add controlled randomness
        if (temperature > 0.01) {
            // Add noise proportional to temperature
            double noise = (dis(gen) - 0.5) * 0.1 * temperature;  
            outputValue = std::min(1.0, std::max(0.0, outputValue + noise));
        }
        
        // Filter to printable ASCII range (32-126) to avoid control characters
        outputValue = 32.0/255.0 + (126.0-32.0)/255.0 * outputValue;
        char predictedChar = static_cast<char>(outputValue * 255.0);
        
        // Ensure character is in printable range (safety check)
        if (predictedChar < 32 || predictedChar > 126) {
            predictedChar = 32 + (predictedChar % 95); // Map to printable ASCII
        }
        
        // Add some variety if we're getting stuck in a loop
        if (i > 5) {
            // Check if we're repeating the same character
            bool repeating = true;
            for (int j = 1; j <= 5; j++) {
                if (generatedText[generatedText.size()-j] != predictedChar) {
                    repeating = false;
                    break;
                }
            }
            
            // If we detect we're in a loop, inject more randomness
            if (repeating) {
                int randomChar = 32 + (std::rand() % 95); // Random printable ASCII
                predictedChar = static_cast<char>(randomChar);
            }
        }
        
        generatedText += predictedChar;
        
        // Update sliding window
        char1 = char2;
        char2 = predictedChar;
    }
    
    return generatedText;
}

// Function to save network weights to a file
bool saveWeights(net& network, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

    try {
        // Write weights for each layer
        for (int i = 0; i < network.layers.size(); i++) {
            for (int j = 0; j < network.layers[i].nodes.size(); j++) {
                for (int k = 0; k < network.layers[i].nodes[j].next.size(); k++) {
                    outFile << network.layers[i].nodes[j].next[k].weight << " ";
                }
                outFile << "\n";
            }
        }
        
        // Write biases for each layer
        for (int i = 0; i < network.layers.size(); i++) {
            for (int j = 0; j < network.layers[i].nodes.size(); j++) {
                outFile << network.layers[i].nodes[j].bias << " ";
            }
            outFile << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error writing weights: " << e.what() << std::endl;
        outFile.close();
        return false;
    }

    outFile.close();
    if (outFile.fail()) {
        std::cerr << "Error closing the weights file!" << std::endl;
        return false;
    }
    
    std::cout << "Network weights successfully saved to " << filename << std::endl;
    return true;
}

// Function to load network weights from a file
bool loadWeights(net& network, const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Error opening weights file: " << filename << std::endl;
        return false;
    }

    try {
        // Read weights for each layer
        for (int i = 0; i < network.layers.size(); i++) {
            for (int j = 0; j < network.layers[i].nodes.size(); j++) {
                for (int k = 0; k < network.layers[i].nodes[j].next.size(); k++) {
                    if (!(inFile >> network.layers[i].nodes[j].next[k].weight)) {
                        std::cerr << "Error reading weight for layer " << i << ", node " << j << ", weight " << k << std::endl;
                        inFile.close();
                        return false;
                    }
                }
            }
        }
        
        // Read biases for each layer
        for (int i = 0; i < network.layers.size(); i++) {
            for (int j = 0; j < network.layers[i].nodes.size(); j++) {
                if (!(inFile >> network.layers[i].nodes[j].bias)) {
                    std::cerr << "Error reading bias for layer " << i << ", node " << j << std::endl;
                    inFile.close();
                    return false;
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        inFile.close();
        return false;
    }

    inFile.close();
    if (inFile.fail() && !inFile.eof()) {
        std::cerr << "Error reading from weights file!" << std::endl;
        return false;
    }
    
    std::cout << "Network weights successfully loaded from " << filename << std::endl;
    return true;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <text_file_path> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --save <weights_file>    Save trained weights to a file" << std::endl;
    std::cout << "  --load <weights_file>    Load weights from a file (skips training)" << std::endl;
    std::cout << "  --generate <length>      Generate text of specified length (default: 200)" << std::endl;
    std::cout << "  --temperature <value>    Set randomness of text generation (default: 1.0)" << std::endl;
    std::cout << "                           Higher values = more random, lower = more deterministic" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    std::string textFilePath = argv[1];
    std::string saveWeightsPath = "";
    std::string loadWeightsPath = "";
    int generateLength = 200;
    double temperature = 1.0;
    bool shouldTrain = true;
    
    // Print all command line arguments for debugging
    std::cout << "Command line arguments:" << std::endl;
    for (int i = 0; i < argc; i++) {
        std::cout << i << ": " << argv[i] << std::endl;
    }
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--save" && i + 1 < argc) {
            saveWeightsPath = argv[++i];
        }
        else if (arg == "--load" && i + 1 < argc) {
            loadWeightsPath = argv[++i];
            shouldTrain = false; // Skip training if loading weights
            std::cout << "Load parameter detected. Training will be skipped." << std::endl;
        }
        else if (arg == "--generate" && i + 1 < argc) {
            try {
                generateLength = std::stoi(argv[++i]);
                if (generateLength <= 0) {
                    std::cerr << "Generate length must be positive" << std::endl;
                    return 1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Invalid generate length: " << argv[i] << std::endl;
                return 1;
            }
        }
        else if (arg == "--temperature" && i + 1 < argc) {
            try {
                temperature = std::stod(argv[++i]);
                if (temperature < 0.0) {
                    std::cerr << "Temperature must be non-negative" << std::endl;
                    return 1;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Invalid temperature value: " << argv[i] << std::endl;
                return 1;
            }
        }
        else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "Training flag: " << (shouldTrain ? "ON (will train)" : "OFF (will skip training)") << std::endl;
    
    // Read text file
    std::string text = readTextFromFile(textFilePath);
    
    if (text.empty()) {
        std::cerr << "Failed to read text or file is empty." << std::endl;
        return 1;
    }
    
    std::cout << "Text loaded: " << text.length() << " characters" << std::endl;
    
    // Create training data
    auto trainingData = createTrainingData(text, 1);
    std::cout << "Training data created: " << trainingData.first.size() << " samples" << std::endl;
    
    // Create network with more hidden nodes for better pattern detection:
    // - 2 input nodes (for two consecutive characters)
    // - 64 hidden nodes (increased from 32)
    // - 1 output node (predicted next character)
    net network({2, 64, 1});
    
    // Configure the network
    network.noActivate(0);
    network.setActivate(1, 1); // Sigmoid for hidden layer
    network.setActivate(2, 1); // Sigmoid for output layer
    network.connectLayers();
    
    if (!loadWeightsPath.empty()) {
        // Load weights from file
        std::cout << "Attempting to load weights from: " << loadWeightsPath << std::endl;
        if (!loadWeights(network, loadWeightsPath)) {
            std::cerr << "Failed to load weights. Exiting." << std::endl;
            return 1;
        }
    } 
    else {
        // Initialize with small random weights
        std::srand(std::time(nullptr));
        for (int i = 0; i < network.layers.size() - 1; i++) {
            for (int j = 0; j < network.layers[i].nodes.size(); j++) {
                for (int k = 0; k < network.layers[i+1].nodes.size(); k++) {
                    double randomWeight = (std::rand() / (double)RAND_MAX) * 0.2 - 0.1; // Random value between -0.1 and 0.1
                    network.setWeight(i, j, i+1, k, randomWeight);
                }
            }
        }
        network.setBiasAll(0);
    }
    
    // Only train if shouldTrain flag is true
    if (shouldTrain) {
        // Train the network
        std::cout << "Training the network..." << std::endl;
        int epochs = 2000;  // Increased from 1000
        double learningRate = 0.05;  // Changed from 0.1
        network.backPropagate_new(trainingData, epochs, learningRate);
        
        // Save weights if requested
        if (!saveWeightsPath.empty()) {
            if (!saveWeights(network, saveWeightsPath)) {
                std::cerr << "Failed to save weights." << std::endl;
            }
        }
    }
    else {
        std::cout << "Training skipped as requested." << std::endl;
    }
    
    // Generate some text with temperature
    std::cout << "\nGenerating text (" << generateLength << " characters, temperature = " << temperature << "):\n" << std::endl;
    std::string generatedText = generateText(network, text[0], text[1], generateLength, temperature);
    std::cout << generatedText << std::endl;
    
    return 0;
}
