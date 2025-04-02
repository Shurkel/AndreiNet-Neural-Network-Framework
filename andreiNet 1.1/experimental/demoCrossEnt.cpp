#include "utils/net.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <string>
#include <sys/stat.h>
#include <direct.h> // For Windows directory creation

// Function to create directory if it doesn't exist
void createDirectoryIfNotExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        #ifdef _WIN32
        _mkdir(path.c_str());
        #else
        mkdir(path.c_str(), 0777);
        #endif
        std::cout << "Created directory: " << path << std::endl;
    }
}

// Function to create output directory structure
void createOutputDirectories() {
    createDirectoryIfNotExists("output");
    createDirectoryIfNotExists("output/network");
    createDirectoryIfNotExists("output/data");
    createDirectoryIfNotExists("output/predictions");
    createDirectoryIfNotExists("output/visuals");
}

// Function to generate a spiral dataset
std::pair<std::vector<std::pair<double, double>>, std::vector<double>> generateSpiralDataset(int points, int classes) {
    std::vector<std::pair<double, double>> coords;
    std::vector<double> labels;
    
    srand(time(NULL));
    
    for (int i = 0; i < points; i++) {
        // Generate points for each class
        for (int c = 0; c < classes; c++) {
            // Spiral formula parameters
            double r = 5.0 * (double)i / points; // Radius
            // Angle with noise
            double angle = c * 4 + 4.0 * (double)i / points + ((double)rand() / RAND_MAX - 0.5) * 0.3;
            
            // Convert polar to cartesian coordinates
            double x = r * cos(angle);
            double y = r * sin(angle);
            
            // Add random noise
            x += ((double)rand() / RAND_MAX - 0.5) * 0.2;
            y += ((double)rand() / RAND_MAX - 0.5) * 0.2;
            
            // Normalize to [-1, 1] range
            x /= 5.0;
            y /= 5.0;
            
            coords.push_back({x, y});
            labels.push_back(c == 0 ? 1.0 : 0.0); // Binary classification
        }
    }
    
    // Shuffle the dataset
    std::vector<int> indices(coords.size());
    for (int i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());
    
    std::vector<std::pair<double, double>> shuffled_coords;
    std::vector<double> shuffled_labels;
    
    for (int i = 0; i < indices.size(); i++) {
        shuffled_coords.push_back(coords[indices[i]]);
        shuffled_labels.push_back(labels[indices[i]]);
    }
    
    return {shuffled_coords, shuffled_labels};
}

// Function to generate a more complex dataset: interlocking moons
std::pair<std::vector<std::pair<double, double>>, std::vector<double>> generateMoonsDataset(int points, double noise) {
    std::vector<std::pair<double, double>> coords;
    std::vector<double> labels;
    
    srand(time(NULL));
    
    // First moon (upper half)
    for (int i = 0; i < points; i++) {
        double angle = M_PI * i / points;
        double x = cos(angle);
        double y = sin(angle);
        
        // Add noise
        x += ((double)rand() / RAND_MAX - 0.5) * noise;
        y += ((double)rand() / RAND_MAX - 0.5) * noise;
        
        coords.push_back({x, y});
        labels.push_back(1.0); // Class 1
    }
    
    // Second moon (lower half, shifted)
    for (int i = 0; i < points; i++) {
        double angle = M_PI * i / points;
        double x = 1 - cos(angle);
        double y = 0.5 - sin(angle);
        
        // Add noise
        x += ((double)rand() / RAND_MAX - 0.5) * noise;
        y += ((double)rand() / RAND_MAX - 0.5) * noise;
        
        coords.push_back({x, y});
        labels.push_back(0.0); // Class 0
    }
    
    // Normalize coordinates to [-1, 1] range
    double maxVal = 0.0;
    for (const auto& coord : coords) {
        maxVal = std::max(maxVal, std::max(std::abs(coord.first), std::abs(coord.second)));
    }
    maxVal = std::max(maxVal, 1.0); // Ensure division doesn't amplify values
    
    for (auto& coord : coords) {
        coord.first /= maxVal;
        coord.second /= maxVal;
    }
    
    // Shuffle the dataset
    std::vector<int> indices(coords.size());
    for (int i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());
    
    std::vector<std::pair<double, double>> shuffled_coords;
    std::vector<double> shuffled_labels;
    
    for (int i = 0; i < indices.size(); i++) {
        shuffled_coords.push_back(coords[indices[i]]);
        shuffled_labels.push_back(labels[indices[i]]);
    }
    
    return {shuffled_coords, shuffled_labels};
}

// Function to generate a complex dataset: concentric circles
std::pair<std::vector<std::pair<double, double>>, std::vector<double>> generateCirclesDataset(int points, double noise) {
    std::vector<std::pair<double, double>> coords;
    std::vector<double> labels;
    
    srand(time(NULL));
    
    // Inner circle
    for (int i = 0; i < points; i++) {
        double angle = 2 * M_PI * i / points;
        double radius = 0.4 + ((double)rand() / RAND_MAX - 0.5) * noise;
        
        double x = radius * cos(angle);
        double y = radius * sin(angle);
        
        coords.push_back({x, y});
        labels.push_back(1.0); // Class 1
    }
    
    // Outer circle
    for (int i = 0; i < points; i++) {
        double angle = 2 * M_PI * i / points;
        double radius = 0.8 + ((double)rand() / RAND_MAX - 0.5) * noise;
        
        double x = radius * cos(angle);
        double y = radius * sin(angle);
        
        coords.push_back({x, y});
        labels.push_back(0.0); // Class 0
    }
    
    // Shuffle the dataset
    std::vector<int> indices(coords.size());
    for (int i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());
    
    std::vector<std::pair<double, double>> shuffled_coords;
    std::vector<double> shuffled_labels;
    
    for (int i = 0; i < indices.size(); i++) {
        shuffled_coords.push_back(coords[indices[i]]);
        shuffled_labels.push_back(labels[indices[i]]);
    }
    
    return {shuffled_coords, shuffled_labels};
}

// Function to generate a complex dataset: checkerboard pattern
std::pair<std::vector<std::pair<double, double>>, std::vector<double>> generateCheckerboardDataset(int points, double noise) {
    std::vector<std::pair<double, double>> coords;
    std::vector<double> labels;
    
    srand(time(NULL));
    
    // Generate points in a grid
    double step = 2.0 / sqrt(points); // Distribute points across [-1,1] x [-1,1]
    
    for (double x = -1.0; x <= 1.0; x += step) {
        for (double y = -1.0; y <= 1.0; y += step) {
            // Add some random offset to avoid perfect grid alignment
            double x_noise = ((double)rand() / RAND_MAX - 0.5) * noise;
            double y_noise = ((double)rand() / RAND_MAX - 0.5) * noise;
            
            double x_pos = x + x_noise;
            double y_pos = y + y_noise;
            
            // Make sure the point stays in the range [-1,1]
            x_pos = std::max(-1.0, std::min(1.0, x_pos));
            y_pos = std::max(-1.0, std::min(1.0, y_pos));
            
            coords.push_back({x_pos, y_pos});
            
            // Checkerboard pattern: determine class based on which quadrant the point is in
            // This creates a 4x4 checkerboard pattern
            int x_quad = (int)((x_pos + 1.0) * 2);
            int y_quad = (int)((y_pos + 1.0) * 2);
            
            // If sum of quadrants is even, class 1, else class 0
            labels.push_back((x_quad + y_quad) % 2 == 0 ? 1.0 : 0.0);
        }
    }
    
    // Shuffle the dataset
    std::vector<int> indices(coords.size());
    for (int i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());
    
    std::vector<std::pair<double, double>> shuffled_coords;
    std::vector<double> shuffled_labels;
    
    for (int i = 0; i < indices.size(); i++) {
        shuffled_coords.push_back(coords[indices[i]]);
        shuffled_labels.push_back(labels[indices[i]]);
    }
    
    // Add more points specifically at the decision boundaries to help the network learn
    int additional_points = points / 4;
    for (int i = 0; i < additional_points; i++) {
        // Generate points near decision boundaries (at x = 0, y = 0, x = 0.5, y = 0.5, etc.)
        double boundaries[] = {-0.5, 0.0, 0.5};
        
        for (double bx : boundaries) {
            for (double by : boundaries) {
                // Add some small random offset
                double x_pos = bx + ((double)rand() / RAND_MAX - 0.5) * 0.1;
                double y_pos = by + ((double)rand() / RAND_MAX - 0.5) * 0.1;
                
                // Make sure the point stays in the range [-1,1]
                x_pos = std::max(-1.0, std::min(1.0, x_pos));
                y_pos = std::max(-1.0, std::min(1.0, y_pos));
                
                coords.push_back({x_pos, y_pos});
                
                // Determine class
                int x_quad = (int)((x_pos + 1.0) * 2);
                int y_quad = (int)((y_pos + 1.0) * 2);
                
                labels.push_back((x_quad + y_quad) % 2 == 0 ? 1.0 : 0.0);
            }
        }
    }
    
    return {shuffled_coords, shuffled_labels};
}

// Function to generate a complex dataset: spiral with 4 classes
std::pair<std::vector<std::pair<double, double>>, std::vector<double>> generateMultiSpiralDataset(int points) {
    std::vector<std::pair<double, double>> coords;
    std::vector<double> labels;
    
    srand(time(NULL));
    
    // Number of spirals (classes)
    const int num_classes = 4;
    
    for (int i = 0; i < points; i++) {
        // Generate points for each class
        for (int c = 0; c < num_classes; c++) {
            // Spiral formula parameters
            double r = 5.0 * (double)i / points; // Radius
            // Angle with noise
            double angle = c * (2 * M_PI / num_classes) + 4.0 * (double)i / points + 
                          ((double)rand() / RAND_MAX - 0.5) * 0.3;
            
            // Convert polar to cartesian coordinates
            double x = r * cos(angle);
            double y = r * sin(angle);
            
            // Add random noise
            x += ((double)rand() / RAND_MAX - 0.5) * 0.15;
            y += ((double)rand() / RAND_MAX - 0.5) * 0.15;
            
            // Normalize to [-1, 1] range
            x /= 5.0;
            y /= 5.0;
            
            coords.push_back({x, y});
            
            // For binary classification, we'll map classes 0,2 -> 1.0 and classes 1,3 -> 0.0
            // This creates a more complex decision boundary
            labels.push_back(c % 2 == 0 ? 1.0 : 0.0);
        }
    }
    
    // Shuffle the dataset
    std::vector<int> indices(coords.size());
    for (int i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    std::random_shuffle(indices.begin(), indices.end());
    
    std::vector<std::pair<double, double>> shuffled_coords;
    std::vector<double> shuffled_labels;
    
    for (int i = 0; i < indices.size(); i++) {
        shuffled_coords.push_back(coords[indices[i]]);
        shuffled_labels.push_back(labels[indices[i]]);
    }
    
    return {shuffled_coords, shuffled_labels};
}

// Function to save network state for visualization
void saveNetworkState(const net& network, int epoch, const std::string& prefix) {
    std::ofstream file("output/network/" + prefix + "_epoch_" + std::to_string(epoch) + ".dat");
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving network state!" << std::endl;
        return;
    }
    
    // Save number of layers and nodes per layer
    file << network.layers.size() << std::endl;
    for (const auto& layer : network.layers) {
        file << layer.nodes.size() << " ";
    }
    file << std::endl;
    
    // Save weights and biases
    for(int i = 0; i < network.layers.size() - 1; i++) {
        for (int j = 0; j < network.layers[i].nodes.size(); j++) {
            for (int k = 0; k < network.layers[i+1].nodes.size(); k++) {
                file << network.layers[i].nodes[j].next[k].weight << " ";
            }
        }
    }
    file << std::endl;
    
    for (int i = 1; i < network.layers.size(); i++) {
        for (int j = 0; j < network.layers[i].nodes.size(); j++) {
            file << network.layers[i].nodes[j].bias << " ";
        }
    }
    file << std::endl;
    
    file.close();
}

// Function to save the dataset for visualization
void saveDataset(const std::pair<std::vector<std::pair<double, double>>, std::vector<double>>& dataset, const std::string& filename) {
    std::ofstream file("output/data/" + filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving dataset!" << std::endl;
        return;
    }
    
    // Save each data point and its label
    for (int i = 0; i < dataset.first.size(); i++) {
        file << dataset.first[i].first << " " << dataset.first[i].second << " " 
             << dataset.second[i] << std::endl;
    }
    
    file.close();
}

// Function to generate a grid of points for decision boundary visualization
void generateGrid(const std::string& filename) {
    std::ofstream file("output/data/" + filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving grid!" << std::endl;
        return;
    }
    
    // Generate a grid of points in the range [-1, 1] x [-1, 1]
    double step = 0.05;
    for (double x = -1.0; x <= 1.0; x += step) {
        for (double y = -1.0; y <= 1.0; y += step) {
            file << x << " " << y << std::endl;
        }
    }
    
    file.close();
}

// Function to run network predictions on grid points
void predictOnGrid(net& network, const std::string& gridFile, const std::string& outFile) {
    std::ifstream gridIn("output/data/" + gridFile);
    std::ofstream predOut("output/predictions/" + outFile);
    
    if (!gridIn.is_open() || !predOut.is_open()) {
        std::cerr << "Failed to open files for grid prediction!" << std::endl;
        return;
    }
    
    double x, y;
    while (gridIn >> x >> y) {
        network.clean();
        
        network.setInputFromVector({x, y});
        network.passValues();
        
        predOut << x << " " << y << " " << network.layers.back().nodes[0].value << std::endl;
    }
    
    gridIn.close();
    predOut.close();
}

// Main function
int main() {
    std::cout << "Cross-Entropy Backpropagation Demo" << std::endl;
    
    // Create output directories
    createOutputDirectories();
    
    // Choose dataset type: 1 = spiral, 2 = moons, 3 = circles, 4 = checkerboard, 5 = multi-spiral
    int datasetType = 1;
    std::pair<std::vector<std::pair<double, double>>, std::vector<double>> dataset;
    
    switch (datasetType) {
        case 1:
            dataset = generateSpiralDataset(100, 2); 
            std::cout << "Generated spiral dataset" << std::endl;
            break;
        case 2:
            dataset = generateMoonsDataset(200, 0.1); 
            std::cout << "Generated moons dataset" << std::endl;
            break;
        case 3:
            dataset = generateCirclesDataset(200, 0.1); 
            std::cout << "Generated circles dataset" << std::endl;
            break;
        case 4:
            dataset = generateCheckerboardDataset(800, 0.05); // More points and less noise
            std::cout << "Generated checkerboard dataset" << std::endl;
            break;
        case 5:
            dataset = generateMultiSpiralDataset(80); 
            std::cout << "Generated multi-spiral dataset" << std::endl;
            break;
        default:
            dataset = generateSpiralDataset(100, 2);
            std::cout << "Generated default spiral dataset" << std::endl;
    }
    
    int datasetSize = dataset.first.size();
    std::cout << "Dataset size: " << datasetSize << " points" << std::endl;
    
    // Save the dataset for visualization
    saveDataset(dataset, "ce_dataset.dat");
    
    // Generate grid for decision boundary visualization
    generateGrid("ce_grid.dat");
    
    // Create the neural network with more complex architecture based on dataset type
    std::vector<int> networkArchitecture;
    
    switch (datasetType) {
        case 1: // Spiral
            networkArchitecture = {2, 10, 1}; 
            break;
        case 2: // Moons
            networkArchitecture = {2, 16, 8, 1};
            break;
        case 3: // Circles
            networkArchitecture = {2, 20, 10, 1};
            break;
        case 4: // Checkerboard - significantly increased complexity
            networkArchitecture = {2, 64, 64, 32, 32, 1};
            break;
        case 5: // Multi-spiral - needs more complexity
            networkArchitecture = {2, 32, 24, 16, 1};
            break;
        default:
            networkArchitecture = {2, 10, 1};
    }
    
    net network(networkArchitecture);
    network.noActivate(0); // No activation for input layer
    
    // Set activation functions for all hidden layers
    for (int i = 1; i < networkArchitecture.size() - 1; i++) {
        network.setActivate(i, 1); // Sigmoid for hidden layers
    }
    
    network.setActivate(networkArchitecture.size() - 1, 1); // Sigmoid for output layer
    
    // Initialize weights with Xavier/Glorot initialization
    srand(time(NULL));
    for (int i = 0; i < network.layers.size() - 1; i++) {
        for (int j = 0; j < network.layers[i].nodes.size(); j++) {
            for (int k = 0; k < network.layers[i+1].nodes.size(); k++) {
                double limit = sqrt(6.0 / (network.layers[i].nodes.size() + network.layers[i+1].nodes.size()));
                double weight = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit;
                network.setWeight(i, j, i+1, k, weight);
            }
        }
    }
    
    // Training parameters - increased epochs and adjusted learning rate
    int epochs = 2000;  // Increased from 500
    double initialLearningRate = 0.05;
    int saveInterval = 50;
    
    std::ofstream lossFile("output/data/ce_loss.dat");
    
    // Training loop with learning rate decay and visualization data collection
    for (int e = 0; e < epochs; e++) {
        // Learning rate decay
        double learningRate = initialLearningRate * (1.0 / (1.0 + 0.0001 * e));
        
        network.clearCrossEntropy();
        double epochLoss = 0.0;
        
        for (int i = 0; i < datasetSize; i++) {
            network.clean();
            network.setInputFromVector({dataset.first[i].first, dataset.first[i].second});
            network.setExpected({dataset.second[i]});
            network.passValues();
            network.getCrossEntropy();
            epochLoss += network.crossEntropy;
            
            if (i == datasetSize - 1 && (e % 50 == 0 || e < 50)) {
                std::cout << "Epoch " << e << ", Loss: " << epochLoss / datasetSize 
                          << ", LR: " << learningRate << std::endl;
            }
        }
        
        // Record the average loss for this epoch
        lossFile << e << " " << epochLoss / datasetSize << std::endl;
        
        // Save the network state periodically
        if (e % saveInterval == 0 || e == epochs - 1) {
            saveNetworkState(network, e, "ce_network");
            
            // Generate predictions on the grid for decision boundary visualization
            predictOnGrid(network, "ce_grid.dat", "ce_pred_" + std::to_string(e) + ".dat");
        }
        
        // Train on all data points with one epoch - with decaying learning rate
        network.backPropagate_crossentropy(dataset, 1, learningRate);
    }
    
    lossFile.close();
    
    // Generate a Python visualization script
    std::ofstream pyScript("visualize_ce.py");
    
    pyScript << R"(
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import os

# Load the dataset
data = np.loadtxt('output/data/ce_dataset.dat')
X = data[:, :2]  # features
y = data[:, 2]   # labels

# Load loss data
loss_data = np.loadtxt('output/data/ce_loss.dat')
epochs = loss_data[:, 0]
losses = loss_data[:, 1]

# Create figure with subplots
fig = plt.figure(figsize=(15, 6))
ax1 = plt.subplot(121)  # For decision boundary
ax2 = plt.subplot(122)  # For loss plot

# Plot the loss curve
ax2.plot(epochs, losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.set_title('Training Loss')
ax2.grid(True)

# Plot the dataset points
class0 = X[y == 0]
class1 = X[y == 1]
ax1.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0', alpha=0.7)
ax1.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1', alpha=0.7)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Decision Boundary Evolution')
ax1.grid(True)
ax1.legend()

# Find all prediction files
pred_files = sorted([f for f in os.listdir('output/predictions') if f.startswith('ce_pred_')])
epochs_saved = [int(f.split('_')[-1].split('.')[0]) for f in pred_files]

# Create colormap for decision boundary
contour = None

def update(frame):
    global contour
    
    # Clear previous contour
    if contour:
        for coll in contour.collections:
            coll.remove()
    
    # Load prediction data
    epoch = epochs_saved[frame]
    pred_data = np.loadtxt(f'output/predictions/ce_pred_{epoch}.dat')
    
    # Reshape data for contour plot
    x_unique = np.sort(np.unique(pred_data[:, 0]))
    y_unique = np.sort(np.unique(pred_data[:, 1]))
    X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
    Z_grid = np.zeros_like(X_grid)
    
    # Fill Z values from predictions
    for i, (x, y, z) in enumerate(pred_data):
        ix = np.where(x_unique == x)[0][0]
        iy = np.where(y_unique == y)[0][0]
        Z_grid[iy, ix] = z
    
    # Plot decision boundary
    contour = ax1.contourf(X_grid, Y_grid, Z_grid, 20, alpha=0.6, cmap=cm.coolwarm)
    ax1.set_title(f'Decision Boundary (Epoch {epoch})')
    
    return contour,

# Create animation
ani = FuncAnimation(fig, update, frames=len(pred_files), interval=300)

# Save animation
ani.save('output/visuals/ce_training_animation.gif', writer='pillow', fps=5)

plt.tight_layout()
plt.savefig('output/visuals/ce_final_plot.png')
plt.show()
)";
    
    pyScript.close();
    
    std::cout << "Demo completed. Run 'python visualize_ce.py' to create visualizations." << std::endl;
    std::cout << "Output files are organized in the 'output' directory." << std::endl;
    
    return 0;
}
