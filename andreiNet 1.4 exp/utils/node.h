#ifndef NODE_H
#define NODE_H

#include "timer.h" // Includes util.h -> includes.h
#include <vector>
#include <string> // For printDetails, etc.

class Node {
public:
    // Forward declare Layer if necessary, though not strictly needed here
    // class Layer;

    // Structure to hold connection info
    struct Connection {
        Node* node = nullptr;
        double weight = 0.0;
        // Could add pointer/index to weight in a central Layer matrix for better cache locality later
    };

    std::vector<Connection> next; // Connections to the *next* layer's nodes
    double value = 0.0;           // Activated value (a)
    double unactivatedValue = 0.0; // Pre-activation value (z)
    double bias = 0.0;
    double delta = 0.0;            // Error signal (dCost/dZ) - IMPORTANT: Often defined wrt Z, not A. Let's assume dC/dZ for standard backprop.
    int id = 0;
    int layerId = 0;
    int activationFunction = -1; // -1: Linear, 0: ReLU, 1: Sigmoid, 2: Softplus

    // Constructor
    Node(double val = 0.0, double b = 0.0) : value(val), unactivatedValue(val), bias(b) {}

    // --- Getters (inline suggested) ---
    inline double getValue() const { return value; }
    inline double getUnactivatedValue() const { return unactivatedValue; }
    inline double getBias() const { return bias; }
    inline double getDelta() const { return delta; }
    inline int getId() const { return id; }
    inline int getLayerId() const { return layerId; }
    inline int getActivationType() const { return activationFunction; } // Renamed

    // --- Setters (inline suggested) ---
    inline void setValue(double val) {
        value = val;
        // Usually, unactivatedValue is set *before* activation during forward pass
        // This setter might be confusing if used incorrectly.
        // Consider separate `setPreActivationValue` and `activateAndUpdateValue`.
        unactivatedValue = val; // For input layer or direct setting? Be cautious.
    }
    inline void setId(int newId) { id = newId; }
    inline void setLayerId(int newLayerId) { layerId = newLayerId; }
    inline void setBias(double b) { bias = b; }
    inline void setDelta(double d) { delta = d; }


    // --- Core Functions ---
    void cleanForForwardPass() {
        value = 0.0;
        unactivatedValue = 0.0;
        // Delta is usually calculated in backprop, might not need clearing here
        // delta = 0.0;
    }

    // Find weight to a specific node in the next layer
    // Potentially slow due to linear search if 'next' is large.
    // Optimize: If nodes/weights were in layer-level arrays, access would be O(1).
    double getWeightTo(const Node* nextNode) const {
        for (const auto& conn : next) {
            if (conn.node == nextNode) {
                return conn.weight;
            }
        }
        // Consider throwing an error or returning NaN if not connected?
        // Returning 0 might hide errors.
        // std::cerr << "Warning: Node " << layerId << ":" << id << " not connected to Node "
        //           << nextNode->layerId << ":" << nextNode->id << std::endl;
        return 0.0; // Or std::numeric_limits<double>::quiet_NaN();
    }

    // Get weight by index (assuming 'next' vector order corresponds to next layer's node IDs)
    // Faster if the order is guaranteed.
    inline double getWeightTo(int nextNodeIndex) const {
        if (nextNodeIndex >= 0 && nextNodeIndex < next.size()) {
             // Assuming next[i] connects to node i in the next layer
            return next[nextNodeIndex].weight;
        }
        // Error handling
        return 0.0; // Or NaN
    }

     // Set weight by index
    inline void setWeightTo(int nextNodeIndex, double w) {
         if (nextNodeIndex >= 0 && nextNodeIndex < next.size()) {
            next[nextNodeIndex].weight = w;
         }
         // Error handling
    }


    void randomiseWeights(double minVal = -1.0, double maxVal = 1.0) {
        for (auto& conn : next) {
            conn.weight = u.randomDouble(minVal, maxVal);
        }
    }

    void setWeightAll(double w) {
        for (auto& conn : next) {
            conn.weight = w;
        }
    }

    // --- Activation ---
    inline void setActivationType(int function) { // Renamed
        activationFunction = function;
    }

    // Calculates activated value from pre-activation input 'z'
    inline double activate(double z) const {
        switch (activationFunction) {
            case 0: return u.relu(z);
            case 1: return u.sigmoid(z);
            case 2: return u.softplus(z);
            case -1: // Linear
            default: return z;
        }
    }

    // Calculates derivative of activation function w.r.t pre-activation input 'z'
    inline double activationDerivative(double z) const {
        switch (activationFunction) {
            case 0: return u.drelu(z);
            case 1: return u.dsigmoid(z); // Derivative w.r.t z
            case 2: return u.dsoftplus(z);
            case -1: // Linear
            default: return 1.0;
        }
    }
     // Calculates derivative w.r.t 'z' using the *activated* value 'a' (useful in backprop)
    inline double activationDerivativeFromOutput(double a) const {
        switch (activationFunction) {
            case 0: return (a > 0.0) ? 1.0 : 0.0; // dReLU from output
            case 1: return u.dsigmoid_from_output(a); // dSigmoid from output: a * (1-a)
            case 2: {
                 // dSoftplus(z) = sigmoid(z) = a_softplus. Need inverse: z = log(exp(a) - 1)
                 // dSoftplus/dz = sigmoid(z). It's simpler to use the z value if available.
                 // If only 'a' is known: Can we express sigmoid(z) using a = log(1+exp(z))?
                 // exp(a) = 1 + exp(z) -> exp(z) = exp(a) - 1
                 // sigmoid(z) = 1 / (1 + exp(-z)) = 1 / (1 + 1/exp(z)) = exp(z) / (exp(z) + 1)
                 // sigmoid(z) = (exp(a) - 1) / (exp(a) - 1 + 1) = (exp(a) - 1) / exp(a) = 1 - exp(-a)
                 if (a < 0) return 0; // Avoid issues with exp(-a) if a is very negative, though softplus output is always > 0
                 return 1.0 - std::exp(-a);
            }
            case -1: // Linear
            default: return 1.0;
        }
    }


    void noActivate() {
        activationFunction = -1; // Use -1 for linear/no activation
    }

    // --- Connections ---
    void connect(Node* nextNode, double w = 1.0) {
        // Check if already connected? Optional.
        next.push_back({nextNode, w});
    }

    void disconnect(Node* nextNode) {
        next.erase(std::remove_if(next.begin(), next.end(),
                   [nextNode](const Connection& c){ return c.node == nextNode; }),
                   next.end());
    }

    void disconnectAll() {
        next.clear();
        next.shrink_to_fit(); // Optional: reclaim memory
    }

    const std::vector<Connection>& getConnections() const { // Renamed
        return next;
    }
    // void setNextNodes(const std::vector<Connection>& new_next) { // Use const ref
    //     next = new_next;
    // }

    // --- Backpropagation Related (Example - adjust based on exact backprop formulas) ---

    // Example: Calculate delta for an output node using MSE loss
    // delta = (dCost/dValue) * (dValue/dUnactivatedValue)
    // delta = (value - target) * activationDerivative(unactivatedValue)
    void calculateOutputDeltaMSE(double target) {
        delta = (value - target) * activationDerivative(unactivatedValue);
    }

    // Example: Calculate delta for an output node using Cross-Entropy loss (for Sigmoid/Softmax)
    // delta = (dCost/dValue) * (dValue/dUnactivatedValue)
    // For CE + Sigmoid, this simplifies nicely: dCost/dZ = (value - target)
    void calculateOutputDeltaCrossEntropy(double target) {
         if (activationFunction == 1) { // Sigmoid
             delta = value - target; // Simplified result for Sigmoid + Cross Entropy
         } else {
             // General case: dCost/dA = -target/value + (1-target)/(1-value) (for binary CE)
             // Need to handle value=0 or value=1 cases carefully
              double dCost_dValue = 0.0;
              // Avoid division by zero - clip value? Or check target value
              if (target == 1.0 && value > 1e-9) {
                  dCost_dValue = -1.0 / value;
              } else if (target == 0.0 && value < 1.0 - 1e-9) {
                  dCost_dValue = 1.0 / (1.0 - value);
              } else if (value > 1e-9 && value < 1.0 - 1e-9) {
                 dCost_dValue = -target / value + (1.0 - target) / (1.0 - value);
              } // else delta remains 0? Or handle error?

             delta = dCost_dValue * activationDerivative(unactivatedValue);
         }
    }


    // Example: Calculate delta for a hidden node
    // delta = (sum over next layer nodes k: delta_k * weight_jk) * activationDerivative(unactivatedValue_j)
    


    // --- Printing ---
    void printNextNodes() const {
        TextTable t('-', '|', '+');
        t.add("LayerID");
        t.add("NodeID");
        t.add("Weight");
        t.endOfRow();
        std::cout << "[x] Node " << id << " at layer " << layerId << " is connected to nodes:\n";
        for (const auto& conn : next) {
            if (conn.node) { // Check if node pointer is valid
                 t.add(std::to_string(conn.node->layerId));
                 t.add(std::to_string(conn.node->id));
                 t.add(std::to_string(conn.weight));
                 t.endOfRow();
            }
        }
        t.setAlignment(2, TextTable::Alignment::RIGHT);
        std::cout << t;
        en; // Use std::cout
    }

    void printDetails() const {
        // Use std::cout
        std::cout
            << '\n'
            << (char)218 << "Node: " << id << '\n'
            << (char)195 << " Value (a): " << value << '\n'
            << (char)195 << " Unactivated (z): " << unactivatedValue << '\n'
            << (char)195 << " Layer: " << layerId << '\n'
            << (char)195 << " Bias: " << bias << "\n"
            << (char)195 << " Delta (dC/dZ): " << delta << "\n" // Added Delta
            << (char)195 << " Activation func: " << activationFunction << "\n"
            << (char)192 << " Connections to next layer: \n";
        if(!next.empty()) {
            for (size_t i = 0; i < next.size(); ++i) {
                const auto& conn = next[i];
                 if (conn.node) { // Check valid pointer
                    std::cout << '\t'; // Use char literal
                    if (i == 0) std::cout << (char)218; // Top corner for first
                    else if (i == next.size() - 1) std::cout << (char)192; // Bottom corner for last
                    else std::cout << (char)195; // Middle connector
                    std::cout << "---> Node " << conn.node->layerId << ":" << conn.node->id
                              << " (Weight: " << conn.weight << ")";
                    en;
                 }
            }
        } else {
             std::cout << "\t" << (char)192 << "---> None\n";
        }
        std::cout.flush();
    }
};

#endif // NODE_H