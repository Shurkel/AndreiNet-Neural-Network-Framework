#ifndef NODE_H
#define NODE_H

#include "timer.h" 
#include <vector>
#include <string> 

class Node {
public:
    struct Connection {
        Node* node = nullptr;
        double weight = 0.0;
    };

    std::vector<Connection> next; 
    double value = 0.0;           // Activated value (a)
    double unactivatedValue = 0.0; // Pre-activation value (z)
    double bias = 0.0;
    double delta = 0.0;            // Error signal (dCost/dZ)
    int id = 0;
    int layerId = 0;
    int activationFunction = -1; // -1: Linear, 0: ReLU, 1: Sigmoid, 2: Softplus

    Node(double val = 0.0, double b = 0.0) : value(val), unactivatedValue(val), bias(b) {}

    inline double getValue() const { return value; }
    inline double getBias() const { return bias; }
    inline int getActivationType() const { return activationFunction; }

    inline void setValue(double val) {
        value = val;
        // For input layer or direct setting, unactivatedValue is same as value.
        // Caution if used on hidden/output layers post-calculation.
        unactivatedValue = val; 
    }
    inline void setId(int newId) { id = newId; }
    inline void setLayerId(int newLayerId) { layerId = newLayerId; }
    inline void setBias(double b) { bias = b; }
    
    void cleanForForwardPass() {
        value = 0.0;
        unactivatedValue = 0.0;
    }

    inline double getWeightTo(int nextNodeIndex) const {
        if (nextNodeIndex >= 0 && nextNodeIndex < next.size()) {
            return next[nextNodeIndex].weight;
        }
        return 0.0; 
    }

    inline void setWeightTo(int nextNodeIndex, double w) {
         if (nextNodeIndex >= 0 && nextNodeIndex < next.size()) {
            next[nextNodeIndex].weight = w;
         }
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

    inline void setActivationType(int function) { 
        activationFunction = function;
    }
    
    inline double activate(double z) const {
        switch (activationFunction) {
            case 0: return u.relu(z);
            case 1: return u.sigmoid(z);
            case 2: return u.softplus(z);
            case -1: 
            default: return z; // Linear
        }
    }

    inline double activationDerivative(double z) const {
        switch (activationFunction) {
            case 0: return u.drelu(z);
            case 1: return u.dsigmoid(z); 
            case 2: return u.dsoftplus(z);
            case -1: 
            default: return 1.0; // Linear
        }
    }

    void noActivate() {
        activationFunction = -1; 
    }

    void connect(Node* nextNode, double w = 1.0) {
        next.push_back({nextNode, w});
    }

    void disconnectAll() {
        next.clear();
        next.shrink_to_fit(); 
    }

    const std::vector<Connection>& getConnections() const { 
        return next;
    }

    void printNextNodes() const {
        TextTable t('-', '|', '+');
        t.add("LayerID");
        t.add("NodeID");
        t.add("Weight");
        t.endOfRow();
        std::cout << "[x] Node " << id << " at layer " << layerId << " is connected to nodes:\n";
        for (const auto& conn : next) {
            if (conn.node) { 
                 t.add(std::to_string(conn.node->layerId));
                 t.add(std::to_string(conn.node->id));
                 t.add(std::to_string(conn.weight));
                 t.endOfRow();
            }
        }
        t.setAlignment(2, TextTable::Alignment::RIGHT);
        std::cout << t;
        en; 
    }

    void printDetails() const {
        std::cout
            << '\n'
            << (char)218 << "Node: " << id << '\n'
            << (char)195 << " Value (a): " << value << '\n'
            << (char)195 << " Unactivated (z): " << unactivatedValue << '\n'
            << (char)195 << " Layer: " << layerId << '\n'
            << (char)195 << " Bias: " << bias << "\n"
            << (char)195 << " Delta (dC/dZ): " << delta << "\n" 
            << (char)195 << " Activation func: " << activationFunction << "\n"
            << (char)192 << " Connections to next layer: \n";
        if(!next.empty()) {
            for (size_t i = 0; i < next.size(); ++i) {
                const auto& conn = next[i];
                 if (conn.node) { 
                    std::cout << '\t'; 
                    if (i == 0) std::cout << (char)218; 
                    else if (i == next.size() - 1) std::cout << (char)192; 
                    else std::cout << (char)195; 
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