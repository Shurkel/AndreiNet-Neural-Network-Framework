#ifndef LAYER_H
#define LAYER_H

#include "node.h" // Includes timer, util, includes
#include <vector>
#include <numeric> // For std::accumulate

class Layer {
public:
    std::vector<Node> nodes;
    // bool activationFunction = true; // This seems redundant with Node's activationFunction
    int layerId = 0;
    Layer* next = nullptr; // Pointer to the next layer
    Layer* prev = nullptr; // Pointer to the previous layer (useful for backprop)

    // Constructor
    Layer(int numNodes, int id = 0) : layerId(id) {
        nodes.reserve(numNodes); // Pre-allocate memory
        for (int i = 0; i < numNodes; ++i) {
            nodes.emplace_back(0.0); // Use emplace_back for potentially better performance
            nodes.back().setId(i);
            nodes.back().setLayerId(layerId);
        }
    }

     // Set previous layer pointer
    void setPrev(Layer* p) {
        prev = p;
    }

    // Call cleanForForwardPass on all nodes
    void cleanNodesForForwardPass() {
        // Don't clean input layer values if they are set from external data
        if (layerId == 0) return;

        for (Node& node : nodes) {
            node.cleanForForwardPass();
        }
    }

    std::vector<double> getValues() const {
        std::vector<double> values;
        values.reserve(nodes.size());
        for (const Node& node : nodes) {
            values.push_back(node.getValue());
        }
        return values;
    }

    // Set values from vector (typically for input layer)
    void setValuesFromVector(const std::vector<double>& values) {
        if (values.size() != nodes.size()) {
            std::cerr << "Error: Layer::setValuesFromVector size mismatch. Layer "
                      << layerId << " has " << nodes.size() << " nodes, input vector has "
                      << values.size() << " elements." << std::endl;
            // Decide how to handle: throw, return, resize layer?
            // For now, just print error and potentially truncate/ignore extra.
             size_t count = std::min(values.size(), nodes.size());
             for (size_t i = 0; i < count; ++i) {
                nodes[i].setValue(values[i]); // Cautious about unactivatedValue here
                nodes[i].unactivatedValue = values[i]; // Input layer: value = unactivatedValue
            }
            return; // Added return
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodes[i].setValue(values[i]);
            nodes[i].unactivatedValue = values[i]; // Input layer: value = unactivatedValue
        }
    }

    // --- Activation Control ---
    void setActivationFunctionAll(int function) {
        for (Node& node : nodes) {
            node.setActivationType(function);
        }
    }

    void noActivateAll() {
        setActivationFunctionAll(-1); // Use -1 for linear
    }


    // --- Value/Bias Setters ---
    void setValueAll(double val) {
        for (Node& node : nodes) {
            node.setValue(val); // Be careful with this - overwrites calculated values
        }
    }
    void setValue(int nodeId, double val) {
         if (nodeId >= 0 && nodeId < nodes.size()) {
             nodes[nodeId].setValue(val);
         } // Add error handling?
    }
    void setBiasAll(double b) {
        for (Node& node : nodes) {
            node.setBias(b);
        }
    }
    void setBias(int nodeId, double b) {
         if (nodeId >= 0 && nodeId < nodes.size()) {
            nodes[nodeId].setBias(b);
         } // Add error handling?
    }


    // --- Connections ---
    // Connects *this* layer's nodes TO all nodes in the *next* layer
    void connectTo(Layer* nextL, double initialWeightMin = -0.1, double initialWeightMax = 0.1, bool randomInit = true) {
        if (!nextL) return;
        next = nextL;
        nextL->setPrev(this); // Set back-pointer

        for (Node& currentNode : nodes) {
            currentNode.disconnectAll(); // Clear existing connections first
            currentNode.next.reserve(nextL->nodes.size()); // Pre-allocate connection vector
            for (Node& nextNode : nextL->nodes) {
                double weight = randomInit ? u.randomDouble(initialWeightMin, initialWeightMax) : 1.0;
                currentNode.connect(&nextNode, weight);
            }
        }
    }

    // Disconnect from the next layer
    void disconnectFromNext() {
        if (!next) return;
        for (Node& node : nodes) {
            node.disconnectAll();
        }
        if(next->prev == this) {
            next->prev = nullptr; // Clear back-pointer if it points here
        }
        next = nullptr;
    }

    // --- Softmax (Apply to this layer's *values*) ---
    void applySoftMax() {
        // Get current pre-activation values (or activated if softmax is post-activation?)
        // Usually, softmax *replaces* the final activation. So we work on unactivated values.
        std::vector<double> preActivationValues;
        preActivationValues.reserve(nodes.size());
        for(const auto& node : nodes) {
            preActivationValues.push_back(node.unactivatedValue); // Use z
        }

        std::vector<double> softmaxValues = u.softMax(preActivationValues);

        if (softmaxValues.size() == nodes.size()) {
            for (size_t i = 0; i < nodes.size(); ++i) {
                nodes[i].value = softmaxValues[i]; // Update activated value
                // Note: Softmax derivative is complex and usually handled within the loss function derivative (e.g., CE+Softmax)
            }
        }
    }

    // --- Forward Pass ---
    // Calculate activations for *this* layer based on the *previous* layer's values
    // Assumes prev layer's values are computed and weights are set correctly FROM prev TO this.
    void calculateActivations() {
        if (!prev) { // Input layer
            // Values should already be set by setValuesFromVector
            // No calculation needed, maybe apply activation if input layer had one? (Uncommon)
             // for (Node& node : nodes) {
             //    node.value = node.activate(node.unactivatedValue); // If input layer needs activation
            // }
            return;
        }

        for (size_t j = 0; j < nodes.size(); ++j) { // Iterate nodes in *this* layer by index j
            Node& currentNode = nodes[j];
            double z_sum = 0.0; // Accumulate weighted sum
    
            for (const Node& prev_node : prev->nodes) { // Iterate nodes in previous layer (i)
                // Weight from prev_node to currentNode (which is nodes[j])
                // is stored in prev_node.next[j].weight
                // This assumes prev_node.next is ordered by the target node's index.
                if (j < prev_node.next.size()) { // Boundary check
                    z_sum += prev_node.value * prev_node.next[j].weight;
                } else {
                    // This case should ideally not happen if layers are correctly connected.
                    // Indicates a mismatch in expected connections.
                    std::cerr << "Warning: Connection missing or index out of bounds in Layer::calculateActivations for L:" 
                              << layerId << " N:" << j << " from prev L:" << prev->layerId << std::endl;
                }
            }
    
            z_sum += currentNode.bias;
            currentNode.unactivatedValue = z_sum;
            currentNode.value = currentNode.activate(z_sum);
        }
    }


    // --- Printing ---
    void printLayer(bool detail = false) const { // Add detail flag
        TextTable t('-', '|', '+');
        std::cout << "+------- Layer " << layerId << " (" << nodes.size() << " nodes) -------+\n";
        if (detail) {
            t.add("ID");
            t.add("Value (a)");
            t.add("Pre-Act (z)");
            t.add("Bias");
            t.add("Delta");
            t.add("Act F");
            t.endOfRow();
            for (const Node& node : nodes) {
                t.add(std::to_string(node.id));
                t.add(std::to_string(node.value));
                t.add(std::to_string(node.unactivatedValue));
                t.add(std::to_string(node.bias));
                t.add(std::to_string(node.delta)); // Show delta
                t.add(std::to_string(node.activationFunction));
                t.endOfRow();
            }
        } else {
             t.add("ID");
             t.add("Value (a)");
             t.add("Bias");
             t.endOfRow();
            for (const Node& node : nodes) {
                t.add(std::to_string(node.id));
                t.add(std::to_string(node.value));
                t.add(std::to_string(node.bias));
                t.endOfRow();
            }
        }

        std::cout << t;
         std::cout << "+------------------------------------+\n";
    }

    // --- Weight Access (Convenience - might be slow) ---
    // Get weight FROM a node in *this* layer TO a node in the *next* layer
    double getWeight(int nodeID, int nextNodeID) const {
        if (nodeID >= 0 && nodeID < nodes.size()) {
            return nodes[nodeID].getWeightTo(nextNodeID); // Assumes index access is okay
        }
        return 0.0; // Or NaN
    }
     // Get weight FROM a node in the *previous* layer TO a node in *this* layer
     double getWeightFromPrev(int prevNodeID, int thisNodeID) const {
         if (prev && prevNodeID >= 0 && prevNodeID < prev->nodes.size() &&
             thisNodeID >= 0 && thisNodeID < nodes.size())
         {
             // Access weight stored in the previous node's connection list
             return prev->nodes[prevNodeID].getWeightTo(thisNodeID); // Assumes index access
         }
         return 0.0; // Or NaN
     }


     // Set weight FROM a node in *this* layer TO a node in the *next* layer
     void setWeight(int nodeID, int nextNodeID, double weight) {
         if (nodeID >= 0 && nodeID < nodes.size()) {
             nodes[nodeID].setWeightTo(nextNodeID, weight); // Assumes index access is okay
         }
          // Error handling?
     }


};

#endif // LAYER_H