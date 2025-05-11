#ifndef LAYER_H
#define LAYER_H

#include "node.h" 
#include <vector>
#include <numeric> 

class Layer {
public:
    std::vector<Node> nodes;
    int layerId = 0;
    Layer* next = nullptr; 
    Layer* prev = nullptr; 

    Layer(int numNodes, int id = 0) : layerId(id) {
        nodes.reserve(numNodes); 
        for (int i = 0; i < numNodes; ++i) {
            nodes.emplace_back(0.0); 
            nodes.back().setId(i);
            nodes.back().setLayerId(layerId);
        }
    }

    void setPrev(Layer* p) {
        prev = p;
    }

    void cleanNodesForForwardPass() {
        if (layerId == 0) return; // no input layer values

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

    void setValuesFromVector(const std::vector<double>& values) {
        if (values.size() != nodes.size()) {
            std::cerr << "Error: Layer::setValuesFromVector size mismatch. Layer "
                      << layerId << " has " << nodes.size() << " nodes, input vector has "
                      << values.size() << " elements." << std::endl;
             size_t count = std::min(values.size(), nodes.size());
             for (size_t i = 0; i < count; ++i) {
                nodes[i].setValue(values[i]); 
                nodes[i].unactivatedValue = values[i];
            }
            return;
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodes[i].setValue(values[i]);
            nodes[i].unactivatedValue = values[i]; // input layer value = unactivatedValue
        }
    }

    void setActivationFunctionAll(int function) {
        for (Node& node : nodes) {
            node.setActivationType(function);
        }
    }

    void noActivateAll() {
        setActivationFunctionAll(-1); //-1 for linear
    }

    void setValueAll(double val) { 
        for (Node& node : nodes) {
            node.setValue(val); 
        }
    }
    void setValue(int nodeId, double val) {
         if (nodeId >= 0 && nodeId < nodes.size()) {
             nodes[nodeId].setValue(val);
         } 
    }
    void setBiasAll(double b) {
        for (Node& node : nodes) {
            node.setBias(b);
        }
    }
    void setBias(int nodeId, double b) {
         if (nodeId >= 0 && nodeId < nodes.size()) {
            nodes[nodeId].setBias(b);
         } 
    }

    void connectTo(Layer* nextL, double initialWeightMin = -0.1, double initialWeightMax = 0.1, bool randomInit = true) {
        if (!nextL) return;
        next = nextL;
        nextL->setPrev(this); 

        for (Node& currentNode : nodes) {
            currentNode.disconnectAll(); 
            currentNode.next.reserve(nextL->nodes.size()); 
            for (Node& nextNode : nextL->nodes) {
                double weight = randomInit ? u.randomDouble(initialWeightMin, initialWeightMax) : 1.0;
                currentNode.connect(&nextNode, weight);
            }
        }
    }

    void disconnectFromNext() {
        if (!next) return;
        for (Node& node : nodes) {
            node.disconnectAll();
        }
        if(next->prev == this) {
            next->prev = nullptr; 
        }
        next = nullptr;
    }

    

    void calculateActivations() {
        if (!prev) { // Input layer
            // Values are set by setValuesFromVector
            return;
        }

        for (size_t j = 0; j < nodes.size(); ++j) { 
            Node& currentNode = nodes[j];
            double z_sum = 0.0; 
    
            for (const Node& prev_node : prev->nodes) { 
                // Assumes prev_node.next is ordered by the target node's index (j).
                if (j < prev_node.next.size()) { 
                    z_sum += prev_node.value * prev_node.next[j].weight;
                } else {
                    std::cerr << "Warning: Connection missing or index out of bounds in Layer::calculateActivations for L:" 
                              << layerId << " N:" << j << " from prev L:" << prev->layerId << std::endl;
                }
            }
    
            z_sum += currentNode.bias;
            currentNode.unactivatedValue = z_sum;
            currentNode.value = currentNode.activate(z_sum);
        }
    }

    void printLayer(bool detail = false) const { 
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
                t.add(std::to_string(node.delta)); 
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

    double getWeight(int nodeID, int nextNodeID) const {
        if (nodeID >= 0 && nodeID < nodes.size()) {
            return nodes[nodeID].getWeightTo(nextNodeID); 
        }
        return 0.0; 
    }
     
     double getWeightFromPrev(int prevNodeID, int thisNodeID) const {
         if (prev && prevNodeID >= 0 && prevNodeID < prev->nodes.size() &&
             thisNodeID >= 0 && thisNodeID < nodes.size())
         {
             return prev->nodes[prevNodeID].getWeightTo(thisNodeID); 
         }
         return 0.0; 
     }

     void setWeight(int nodeID, int nextNodeID, double weight) {
         if (nodeID >= 0 && nodeID < nodes.size()) {
             nodes[nodeID].setWeightTo(nextNodeID, weight); 
         }
     }
};

#endif // LAYER_H