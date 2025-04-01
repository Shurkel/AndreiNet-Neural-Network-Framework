
#include "timer.h"



class Node
{// ? DONE 12/24/24

private:

    double value;
    double unactivatedValue;

    double bias = 0.0;
    double delta = 0.0; 
    int id = 0;
    int layerId = 0;
    int activationFunction = -1;
public:
    struct nextNodes
    {
        Node *node;
        double weight;
    };

    vector<nextNodes> next;
    // * CONSTRUCTORS

    Node(double value, double bias = 0.0) : value(value), unactivatedValue(value), bias(bias) {};
    Node() : value(0.0), unactivatedValue(0.0), bias(0.0) {};
    Node(int id, int lID = 0) : id(id), layerId(lID) {};
    // * GETTERS AND SETTERS

    //VALUE 
    double getValue() const { return value; }
    void setValue(double val) { value = unactivatedValue = val; }

    //UNACTIVATED VALUE
    double getUnactivatedValue() { return unactivatedValue; }
    void setUnactivatedValue(double val) { unactivatedValue = val; }
    
    //BIAS
    double getBias() { return bias; }
    void setBias(double b) { bias = b; }

    //DELTA
    double getDelta() { return delta; }
    void setDelta(double d) { delta = d; }

    //ID
    int getId() { return id; }
    void setId(int id) { this->id = id; }

    //LAYER ID
    int getLayerId() { return layerId; }
    void setLayerId(int id) { layerId = id; }

    //ACTIVATION FUNCTION
    int getActivationFunction() { return activationFunction; }
    void setActivationFunction(int function) { activationFunction = function; }

    //NEXT NODES
    vector<nextNodes> getNextNodes() { return next; }
    void setNextNodes(vector<nextNodes> n) { next = n; }

    //WEIGHT
    double getWeight(int nextNodeID) { return next[nextNodeID].weight; }
    double getWeight(Node *nextNode)
    {
        for(auto n : next)
            if (n.node == nextNode)
                return n.weight;
        cerr << "\nNode " << id << " is not connected to node " << nextNode->id << ".";
    }
    void setWeight(int nextNodeID, double w) { next[nextNodeID].weight = w; }
    void setWeight(Node *nextNode, double w)
    {
        for(auto n : next)
            if (n.node == nextNode)
                n.weight = w;
        cerr << "\nNode " << id << " is not connected to node " << nextNode->id << ".";
    }
    void setWeight(double w)
    {//set weight for all next nodes
        for (auto &n : next)
            n.weight = w;
    }
    
    void randomiseWeights()
    {
        for (int i = 0; i < next.size(); i++)
        {
            next[i].weight = u.randomDouble(-1, 1);
        }
    }

    // * * FUNCTIONS
    
    void clean()
    {
        value = 0;
        unactivatedValue = 0;
    }
    
    // * LOGIC
    //CONNECT
    void connect(Node *nextNode, double w=1.0)
    {
        next.push_back({nextNode, w});
    }
    void disconnect(Node *nextNode)
    {
        for (int i = 0; i < next.size() && next[i].node == nextNode; i++)
                next.erase(next.begin() + i);
    }
    void disconnect() { next.clear(); }
    //ACTIVATE
    double activateNode(double val, bool deriv=false)
    {

        //relu
        if (activationFunction == 0)
        {
            if(deriv)
                return u.drelu(val);
            return u.relu(val);
        }
        //sigmoid
        else if (activationFunction == 1)
        {
            if(deriv)
                return u.dsigmoid(val);
            return u.sigmoid(val);
        }
        //softmax
        else if (activationFunction == 2)
        {
            if(deriv)
                return u.dsoftplus(val);
            return u.softplus(val);
        }
        else
        {
            if(deriv)
                return 1;
            return val;
        }
    }
    void noActivate(){ activationFunction = -1; }
    //PASS VALUE
    void passValueTo(Node *nextNode)
    {
        //next node value = this value * weight
        nextNode->value += value * getWeight(nextNode);
        nextNode->unactivatedValue += value * getWeight(nextNode);
    }
    
    // * PRINT
    
    void printNextNodes()
    {
        TextTable t('-', '|', '+');
        t.add("LayerID");
        t.add("nodeID");
        t.add("weight");
        t.endOfRow();
        cout << "[X] Node " << id << " at layer " << layerId << " is connected to nodes:\n";
        for (int i = 0; i < next.size(); i++)
        {
            t.add(to_string(next[i].node->layerId));
            t.add(to_string(next[i].node->id));
            t.add(to_string(next[i].weight));
            t.endOfRow();   
        }
        t.setAlignment(2, TextTable::Alignment::RIGHT);
        cout << t;
        en
    }
    void print()
    {
        cout 
        << '\n'
        << (char)218 << "Node: " << id << '\n' 
        << (char)195 << " Value: " << value << '\n'
        << (char)195 << " Unactivated value: " << unactivatedValue << '\n'
        << (char)195 << " Layer: " << layerId << '\n'
        << (char)195 << " Bias: " << bias << "\n"
        << (char)195 << " Activation function: " << activationFunction << "\n"
        << (char)192 << " Next nodes: \n";
        if(next.size() != 0)
        {
            cout << (char)9 << (char)192 << "Node " << next[0].node->id << " -->  weight " << next[0].weight;
            en
            for (int i = 1; i < next.size(); i++)
            {
                cout << (char)9 << (char)195 << "Node " << next[i].node->id << " -->  weight = " << next[i].weight;
                en
            }
        }
        cout.flush();
    }

};