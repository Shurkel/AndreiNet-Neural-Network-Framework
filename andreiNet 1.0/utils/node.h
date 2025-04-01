
#include "timer.h"



class Node
{

public:
    struct nextNodes
    {
        Node *node;
        double weight;
    };
    vector<nextNodes> next;//<node, weight>
    double value;
    double unactivatedValue;
    double bias = 0.0;
    int id = 0;
    int layerId = 0;
    bool log = false;
    bool yesActivate = true;
    int activationFunction = 0;
    

    
    
    

    Node(double val = 0.0, double b = 0.0)
    {
        value = val;
        bias = b;
    }

    //////////////////////////
    
    // * *NODE FUNCTIONS
    double getValue()
    {
        return value;
    }
    void setValue(double val)
    {
        value = val;
        unactivatedValue = val;
        if(log)
            l << "[+] Node " << id << " value set to " << val << "\n";
    }
    
    int getId()
    {
        return id;
    }
    void setId(int id)
    {
        this->id = id;
        if(log)
            l << "[+] Node " << id << " ID set to " << id << "\n";

    }

    double getBias()
    {
        return bias;
    }
    void setBias(double b)
    {
        bias = b;
        if(log)
            l << "[+] Node " << id << " bias set to " << b << "\n";
    }

    void clean()
    {
        value = 0;
    }
    
    double weight(int nextNodeID)
    {
        return next[nextNodeID].weight;
    }
    double weight(Node *nextNode)
    {
        for (int i = 0; i < next.size(); i++)
        {
            if (next[i].node == nextNode)
            {
                return next[i].weight;
            }
        }
        return 0;
    }
    void randomiseWeights()
    {
        for (int i = 0; i < next.size(); i++)
        {
            next[i].weight = u.randomDouble(-1, 1);
        }
    }

    int getActivate()
    {
        return activationFunction;
    }
    void setActivate(int function)
    {
        activationFunction = function;
        if(log)
        {
            l << "[+] Activation function set to " << function << " for node " << id << "\n";
        }
    }
    void activate(int function)
    {
        // * * add more functions here
        //relu
        if (function == 0)
            value = u.relu(value);
        //sigmoid
        else if (function == 1)
            value = u.sigmoid(value);
        //softmax
        else if (function == 2)
            value = u.softmax(value);
    }
    void noActivate()
    {
        if(yesActivate)
            yesActivate = false;
        else
            yesActivate = true;
        if(log)
        {
            l << "[-] Node " << id << " deactivated\n";
        }
    }
    
    void logging()
    {
        if(log)
        {
            
            l << "[-] Logging disabled for node with id " << id << "\n";
            log = false;
        }
        else
        {
            l << "[+] Logging enabled for node with id " << id << ", layer: " << layerId << "\n";
            log = true;
        }            
    }
    
    void setWeight(int nextNodeID, int nextLayerID, double w)
    {
        next[nextNodeID].weight = w;
        if(log)
        {
            l << "[+] Weight set to " << w << " for node " << next[nextNodeID].node->id << " from node " << id << "\n";
        }
    }
    void setWeightAll(double w)
    {
        for (int i = 0; i < next.size(); i++)
        {
            next[i].weight = w;
            if(log)
                l << "[+] Weight set to " << w << " for node " << next[i].node->id << " from node " << id << "\n";
        }
    }
    double getWeight(int nextNodeID)
    {
        return next[nextNodeID].weight;
    }

    void passValues()
    {
        value += bias;
        unactivatedValue = value;
        
        if(yesActivate)
        {
            //cout << "Activating node " << id << " with value " << value << "\n";
            activate(activationFunction);
        }    
            
        for (int i = 0; i < next.size(); i++)
        {

            next[i].node->value = value*next[i].weight + next[i].node->getValue();// next node value += this value * weight 
            if(log)
            {
                l << "[+] Node " << id << " passed value " << value << " to node " << next[i].node->id << " with weight " << next[i].weight << "\n";
            }
        }
        

    }
    
    void connect(Node *nextNode, double w=1.0)
    {
        next.push_back({nextNode, w});
        //log
        if(log)
        {
            l << "[+] Node " << id << " connected to node " << nextNode->id << " with weight " << w << "\n";
        }
    }
    void disconnect(Node *nextNode)
    {
        for (int i = 0; i < next.size(); i++)
        {
            if (next[i].node == nextNode)
            {
                next.erase(next.begin() + i);
                if(log)
                    l << "[-] Node " << id << " disconnected from node " << nextNode->id << "\n";
            }
        }
    }
    void disconnectAll()
    {
        next.clear();
        if(log)
            l << "[-] Cleared all connections for node " << id << "\n";
    }
    
    vector<nextNodes> getNextNodes()
    {
        return next;
    }
    void setNextNodes(vector<nextNodes> next)
    {
        this->next = next;
    }
    
    void printNextNodes()
    {
        TextTable t('-', '|', '+');
        t.add("LayerID");
        t.add("nodeID");
        t.add("weight");
        t.endOfRow();
        cout << "[x] Node " << id << " at layer " << layerId << " is connected to nodes:\n";
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
    void printDetails()
    {
        cout 
        << '\n'
        << (char)218 << "Node: " << id << '\n' 
        << (char)195 << " Value: " << value << '\n'
        << (char)195 << " Unactivated value: " << unactivatedValue << '\n'
        << (char)195 << " Layer: " << layerId << '\n'
        << (char)195 << " Bias: " << bias << "\n"
        << (char)195 << " Activation function: " << activationFunction << "\n"
        << (char)195 << " Logging: " << log << "\n"
        << (char)192 << " Next nodes: \n";
        if(next.size() != 0)
        {
            cout << (char)9 << (char)218 << "Node " << next[0].node->id << " -->  weight " << next[0].weight;
            en
            for (int i = 1; i < next.size(); i++)
            {
                cout << (char)9 << (char)195 << "Node " << next[i].node->id << " -->  weight " << next[i].weight;
                en
            }
        }
        
        
        cout.flush();
    }

};