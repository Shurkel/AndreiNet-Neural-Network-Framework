#include "node.h"

class Layer
{
public:
    vector<Node> nodes;
    bool activationFunciton = true;
    int layerId = 0;
    //next layer
    Layer *next;

    Layer(int n)
    {
        
        for (int i = 0; i < n; i++)
        {
            nodes.push_back(Node(0));
        }
    }

    void clean()
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].clean();
        }
    }
    void clean(int i)
    {
        if(layerId == 0)
            return;
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].clean();
        }
    }

    void setValueFromVector(vector<double> values)
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].setValue(values[i]);
        }
    }

    void noActivate(int nodeId)
    {
        nodes[nodeId].noActivate();
    }
    void noActivateAll()
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].noActivate();
        }
    }
    void setActivateAll(int function)
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].setActivate(function);
        }
    }
    void setActivate(int function)
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].activationFunction = function;
        }
    }
    void setValueAll(double val)
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].setValue((double)val);
        }
    }
    void setValue(int nodeId, double val)
    {
        nodes[nodeId].setValue(val);
    }
    void setBiasAll(double w)
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].setBias(w);
        }
    }
    void setIdAll(int id)
    {

        layerId = id;
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].setId(i);
            nodes[i].layerId = layerId;
        }
    }
    
    void connect(Layer *nextL)
    {
        next = nextL;
        cout << "Connected layer " << layerId << " to layer " << next->layerId << endl;
        for (int i = 0; i < nodes.size(); i++)
        {
            for (int j = 0; j < next->nodes.size(); j++)
            {
                nodes[i].connect(&next->nodes[j]);
            }
        }
    }
    void disconnect(Layer *next)
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            for (int j = 0; j < next->nodes.size(); j++)
            {
                nodes[i].disconnect(&next->nodes[j]);
            }
        }
    }
    void passValuesOld()
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].passValueOld();

        }
    }

    void passValues()
    {
        for(int j = 0; j < next->nodes.size(); j++)
        {
            for (int i = 0; i < nodes.size(); i++)
            {
                nodes[i].passValueTo(&next->nodes[j]);
                cout << "Passed from n: " << nodes[i].getId() << " l: " << layerId << " to n: " << next->nodes[j].getId() << " l: " << next->layerId << endl;
                cout.flush();
                if(i == nodes.size() - 1)
                {
                    nodes[i].next[j].node->value += nodes[i].next[j].node->bias;
                    nodes[i].next[j].node->unactivatedValue = nodes[i].next[j].node->value;
                    nodes[i].next[j].node->activate();
                }
            }
        }
        
    }

    void printLayer()
    {
       //with cout
        en
        TextTable t('-', '|', '+');
        cout << "+-------+\n";
        cout << "|Layer " << layerId << "|\n";
        t.add("Node Id");
        t.add("Value");
        t.add("Bias");
        t.endOfRow();
        for (int i = 0; i < nodes.size(); i++)
        {
            t.add(to_string(nodes[i].getId()));
            t.add(to_string(nodes[i].getValue()));
            t.add(to_string(nodes[i].bias));
            t.endOfRow();
        }
        cout << t;
        
    }
    
    double weight(int nodeID, int nextNodeID)
    {
        return nodes[nodeID].next[nextNodeID].weight;
    }
    double weight(Node *node, Node *nextNode)
    {
        return node->next[nextNode->getId()].weight;
    }
    
    
};
