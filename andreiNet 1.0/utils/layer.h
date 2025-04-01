#include "node.h"

class Layer
{
public:
    vector<Node> nodes;
    bool yesActivate = true;
    int layerId = 0;

    Layer(int n)
    {
        
        for (int i = 0; i < n; i++)
        {
            nodes.push_back(Node(0));
            nodes[i].yesActivate = yesActivate;
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
            //cout << "Node " << nodes[i].getId() << " set to " << nodes[i].getValue() << "\n";
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
    
    void connect(Layer *next)
    {
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
    void passValues()
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            nodes[i].passValues();

        }
    }
    void logLayer()
    {
        len
        TextTable t('-', '|', '+');
        l << "+-------+\n";
        l << "|Layer " << layerId << "|\n";
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
        l << t;
        
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
