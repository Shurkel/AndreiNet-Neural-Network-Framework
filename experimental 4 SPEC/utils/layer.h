#include "node.h"

class Layer
{
public:
    vector<Node> nodes;
    int layerId = 0;
    Layer *next;


    // * CONSTRUCTORS
    Layer () {}
    Layer(int n)
    {
        for (int i = 0; i < n; i++)
            nodes.push_back(Node(i, layerId));
    }
    
    // * GETTERS AND SETTERS

    //VALUE
    vector<double> getValues()
    {
        vector<double> values;
        values.reserve(nodes.size());
        for (const auto& node : nodes)
            values.push_back(node.getValue());
        return values;
    }
    void setValues(vector<double> values)
    {
        for (int i = 0; i < nodes.size(); i++)
            nodes[i].setValue(values[i]);
    }

    //UNACTIVATED VALUE
    vector<double> getUnactivatedValues()
    {
        vector<double> values;
        values.reserve(nodes.size());
        for (auto& node : nodes)
            values.push_back(node.getUnactivatedValue());
        return values;
    }
    void setUnactivatedValues(vector<double> values)
    {
        for (int i = 0; i < nodes.size(); i++)
            nodes[i].setUnactivatedValue(values[i]);
    }

    //BIAS
    void setBias(double b)
    {//set bias for all nodes
        for (auto &node : nodes)
            node.setBias(b);
    }

    //DELTA
    double getDelta(int nodeId) { return nodes[nodeId].getDelta(); }
    void getDelta(Node *node) { node->getDelta(); }
    void setDelta(double d)
    {//set delta for all nodes
        for (auto &node : nodes)
            node.setDelta(d);
    }

    //ID
    int getId() { return layerId; }
    void setId(int id)
    {
        layerId = id;
        for (int i = 0; i < nodes.size(); i++)
            nodes[i].setId(i);

    }

    //ACTIVATION FUNCTION
    void setActivationFunction(int function)
    {
        for (auto &node : nodes)
            node.setActivationFunction(function);
    }
    void noActivateAll()
    {
        for (auto &node : nodes)
            node.noActivate();
    }
    //set all weights to w
    void setWeight(double w)
    {
        for (auto &node : nodes)
            node.setWeight(w);
    }

    //NEXT LAYER
    void setNextLayer(Layer *nextLayer) { next = nextLayer; }
    

    // * FUNCTIONS
    
    void clean()
    {
        for (auto &node : nodes)
            node.clean();
    }
    
    void connect(Layer *nextL)
    {
        setNextLayer(nextL);
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
        setNextLayer(nullptr);
        for (int i = 0; i < nodes.size(); i++)
        {
            for (int j = 0; j < next->nodes.size(); j++)
            {
                nodes[i].disconnect(&next->nodes[j]);
            }
        }
    }
    void softMaxLayer() { setValues(u.softMax(getValues())); }

    void passValues()
    {
        for(int j = 0; j < next->nodes.size(); j++)
            for (int i = 0; i < nodes.size(); i++)
            {
                nodes[i].passValueTo(&next->nodes[j]);
                if(i == nodes.size() - 1)
                {
                    nodes[i].next[j].node->setValue(
                        nodes[i].next[j].node->getValue() + nodes[i].next[j].node->getBias()
                    );
                    
                    nodes[i].next[j].node->setUnactivatedValue(nodes[i].next[j].node->getValue());

                    
                    nodes[i].next[j].node->setValue(
                        nodes[i].next[j].node->activateNode(nodes[i].next[j].node->getValue())
                    );
                }
            }
    }

    void printLayer()
    {
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
            t.add(to_string(nodes[i].getBias()));
            t.endOfRow();
        }
        cout << t;
        
    }
};
