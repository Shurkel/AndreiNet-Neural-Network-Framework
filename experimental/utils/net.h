#include "layer.h"

class net
{
public:
    vector<Layer> layers;
    vector<double> expected;
    vector<double> costs; 
    
    //sum of squared residuals
    double SSR = 0.0;
    //derivate von SSR
    double dSSR = 0.0;


    net(vector<int> layerSizes)
    {
        
        
        
        for (int i = 0; i < layerSizes.size(); i++)
        {
            layers.push_back(Layer(layerSizes[i])); // Create the layer first
            layers[i].setIdAll(i);   
        }
        connectLayers();
        clearCosts();
    }
    void clean()
    {
        for(int i = 0; i < layers.size(); i++)
        {
            layers[i].clean();
        }
    }

    void clean(int i)
    {
        for(int i = 0; i < layers.size(); i++)
        {
            if(layers[i].layerId == 0)
                continue;
            layers[i].clean();
        }
    }
    
    double weight(int layerId, int nodeId, int nextLayerId, int nextNodeId)
    {
        return layers[layerId].nodes[nodeId].weight(&layers[nextLayerId].nodes[nextNodeId]);
    }
    
    double weight(Node *node, Node *nextNode)
    {
        return node->weight(nextNode);
    }

    void setExpected(vector<double> expectedValues)
    {
        expected.clear();
        for (int i = 0; i < layers.back().nodes.size(); i++)
        {
            expected.push_back(expectedValues[i]);
        }
    }
    
    void getCosts()
    {
        costs.clear();
        if (expected.empty())
        {
            cerr << (char)218 << "[x] Error: Costs vector is empty.\n"
            << (char)192 << "Please use setExpected() before calling getCosts().\n";
            return;
        }
        for(int i = 0; i < layers.back().nodes.size(); i++)
        {
            costs.push_back(layers.back().nodes[i].value - expected[i]);
        }
        
    }
    
    void getSSR()
    {
        clearCosts();
        getCosts();
        for (int i = 0; i < costs.size(); i++)
        {
            SSR += pow(costs[i], 2);
            dSSR += -2 * costs[i];
        }
        
    }

    void clearCosts()
    {
        costs.clear();
    }
    
    void clearSSR()
    {
        SSR = 0.0;
        dSSR = 0.0;
    }
    
    void printInput()
    {
        cout << "\n[+]Input: ";
        for (int i = 0; i < layers[0].nodes.size(); i++)
        {
            cout << layers[0].nodes[i].value << " ";
        }
        cout.flush();
    }

    void printCosts()
    {
        cout << MAGENTA << "\n    [+]Costs: ";
        for (int i = 0; i < costs.size(); i++)
        {
            cout << costs[i] << " ";
        }
        cout << RESET;
        cout.flush();
    }
    
    void printExpected()
    {
        cout << "\n[+]Expected: ";
        for (int i = 0; i < expected.size(); i++)
        {
            cout << expected[i] << " ";
        }
        cout.flush();
    }

    void printActualOutput()
    {
        cout << "\n[+]Actual output: ";
        for (int i = 0; i < layers.back().nodes.size(); i++)
        {
            cout << layers.back().nodes[i].value << " ";
        }
        cout.flush();
    }
    
    void setValueAll( double val)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            layers[i].setValueAll(val);
        }
    }
    
    void setValue(int layerId, int nodeId, double val)
    {
        layers[layerId].setValue(nodeId, val);
    }
    
    void setInputFromVector(pair<double, double> values)
    {
        layers[0].setValueFromVector({values.first, values.second});
    }
    
    void setBiasAll(int layerId, double w)
    {
        layers[layerId].setBiasAll(w);
    }

    void setIdAll(int layerId, int id)
    {
        layers[layerId].setIdAll(id);
    }

    void setActivateAll(int function)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            layers[i].setActivateAll(function);
        }
    }

    void noActivate(int layerId, int nodeId)
    {
        layers[layerId].noActivate(nodeId);
    }
    void noActivate(int layerId)
    {
        for(int i = 0; i < layers[layerId].nodes.size(); i++)
        {
            layers[layerId].noActivate(i);
        }
    }

    void noActivateAll()
    {
        for (int i = 0; i < layers.size(); i++)
        {
            layers[i].noActivateAll();
        }
    }

    void setActivate(int layerId, int function)
    {
        layers[layerId].setActivate(function);
    }

    void printLayer(int id)
    {
        layers[id].printLayer();
    }

    void printNet()
    {
        cout << (char)218 << "Layer count: " << layers.size() << '\n';
        cout << (char)195 << "SSR: " << SSR << '\n';
        cout << (char)195 << "Layer | Size\n";

        cout << char(195);
        cout << "  " << layers[0].layerId << "   |    " << layers[0].nodes.size() << "\n";
        
        for(int i = 1; i < layers.size()-1; i++)
        {
            cout << char(195) << "  ";
            cout << layers[i].layerId << "   |    " << layers[i].nodes.size() << "\n";
        }

        cout << char(192);
        cout << "  " << layers.back().layerId << "   |    " << layers.back().nodes.size();
        cout.flush();
        
    }
    
    void printLayers()
    {
        for (int i = 0; i < layers.size(); i++)
        {
            layers[i].printLayer();
        }
    }

    void setWeight(int layerID, int nodeID, int nextLayerID, int nextNodeID, double w)
    {
        layers[layerID].nodes[nodeID].setWeight(nextNodeID, nextLayerID, w);
    }

    void setBias(int layerID, int nodeID, double b)
    {
        layers[layerID].nodes[nodeID].setBias(b);
    }   
    
    void printNextNodes(int layerId, int nodeId)
    {
        layers[layerId].nodes[nodeId].printNextNodes();
    }

    void connectLayers()
    {
        for (int i = 0; i < layers.size() - 1; i++)
        {
            layers[i].connect(&layers[i + 1]);
            layers[i].next = &layers[i + 1];
        }
    }
    void disconnectLayers()
    {
        for (int i = 0; i < layers.size() - 1; i++)
        {
            layers[i].disconnect(&layers[i + 1]);
        }
    }
    void passValuesOld()
    {
        for (int i = 0; i < layers.size(); i++)
        {
            layers[i].passValuesOld();
            //cout layer with size passed values
            cout << "Layer " << i << " with size " << layers[i].nodes.size() << " passed values\n";
        }
            

    }
    void passValues()
    {
        for (int i = 0; i < layers.size()-1; i++)
        {
            layers[i].passValues();
            //cout layer with size passed values
            cout << "Layer " << i << " with size " << layers[i].nodes.size() << " passed values\n";
        }
            

    }

    void setWeightAll(double w)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            for (int j = 0; j < layers[i].nodes.size(); j++)
            {
                layers[i].nodes[j].setWeightAll(w);
            }
        }
    }

    void setBiasAll(double b)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            for (int j = 0; j < layers[i].nodes.size(); j++)
            {
                layers[i].nodes[j].setBias(b);
            }
        }
    }

    void ranomiseAllWeights()
    {
        // ! not working
        for (int i = 0; i < layers.size(); i++)
        {
            for (int j = 0; j < layers[i].nodes.size(); j++)
            {
                layers[i].nodes[j].randomiseWeights();
            }
        }
    }
    
    void printNodeDetails(int nodeId, int layerId)
    {
        layers[layerId].nodes[nodeId].printDetails();
    }

    void printSSR()
    {
        cout << MAGENTA << "\n    [+]SSR: " << SSR << RESET;
        cout.flush();}

    void testNet(pair<vector< pair<double, double>>,vector<double>> trainingData, bool brief)
    {
        clearSSR();
        
        //go trough all the training data and calculate the SSR
        for(int i = 0; i < trainingData.first.size(); i++)
        {
            
            clean();
            setInputFromVector(trainingData.first[i]);
            setExpected({trainingData.second[i]});
            passValues();
            getSSR();
            if(!brief)
            {
                en
                tab 
                cout << "//// ";
                tab
                printInput();
                tab
                printExpected();
                tab
                printActualOutput();
                tab
                printSSR();
            }
            
        }
        if(brief)
        {
            tab
            printSSR();
        }
        /* //timer the network
        chrono::time_point<chrono::high_resolution_clock> startTime, endTime;
        chrono::duration<float> duration;
        startTime = chrono::high_resolution_clock::now();
        for(int j = 0; j < 10000;j++)
        {
            for(int i = 0; i < trainingData.first.size(); i++)
            {
            
            clean();
            setInputFromVector(trainingData.first[i]);
            setExpected({trainingData.second[i]});
            passValues();
            getSSR();         
            }
        }
        
        endTime = chrono::high_resolution_clock::now();
        duration = endTime - startTime;
        cout << "Time: " << duration.count() << "s" << endl; */
        clean();
        clearSSR();
    }


    void useDefaults()
    {
        noActivate(0);    
        setActivate(1, 1); 
        setActivate(2, 1); 
        connectLayers();  
        setWeightAll(1); 
        setBiasAll(0);  
    }

    void backPropagate_old(pair<vector< pair<double, double>>,vector<double>> trainingData, int epochs, double learningRate)
    {
        
        for(int i = 0; i < epochs; i++)
        {
            clearSSR();
            for(int j = 0; j < trainingData.first.size(); j++)
            {
                clean();
                setInputFromVector(trainingData.first[j]);
                setExpected({trainingData.second[j]});
                passValues();
                getSSR();
                double stepSize = 0.0;
                double dSSR_dn7 = 2*(layers.back().nodes[0].value - expected[0]);
                double dn7_du7 = u.dsigmoid(layers.back().nodes[0].unactivatedValue);
                //layer 2 bias
                double dSSR_dBias = dSSR_dn7 * dn7_du7;
                stepSize = dSSR_dBias * learningRate;
                layers.back().nodes[0].bias -= stepSize;

                //layer 1 -> layer 2 weights
                double dSSR_dWXY[4];//Y = 7, X = (3 4 5 6)
                double du7_dWX7[4];
                
                for(int i = 0; i < 4; i++)
                {
                    du7_dWX7[i] = layers[1].nodes[i].value;
                    dSSR_dWXY[i] = dSSR_dn7 * dn7_du7 * du7_dWX7[i];
                    stepSize = dSSR_dWXY[i] * learningRate;
                    layers[1].nodes[i].next[0].weight -= stepSize;
                }

                //layer 1 bias
                for(int i = 0 ; i < 4; i++)
                {

                    dSSR_dBias = dSSR_dn7 * dn7_du7 * weight(1, i, 2, 0) * u.dsigmoid(layers[1].nodes[i].unactivatedValue);
                    stepSize = dSSR_dBias * learningRate;
                    layers[1].nodes[i].bias -= stepSize;
                }
                //layer 0 -> layer 1 weights
                //X = (1 2), Y = (3 4 5 6) dSSR/dWXY     
                for(int i = 0; i < 4; i++)
                {
                    dSSR_dWXY[i] = dSSR_dn7 * dn7_du7 * weight(1, i, 2, 0) * u.dsigmoid(layers[1].nodes[i].unactivatedValue) * layers[0].nodes[1].value;
                    stepSize = dSSR_dWXY[i] * learningRate;
                    layers[0].nodes[1].next[i].weight -= stepSize;
                }
                for(int i = 0; i < 4; i++)
                {
                    //x = 1
                    dSSR_dWXY[i] = dSSR_dn7 * dn7_du7 * weight(1, i, 2, 0) * u.dsigmoid(layers[1].nodes[i].unactivatedValue) * layers[0].nodes[0].value;
                    stepSize = dSSR_dWXY[i] * learningRate;
                    layers[0].nodes[0].next[i].weight -= stepSize;
                }
                
                

            }

        }

        


        
    }
    

    void backPropagate_new(pair<vector< pair<double, double>>,vector<double>> trainingData, int epochs, double learningRate)
    {
        //? STILL IN PROGRESS
        
        for(int i = 0; i < epochs; i++)
        {
            clearSSR();
            for(int j = 0; j < trainingData.first.size(); j++)
            {
                clean();
                setInputFromVector(trainingData.first[j]);
                setExpected({trainingData.second[j]});
                passValues();
                getSSR();
                double stepSize = 0.0;

                //calculate delta for each node
                    //step 1 last layer
                for(int i = 0; i < layers.back().nodes.size(); i++)
                {
                    //layers.back().nodes[i].delta = 
                }



                double dSSR_dn7 = 2*(layers.back().nodes[0].value - expected[0]);
                double dn7_du7 = u.dsigmoid(layers.back().nodes[0].unactivatedValue);
                //layer 2 bias
                double dSSR_dBias = dSSR_dn7 * dn7_du7;
                //!stepSize = dSSR_dBias * learningRate;
                layers.back().nodes[0].bias -= stepSize;

                //layer 1 -> layer 2 weights
                double dSSR_dWXY[4];//Y = 7, X = (3 4 5 6)
                double du7_dWX7[4];
                
                for(int i = 0; i < 4; i++)
                {
                    du7_dWX7[i] = layers[1].nodes[i].value;
                    dSSR_dWXY[i] = dSSR_dn7 * dn7_du7 * du7_dWX7[i];
                    stepSize = dSSR_dWXY[i] * learningRate;
                    layers[1].nodes[i].next[0].weight -= stepSize;
                }

                //layer 1 bias
                for(int i = 0 ; i < 4; i++)
                {

                    dSSR_dBias = dSSR_dn7 * dn7_du7 * weight(1, i, 2, 0) * u.dsigmoid(layers[1].nodes[i].unactivatedValue);
                    stepSize = dSSR_dBias * learningRate;
                    layers[1].nodes[i].bias -= stepSize;
                }
                //layer 0 -> layer 1 weights
                //X = (1 2), Y = (3 4 5 6) dSSR/dWXY     
                for(int i = 0; i < 4; i++)
                {
                    dSSR_dWXY[i] = dSSR_dn7 * dn7_du7 * weight(1, i, 2, 0) * u.dsigmoid(layers[1].nodes[i].unactivatedValue) * layers[0].nodes[1].value;
                    stepSize = dSSR_dWXY[i] * learningRate;
                    layers[0].nodes[1].next[i].weight -= stepSize;
                }
                for(int i = 0; i < 4; i++)
                {
                    //x = 1
                    dSSR_dWXY[i] = dSSR_dn7 * dn7_du7 * weight(1, i, 2, 0) * u.dsigmoid(layers[1].nodes[i].unactivatedValue) * layers[0].nodes[0].value;
                    stepSize = dSSR_dWXY[i] * learningRate;
                    layers[0].nodes[0].next[i].weight -= stepSize;
                }
                
                

            }

        }

        


        
    }


};
