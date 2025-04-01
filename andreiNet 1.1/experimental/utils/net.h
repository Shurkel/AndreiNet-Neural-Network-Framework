#include "layer.h"
#include <iostream> // Add this include for std::cout
#include <fstream> // Add this include for std::ofstream
#include <cmath> // Add this explicitly for log function


std::ofstream tst("test.txt");
class net
{
public:
    vector<Layer> layers;
    vector<double> expected;
    vector<double> costs; 
    
    //sum of squared residuals
    double SSR = 0.0;
    //derivate von SSR

    // Cross-entropy loss calculation
    double crossEntropy = 0.0;

    net(){}
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
        }
        SSR*=0.5;
        
        
    }

    void clearCosts()
    {
        costs.clear();
    }
    
    void clearSSR()
    {
        SSR = 0.0;
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
    
    void setInputFromVector(vector<double> values)
    {
        layers[0].setValueFromVector(values);
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
            //cout << "Layer " << i << " with size " << layers[i].nodes.size() << " passed values\n";
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

    void testNet(pair<vector<pair<double,double>>,vector<double>> trainingData, bool brief)
    {
        clearSSR();
        
        //go trough all the training data and calculate the SSR
        for(int i = 0; i < trainingData.first.size(); i++)
        {
            
            clean();
            setInputFromVector({trainingData.first[i].first, trainingData.first[i].second});
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

    void backPropagate_old(pair<vector<double>,vector<double>> trainingData, int epochs, double learningRate)
    {
        int counter = 0;
        for(int i = 0; i < epochs; i++)
        {
            clearSSR();
            for(int j = 0; j < trainingData.first.size(); j++)
            {
                clean();
                setInputFromVector(trainingData.first);
                setExpected({trainingData.second[j]});
                passValues();
                getSSR();
                if(j == 0)
                    tst << SSR << endl;
                if (i % 500 == 0 && j == 3)
                {   
                    std::cout << "Epoch " << i << ", SSR for 4th element: " << SSR << std::endl;
                    printCosts();
                }
            
                double stepSize = 0.0;
                double dSSR_dn7 = (layers.back().nodes[0].value - expected[0]);
                
                double dn7_du7 = u.dsigmoid(layers.back().nodes[0].unactivatedValue);
                
                //layer 2 bias
                double dSSR_dBias = dSSR_dn7 * dn7_du7;
                stepSize = dSSR_dBias * learningRate;
                //layers.back().nodes[0].bias -= stepSize;

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

                    dSSR_dBias = dSSR_dn7 * dn7_du7 * weight(1, i, 2, 0);
                    stepSize = dSSR_dBias * learningRate;
                    //layers[1].nodes[i].bias -= stepSize;
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
    

    void backPropagate_new(pair<vector<pair<double, double>>, vector<double>> trainingData, int epochs, double learningRate)
{
    int counter = 0;

    for (int e = 0; e < epochs; e++)
    {
        clearSSR();
        
        for (int t = 0; t < trainingData.first.size(); t++)
        {
            clean();
            setInputFromVector({trainingData.first[t].first, trainingData.first[t].second});
            setExpected({trainingData.second[t]});
            passValues();

            getSSR();
            if(t == 0 && e != 0)
                tst << SSR << endl;
            // Print SSR value every 500th epoch for the 4th element in the training data
            if (e % 50 == 0 && t == 0)
            {
                
                std::cout << "Epoch " << e << ", SSR for 4th element: " << SSR << std::endl;
                printCosts();
            }
            

            // Output layer delta calculation
            for (int i = 0; i < layers.back().nodes.size(); i++)
            {
                double error = layers.back().nodes[i].value - expected[i]; //SSR
                double activationDerivative = layers.back().nodes[i].activate(layers.back().nodes[i].unactivatedValue, true);
                layers.back().nodes[i].delta = error * activationDerivative;
            }

            // Hidden layers delta calculation (backpropagation)
            for (int k = layers.size() - 2; k >= 0; k--) // Iterate from last hidden layer to first
            {
                for (int j = 0; j < layers[k].nodes.size(); j++) // Iterate over nodes in layer k
                {
                    double sum = 0.0;
                    for (int i = 0; i < layers[k + 1].nodes.size(); i++) // Iterate over nodes in the next layer (k+1)
                    {
                        sum += layers[k + 1].nodes[i].delta * weight(k, j, k + 1, i);
                    }

                    // Apply the derivative of the activation function for the current node in layer k
                    layers[k].nodes[j].delta = sum * layers[k].nodes[j].activate(layers[k].nodes[j].unactivatedValue, true);
                }
            }

            // Update weights and biases from hidden layer to output layer
            for (int j = 0; j < layers[layers.size() - 2].nodes.size(); j++) // For each node in second to last layer
            {
                for (int i = 0; i < layers.back().nodes.size(); i++) // For each node in the output layer
                {
                    double delta = layers.back().nodes[i].delta;
                    double value = layers[layers.size() - 2].nodes[j].value;
                    double stepSize = delta * value * learningRate;

                    // Update the weight between the current node and the output node
                    layers[layers.size() - 2].nodes[j].next[i].weight += stepSize;

                    // Update the bias for the output node
                    layers.back().nodes[i].bias -= delta * learningRate;
                }
            }

            // Update weights and biases between hidden layers
            for (int k = layers.size() - 2; k > 0; k--) // Iterate from second-to-last hidden layer back to first hidden layer
            {
                for (int j = 0; j < layers[k - 1].nodes.size(); j++) // Iterate over nodes in the previous layer (k-1)
                {
                    for (int i = 0; i < layers[k].nodes.size(); i++) // Iterate over nodes in the current layer (k)
                    {
                        double delta = layers[k].nodes[i].delta;
                        double value = layers[k - 1].nodes[j].value;
                        double stepSize = delta * value * learningRate;

                        // Update the weight between the nodes from layer k-1 to layer k
                        layers[k - 1].nodes[j].next[i].weight -= stepSize;

                        // Update the bias for the current node in layer k
                        layers[k].nodes[i].bias -= delta * learningRate;
                    }
                }
            }
        }
    }
}

    void getCrossEntropy()
    {
        crossEntropy = 0.0;
        if (expected.empty())
        {
            cerr << (char)218 << "[x] Error: Expected vector is empty.\n"
            << (char)192 << "Please use setExpected() before calling getCrossEntropy().\n";
            return;
        }
        
        for(int i = 0; i < layers.back().nodes.size(); i++)
        {
            // Clip predictions to avoid log(0) and log(1)
            double pred = std::max(1e-15, std::min(1.0 - 1e-15, layers.back().nodes[i].value));
            
            // Keep the original cross-entropy formula: -Sum(expected[i] * log(pred))
            crossEntropy -= expected[i] * std::log(pred);
            
            // Debug output (uncomment if needed)
            // std::cout << "Example: " << i << ", Expected: " << expected[i] << ", Pred: " << pred 
            //          << ", CE contribution: " << (expected[i] * std::log(pred)) << std::endl;
        }
    }
    
    void clearCrossEntropy()
    {
        crossEntropy = 0.0;
    }
    
    void printCrossEntropy()
    {
        cout << MAGENTA << "\n    [+]Cross-Entropy: " << crossEntropy << RESET;
        cout.flush();
    }

    void backPropagate_crossentropy(pair<vector<pair<double, double>>, vector<double>> trainingData, int epochs, double learningRate)
    {
        ofstream tst_ce("test_ce.txt");
        
        for (int e = 0; e < epochs; e++)
        {
            clearCrossEntropy();
            
            for (int t = 0; t < trainingData.first.size(); t++)
            {
                clean();
                setInputFromVector({trainingData.first[t].first, trainingData.first[t].second});
                setExpected({trainingData.second[t]});
                passValues();

                getCrossEntropy();
                // Log values for all examples, not just t=51
                if(e != 0)  // Log every 10th example
                {
                    cout << "Epoch " << e << ", Example " << t << ", Cross-Entropy: " << crossEntropy << std::endl;
                    tst_ce << crossEntropy << '\n';
                }
                
                // Print cross-entropy value periodically
                if (e % 50 == 0 && t == trainingData.first.size()-1)
                {
                    std::cout << "Epoch " << e << ", Cross-Entropy: " << crossEntropy << std::endl;
                    printCosts();
                }
                
                // Output layer delta calculation for cross-entropy
                for (int i = 0; i < layers.back().nodes.size(); i++)
                {
                    // For binary cross-entropy, the derivative is (p-y)/(p*(1-p))
                    // But since sigmoid's derivative is p*(1-p), these terms cancel out
                    // So delta becomes simply (prediction - target)
                    layers.back().nodes[i].delta = layers.back().nodes[i].value - expected[i];
                }

                // Hidden layers delta calculation (backpropagation) - same as before
                for (int k = layers.size() - 2; k >= 0; k--)
                {
                    for (int j = 0; j < layers[k].nodes.size(); j++)
                    {
                        double sum = 0.0;
                        for (int i = 0; i < layers[k + 1].nodes.size(); i++)
                        {
                            sum += layers[k + 1].nodes[i].delta * weight(k, j, k + 1, i);
                        }
                        layers[k].nodes[j].delta = sum * layers[k].nodes[j].activate(layers[k].nodes[j].unactivatedValue, true);
                    }
                }

                // Update weights and biases - negative gradients for weight updates now
                // For output to hidden layer
                for (int j = 0; j < layers[layers.size() - 2].nodes.size(); j++)
                {
                    for (int i = 0; i < layers.back().nodes.size(); i++)
                    {
                        double delta = layers.back().nodes[i].delta;
                        double value = layers[layers.size() - 2].nodes[j].value;
                        double stepSize = delta * value * learningRate;

                        // Update the weight (negative gradient descent)
                        layers[layers.size() - 2].nodes[j].next[i].weight -= stepSize;
                        
                        // Update the bias
                        layers.back().nodes[i].bias -= delta * learningRate;
                    }
                }

                // For hidden layers
                for (int k = layers.size() - 2; k > 0; k--)
                {
                    for (int j = 0; j < layers[k - 1].nodes.size(); j++)
                    {
                        for (int i = 0; i < layers[k].nodes.size(); i++)
                        {
                            double delta = layers[k].nodes[i].delta;
                            double value = layers[k - 1].nodes[j].value;
                            double stepSize = delta * value * learningRate;

                            // Update weights
                            layers[k - 1].nodes[j].next[i].weight -= stepSize;
                            
                            // Update biases
                            layers[k].nodes[i].bias -= delta * learningRate;
                        }
                    }
                }
            }
        }
        tst_ce.close();
    }

};
