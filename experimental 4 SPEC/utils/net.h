#include "layer.h"
#include "data.h"

// ! fix
ofstream tst("test.txt");
class net
{
public:
    vector<Layer> layers;
    
    dataSet trainingData();
    vector<double> costs;
    double SSR = 0.0;

    net(){}
    net(vector<int> layerSizes)
    {
        for (auto i : layerSizes)
            layers.push_back(Layer(layerSizes[i], i)); // Create the layer first 
        connectLayers();
        clearCosts();
    }
    
    
    
    void clean(int i = -1)
    {   
        if(i == -1)
            for(int j = 0; j < layers.size(); j++)
                layers[j].clean();
        else
            for(int j = 0; j < layers[i].nodes.size() && j != i; j++)
                layers[j].clean();
    }
    
    

    void setExpected(vector<double> expectedValues) { trainingData().setExpected(expectedValues); }
    
    void updateCosts()
    {
        costs.clear();
        if (trainingData().getExpected().size() == 0)
        {
            cerr << (char)218 << "[x] Error: Costs vector is empty.\n"
            << (char)192 << "Please use setExpected() before calling getCosts().\n";
            return;
        }
        for(int i = 0; i < layers.back().nodes.size(); i++)
            costs.push_back(layers.back().nodes[i].getValue() - trainingData().getExpected()[i]);
    }

    vector<double> getCosts() { return costs; }

    void setSSR(double ssr) { SSR = ssr; }

    double getSSR() { return SSR; }

    void updateSSR()
    {
        clearCosts();
        updateCosts();
        
        for (int i = 0; i < costs.size(); i++)
            setSSR(getSSR() + pow(getCosts()[i], 2));
        SSR *= 0.5;
    }

    void clearCosts() { costs.clear(); }
    
    void resetCosts() { clearCosts(); }
    void resetSSR() { setSSR(0.0); }
    

    void printCosts()
    {
        cout  << "[+]Costs: ";
        for (int i = 0; i < costs.size(); i++)
            cout << costs[i] << " ";
        en
    }
    


    void printActualOutput()
    {
        cout << "[+]Actual output: ";
        for (int i = 0; i < layers.back().nodes.size(); i++)
            cout << layers.back().nodes[i].getValue() << " ";
        en
    }
    
    void setInputData(vector<double> data)
    {
        trainingData().setInputs(data);
    }

    void setInput(int i)
    {
        if(trainingData().getInputs().size() == 0)
        {
            cerr << (char)218 << "[x] Error: Inputs vector is empty.\n"
            << (char)192 << "Please use setInputData() before calling setInput().\n";
            return;
        }
        layers[0].setValues(trainingData().getInput(i));
    }
    
    void setBias(double b)
    {
        for(int i = 0; i < layers.size(); i++)
            layers[i].setBias(b);
    }

    void setActivationFunction(int function)
    {
        for (int i = 0; i < layers.size(); i++)
            layers[i].setActivationFunction(function);
    }


    void noActivate()
    {
        for (int i = 0; i < layers.size(); i++)
            layers[i].noActivateAll();
    }

    void setActivate(int layerId, int function)
    {
        layers[layerId].setActivationFunction(function);
    }

    void reset(int i = -1)
    {
        clean(i);
        resetCosts();
        resetSSR();
    }

    void print()
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
        flush
        
    }
    
    void printLayers()
    {
        for (int i = 0; i < layers.size(); i++)
            layers[i].printLayer();
    } 
    
    void printNextNodes(int layerId, int nodeId)
    {
        layers[layerId].nodes[nodeId].printNextNodes();
    }

    void connectLayers()
    {
        for (int i = 0; i < layers.size() - 1; i++)
            layers[i].connect(&layers[i + 1]);
    }
    void disconnectLayers()
    {
        for (int i = 0; i < layers.size() - 1; i++)
            layers[i].disconnect(&layers[i + 1]);
    }

    void passValues()
    {
        for (int i = 0; i < layers.size()-1; i++)
            layers[i].passValues();
    }

    void setWeight(double w)
    {
        for (int i = 0; i < layers.size(); i++)
                layers[i].setWeight(w);
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
        layers[layerId].nodes[nodeId].print();
    }

    void printSSR()
    {
        cout << "[+]SSR: " << SSR;
        en flush
    }

    void testNet()
    {
        for(int i = 0; i < trainingData().getInputs().size(); i++)
        {
            reset();
            setInput(trainingData.first);
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
    

    void backPropagate_new(pair<vector<double>, vector<double>> trainingData, int epochs, double learningRate)
{
    int counter = 0;

    for (int e = 0; e < epochs; e++)
    {
        clearSSR();
        
        for (int t = 0; t < trainingData.first.size(); t++)
        {
            clean();
            setInputFromVector(trainingData.first);
            setExpected({trainingData.second[t]});
            passValues();

            getSSR();
            if(t == 0)
                tst << SSR << endl;
            // Print SSR value every 500th epoch for the 4th element in the training data
            if (e % 500 == 0 && t == 3)
            {
                std::cout << "Epoch " << e << ", SSR for 4th element: " << SSR << std::endl;
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

            // Update weights from hidden layer to output layer
            for (int j = 0; j < layers[layers.size() - 2].nodes.size(); j++) // For each node in second to last layer
            {
                for (int i = 0; i < layers.back().nodes.size(); i++) // For each node in the output layer
                {
                    double delta = layers.back().nodes[i].delta;
                    double value = layers[layers.size() - 2].nodes[j].value;
                    double stepSize = delta * value * learningRate;

                    // Update the weight between the current node and the output node
                    layers[layers.size() - 2].nodes[j].next[i].weight -= stepSize;
                }
            }

            // Update weights between hidden layers
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
                    }
                }
            }
        }
    }
}



};
