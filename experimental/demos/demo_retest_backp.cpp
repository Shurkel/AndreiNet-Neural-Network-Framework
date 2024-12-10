#include "../utils/andreinet.h"
#include <fstream>

net n({3, 6, 4, 2});

void saveWeights()
{
    ofstream wo("save.txt");
    if (!wo.is_open())
    {
        cerr << "Error opening save file!" << endl;
        throw runtime_error("Failed to open save file.");
    }

    try
    {
        // Write weights
        for (int i = 0; i < n.layers.size(); i++)
        {
            for (int j = 0; j < n.layers[i].nodes.size(); j++)
            {
                for (int k = 0; k < n.layers[i].nodes[j].next.size(); k++)
                {
                    wo << n.layers[i].nodes[j].next[k].weight << " ";
                }
                wo << "\n";
            }
        }
        // Write biases
        for (int i = 0; i < n.layers.size(); i++)
        {
            for (int j = 0; j < n.layers[i].nodes.size(); j++)
            {
                wo << n.layers[i].nodes[j].bias << " ";
            }
            wo << "\n";
        }
    }
    catch (const ios_base::failure &e)
    {
        cerr << "Error writing to save file: " << e.what() << endl;
        wo.close();
        throw;
    }

    wo.close();
    if (wo.fail())
    {
        cerr << "Error closing the save file!" << endl;
        throw runtime_error("Failed to close save file properly.");
    }
}

void importWeights()
{
    ifstream wi("save.txt");
    if (!wi.is_open())
    {
        cout << "Error opening save file!" << endl;
        return;
    }

    for (int i = 0; i < n.layers.size(); i++)
    {
        for (int j = 0; j < n.layers[i].nodes.size(); j++)
        {
            for (int k = 0; k < n.layers[i].nodes[j].next.size(); k++)
            {
                if (!(wi >> n.layers[i].nodes[j].next[k].weight))
                {
                    cout << "Error reading weight for layer " << i << ", node " << j << ", weight " << k << endl;
                    wi.close();
                    return;
                }
            }
        }
    }
    // biases
    for (int i = 0; i < n.layers.size(); i++)
    {
        for (int j = 0; j < n.layers[i].nodes.size(); j++)
        {
            if (!(wi >> n.layers[i].nodes[j].bias))
            {
                cout << "Error reading bias for layer " << i << ", node " << j << endl;
                wi.close();
                return;
            }
        }
    }
    if (wi.fail() && !wi.eof())
    {
        cout << "Error reading from save file!" << endl;
    }
    wi.close();
}



void runCustomTraining() {
    vector<pair<vector<double>, vector<double>>> trainingData = {
        {{1.0, 2.0, 3.0}, {1, 0}},
        {{3.0, 1.5, 2.5}, {1, 0}},
        {{2.0, 1.0, 2.0}, {0, 1}},
        {{4.0, 3.0, 1.0}, {1, 0}},
        {{0.5, 1.5, 3.5}, {0, 1}},
        {{3.5, 2.5, 0.5}, {1, 0}},
        {{1.0, 3.0, 1.0}, {0, 1}},
        {{2.5, 2.0, 3.0}, {1, 0}},
        {{3.5, 0.5, 2.0}, {0, 1}},
        {{2.0, 2.0, 2.0}, {1, 0}}};
    
    int epochs = 50000;
    double learningRate = 0.03;

    cout << "\n[+] Training with new setup";
    n.setActivate(1, 1); // Sigmoid activation for hidden layer 1
    n.setActivate(2, 1); // Sigmoid activation for hidden layer 2
    n.setActivate(3, 2); // Softmax activation for output layer

    chrono::time_point<chrono::high_resolution_clock> startTime, endTime;
    chrono::duration<float> duration;
    startTime = chrono::high_resolution_clock::now();

    n.backPropagate_new(trainingData, epochs, learningRate);

    endTime = chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    cout << "Time taken: " << duration.count() << " seconds\n";

    cout << "[+] Final Testing Results:\n";
    n.testNet(trainingData, true); // Test with verbose output
}