#include "../utils/andreinet.h"
#include <fstream>

net n({2, 4, 1});

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

pair<vector<pair<double, double>>, vector<double>> generateTrainingData(int treshold = 10, int maxn = 10, int samples = 50){
    vector<pair<double, double>> inputData;
    vector<double> expectedOutput;

    // Seed random number generator
    srand(time(0));

    // Generate 50 samples
    for (int i = 0; i < samples; ++i) {
        // Random input values between 0 and 10
        double x1 = rand() % maxn;
        double x2 = rand() % maxn;

        // Simple rule: sum of inputs > 10, output 1; otherwise, output 0
        double output = (x1 + x2 > treshold) ? 1.0 : 0.0;

        // Store the generated input and expected output
        inputData.push_back({x1, x2});
        expectedOutput.push_back(output);
    }

    // Return the training data as a pair
    return {inputData, expectedOutput};
}

void saveTrainingData(int treshold = 10, int maxn=10,  int samples = 50)
{
    ofstream wo("trainingData.txt");
    if (!wo.is_open())
    {
        cerr << "Error opening save file!" << endl;
        throw runtime_error("Failed to open save file.");
    }

    try
    {
        pair<vector<pair<double, double>>, vector<double>> trainingData = generateTrainingData(treshold, maxn, samples);
        // Write training data
        for (int i = 0; i < trainingData.first.size(); i++)
        {
            wo << trainingData.first[i].first << " " << trainingData.first[i].second << " " << trainingData.second[i] << endl;
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

pair<vector<pair<double, double>>, vector<double>> readTrainingData()
{
    ifstream wi("trainingData.txt");
    if (!wi.is_open())
    {
        cout << "Error opening save file!" << endl;
        return {{}, {}};
    }

    vector<pair<double, double>> inputData;
    vector<double> expectedOutput;

    double x1, x2, output;
    while (wi >> x1 >> x2 >> output)
    {
        inputData.push_back({x1, x2});
        expectedOutput.push_back(output);
    }

    if (wi.fail() && !wi.eof())
    {
        cout << "Error reading from save file!" << endl;
    }
    wi.close();

    return {inputData, expectedOutput};
}

int main()
{

    /*
    Aktivierungsfunktionen:
    0 - ReLU
    1 - Sigmoid
    2 - Softmax
    */
    t.start();



    n.setActivateAll(1);

    n.noActivate(0);

    n.setWeightAll(1);
    n.setBiasAll(0);


    vector<pair<double, double>> input = {
    // Input data for output 0
    {1, 1}, {2, 3}, {3, 5}, {1, 4}, {2, 5},
    {3, 6}, {4, 5}, {1, 6}, {2, 7}, {3, 7},
    {1, 8}, {4, 4}, {3, 4}, {5, 5}, {2, 6},
    {1, 9}, {3, 3}, {4, 6}, {2, 4}, {1, 7},
    {2, 2}, {1, 3}, {2, 8}, {4, 3}, {3, 2},
    {5, 3}, {1, 5}, {6, 2}, {4, 2}, {3, 3},
    {2, 1}, {1, 2}, {4, 1}, {2, 9}, {1, 10},
    {5, 2}, {3, 1}, {6, 3}, {4, 7}, {3, 6},
    {2, 10}, {1, 11}, {5, 1}, {6, 4}, {7, 3},

    // Input data for output 1
    {9, 8}, {10, 7}, {8, 9}, {9, 9}, {10, 6},
    {9, 10}, {8, 8}, {10, 8}, {9, 11}, {10, 9},
    {7, 10}, {10, 10}, {11, 6}, {8, 10}, {9, 7},
    {10, 11}, {9, 6}, {7, 12}, {8, 11}, {11, 5},
    {10, 10}, {12, 6}, {11, 7}, {8, 12}, {9, 10},
    {10, 8}, {11, 8}, {12, 7}, {10, 12}, {11, 6},
    {9, 12}, {8, 13}, {9, 11}, {10, 12}, {12, 5},
    {11, 9}, {10, 13}, {12, 8}, {9, 13}, {13, 7},
    {8, 14}, {9, 14}, {12, 9}, {11, 10}, {10, 14},
    {13, 6}, {12, 10}, {13, 8}, {14, 7}, {11, 11}
};

vector<double> expected = {
    // Expected output for 0
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    // Expected output for 1
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
};

    pair<
        vector<
            pair<double, double>>,
        vector<double>>
        trainingData  = make_pair(input, expected);
    //trainingData = generateTrainingData(15, 5);
    //saveTrainingData(5, 5, 1000);

    /* //print training data
    cout << "\n[+]Training data: ";
    for (int i = 0; i < trainingData.first.size(); i++)
    {
        cout << "Input: " << trainingData.first[i].first << " " << trainingData.first[i].second << " Output: " << trainingData.second[i] << endl;
    }
    */



    int epochs = 3500;
    double learningRate = 0.001;

    n.backPropagate_new(trainingData, epochs, learningRate);
    //n.backPropagate_old(trainingData, epochs, learningRate);
    n.clean();
    n.setInputFromVector({10, 5});
    n.passValues();
    n.printActualOutput();

    n.clean();
    n.setInputFromVector({1, 11});
    n.passValues();
    n.printActualOutput();

    system("python draw_ssr_progress.py");

    t.stop();
}