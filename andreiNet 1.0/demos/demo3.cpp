#include "../utils/andreinet.h"


net n({2, 4, 1});

ofstream py("../python/data.txt");

void saveWeights()
{
    ofstream wo("save.txt");
    if (!wo.is_open()) {
        cerr << "Error opening save file!" << endl;
        throw runtime_error("Failed to open save file.");
    }

    try {
        // Write weights
        for (int i = 0; i < n.layers.size(); i++) {
            for (int j = 0; j < n.layers[i].nodes.size(); j++) {
                for (int k = 0; k < n.layers[i].nodes[j].next.size(); k++) {
                    wo << n.layers[i].nodes[j].next[k].weight << " ";
                }
                wo << "\n";
            }
        }
        // Write biases
        for (int i = 0; i < n.layers.size(); i++) {
            for (int j = 0; j < n.layers[i].nodes.size(); j++) {
                wo << n.layers[i].nodes[j].bias << " ";
            }
            wo << "\n";
        }
    } catch (const ios_base::failure& e) {
        cerr << "Error writing to save file: " << e.what() << endl;
        wo.close();
        throw;
    }

    wo.close();
    if (wo.fail()) {
        cerr << "Error closing the save file!" << endl;
        throw runtime_error("Failed to close save file properly.");
    }
}

void importWeights()
{
    ifstream wi("save.txt");
    if (!wi.is_open()) {
        cout << "Error opening save file!" << endl;
        return;
    }

    for (int i = 0; i < n.layers.size(); i++)
    {
        for (int j = 0; j < n.layers[i].nodes.size(); j++)
        {
            for (int k = 0; k < n.layers[i].nodes[j].next.size(); k++)
            {
                if (!(wi >> n.layers[i].nodes[j].next[k].weight)) {
                    cout << "Error reading weight for layer " << i << ", node " << j << ", weight " << k << endl;
                    wi.close();
                    return;
                }
            }
        }
    }
    //biases
    for(int i = 0; i < n.layers.size(); i++)
    {
        for(int j = 0; j < n.layers[i].nodes.size(); j++)
        {
            if(!(wi >> n.layers[i].nodes[j].bias))
            {
                cout << "Error reading bias for layer " << i << ", node " << j << endl;
                wi.close();
                return;
            }
        }
    }
    if (wi.fail() && !wi.eof()) {
        cout << "Error reading from save file!" << endl;
    }
    wi.close();
    
    
}




void runDemo3(string outputfile)
{
    cout << BOLDGREEN << "\n[+]INIT DEMO3" << RESET;
    
    /*
    Aktivierungsfunktionen:
    0 - ReLU
    1 - Sigmoid
    2 - Softmax
    */
    t.start(); 
    
    
    n.useDefaults();
    

    vector<pair<double, double>> input = {
        // Durchfallwerte
        {2, 4},
        {3, 5},
        {1, 3},
        {4, 2},
        {2, 1},
        {3, 2},
        {1, 5},
        {4, 3},
        {2, 6},
        {3, 1},
        {2, 2},
        {1, 4},
        {4, 5},
        {3, 6},
        {1, 1},
        {4, 1},
        {2, 2},
        {3, 3},
        {1, 2},
        {4, 4},
        // Bestandene Werte
        {8, 10},
        {9, 11},
        {7, 9},
        {10, 8},
        {8, 7},
        {9, 8},
        {7, 11},
        {10, 9},
        {8, 12},
        {9, 7},
        {8, 13},
        {7, 10},
        {10, 11},
        {9, 12},
        {7, 7},
        {10, 7},
        {8, 8},
        {9, 9},
        {7, 8},
        {10, 10},
        {6, 6},
        {11, 11},
        {5, 5},
        {12, 12},
        {6, 5}};

    // Erwartete Ausgaben: 0 für Durchfall, 1 für Bestanden
    vector<double> expected = {
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1};

    pair<
        vector<
            pair<double, double>>,
        vector<double>>
        trainingData;
    trainingData.first = input;
    trainingData.second = expected;

    int epochs = 100000;
    double learningRate = 0.06;

    std::cin.get(); //aici ai optiunea sa continui sau sa te sinucizi
    
    t.stop();
    u.logTime();
    l.flush();
}
