    // layer funktionen
    void logNet();
    void clean();
    void clean(int i);
    void loggingGlobal();
    void loggingLayer(int layerId);
    double weight(int layerId, int nodeId, int nextLayerId, int nextNodeId);
    double weight(Node *node, Node *nextNode);
    void setExpected(vector<double> expectedValues);
    void getCosts();
    void getSSR();
    void clearCosts();
    void clearSSR();
    void logCosts();
    void printInput();
    void printCosts();
    void printExpected();
    void printActualOutput();
    void setValueAll(double val);
    void setValue(int layerId, int nodeId, double val);
    void setInputFromVector(pair<double, double> values);
    void setBiasAll(int layerId, double w);
    void setIdAll(int layerId, int id);
    void setActivateAll(int function);
    void noActivate(int layerId, int nodeId);
    void noActivate(int layerId);
    void noActivateAll();
    void setActivate(int layerId, int function);
    void printLayer(int id);
    void printNet();
    void printLayers();
    void logLayers();
    
    // node funktionen
    void setWeight(int layerID, int nodeID, int nextLayerID, int nextNodeID, double w);
    void setBias(int layerID, int nodeID, double b);
    void printNextNodes(int layerId, int nodeId);

    // net funktionen
    void connectLayers();
    void passValues();
    void updateGradient();
    void setWeightAll(double w);
    void setBiasAll(double b);
    void randomiseAllWeights();
    void printNodeDetails(int nodeId, int layerId);
    void printSSR();
    void testNet(pair<vector<pair<double, double>>, vector<double>> trainingData, bool brief);
    void backPropagate(pair<vector<pair<double, double>>, vector<double>> trainingData, int epochs, double learningRate);
