#include "includes.h"
class dataSet
{
private:
    vector<vector<double>> inputs;
    vector<vector<double>> outputs;
    vector<vector<double>> expected;

public:
    dataSet() {}
    dataSet(vector<vector<double>> inputs, vector<vector<double>> outputs, vector<vector<double>> expected) : inputs(inputs), outputs(outputs), expected(expected) {}

    // * GETTERS AND SETTERS

    //INPUTS
    vector<vector<double>> getInputs() { return inputs; }
    vector<double> getInput(int i) { return inputs[i]; }
    void setInputs(vector<vector<double>> inputs) { this->inputs = inputs; }
    void printInputs()
    {
        cout << "[+]Inputs: ";
        for (int i = 0; i < getInputs().size(); i++)
            for (int j = 0; j < getInputs()[i].size(); j++)
                cout << inputs[i][j] << " ";
        en
    }

    //OUTPUTS
    vector<vector<double>> getOutputs() { return outputs; }
    void setOutputs(vector<vector<double>> outputs) { this->outputs = outputs; }

    //EXPECTED
    vector<vector<double>> getExpected() { return expected; }
    void setExpected(vector<vector<double>> expected) { this->expected = expected; }
    void printExpected()
    {
        cout << "[+]Expected: ";
        for (int i = 0; i < getExpected().size(); i++)
            for (int j = 0; j < getExpected()[i].size(); j++)
                cout << getExpected()[i][j] << " ";
        en
    }

};