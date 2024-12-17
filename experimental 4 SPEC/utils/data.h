#include "includes.h"
class data
{
private:
    vector<double> inputs;
    vector<double> outputs;
    vector<double> expected;

public:
    data() {}
    data(vector<double> inputs, vector<double> outputs, vector<double> expected) : inputs(inputs), outputs(outputs), expected(expected) {}

    // * GETTERS AND SETTERS

    //INPUTS
    vector<double> getInputs() { return inputs; }
    void setInputs(vector<double> inputs) { this->inputs = inputs; }

    //OUTPUTS
    vector<double> getOutputs() { return outputs; }
    void setOutputs(vector<double> outputs) { this->outputs = outputs; }

    //EXPECTED
    vector<double> getExpected() { return expected; }
    void setExpected(vector<double> expected) { this->expected = expected; }

};