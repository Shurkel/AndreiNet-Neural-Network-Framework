#include "includes.h"
class util
{
public:
    double relu(double x)
    {
        if (x > 0)
        {
            return x;
        }
        else
        {
            return 0;
        }
    }
    double sigmoid(double x)
    {
        return 1 / (1 + exp(-x));
    }
    double softmax(double x)
    {
        //f(x) = log(1 + e^x)
        return log(1 + exp(x));
    }
    double drelu(double x)
    {
        if (x > 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    double dsigmoid(double x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }
    
    void logVector(vector<double> v)
    {
        for (int i = 0; i < v.size(); i++)
        {
            l << v[i] << " ";
        }
        len
    }
    void logTime()
    {
        // curent time and date
        l << "Last Build: " << __TIME__ << ", " << __DATE__ << "\n";
    }
    double randomDouble(double fMin, double fMax)
    {
        double f = (double)rand() / RAND_MAX;
        return fMin + f * (fMax - fMin);
    }
    
    double normalize(double value, double minVal, double maxVal) {
        return (value - minVal) / (maxVal - minVal);
    }

} u;