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
    double softplus(double x)
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
    double dsoftplus(double x)
    {
        return 1 / (1 + exp(-x));
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