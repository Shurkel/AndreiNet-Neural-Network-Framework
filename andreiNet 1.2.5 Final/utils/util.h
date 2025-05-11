#ifndef UTIL_H
#define UTIL_H

#include "includes.h" 
#include <random>    
#include <cmath>     
#include <vector>
#include <numeric>   
#include <limits>    

class Util
{
private:
    std::mt19937 rng; 
    std::uniform_real_distribution<double> unif;

    public:
    Util() : rng(std::random_device{}()), unif(-1.0, 1.0) {}

    std::mt19937& getRng() { 
        return rng;
    }

    inline double relu(double x) const {
        return std::max(0.0, x); 
    }

    inline double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double softplus(double x) const {
        return std::log1p(std::exp(x)); // log1p(y) = log(1+y), more accurate for small exp(x)
    }

    inline double drelu(double x) const {
        return (x > 0.0) ? 1.0 : 0.0;
    }

    inline double dsigmoid(double x) const {
        const double sig_x = sigmoid(x);
        return sig_x * (1.0 - sig_x);
    }
    
    inline double dsigmoid_from_output(double y) const { // Derivative of sigmoid from activated value y = sigmoid(x)
        return y * (1.0 - y);
    }

    inline double dsoftplus(double x) const {
        return sigmoid(x); // Derivative is sigmoid(x)
    }

    double randomDouble(double fMin, double fMax) {
        std::uniform_real_distribution<double> dist(fMin, fMax);
        return dist(rng);
    }

    std::vector<double> softMax(const std::vector<double>& x) const {
        if (x.empty()) return {};
        std::vector<double> y;
        y.reserve(x.size());
        double max_val = -std::numeric_limits<double>::infinity();
        for (double val : x) {
            if (val > max_val) max_val = val;
        }

        double sum = 0.0;
        for (double val : x) {
            const double exp_val = std::exp(val - max_val); // Subtract max_val for numerical stability
            y.push_back(exp_val);
            sum += exp_val;
        }

        if (sum == 0) sum = 1.0; // Avoid division by zero

        for (double& val : y) {
            val /= sum;
        }
        return y;
    }
};

inline Util u; 

#endif // UTIL_H