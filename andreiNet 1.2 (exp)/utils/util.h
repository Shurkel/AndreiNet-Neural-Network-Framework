#ifndef UTIL_H
#define UTIL_H

#include "includes.h" // Include necessary headers
#include <random>    // For modern C++ random numbers
#include <cmath>     // For exp, log
#include <vector>
#include <numeric>   // For std::accumulate
#include <limits>    // For numeric_limits

class Util
{
private:
    // Use a better random number generator setup
    std::mt19937 rng; // Mersenne Twister engine
    std::uniform_real_distribution<double> unif;

    public:
    // Seed the generator once
    Util() : rng(std::random_device{}()), unif(-1.0, 1.0) {}

    // --- GETTER FOR RNG ---
    // Provide access to the random number generator engine
    std::mt19937& getRng() { // Return by reference
        return rng;
    }
    // --- Activation Functions ---
    inline double relu(double x) const {
        return std::max(0.0, x); // Simpler than if/else
    }

    inline double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double softplus(double x) const {
        // f(x) = log(1 + e^x)
        return std::log1p(std::exp(x)); // log1p(y) = log(1+y), more accurate for small y (exp(x))
    }

    // --- Activation Derivatives ---
    inline double drelu(double x) const {
        return (x > 0.0) ? 1.0 : 0.0;
    }

    inline double dsigmoid(double x) const {
        // Calculate sigmoid(x) only once if possible within node/layer
        // Or calculate from the *activated* value 'a': return a * (1.0 - a);
        // Here we use the input 'x' to the sigmoid (pre-activation)
        const double sig_x = sigmoid(x);
        return sig_x * (1.0 - sig_x);
    }
    // Derivative of sigmoid from *activated* value y = sigmoid(x)
     inline double dsigmoid_from_output(double y) const {
        return y * (1.0 - y);
    }


    inline double dsoftplus(double x) const {
        // Derivative is sigmoid(x)
        return sigmoid(x);
    }

    // --- Random Numbers ---
    double randomDouble(double fMin, double fMax) {
        // Adjust the distribution range if needed, or create a new one
        std::uniform_real_distribution<double> dist(fMin, fMax);
        return dist(rng);
    }

    // --- Vector Operations ---
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
            // Subtract max_val for numerical stability (avoids large exp values)
            const double exp_val = std::exp(val - max_val);
            y.push_back(exp_val);
            sum += exp_val;
        }

        if (sum == 0) sum = 1.0; // Avoid division by zero if all inputs were -inf

        for (double& val : y) {
            val /= sum;
        }
        return y;
    }

};

// Create a single global instance (or pass it around/use dependency injection)
// Global instance is simpler for this structure but less flexible.
inline Util u; // Use inline to avoid multiple definition errors if included in multiple TUs

#endif // UTIL_H