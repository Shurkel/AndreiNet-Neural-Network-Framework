#ifndef UTIL_H
#define UTIL_H

#include "includes.h" // Include necessary headers
#include <random>     // For modern C++ random numbers
#include <cmath>      // For exp, log
#include <vector>
#include <numeric>                   // For std::accumulate
#include <limits>                    // For numeric_limits

class Util
{
private:
    // Use a better random number generator setup
    std::mt19937 rng; // Mersenne Twister engine
    std::uniform_real_distribution<double> unif;

public:
    // Seed the generator once
    Util() : rng(std::random_device{}()), unif(-1.0, 1.0) {} // Constructor needed

    // --- GETTER FOR RNG ---
    // Provide access to the random number generator engine
    std::mt19937 &getRng()
    { // Return by reference
        return rng;
    }

    // --- Eigen-based Activation Functions ---

    // Apply activation function element-wise to an Eigen vector/array
    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    activateEigen(const Eigen::MatrixBase<Derived> &z) const
    {
        // Example for sigmoid - adapt for others using a switch or similar
        return z.unaryExpr([](double x)
                           { return 1.0 / (1.0 + std::exp(-x)); }); // Sigmoid
        // For ReLU: return z.unaryExpr([](double x){ return std::max(0.0, x); });
        // For Linear: return z;
    }

    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    activateEigen(const Eigen::MatrixBase<Derived> &z, int activationType) const
    {
        switch (activationType)
        {
        case 0: // ReLU
            return z.unaryExpr([](double x)
                               { return std::max(0.0, x); });
        case 1: // Sigmoid
            return z.unaryExpr([](double x)
                               { return 1.0 / (1.0 + std::exp(-x)); });
        case 2: // Softplus
            return z.unaryExpr([](double x)
                               { return std::log1p(std::exp(x)); });
        case -1: // Linear
        default:
            return z;
        }
    }

     // Apply activation derivative element-wise
     template<typename Derived>
     Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
     activationDerivativeEigen(const Eigen::MatrixBase<Derived>& z, int activationType) const {
          switch (activationType) {
              case 0: // dReLU
                  return z.unaryExpr([](double x){ return (x > 0.0) ? 1.0 : 0.0; });
              case 1: // dSigmoid (from z)
                  return z.unaryExpr([this](double x){ double sig = sigmoid(x); return sig * (1.0 - sig); });
              case 2: // dSoftplus (which is sigmoid(z))
                  return z.unaryExpr([this](double x){ return sigmoid(x); });
              case -1: // dLinear
              default:
                  // Return a vector/matrix of ones with the same dimensions
                  return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Ones(z.rows(), z.cols());
          }
     }

     // Apply activation derivative using the *activated* value 'a' (faster if 'a' is known)
     template<typename Derived>
     Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
     activationDerivativeFromOutputEigen(const Eigen::MatrixBase<Derived>& a, int activationType) const {
         switch (activationType) {
             case 0: // dReLU from output a
                 return a.unaryExpr([](double x){ return (x > 0.0) ? 1.0 : 0.0; });
             case 1: // dSigmoid from output a: a * (1 - a)
                  // Use Eigen's array() for element-wise operations
                 return (a.array() * (1.0 - a.array())).matrix();
             case 2: // dSoftplus from output a: 1 - exp(-a)
                  return (1.0 - (-a.array()).exp()).matrix();
             case -1: // dLinear
             default:
                 return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Ones(a.rows(), a.cols());
         }
     }

    // --- Activation Functions ---
    inline double relu(double x) const
    {
        return std::max(0.0, x); // Simpler than if/else
    }

    inline double sigmoid(double x) const
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double softplus(double x) const
    {
        // f(x) = log(1 + e^x)
        return std::log1p(std::exp(x)); // log1p(y) = log(1+y), more accurate for small y (exp(x))
    }

    // --- Activation Derivatives ---
    inline double drelu(double x) const
    {
        return (x > 0.0) ? 1.0 : 0.0;
    }

    inline double dsigmoid(double x) const
    {
        // Calculate sigmoid(x) only once if possible within node/layer
        // Or calculate from the *activated* value 'a': return a * (1.0 - a);
        // Here we use the input 'x' to the sigmoid (pre-activation)
        const double sig_x = sigmoid(x);
        return sig_x * (1.0 - sig_x);
    }
    // Derivative of sigmoid from *activated* value y = sigmoid(x)
    inline double dsigmoid_from_output(double y) const
    {
        return y * (1.0 - y);
    }

    inline double dsoftplus(double x) const
    {
        // Derivative is sigmoid(x)
        return sigmoid(x);
    }

    // --- Random Numbers ---
    double randomDouble(double fMin, double fMax)
    {
        // Adjust the distribution range if needed, or create a new one
        std::uniform_real_distribution<double> dist(fMin, fMax);
        return dist(rng);
    }

    // --- Vector Operations ---
    std::vector<double> softMax(const std::vector<double> &x) const
    {
        if (x.empty())
            return {};
        std::vector<double> y;
        y.reserve(x.size());
        double max_val = -std::numeric_limits<double>::infinity();
        for (double val : x)
        {
            if (val > max_val)
                max_val = val;
        }

        double sum = 0.0;
        for (double val : x)
        {
            // Subtract max_val for numerical stability (avoids large exp values)
            const double exp_val = std::exp(val - max_val);
            y.push_back(exp_val);
            sum += exp_val;
        }

        if (sum == 0)
            sum = 1.0; // Avoid division by zero if all inputs were -inf

        for (double &val : y)
        {
            val /= sum;
        }
        return y;
    }
};

// Create a single global instance (or pass it around/use dependency injection)
// Global instance is simpler for this structure but less flexible.
inline Util& getUtil() {
    static Util u_instance; // Meyers' Singleton
    return u_instance;
}
#endif // UTIL_H