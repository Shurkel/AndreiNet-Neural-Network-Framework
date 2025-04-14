// --- START OF FILE util.h ---

#ifndef UTIL_H
#define UTIL_H

#include "includes.h" // Include necessary headers like iostream, cmath, vector, limits, etc.
#include <random>     // For modern C++ random numbers
#include <cmath>      // For exp, log, tanh, max, std::pow, sqrt
#include <vector>
#include <numeric>    // For std::accumulate
#include <limits>     // For numeric_limits
#include <algorithm>  // For std::max used in relu

// Ensure Eigen Core is included (it should be pulled in via includes.h -> layer.h -> includes.h)
// If not, uncomment the following lines and adjust the path as needed:
/*
#ifndef EIGEN_CORE_H
#include "../eigen-3.4.0/Eigen/Core" // Make sure path is correct
#define EIGEN_CORE_H // Basic include guard for Eigen Core
#endif
*/


class Util
{
private:
    // Use a better random number generator setup
    std::mt19937 rng; // Mersenne Twister engine
    // Pre-define a standard distribution for frequent use (e.g., weights init)
    std::uniform_real_distribution<double> unif;

public:
    // Seed the generator once during construction
    Util() : rng(std::random_device{}()), unif(-1.0, 1.0) {} // Default range [-1, 1]

    // --- GETTER FOR RNG ---
    // Provide access to the random number generator engine if needed externally (e.g., shuffling data)
    std::mt19937 &getRng()
    { // Return by reference
        return rng;
    }

    // --- Activation Functions (Scalar) ---
    // These functions operate on single double values.

    // Linear (identity) - often represented by activationType = -1
    // No function needed, the operation is just returning the input value.

    // ReLU (Rectified Linear Unit) - activationType = 0
    inline double relu(double x) const
    {
        return std::max(0.0, x);
    }

    // Sigmoid (Logistic) - activationType = 1
    inline double sigmoid(double x) const
    {
        // Prevent overflow/underflow for large magnitude inputs
        if (x > 35.0) return 1.0;
        if (x < -35.0) return 0.0;
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Softplus - activationType = 2
    inline double softplus(double x) const
    {
        // f(x) = log(1 + e^x)
        // Use numerical stability tricks:
        // For large x, log(1 + e^x) ≈ log(e^x) = x
        // For very negative x, e^x is small, log(1 + e^x) ≈ e^x
        // Use log1p(y) = log(1+y) for accuracy when e^x is small.
        if (x > 30.0) { // Threshold where exp(x) might overflow or 1 is negligible
            return x;
        } else if (x < -30.0) { // Threshold where exp(x) is very small
             return std::exp(x);
        }
        return std::log1p(std::exp(x));
    }

    // Tanh (Hyperbolic Tangent) - activationType = 3
    inline double tanh(double x) const
    {
        return std::tanh(x);
    }


    // --- Activation Derivatives (Scalar) ---
    // Derivatives calculated with respect to the pre-activation input 'z'.

    // dLinear/dz - activationType = -1
    inline double dlinear(double /*z*/) const { // Parameter z is unused but kept for signature consistency
        return 1.0;
    }

    // dReLU/dz - activationType = 0
    inline double drelu(double z) const
    {
        return (z > 0.0) ? 1.0 : 0.0;
    }

    // dSigmoid/dz - activationType = 1
    inline double dsigmoid(double z) const
    {
        const double sig_z = sigmoid(z); // Calculate sigmoid only once
        return sig_z * (1.0 - sig_z);
    }

    // dSoftplus/dz - activationType = 2
    inline double dsoftplus(double z) const
    {
        // The derivative of softplus(z) is sigmoid(z)
        return sigmoid(z);
    }

    // dTanh/dz - activationType = 3
    inline double dtanh(double z) const
    {
        const double tanh_z = std::tanh(z);
        // Derivative is 1 - tanh^2(z), also known as sech^2(z)
        return 1.0 - tanh_z * tanh_z;
    }


    // --- Activation Derivatives (Scalar) from *Output* 'a' ---
    // Calculate derivative w.r.t 'z' using the activated output 'a'. Useful in backpropagation.

    // dLinear/dz from output 'a' - activationType = -1
    inline double dlinear_from_output(double /*a*/) const {
        return 1.0;
    }

    // dReLU/dz from output 'a' - activationType = 0
    inline double drelu_from_output(double a) const {
        // If the output a > 0, the derivative (w.r.t z) was 1. Otherwise 0.
        return (a > 0.0) ? 1.0 : 0.0;
    }

    // dSigmoid/dz from output 'a' - activationType = 1
    inline double dsigmoid_from_output(double a) const
    {
        // If a = sigmoid(z), then dsigmoid/dz = a * (1 - a)
        return a * (1.0 - a);
    }

    // dSoftplus/dz from output 'a' - activationType = 2
    inline double dsoftplus_from_output(double a) const
    {
        // If a = softplus(z), then dsoftplus/dz = sigmoid(z).
        // We need sigmoid(z) in terms of a = log(1 + exp(z)).
        // From derivation: sigmoid(z) = 1 - exp(-a)
        // Check for potential issues: a = softplus(z) is always > 0.
        // If a is extremely close to 0 (z was very negative), exp(-a) -> exp(0) = 1, result is 0. Correct.
        // If a is very large (z was very large), exp(-a) -> 0, result is 1. Correct.
        if (a < std::numeric_limits<double>::epsilon()) return 0.0; // Handle very small 'a' if necessary
        return 1.0 - std::exp(-a);
    }

    // dTanh/dz from output 'a' - activationType = 3
    inline double dtanh_from_output(double a) const
    {
        // If a = tanh(z), then dtanh/dz = 1 - a^2
        return 1.0 - a * a;
    }


    // --- Eigen-based Activation Functions ---
    // Apply activation function element-wise to an Eigen matrix or vector.
    // Uses Eigen's .unaryExpr for efficient element-wise operations.

    template <typename Derived> // Works for VectorXd, MatrixXd, ArrayXd, etc.
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    activateEigen(const Eigen::MatrixBase<Derived>& z, int activationType) const
    {
        // Use lambdas capturing 'this' to call the scalar member functions
        switch (activationType)
        {
        case 0: // ReLU
            return z.unaryExpr([this](double x){ return this->relu(x); });
        case 1: // Sigmoid
            return z.unaryExpr([this](double x){ return this->sigmoid(x); });
        case 2: // Softplus
            return z.unaryExpr([this](double x){ return this->softplus(x); });
        case 3: // Tanh
             return z.unaryExpr([this](double x){ return this->tanh(x); });
        case -1: // Linear
        default:
            // Return a copy of the input matrix/vector
            // Using .derived() gets the actual object type (MatrixXd, VectorXd, etc.)
            return z.derived();
        }
    }

    // --- Eigen-based Activation Derivatives ---
    // Apply activation derivative (w.r.t 'z') element-wise to an Eigen matrix or vector.

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    activationDerivativeEigen(const Eigen::MatrixBase<Derived>& z, int activationType) const {
         switch (activationType) {
             case 0: // dReLU/dz
                 return z.unaryExpr([this](double x){ return this->drelu(x); });
             case 1: // dSigmoid/dz
                 return z.unaryExpr([this](double x){ return this->dsigmoid(x); });
             case 2: // dSoftplus/dz
                 return z.unaryExpr([this](double x){ return this->dsoftplus(x); });
             case 3: // dTanh/dz
                 return z.unaryExpr([this](double x){ return this->dtanh(x); });
             case -1: // dLinear/dz
             default:
                 // Return a matrix/vector of ones with the same dimensions as z
                 return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Ones(z.rows(), z.cols());
         }
    }

    // Apply activation derivative (w.r.t 'z') element-wise using the *activated* value 'a'.

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    activationDerivativeFromOutputEigen(const Eigen::MatrixBase<Derived>& a, int activationType) const {
        // Use Eigen's array() methods for efficient element-wise arithmetic where possible
        switch (activationType) {
            case 0: // dReLU/dz from a
                return a.unaryExpr([this](double x){ return this->drelu_from_output(x); });
            case 1: // dSigmoid/dz from a: a * (1 - a)
                return (a.array() * (1.0 - a.array())).matrix(); // Use array ops, convert back to matrix
            case 2: // dSoftplus/dz from a: 1 - exp(-a)
                 return (1.0 - (-a.array()).exp()).matrix();
            case 3: // dTanh/dz from a: 1 - a^2
                return (1.0 - a.array().square()).matrix();
            case -1: // dLinear/dz from a
            default:
                // Return a matrix/vector of ones with the same dimensions as a
                return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Ones(a.rows(), a.cols());
        }
    }


    // --- Random Numbers ---

    // Generate a random double within a specified range [fMin, fMax]
    double randomDouble(double fMin, double fMax)
    {
        // Create a temporary distribution for the specific range
        std::uniform_real_distribution<double> dist(fMin, fMax);
        // Generate number using the class member 'rng' engine
        return dist(rng);
    }

    // Generate random double using the default distribution range [-1, 1] stored in 'unif'
    double randomDouble()
    {
        return unif(rng);
    }


    // --- Vector Operations (std::vector based) ---
    // Included for completeness or if non-Eigen parts of the code use it.

    // Softmax for std::vector<double>
    std::vector<double> softMax(const std::vector<double> &x) const
    {
        if (x.empty())
            return {}; // Return empty vector if input is empty

        std::vector<double> y;
        y.reserve(x.size()); // Reserve space for efficiency

        // Find max element for numerical stability (subtracting max avoids large exp values)
        double max_val = -std::numeric_limits<double>::infinity();
        for (double val : x) {
            if (val > max_val)
                max_val = val;
        }
         // Handle case where all inputs might be -infinity
         if (!std::isfinite(max_val)) {
             // Return uniform distribution or zeros? Uniform seems more reasonable.
              double uniform_prob = 1.0 / x.size();
              y.assign(x.size(), uniform_prob);
              return y;
         }


        // Calculate exponentials (shifted by max_val) and their sum
        double sum = 0.0;
        for (double val : x) {
            const double exp_val = std::exp(val - max_val);
            y.push_back(exp_val);
            sum += exp_val;
        }

        // Normalize by the sum
        // Avoid division by zero if sum is extremely small (e.g., all shifted inputs were very negative)
        if (sum <= std::numeric_limits<double>::epsilon()) {
             // Assign uniform probability
             double uniform_prob = 1.0 / x.size();
             std::fill(y.begin(), y.end(), uniform_prob);
        } else {
             for (double &val : y) {
                 val /= sum;
             }
        }
        return y;
    }

    // --- Eigen-based Softmax ---
    // Softmax for Eigen::VectorXd (more commonly used with the Eigen network structure)
    Eigen::VectorXd softmaxEigen(const Eigen::VectorXd& x) const {
        if (x.size() == 0) return Eigen::VectorXd(); // Handle empty input

        // Subtract max coefficient for numerical stability
        double maxCoeff = x.maxCoeff();
         if (!std::isfinite(maxCoeff)) {
             // Handle case where input contains NaN or +/-Inf leading to invalid maxCoeff
             // Or if all inputs were -inf. Return uniform.
              return Eigen::VectorXd::Constant(x.size(), 1.0 / x.size());
         }

        Eigen::VectorXd exp_x = (x.array() - maxCoeff).exp(); // Element-wise exponentiation

        // Sum of exponentials
        double sum_exp = exp_x.sum();

        // Normalize
        if (sum_exp <= std::numeric_limits<double>::epsilon() || !std::isfinite(sum_exp)) {
            // Handle degenerate case (sum is zero/small or NaN/Inf) -> return uniform distribution
            return Eigen::VectorXd::Constant(x.size(), 1.0 / x.size());
        } else {
            return exp_x / sum_exp; // Element-wise division by the sum
        }
    }


}u; // End class Util

// --- Global Accessor Function (Meyers' Singleton) ---
// Provides a single, thread-safe (since C++11) instance of the Util class.
inline Util& getUtil() {
    // Static local variable initialization is guaranteed to happen only once.
    static Util u_instance;
    return u_instance;
}

#endif // UTIL_H
// --- END OF FILE util.h ---