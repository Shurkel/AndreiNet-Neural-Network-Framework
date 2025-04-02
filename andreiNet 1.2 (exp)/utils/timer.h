#ifndef TIMER_H
#define TIMER_H

#include "util.h" // Include util which includes includes.h
#include <chrono>
#include <iostream> // For cout

class Timer { // Capitalized class name convention
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime, endTime;
    std::chrono::duration<double, std::milli> duration; // Use ms directly

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    void stop(bool print = true) { // Add option to suppress print
        endTime = std::chrono::high_resolution_clock::now();
        duration = endTime - startTime;

        if (print) {
            std::cout << "Runtime: " << duration.count() << " ms\n";
        }
    }

    double getDurationMs() const {
        return duration.count();
    }
};

// Avoid creating global instances in headers if possible.
// Let the user create a Timer object where needed.
// inline Timer t; // Removed global instance

#endif // TIMER_H