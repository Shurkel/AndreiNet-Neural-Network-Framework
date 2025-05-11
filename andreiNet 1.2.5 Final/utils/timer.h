#ifndef TIMER_H
#define TIMER_H

#include "util.h" 
#include <chrono>
#include <iostream> 

class Timer { 
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime, endTime;
    std::chrono::duration<double, std::milli> duration; 

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    void stop(bool print = true) { 
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

#endif // TIMER_H