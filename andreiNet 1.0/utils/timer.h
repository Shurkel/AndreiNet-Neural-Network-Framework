#include "util.h"
class timer {
public:
    chrono::time_point<chrono::high_resolution_clock> startTime, endTime;
    chrono::duration<float> duration;

    void start() {
        startTime = chrono::high_resolution_clock::now();
    }

    void stop() {
        endTime = chrono::high_resolution_clock::now();
        duration = endTime - startTime;

        //time in milliseconds
        l << "Runtime: " << duration.count() * 1000 << "ms\n";
    }
}t;