#ifndef INCLUDES_H
#define INCLUDES_H

#include <iostream>
#include <vector>
//#include <windows.h> // Removed - non-portable and slow system calls
#include <random>
#include <chrono>
#include <string>
#include <cstdlib> // For std::exit, etc., if needed, but avoid system()
#include <cmath>   // For std::exp, std::log, std::pow
#include <numeric> // For std::accumulate if needed
#include <iomanip> // For std::setw, std::left, std::right
#include <fstream> // For file I/O (use outside critical loops)
#include <algorithm> // For std::max, std::min
#include <map>       // For TextTable alignment
#include "../../eigen-3.4.0/Eigen/Core" // Include Eigen Core
#include "TextTable.h" // Keep TextTable include


// DEFINES
#define en std::cout << '\n' // Use std namespace

// #define cls system("cls"); // Removed - slow and non-portable
#define cls std::cout << "\033[2J\033[1;1H"; // ANSI escape code for clearing (more portable)

#define test std::cout << "test\n"

#define tab std::cout << '\t' // Use char literal

// ANSI Color Codes (Good as they are)
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#endif // INCLUDES_H