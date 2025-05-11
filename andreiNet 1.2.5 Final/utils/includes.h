#ifndef INCLUDES_H
#define INCLUDES_H

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cstdlib> 
#include <cmath>   
#include <numeric> 
#include <iomanip> 
#include <fstream> 
#include <algorithm> 
#include <map>       

#include "TextTable.h" 


// DEFINES
#define en std::cout << '\n' 

#define cls std::cout << "\033[2J\033[1;1H"; // ANSI escape code for clearing

#define test std::cout << "test\n"

#define tab std::cout << '\t' 

// ANSI Color Codes
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