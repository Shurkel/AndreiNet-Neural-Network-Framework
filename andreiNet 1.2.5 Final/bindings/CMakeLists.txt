cmake_minimum_required(VERSION 3.12)
project(andreinet_py)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add the path to your C++ library headers
# Adjust this path if your cpp_lib is elsewhere relative to this CMakeLists.txt
include_directories(../cpp_lib) # Assumes cpp_lib is one level up

# Define the Python extension module
# The first argument is the name of the target library (doesn't have to match module name)
# The second argument is the source file for the bindings
pybind11_add_module(andreinet_py bindings.cpp)
# The target name above (andreinet_py) will become `andreinet_py.so` or `andreinet_py.pyd`

# Link Python libraries
target_link_libraries(andreinet_py PRIVATE Python::Python)

# Specify C++ standard for the target (pybind11 might do this, but explicit is good)
set_target_properties(andreinet_py PROPERTIES CXX_STANDARD 17)

# Optional: Add compiler flags for optimization, debugging
# target_compile_options(andreinet_py PRIVATE -O3) # Example for GCC/Clang
# For MSVC:
# if(MSVC)
#   target_compile_options(andreinet_py PRIVATE /O2)
# endif()