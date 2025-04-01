#!/bin/bash

# Compile the demo
g++ -std=c++11 demoCrossEnt.cpp -o demoCrossEnt

# Run the demo
./demoCrossEnt

# Run the visualization script
python visualize_ce.py

echo "Demo completed. Check the generated ce_training_animation.gif and ce_final_plot.png files."
