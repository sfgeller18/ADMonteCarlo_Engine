#!/bin/bash

cd ..
# Run CMake to generate build files in a 'build' directory
cmake -B build

# Move to the 'build' directory
cd build

# Build the project using make
make

# Run the compiled executable
./ADMC

gimp ../output/simulation_plot.png
