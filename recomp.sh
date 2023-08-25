#!/bin/bash

# Clean up the current directory
rm -rf build

mkdir build

# Run CMake to generate build files in a 'build' directory
cmake -B build

# Move to the 'build' directory
cd build

# Build the project using make
make

# Run the compiled executable
./ADMC
