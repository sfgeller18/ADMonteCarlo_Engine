#!/bin/bash
cd ..
# Run CMake to generate build files in a 'build' directory
rm -rf build

mkdir build

cmake -B build

# Move to the 'build' directory
cd build

# Build the project using make
make

# Run the compiled executable
./ADMC

cd scripts
