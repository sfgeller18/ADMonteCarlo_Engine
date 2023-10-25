#!/bin/bash
cd ../src
# Compile your C++ project with g++ and create an AST dump
g++ -std=c++17 -fdump-tree-all -c -o /dev/null HestonProcess.cpp 2> ast_dump.txt

# Replace "your_source_file.cpp" with the name of your C++ source file

if [ $? -eq 0 ]; then
    echo "Compilation successful. AST dumped to ast_dump.txt"
else
    echo "Compilation failed."
fi

# Exit with success status
exit 0
