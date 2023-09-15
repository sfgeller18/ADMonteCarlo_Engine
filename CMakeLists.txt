cmake_minimum_required(VERSION 3.9)

project(ADMC LANGUAGES CUDA CXX)

# Set the minimum required version of the CUDA Toolkit
find_package(CUDA 11.3 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_BUILD_TYPE Debug)

# Set the C++ standard for the CUDA code
set(CMAKE_CUDA_STANDARD 20)
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Add the executable using the collected sources
add_executable(ADMC ${SOURCES})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")
find_package(MPFR REQUIRED)
target_link_libraries(ADMC PRIVATE ${MPFR_LIBRARIES})

# Find GMP
find_package(GMP REQUIRED)
target_link_libraries(ADMC PRIVATE ${GMP_LIBRARIES})

# To disable some warnings relating to the HyperComplex class
target_compile_options(ADMC PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=186>)
