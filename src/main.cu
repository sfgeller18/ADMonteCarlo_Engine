#include <iostream>
#include <StochasticProcess.h>
#include <HestonProcess.h>
#include "ProcessGrapher.hpp"
#include <unistd.h> 
#include "duals.hpp"
#include "deviceDualFuncs.cuh"
#include "cuRandSamples.cuh"
#include "deviceMPFR.hpp"

# define precision 12

// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


int main() {
    const int numElements = 128;
    const size_t size = numElements * sizeof(DualNumber<double>);

    // Allocate memory on the host and initialize data
    DualNumber<double>* h_input = new DualNumber<double>[numElements];
    DualNumber<double>* h_output = new DualNumber<double>[numElements];
    
    for (int i = 0; i < numElements; ++i) {
        h_input[i] = DualNumber<double>(i, 1.0);
    }

    // Allocate memory on the device
    DualNumber<double>* d_input = nullptr;
    DualNumber<double>* d_output = nullptr;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    dual_pow<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, 0.5);

    // Copy the result from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < numElements; ++i) {
        std::cout << "Input: (" << h_input[i].real << ", " << h_input[i].dual << "), ";
        std::cout << "Squared: (" << h_output[i].real << ", " << h_output[i].dual << ")\n";
    }

    // Clean up
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);



return EXIT_SUCCESS;


}