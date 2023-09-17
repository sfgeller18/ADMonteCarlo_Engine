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
      const int numSamples = NUM_BLOCKS * NUM_THREADS_PER_BLOCK * NUM_SAMPLES_PER_THREAD;
    const int numSamplesToPrint = 20;

    // Allocate host memory to store the random samples
    double* host_samples = new double[numSamples];

    // Allocate device memory to store the random samples
    double* device_samples;
    cudaMalloc(&device_samples, numSamples * sizeof(double));

    // Call the CUDA function to generate random samples
    cudaError_t cudaStatus = CudaNormalSamples(device_samples);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "CudaNormalSamples failed with error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // Copy the generated samples from the device to the host
    cudaMemcpy(host_samples, device_samples, numSamples * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the first 20 samples
    std::cout << "First " << numSamplesToPrint << " samples:" << std::endl;
    for (int i = 0; i < numSamplesToPrint; i++) {
        std::cout << std::fixed << host_samples[i] << " ";
    }
    std::cout << std::endl;

    double mean, variance;
    msdArray<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(device_samples, numSamples, mean, variance);
 // Print the calculated mean and standard deviation
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Variance: " << variance << std::endl;

    // Cleanup: Free memory
    delete[] host_samples;
    cudaFree(device_samples);


return EXIT_SUCCESS;


}