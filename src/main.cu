#include <iostream>
#include <StochasticProcess.h>
#include <HestonProcess.h>
#include <mpfr.h>
#include <fstream>
#include <unistd.h> 
#include "duals.hpp"
#include "ProcessGrapher.hpp"
#include "cuRandSamples.hpp"
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
 cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Set the number of samples you want to generate
    const long numSamples = (long)NUM_BLOCKS * NUM_THREADS_PER_BLOCK * NUM_SAMPLES_PER_THREAD * KERNEL_ITERATIONS;

    // Allocate memory for the generated samples on the host
    double* hostSamples = new double[numSamples];

    // Allocate memory for the generated samples on the device
    double* deviceSamples;
    cudaMalloc((void**)&deviceSamples, sizeof(double) * numSamples);

    // Start measuring time
    cudaEventRecord(start);

    // Generate normal samples on the GPU
    cudaError_t cudaStatus = CudaNormalSamples(deviceSamples);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // Stop measuring time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Calculate and display the elapsed time
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Elapsed Time: " << milliseconds << " ms" << std::endl;

    // Copy the generated samples from the device to the host
    cudaMemcpy(hostSamples, deviceSamples, sizeof(double) * numSamples, cudaMemcpyDeviceToHost);

    // Display the first few samples as a test
    const int numSamplesToShow = 10;
    std::cout << "Generated Samples:" << std::endl;
    for (int i = 0; i < numSamplesToShow; ++i) {
        std::cout << hostSamples[i] << " ";
    }
    std::cout << std::endl;

    // Clean up memory
    delete[] hostSamples;
    cudaFree(deviceSamples);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(end);






return EXIT_SUCCESS;


}