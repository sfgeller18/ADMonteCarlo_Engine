#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


#define NUM_BLOCKS 1028
#define NUM_THREADS_PER_BLOCK 128
constexpr auto NUM_SAMPLES_PER_THREAD = 1;
#define KERNEL_ITERATIONS 5

// Function to generate a random number from a standard normal distribution using the Box-Muller transform
__device__ double normalRandGen(curandState* state) {
    double u1 = curand_uniform(state);
    double u2 = curand_uniform(state);
    double rand_std_normal = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return rand_std_normal;
}

__global__ void setup_kernel(curandState* states, unsigned long long seed, unsigned long long sequence, time_t* deviceTime)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Retrieve the timestamp from deviceTime
    time_t currentTime = *deviceTime;

    // Calculate the unique seed incorporating the timestamp
    unsigned long long uniqueSeed = seed + id * 12345ULL + currentTime;

    // Initialize the random number generator state
    curand_init(uniqueSeed, id, sequence, &states[id]);
}

__global__ void generate_normal_samples(double* samples, curandState* states)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = states[tid];
    double2 random_numbers;

    for (int i = 0; i < NUM_SAMPLES_PER_THREAD * 2; i += 2) {
        // Generate two independent standard normal random numbers using Box-Muller
        random_numbers.x = normalRandGen(&localState);
        random_numbers.y = normalRandGen(&localState);

        // Store the generated random numbers
        int sampleIdx = tid * NUM_SAMPLES_PER_THREAD * 2 + i;
        samples[sampleIdx] = random_numbers.x;
        samples[sampleIdx + 1] = random_numbers.y;
    }

    // Update the state for the thread
    states[tid] = localState;
}



cudaError_t CudaNormalSamples(double* device_samples)
{
    curandState* states;
    cudaMalloc(&states, NUM_BLOCKS * NUM_THREADS_PER_BLOCK * sizeof(curandState));

    // Initialize random number generator states
    unsigned long long seed = 1234ULL;
    unsigned long long sequence = 0ULL;
    time_t currentTime;
    time(&currentTime);

    // Copy the timestamp to the device
    time_t* deviceTime;
    cudaMalloc((void**)&deviceTime, sizeof(time_t));
    cudaMemcpy(deviceTime, &currentTime, sizeof(time_t), cudaMemcpyHostToDevice);

    setup_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(states, seed, sequence, deviceTime);

    // Launch kernel to generate normal samples
    generate_normal_samples<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(device_samples, states);

    cudaFree(deviceTime);
    cudaFree(states);

    return cudaGetLastError();
}

// Kernel to calculate the sum of an array
template <typename T>
__global__ void GPUmsdArray(const T* array, const int numSamples, T& mean, T& sdev) {
    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize the shared memory with zero
    sdata[tid] = 0;

    // Calculate the local sum and local squared sum for this thread
    T localSum = 0;
    T localSquaredSum = 0;
    while (idx < numSamples) {
        localSum += array[idx];
        localSquaredSum += array[idx] * array[idx];
        idx += blockDim.x * gridDim.x;
    }

    // Store the local sum and local squared sum in shared memory
    sdata[tid] = localSum;
    sdata[blockDim.x + tid] = localSquaredSum;

    // Synchronize threads within the block
    __syncthreads();

    // Perform parallel reduction on shared memory for both sum and squared sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    // Only the first thread in the block will store the results
    if (tid == 0) {
        mean = sdata[0] / numSamples;
        T variance = (sdata[blockDim.x] / numSamples) - pow(mean, 2);
        sdev = sqrt(variance);
    }
}

template <typename T>
void CPUmsdArray(const T* array, const int numSamples, T& mean, T& sdev) {
    if (numSamples <= 0) {
        std::cerr << "Invalid number of samples." << std::endl;
        return;
    }

    // Calculate the mean
    for (int i = 0; i < numSamples; ++i) {
        mean += array[i]/numSamples;
    }

    // Calculate the standard deviation (sdev)
    sdev = 0.0;
    for (int i = 0; i < numSamples; ++i) {
        T deviation = array[i] - mean;
        sdev += pow((deviation -mean), 2)/numSamples;
    }
    sdev = std::sqrt(sdev);
}


// Kernel to calculate the sum of squared differences from the mean

#endif