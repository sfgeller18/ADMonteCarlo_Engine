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


#define NUM_BLOCKS 64
#define NUM_THREADS_PER_BLOCK 32
constexpr auto NUM_SAMPLES_PER_THREAD = 4000;
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

    for (int i = 0; i < NUM_SAMPLES_PER_THREAD; i += 2) {
        // Generate two independent standard normal random numbers using Box-Muller
        random_numbers.x = normalRandGen(&localState);
        random_numbers.y = normalRandGen(&localState);

        // Store the generated random numbers
        int sampleIdx = tid * NUM_SAMPLES_PER_THREAD + i;
        samples[sampleIdx] = random_numbers.x;
        samples[sampleIdx + 1] = random_numbers.y;
    }

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

#endif