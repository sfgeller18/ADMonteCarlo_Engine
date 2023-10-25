#ifndef cuda_helpers_H
#define cuda_helpers_H

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
    const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(T (*bound_function)(cudaStream_t), cudaStream_t stream, int num_repeats = 100, int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

__global__ void reduceKernel(float* d_vec, float* d_vec_2, int n) {
    extern __shared__ float mem[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    mem[threadIdx.x] = (id < n) ? d_vec[id] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mem[threadIdx.x] += mem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_vec_2[blockIdx.x] = mem[0];
    }
}

void arr_avg(float* d_vec, int n, float* sum) {
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float* d_vec_2;
    CHECK_CUDA_ERROR(cudaMalloc(&d_vec_2, blocks * sizeof(float)));

    reduceKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_vec, d_vec_2, n);

    float* h_ptr;
    unsigned long long m = blocks * sizeof(float);
    CHECK_CUDA_ERROR(cudaMallocHost(&h_ptr, m));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_ptr, d_vec_2, m, cudaMemcpyDeviceToHost));

    *sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        *sum += h_ptr[i];
    }
    *sum /= n;

    CHECK_CUDA_ERROR(cudaFreeHost(h_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_vec_2));
}

__global__ void varKernel(float* d_vec, float* d_vec_2, int n, float* mu) {
    extern __shared__ float mem[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    mem[threadIdx.x] = (id < n) ? (d_vec[id] - *mu) * (d_vec[id] - *mu) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mem[threadIdx.x] += mem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_vec_2[blockIdx.x] = mem[0];
    }
}

void arr_var(float* d_vec, int n, float* var, float* mu) {
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float* d_vec_2;
    CHECK_CUDA_ERROR(cudaMalloc(&d_vec_2, blocks * sizeof(float)));

    varKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_vec, d_vec_2, n, mu);

    float* h_ptr;
    unsigned long long m = blocks * sizeof(float);
    CHECK_CUDA_ERROR(cudaMallocHost(&h_ptr, m));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_ptr, d_vec_2, m, cudaMemcpyDeviceToHost));

    *var = 0.0f;
    for (int i = 0; i < blocks; i++) {
        *var += h_ptr[i];
    }
    *var /= n;

    CHECK_CUDA_ERROR(cudaFreeHost(h_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_vec_2));
}

__global__ void corKernel(float* d_vec_1, float* d_vec_2, float* d_vec_3, int n, float* mu1, float* mu2) {
    extern __shared__ float mem[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    mem[threadIdx.x] = (id < n) ? (d_vec_1[id] - *mu1) * (d_vec_2[id] - *mu2) : 0.0f;
    mem[256 + threadIdx.x] = (id < n) ? d_vec_2[id] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mem[threadIdx.x] += mem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_vec_3[blockIdx.x] = mem[0];
    }
}

void arr_cor(float* d_vec_1, float* d_vec_2, int n, float* cor, float* mu1, float* mu2, float* sigma1, float* sigma2) {
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float* d_vec_3;
    CHECK_CUDA_ERROR(cudaMalloc(&d_vec_3, blocks * sizeof(float)));

    corKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float) * 2>>>(d_vec_1, d_vec_2, d_vec_3, n, mu1, mu2);

    float* h_ptr;
    unsigned long long m = blocks * sizeof(float);
    CHECK_CUDA_ERROR(cudaMallocHost(&h_ptr, m));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_ptr, d_vec_3, m, cudaMemcpyDeviceToHost));

    *cor = 0.0f;
    for (int i = 0; i < blocks; i++) {
        *cor += h_ptr[i];
    }
    *cor /= (256 * blocks * sqrt((*sigma1) * (*sigma2)));

    CHECK_CUDA_ERROR(cudaFreeHost(h_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_vec_3));
}
	
float host_arr_avg(float* d_vec, int n) {
    int blocks = ceil((n + 255) / 256);
    float sum{0.0f};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void (*avgFunction)(float*, int, float*);
    avgFunction = &arr_avg;

    avgFunction(d_vec, n, &sum);

    cudaStreamDestroy(stream);
    return sum;
}

float host_arr_var(float* d_vec, int n, float* mu) {
    int blocks = ceil((n + 255) / 256);
    float var{0.0f};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void (*varFunction)(float*, int, float*, float*);
    varFunction = &arr_var;

    varFunction(d_vec, n, &var, mu);

    cudaStreamDestroy(stream);
    return var;
}

float host_arr_cor(float* d_vec_1, float* d_vec_2, int n, float *mu1, float *mu2, float* sigma1, float* sigma2) {
    int blocks = ceil((n + 255) / 256);
    float cor{0.0f};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void (*corFunction)(float*, float*, int, float*, float*, float*, float*, float*);
    corFunction = &arr_cor;

    corFunction(d_vec_1, d_vec_2, blocks, &cor, mu1, mu2, sigma1, sigma2);

    cudaStreamDestroy(stream);
    return cor;
};




#endif