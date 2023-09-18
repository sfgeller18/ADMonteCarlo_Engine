#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuRandSamples.cuh>
#include "cuda_runtime.h"
#include "duals.hpp"

enum OptionType {
    VanillaCall,
    VanillaPut
};

struct Asset {
    double S0;
    double K;
    double r;
    double y;
    double T;
    double sigma;
};

template <typename N>
struct DualAsset {
    DualNumber<N> S0;
    DualNumber<N> K;
    DualNumber<N> r;
    DualNumber<N> y;
    DualNumber<N> T;
    DualNumber<N> sigma;
};

template <typename T>
__device__ T Max(T a, T b) {
    return (a > b) ? a : b;
}

template <typename T>
__device__ T Min(T a, T b) {
    return (a < b) ? a : b;
}

__device__ double optionPayoff(OptionType optionType, double stockPrice, double K){
    if (optionType == VanillaCall) {double payout = Max(stockPrice - K, 0.0); return payout;} 
    if (optionType == VanillaPut) {return Max(K - stockPrice, 0.0);}
}

template <typename T>
__device__ void DualOptionPayoff(DualNumber<T> payoff, OptionType optionType, DualNumber<T> stockPrice, DualNumber<T> K) {
    if (optionType == VanillaCall) {
        payoff.real = (stockPrice.real >= K.real) ? stockPrice.dual*K.dual : 0.0;
        payoff.dual = Max(stockPrice.real - K.real, 0.0);
        }
    if (optionType == VanillaPut) {
        payoff.real = (stockPrice.real <= K.real) ? -stockPrice.dual*K.dual : 0.0;
        payoff.dual = Max(stockPrice.real - K.real, 0.0);
        }
}

__device__ double stockPrice(double S0, double dt, double drift, double vol, double sample) {
    double temp = S0 * exp(drift * dt + sample * vol);
    return temp;
}

template <typename T>
__device__ void  DualStockPrice(DualNumber<T> stockPrice, DualNumber<T> S0, DualNumber<T> dt, DualNumber<T> drift, DualNumber<T> vol, T sample) {
    scalar_mul(&vol, sample, &stockPrice);
    dual_mul(&drift, &dt, &drift);
    dual_add(&stockPrice, &drift, &stockPrice);
    dual_exp(&stockPrice, &stockPrice);
    dual_mul(&stockPrice, &S0, &stockPrice);
}

__global__ void OptionPricingMC(double* optionPrices, double* deviceSamples, Asset _asset, OptionType _optionType, int numSimulations, double dt, double drift, double vol) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < numSimulations) {
        double sampledPrice = stockPrice(_asset.S0, dt, drift, vol, deviceSamples[idx]);
        optionPrices[idx] = optionPayoff(_optionType, sampledPrice, _asset.K);
    }
}

//This can be parallelized in a smarter way but this is just a naive implementation for now
template <typename T>
__global__ void DualOptionPricingMC(DualNumber<T>* optionPrices, DualNumber<T>* stockPrices, T* deviceSamples, DualAsset<T> _asset, OptionType _optionType, int numSimulations, DualNumber<T> dt, DualNumber<T> drift, DualNumber<T> vol) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < numSimulations) {
        DualStockPrice(stockPrices[idx], _asset.S0, dt, drift, vol, deviceSamples[idx]);
        DualOptionPayoff(optionPrices[idx], _optionType, stockPrices[idx], _asset.K);
    }
}

void MonteCarloSimulator(double* optionPrices, OptionType _optionType, int numSimulations, const Asset& _asset)
{
    double dt = _asset.T/252;
    double drift = _asset.r - _asset.y - pow(_asset.sigma, 2)/2;
    double vol = _asset.sigma*sqrt(dt);

    double* d_optionPrices = nullptr;
    double* d_deviceSamples = nullptr;


    // Allocate GPU memory for option prices
    cudaMalloc((void**)&d_optionPrices, numSimulations * sizeof(double));
    cudaMalloc((void**)&d_deviceSamples, numSimulations * sizeof(double));

    // Set up the random number generator for deviceSamples using your existing CudaNormalSamples function
    CudaNormalSamples(d_deviceSamples);

    // Launch the OptionPricingMC kernel
    OptionPricingMC<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_optionPrices, d_deviceSamples, _asset, _optionType, numSimulations, dt, drift, vol);

    cudaMemcpy(optionPrices, d_optionPrices, numSimulations * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(d_optionPrices);
    cudaFree(d_deviceSamples);
}

template <typename T>
void DualMonteCarloSimulator(DualNumber<T>* optionPrices, OptionType _optionType, int numSimulations, const DualAsset<T>& _asset)
{
    DualNumber<T> dt(_asset.T.real/252, 0);
    DualNumber<T>drift(_asset.r.real - _asset.y.real - pow(_asset.sigma.real, 2)/2, 0);
    T sqrt_dt = sqrt(dt.real);
    DualNumber<T>vol(_asset.sigma.real*sqrt_dt, 0);

    DualNumber<T>* d_optionPrices = nullptr;
    T* d_deviceSamples = nullptr;
    DualNumber<T>* stockPrices = nullptr;

    // Allocate GPU memory for option prices
    cudaMalloc((void**)&d_optionPrices, numSimulations * sizeof(DualNumber<T>));
    cudaMalloc((void**)&d_deviceSamples, numSimulations * sizeof(T));
    cudaMalloc((void**)&stockPrices, numSimulations * sizeof(DualNumber<T>));


    // Set up the random number generator for deviceSamples using your existing CudaNormalSamples function
    CudaNormalSamples(d_deviceSamples);

    // Launch the OptionPricingMC kernel
    DualOptionPricingMC<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_optionPrices, stockPrices, d_deviceSamples, _asset, _optionType, numSimulations, dt, drift, vol);

    cudaMemcpy(optionPrices, d_optionPrices, numSimulations * sizeof(DualNumber<T>), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(stockPrices);
    cudaFree(d_optionPrices);
    cudaFree(d_deviceSamples);
}