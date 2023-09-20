#ifndef CUDAMC_H
#define CUDAMC_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "deviceDualFuncs.cuh"
#include <cuRandSamples.cuh>
#include "cuda_runtime.h"

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

struct DualAsset {
    DualNumber<double> S0;
    DualNumber<double> K;
    DualNumber<double> r;
    DualNumber<double> y;
    DualNumber<double> T;
    DualNumber<double> sigma;
};



__device__ double optionPayoff(OptionType optionType, double stockPrice, double K) {
    if (optionType == VanillaCall) {
        double payout = max(stockPrice - K, 0.0);
        return payout;
    } 
    if (optionType == VanillaPut) {
        return max(K - stockPrice, 0.0);
    }
}

__device__ double stockPrice(double S0, double dt, double drift, double vol, double sample) {
    double temp = S0 * exp(drift * dt + sample * vol);
    return temp;
}

__global__ void OptionPricingMC(double* optionPrices, double* deviceSamples, Asset _asset, OptionType _optionType, int numSimulations, double dt, double drift, double vol) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < numSimulations) {
        double sampledPrice = stockPrice(_asset.S0, dt, drift, vol, deviceSamples[idx]);
        optionPrices[idx] = optionPayoff(_optionType, sampledPrice, _asset.K);
    }
}

void MonteCarloSimulator(double* optionPrices, OptionType _optionType, int numSimulations, const Asset& _asset) {
    double dt = _asset.T / 252;
    double drift = _asset.r - _asset.y - pow(_asset.sigma, 2) / 2;
    double vol = _asset.sigma * sqrt(dt);

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

__device__ void DualOptionPayoff(DualNumber<double>& _payoff, OptionType optionType, const DualNumber<double>& stockPrice, const DualNumber<double>& K) {
    if (optionType == VanillaCall) {
        double payoff_real = (stockPrice.real >= K.real) ? stockPrice.real - K.real : 0.0;
        _payoff.dual = (stockPrice.real >= K.real) ? stockPrice.dual * 1 : 0.0;
        _payoff.real = payoff_real;
    }
    if (optionType == VanillaPut) {
        _payoff.real = (stockPrice.real <= K.real) ? -stockPrice.dual * K.dual : 0.0;
        _payoff.dual = max(stockPrice.real - K.real, 0.0);
    }
}

__device__ void DualStockPrice(DualNumber<double>& stockPrice, DualNumber<double>& S0, DualNumber<double>& dt, DualNumber<double>& drift, DualNumber<double>& vol, double sample) {
    scalar_mul(&vol, sample, &stockPrice);
    dual_mul(&drift, &dt, &drift);
    dual_add(&stockPrice, &drift, &stockPrice);
    dual_exp(&stockPrice, &stockPrice);
    dual_mul(&stockPrice, &S0, &stockPrice);
}

__global__ void DualOptionPricingMC(DualNumber<double>* optionPrices, DualNumber<double>* stockPrices, double* deviceSamples, DualAsset _asset, OptionType _optionType, int numSimulations, DualNumber<double> dt, DualNumber<double> drift, DualNumber<double> vol) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < numSimulations) {
        DualStockPrice(stockPrices[idx], _asset.S0, dt, drift, vol, deviceSamples[idx]);
        DualOptionPayoff(optionPrices[idx], _optionType, stockPrices[idx], _asset.K);
    }
}


void DualMonteCarloSimulator(DualNumber<double>* optionPrices, OptionType _optionType, int numSimulations, const DualAsset& _asset) {
    DualNumber<double> dt(_asset.T.real / 252, _asset.T.dual);
    DualNumber<double> drift(_asset.r.real - _asset.y.real - pow(_asset.sigma.real, 2) / 2, 0);
    DualNumber<double> sqrt_dt(pow(dt.real, 0.5), dt.dual/(2*pow(dt.real, 0.5)));
    DualNumber<double> vol(_asset.sigma * sqrt_dt);
    DualNumber<double>* d_optionPrices = new DualNumber<double>[numSimulations];
    double* d_deviceSamples = new double[numSimulations];
    DualNumber<double>* stockPrices = new DualNumber<double>[numSimulations];

    // Allocate GPU memory for option prices
    cudaMalloc((void**)&d_optionPrices, numSimulations * sizeof(DualNumber<double>));
    cudaMalloc((void**)&d_deviceSamples, numSimulations * sizeof(double));
    cudaMalloc((void**)&stockPrices, numSimulations * sizeof(DualNumber<double>));

    // Set up the random number generator for deviceSamples using your existing CudaNormalSamples function
    CudaNormalSamples(d_deviceSamples);

    // Launch the OptionPricingMC kernel
    DualOptionPricingMC<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_optionPrices, stockPrices, d_deviceSamples, _asset, _optionType, numSimulations, dt, drift, vol);
    cudaMemcpy(optionPrices, d_optionPrices, numSimulations * sizeof(DualNumber<double>), cudaMemcpyDeviceToHost);

    // Clean up GPU memory
    cudaFree(stockPrices);
    cudaFree(d_optionPrices);
    cudaFree(d_deviceSamples);
}
   


#endif