#include <iostream>
#include <StochasticProcess.h>
#include <HestonProcess.h>
#include "ProcessGrapher.hpp"
#include <unistd.h> 
#include "duals.hpp"
#include "deviceDualFuncs.cuh"
#include "cuRandSamples.cuh"
#include "deviceMPFR.hpp"
#include "cuMonteCarlo.hpp"

# define precision 12





int main() {

// Define the option and asset
OptionType optionType = VanillaCall;
DualAsset asset;
asset.S0 = DualNumber<double>(3975, 1);  // Initial stock price with dual part initialized to 0
asset.K = DualNumber<double>(3975, 0);   // Strike price with dual part initialized to 0
asset.r = DualNumber<double>(0.048, 0);  // Risk-free interest rate with dual part initialized to 0
asset.y = DualNumber<double>(0.015, 0);  // Dividend yield with dual part initialized to 0
asset.T = DualNumber<double>(252, 0);    // Time to expiration in days (252d=1y) with dual part initialized to 0
asset.sigma = DualNumber<double>(0.2, 0); // Volatility with dual part initialized to 0

// Define the number of simulations
size_t numSimulations = NUM_BLOCKS * NUM_THREADS_PER_BLOCK * NUM_SAMPLES_PER_THREAD;

// Create an array of DualNumber<double> to store option prices
DualNumber<double>* optionPrices = new DualNumber<double>[numSimulations];

// Initialize optionPrices with real values and dual parts set to 0
for (size_t i = 0; i < numSimulations; ++i) {
    optionPrices[i] = DualNumber<double>(0.0, 0.0);
}

// Call the Monte Carlo simulator
DualMonteCarloSimulator(optionPrices, optionType, numSimulations, asset);

// Print the first 10 option prices and their mean
std::cout << "First 10 Option Prices: ";
for (int i = 0; i < std::min(10, static_cast<int>(numSimulations)); ++i) {
    std::cout << optionPrices[i].real << " "; // Print the real part
}
std::cout << std::endl;

// Compute the mean and variance
DualNumber<double> mean;
for (size_t i = 0; i < numSimulations; ++i) {
    mean = mean + optionPrices[i];
}
mean = mean / static_cast<double>(numSimulations);

std::cout << "Price is: " << exp(-asset.r.real * asset.T.real / 252) * mean.real << std::endl;
std::cout << "Delta is: " << exp(-asset.r.real * asset.T.real / 252) * mean.dual << std::endl;

return EXIT_SUCCESS;


}