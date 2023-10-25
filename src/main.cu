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

OptionType optionType = VanillaCall;
DualAsset asset;
asset.S0 = DualNumber<double>(100, 1);  // Initial stock price with dual part initialized to 1 (for option delta calculation)
asset.K = DualNumber<double>(100, 0);   // Strike price with dual part initialized to 0
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
DualNumber<double> result = DualMonteCarloSimulator(optionPrices, optionType, numSimulations, asset);

std::cout << "Price is: " << result.real << std::endl;
std::cout << "Delta is: " << result.dual << std::endl;

return EXIT_SUCCESS;
}