#include <iostream>
#include <StochasticProcess.h>
#include <HestonProcess.h>
#include <mpfr.h>
#include <fstream>
#include <unistd.h> 
#include "HyperComplex.hpp"
#include "ProcessGrapher.hpp"
#include "cuda_helpers.hpp"



// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    std::cout << "NVCC C++ Standard Version: " << __cplusplus << std::endl;
    mpfr_t x;
    mpfr_init2(x, 128);

    // Set the value of x to 2.0
    mpfr_set_d(x, 2.0, MPFR_RNDN);

    // Calculate the square root of x
    mpfr_sqrt(x, x, MPFR_RNDN);

    // Print the result
    mpfr_exp_t exp;
    char* str = mpfr_get_str(NULL, &exp, 10, 0, x, MPFR_RNDN);
    std::cout << "Square root of 2.0 is approximately " << str << " x 2^" << exp << std::endl;

    // Clean up
    mpfr_free_str(str);
    mpfr_clear(x);

    double initialPosition = 100.0;
    double kappa = 1.0;
    double theta = 0.2;
    double initVol = 0.05;
    double zeta = 0.3;
    double mu = 0.05;
    double timeStep = 0.01;
    int numSteps = 100;

    HestonProcess process(initialPosition, mu, initVol, kappa, theta, zeta);

    std::cout << "Initial HestonProcess Properties:" << std::endl;
    process.printProperties();
    std::cout << "-----------------------------" << std::endl;

    printHestonEvolution(process, timeStep, numSteps, "time_evolution.csv");    
    return EXIT_SUCCESS;

}
