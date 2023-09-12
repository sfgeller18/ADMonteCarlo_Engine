#include <iostream>
#include <StochasticProcess.h>
#include <HestonProcess.h>
#include <mpfr.h>
#include <fstream>
#include <unistd.h> 
#include "HyperComplex.hpp"
#include "ProcessGrapher.hpp"


// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    
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
mpfr_clear(x);  // Clear x

mpfr_t initialPosition;
mpfr_t kappa;
mpfr_t theta;
mpfr_t initVol;
mpfr_t zeta;
mpfr_t mu;
mpfr_t timeStep;

mpfr_init2(initialPosition, 12);  // Initialize to 128-bit precision
mpfr_init2(kappa, 12);
mpfr_init2(theta, 12);
mpfr_init2(initVol, 12);
mpfr_init2(zeta, 12);
mpfr_init2(mu, 12);
mpfr_init2(timeStep, 12);

mpfr_set_d(initialPosition, 100.0, MPFR_RNDN);
mpfr_set_d(kappa, 1.0, MPFR_RNDN);
mpfr_set_d(theta, 0.2, MPFR_RNDN);
mpfr_set_d(initVol, 0.05, MPFR_RNDN);
mpfr_set_d(zeta, 0.3, MPFR_RNDN);
mpfr_set_d(mu, 0.05, MPFR_RNDN);
mpfr_set_d(timeStep, 0.01, MPFR_RNDN);
int numSteps = 100;

HestonProcess process(initialPosition, mu, initVol, kappa, theta, zeta);

std::cout << "Initial HestonProcess Properties:" << std::endl;
process.printProperties();
std::cout << "-----------------------------" << std::endl;

printHestonEvolution(process, timeStep, numSteps, "time_evolution.csv");

// Clean up
mpfr_clear(initialPosition);
mpfr_clear(kappa);
mpfr_clear(theta);
mpfr_clear(initVol);
mpfr_clear(zeta);
mpfr_clear(mu);
mpfr_clear(timeStep);

std::cout << "Final HestonProcess Properties:" << std::endl;
process.printProperties();
std::cout << "-----------------------------" << std::endl;

return EXIT_SUCCESS;


}
