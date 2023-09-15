#include <iostream>
#include <StochasticProcess.h>
#include <HestonProcess.h>
#include <mpfr.h>
#include <fstream>
#include <unistd.h> 
#include "duals.hpp"
#include "ProcessGrapher.hpp"
#include "cuda_runtime.h"


// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    mpfr_t myFloat;
    mpfr_init_set_d(myFloat, 1.0, MPFR_RNDN);  // Replace 123.456 with your desired float
    mpfr_prec_t prec = mpfr_get_prec(myFloat);
    std::cout<<"Decimal"<<mpfrToString(myFloat, prec)<<std::endl;
    mpz_t mantissa;
    mpz_init(mantissa);
    // Initialize DeviceMpfr object and convert mpfr_t to DeviceMpfr
    DeviceMpfr deviceVar(myFloat);
    mpfr_get_mantissa(myFloat, mantissa);
    std::cout << "Mantissa: " << mpz_get_str(nullptr, 10, mantissa) << std::endl;

    // Convert DeviceMpfr back to mpfr_t
    mpfr_t myFloatConverted;
    mpfr_init(myFloatConverted);
    deviceVar.deviceMpfrToMpfr(myFloatConverted);

    mpfr_get_mantissa(myFloat, mantissa);
    std::cout << "Mantissa (Converted): " << mpz_get_str(nullptr, 10, mantissa) << std::endl;

    // Print the mantissa of the converted mpfr_t
    mpfr_clear(myFloatConverted);
    mpfr_clear(myFloat);
    mpz_clear(mantissa);

mpfr_t x;
mpfr_init2(x, 128);
mpfr_set_d(x, 2.0, MPFR_RNDN);
mpfr_sqrt(x, x, MPFR_RNDN);
std::cout << "Square root of 2.0 is approximately " << mpfrToString(x, 128) << std::endl;
mpfr_clear(x);  // Clear x

    mpfr_t a, b;
    mpfr_init_set_d(a, 2.0, MPFR_RNDN); // Real part
    mpfr_init_set_d(b, 1.0, MPFR_RNDN); // Dual part

    DualNumber dual_a(a, b);

    DualNumber dual_b;
    mpfr_set_d(b, 3.0, MPFR_RNDN); // Change the dual part
    dual_b.setReal(a);
    dual_b.setDual(b);

    DualNumber result = dual_a * dual_b;

    std::cout << "Result:" << std::endl;
    result.print();

    mpfr_clear(a);
    mpfr_clear(b);

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
int numSteps = 253;

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