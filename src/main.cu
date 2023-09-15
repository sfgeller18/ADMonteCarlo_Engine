#include <iostream>
#include <StochasticProcess.h>
#include <HestonProcess.h>
#include <mpfr.h>
#include <fstream>
#include <unistd.h> 
#include "duals.hpp"
#include "ProcessGrapher.hpp"
#include "cuda_kernels.hpp"
#include "deviceMPFR.hpp"


// CUDA kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    cudaSetDevice(0);

    const int n = 10; // Number of DeviceMpfr objects

    // Allocate memory on the host and device for DeviceMpfr objects
    DeviceMpfr* host_ap = new DeviceMpfr[n];
    DeviceMpfr* host_bp = new DeviceMpfr[n];
    DeviceMpfr* host_rp = new DeviceMpfr[n];
    DeviceMpfr* device_ap;
    DeviceMpfr* device_bp;
    DeviceMpfr* device_rp;

    // Initialize DeviceMpfr objects on the host
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < n; i++) {
        host_ap[i].prec = 64; // Set precision
        host_ap[i].mantissa = rand(); // Set a random mantissa
        host_bp[i].prec = 64; // Set precision
        host_bp[i].mantissa = rand(); // Set a random mantissa
    }

    // Allocate memory on the device
    cudaMalloc((void**)&device_ap, n * sizeof(DeviceMpfr));
    cudaMalloc((void**)&device_bp, n * sizeof(DeviceMpfr));
    cudaMalloc((void**)&device_rp, n * sizeof(DeviceMpfr));

    // Copy DeviceMpfr objects from host to device
    cudaMemcpy(device_ap, host_ap, n * sizeof(DeviceMpfr), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bp, host_bp, n * sizeof(DeviceMpfr), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    device_adder<<<numBlocks, blockSize>>>(device_rp, device_ap, device_bp, n);

    // Copy results back from device to host
    cudaMemcpy(host_rp, device_rp, n * sizeof(DeviceMpfr), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < n; i++) {
        std::cout << "Result " << i << ": " << host_rp[i].mantissa << std::endl;
    }

    // Clean up
    cudaFree(device_ap);
    cudaFree(device_bp);
    cudaFree(device_rp);
    delete[] host_ap;
    delete[] host_bp;
    delete[] host_rp;

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