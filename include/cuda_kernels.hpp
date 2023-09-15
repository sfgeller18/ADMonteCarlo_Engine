#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "cuda_runtime.h"
#include "deviceMPFR.hpp"

__global__ void device_adder(DeviceMpfr* rp, DeviceMpfr* ap, DeviceMpfr* bp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int size = min(ap[idx].prec, bp[idx].prec);
        int cy = 0;

        for (int i = 0; i < size; i++) {
            unsigned long int a = ap[idx].mantissa;
            unsigned long int b = bp[idx].mantissa;
            unsigned long int r = a + cy;
            cy = (r < cy);
            r += b;
            cy += (r < b);
            rp[idx].mantissa = r;
        }

        // Synchronize threads within the block
        __syncthreads();
    }
}

#endif
