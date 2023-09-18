#ifndef DEVICE_DUAL_FUNCS_H
#define DEVICE_DUAL_FUNCS_H

# include "duals.hpp"
#include <cuda_runtime.h>

template <typename T>
__device__ void dual_pow(DualNumber<T>* input, DualNumber<T>* output, double exponent, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        output[idx].real = pow(input[idx].real, exponent);
        if (exponent != (T)1.0) {
        output[idx].dual = exponent * pow(input[idx].real, exponent-(T)1.0) * input[idx].dual;
        }
    }
}

template <typename T>
__device__ void dual_add(const DualNumber<T>* a, const DualNumber<T>* b, DualNumber<T>* result, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        result[idx].real = a[idx].real + b[idx].real;
        result[idx].dual = a[idx].dual + b[idx].dual;
    }
}

template <typename T>
__device__ void dual_sub(const DualNumber<T>* a, const DualNumber<T>* b, DualNumber<T>* result, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        result[idx].real = a[idx].real - b[idx].real;
        result[idx].dual = a[idx].dual - b[idx].dual;
    }
}

template <typename T>
__device__ void dual_mul(const DualNumber<T>* a, const DualNumber<T>* b, DualNumber<T>* result, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        result[idx].real = a[idx].real * b[idx].real;
        result[idx].dual = a[idx].real * b[idx].dual + a[idx].dual * b[idx].real;
    }
}

template <typename T>
__device__ void scalar_mul(const DualNumber<T>* a, const T& b, DualNumber<T>* result, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        result[idx].real = a[idx].real * b;
        result[idx].dual = a[idx].dual * b;
    }
}

template <typename T>
__device__ void dual_div(const DualNumber<T>* a, const DualNumber<T>* b, DualNumber<T>* result, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        T b_real_squared = b[idx].real * b[idx].real;
        result[idx].real = a[idx].real / b[idx].real;
        result[idx].dual = (a[idx].dual * b[idx].real - a[idx].real * b[idx].dual) / b_real_squared;
    }
}

template <typename T>
__device__ void dual_sin(const DualNumber<T>* input, DualNumber<T>* output, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        output[idx].real = sin(input[idx].real);
        output[idx].dual = cos(input[idx].real) * input[idx].dual;
    }
}

template <typename T>
__device__ void dual_exp(const DualNumber<T>* input, DualNumber<T>* output, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        output[idx].real = exp(input[idx].real);
        output[idx].dual = exp(input[idx].real) * input[idx].dual;
    }
}


template <typename T>
__device__ void dual_cos(const DualNumber<T>* input, DualNumber<T>* output, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        output[idx].real = cos(input[idx].real);
        output[idx].dual = -sin(input[idx].real) * input[idx].dual;
    }
}

template <typename T>
__device__ void dual_tan(const DualNumber<T>* input, DualNumber<T>* output, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        T cos_x = cos(input[idx].real);
        output[idx].real = tan(input[idx].real);
        output[idx].dual = input[idx].dual / (cos_x * cos_x);
    }
}

template <typename T>
__device__ void dual_acos(const DualNumber<T>* input, DualNumber<T>* output, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        T one_minus_x_squared = 1.0 - input[idx].real * input[idx].real;
        T sqrt_term = sqrt(one_minus_x_squared);

        output[idx].real = acos(input[idx].real);
        output[idx].dual = -input[idx].dual / sqrt_term;
    }
}

template <typename T>
__device__ void dual_asin(const DualNumber<T>* input, DualNumber<T>* output, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        T one_minus_x_squared = 1.0 - input[idx].real * input[idx].real;
        T sqrt_term = sqrt(one_minus_x_squared);

        output[idx].real = asin(input[idx].real);
        output[idx].dual = input[idx].dual / sqrt_term;
    }
}

template <typename T>
__device__ void dual_atan(const DualNumber<T>* input, DualNumber<T>* output, int numElements = 1) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numElements) {
        T one_plus_x_squared = 1.0 + input[idx].real * input[idx].real;

        output[idx].real = atan(input[idx].real);
        output[idx].dual = input[idx].dual / one_plus_x_squared;
    }
}

#endif