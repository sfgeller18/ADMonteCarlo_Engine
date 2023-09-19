#ifndef DEVICE_DUAL_FUNCS_H
#define DEVICE_DUAL_FUNCS_H

# include "duals.hpp"
#include <cuda_runtime.h>

__device__ void dual_pow(DualNumber<double> input, DualNumber<double> output, double exponent) {
        output.real = pow(input.real, exponent);
        if (exponent != 1.0) {output.dual = exponent * pow(input.real, exponent - 1.0) * input.dual;}
        else {output.dual = input.dual;}
        }

__device__ void dual_add(const DualNumber<double>* a, const DualNumber<double>* b, DualNumber<double>* result) {
    result->real = a->real + b->real;
    result->dual = a->dual + b->dual;
}

__device__ void dual_sub(const DualNumber<double>* a, const DualNumber<double>* b, DualNumber<double>* result) {
    result->real = a->real - b->real;
    result->dual = a->dual - b->dual;
}

__device__ void dual_mul(const DualNumber<double>* a, const DualNumber<double>* b, DualNumber<double>* result) {
    result->real = a->real * b->real;
    result->dual = a->real * b->dual + a->dual * b->real;
}

__device__ void scalar_mul(const DualNumber<double>* a, const double& b, DualNumber<double>* result) {
    result->real = a->real * b;
    result->dual = a->dual * b;
}

__device__ void dual_div(const DualNumber<double>* a, const DualNumber<double>* b, DualNumber<double>* result) {
    double b_real_squared = b->real * b->real;
    result->real = a->real / b->real;
    result->dual = (a->dual * b->real - a->real * b->dual) / b_real_squared;
}

__device__ void dual_sin(const DualNumber<double>* input, DualNumber<double>* output) {
    output->real = sin(input->real);
    output->dual = cos(input->real) * input->dual;
}

__device__ void dual_exp(const DualNumber<double>* input, DualNumber<double>* output) {
    output->real = exp(input->real);
    output->dual = exp(input->real) * input->dual;
}

__device__ void dual_cos(const DualNumber<double>* input, DualNumber<double>* output) {
    output->real = cos(input->real);
    output->dual = -sin(input->real) * input->dual;
}

__device__ void dual_tan(const DualNumber<double>* input, DualNumber<double>* output) {
    double cos_x = cos(input->real);
    output->real = tan(input->real);
    output->dual = input->dual / (cos_x * cos_x);
}

__device__ void dual_acos(const DualNumber<double>* input, DualNumber<double>* output) {
    double one_minus_x_squared = 1.0 - input->real * input->real;
    double sqrt_term = sqrt(one_minus_x_squared);

    output->real = acos(input->real);
    output->dual = -input->dual / sqrt_term;
}

__device__ void dual_asin(const DualNumber<double>* input, DualNumber<double>* output) {
    double one_minus_x_squared = 1.0 - input->real * input->real;
    double sqrt_term = sqrt(one_minus_x_squared);

    output->real = asin(input->real);
    output->dual = input->dual / sqrt_term;
}

__device__ void dual_atan(const DualNumber<double>* input, DualNumber<double>* output) {
    double one_plus_x_squared = 1.0 + input->real * input->real;

    output->real = atan(input->real);
    output->dual = input->dual / one_plus_x_squared;
}

#endif