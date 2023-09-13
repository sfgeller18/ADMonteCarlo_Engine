#ifndef MPFR_HELPERS_H
#define MPFR_HELPERS_H

#include <mpfr.h>
#include <iostream>
#include <string>

std::string mpfrToString(const mpfr_t& value, int precision) {
    mpfr_exp_t exponent;
    char* decimalStr = nullptr;

    // Determine the size of the buffer needed
    size_t size = mpfr_snprintf(NULL, 0, "%.*Rf", precision, value);

    // Allocate memory for the buffer
    decimalStr = (char*)malloc(size + 1); // +1 for null terminator

    // Format the mpfr variable to the buffer with the desired precision
    mpfr_snprintf(decimalStr, size + 1, "%.*Rf", precision, value);

    std::string result(decimalStr);

    // Free the allocated memory
    free(decimalStr);

    return result;
}


#endif