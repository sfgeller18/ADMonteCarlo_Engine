#ifndef MPFR_HELPERS_H
#define MPFR_HELPERS_H

#include <mpfr.h>
#include <iostream>
#include <string>
#include <assert.h>

std::string mpfrToString(const mpfr_t& value, int prec) {
    char* decimalStr = nullptr;

    // Determine the size of the buffer needed
    size_t size = mpfr_snprintf(NULL, 0, "%.*Rf", prec, value);

    // Allocate memory for the buffer
    decimalStr = (char*)malloc(size + 1); // +1 for null terminator

    // Format the mpfr variable to the buffer with the desired precision
    mpfr_snprintf(decimalStr, size + 1, "%.*Rf", prec, value);

    std::string result(decimalStr);

    // Free the allocated memory
    free(decimalStr);

    return result;
}

void mpfr_get_mantissa(const mpfr_t& src, mpz_t mantissa) {
    assert(src != NULL);
    mpz_t rop;
    mpz_init(rop);  // Initialize the mpz_t object
    mpfr_exp_t exp = mpfr_get_z_2exp(rop, src);  // Get mantissa and exponent
    mpz_set(mantissa, rop);
    mpz_clear(rop);
}



#endif