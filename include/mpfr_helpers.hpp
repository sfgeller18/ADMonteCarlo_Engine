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

class DeviceMpfr {
public:
    // Default constructor
    DeviceMpfr() {
        mpfr_prec = 12;
        mpfr_exp = 1;
        mpz_init(mpfr_d);
    }

    // Constructor with custom precision and exponent
    DeviceMpfr(mpfr_prec_t _precision, mpfr_exp_t _exponent) {
        mpfr_prec = _precision;
        mpfr_exp = _exponent;  
        mpz_init(mpfr_d);
    }

    DeviceMpfr(const mpfr_t& src) {
        this->setPrecision(mpfr_get_prec(src));
        this->setExponent(mpfr_get_exp(src));
        mpz_t mantissa;
        mpz_init(mantissa);
        mpfr_get_mantissa(src, mantissa);
        this->setMantissa(mantissa);
        mpz_clear(mantissa);
    }

    // Getter for precision
    mpfr_prec_t getPrecision() const {
        return mpfr_prec;
    }

    // Setter for precision
    void setPrecision(mpfr_prec_t _precision) {
        mpfr_prec = _precision;
    }

    // Getter for exponent
    mpfr_exp_t getExponent() const {
        return mpfr_exp;
    }

    // Setter for exponent
    void setExponent(mpfr_exp_t _exponent) {
        mpfr_exp = _exponent;
    }
    
    void setMantissa(mpz_t _mantissa) {
        mpz_set(mpfr_d, _mantissa);
    }

    // Destructor to release resources
    ~DeviceMpfr() {
        mpz_clear(mpfr_d);
    }

void deviceMpfrToMpfr(mpfr_t& result) const {
    // Initialize mpfr_t with the precision of the DeviceMpfr object
    mpfr_set_prec(result, this->mpfr_prec);
    mpfr_set_exp(result, this->mpfr_exp);
    mpfr_set_z(result, this->mpfr_d, MPFR_RNDN);
}

private:
    mpfr_prec_t mpfr_prec;
    mpfr_exp_t mpfr_exp;
    mpz_t mpfr_d;
};





#endif