#ifndef DEVICE_MPFR_H
#define DEVICE_MPFR_H

#include "mpfr_helpers.hpp"

#define precision 12

class DeviceMpfr {
public:
    // Default constructor
    DeviceMpfr() {
        prec = precision; // Changed from mpfr_prec to precision
        exponent = 1;  // Changed from mpfr_exp to exponent
        mantissa = 0;  // Changed from mpfr_d to mantissa; Initialize as zero
    }

    // Constructor with custom precision and exponent
    DeviceMpfr(mpfr_prec_t _precision, mpfr_exp_t _exponent) {
        prec = _precision; // Changed from mpfr_prec to precision
        exponent = _exponent;   // Changed from mpfr_exp to exponent
        mantissa = 0;          // Changed from mpfr_d to mantissa; Initialize as zero
    }

    DeviceMpfr(const mpfr_t& src) {
        this->setPrecision(mpfr_get_prec(src));
        this->setExponent(mpfr_get_exp(src));
        mpz_t mant;
        mpz_init(mant);
        mpfr_get_mantissa(src, mant);
        this->setMantissa(mpz_get_ui(mant));
        mpz_clear(mant);
    }

    std::string toString() {
        // Convert unsigned long int to string
        unsigned long int temp = mantissa*pow(2, exponent);
        return std::to_string(temp);
    }

    // ... (Other members and functions as defined earlier)

    // Getter for precision
    mpfr_prec_t getPrecision() const {
        return prec; // Changed from mpfr_prec to precision
    }

    // Setter for precision
    void setPrecision(mpfr_prec_t _precision) {
        prec = _precision; // Changed from mpfr_prec to precision
    }

    // Getter for exponent
    mpfr_exp_t getExponent() const {
        return exponent; // Changed from mpfr_exp to exponent
    }

    // Setter for exponent
    void setExponent(mpfr_exp_t _exponent) {
        exponent = _exponent; // Changed from mpfr_exp to exponent
    }

    unsigned long int getMantissa() const {
        return mantissa; // Changed from mpfr_d to mantissa
    }

    void setMantissa(unsigned long int _mantissa) {
        mantissa = _mantissa; // Changed from mpfr_d to mantissa
    }

    // Destructor to release resources (nothing to clear for unsigned long int)
    ~DeviceMpfr() {
    }

    void deviceMpfrToMpfr(mpfr_t& result) const {
        // Initialize mpfr_t with the precision of the DeviceMpfr object
        mpfr_set_prec(result, this->prec); // Changed from mpfr_prec to precision
        mpfr_set_ui(result, this->mantissa, MPFR_RNDN); // Changed from mpfr_d to mantissa
        mpfr_set_exp(result, this->exponent); // Changed from mpfr_exp to exponent
    }

public:
    mpfr_prec_t prec; // Updated variable name
    mpfr_exp_t exponent;   // Updated variable name
    unsigned long int mantissa; // Updated variable name
};

#endif

