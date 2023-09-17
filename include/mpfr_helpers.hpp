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



class mpfr_DualNumber {
private:
    mpfr_t real;
    mpfr_t dual;

public:
    // Constructors
    mpfr_DualNumber() {
        mpfr_init(real);
        mpfr_init(dual);
    }

    mpfr_DualNumber(const mpfr_t& real_value, const mpfr_t& dual_value) {
        mpfr_init_set(real, real_value, MPFR_RNDN);
        mpfr_init_set(dual, dual_value, MPFR_RNDN);
    }

    // Copy constructor
    mpfr_DualNumber(const mpfr_DualNumber& other) {
        mpfr_init_set(real, other.real, MPFR_RNDN);
        mpfr_init_set(dual, other.dual, MPFR_RNDN);
    }

    // Destructor
    ~mpfr_DualNumber() {
        mpfr_clear(real);
        mpfr_clear(dual);
    }

    // Accessor functions
    mpfr_t& getReal() {
        return real;
    }

    mpfr_t& getDual() {
        return dual;
    }

    void setReal(mpfr_t& _real) {
        mpfr_set(real, _real, MPFR_RNDN);
    }

    void setDual(mpfr_t& _dual) {
        mpfr_set(dual, _dual, MPFR_RNDN);
    }

    // Overloaded operators for basic arithmetic
    mpfr_DualNumber operator+(const mpfr_DualNumber& other) const {
        mpfr_DualNumber result;
        mpfr_add(result.real, real, other.real, MPFR_RNDN);
        mpfr_add(result.dual, dual, other.dual, MPFR_RNDN);
        return result;
    }

    mpfr_DualNumber operator-(const mpfr_DualNumber& other) const {
        mpfr_DualNumber result;
        mpfr_sub(result.real, real, other.real, MPFR_RNDN);
        mpfr_sub(result.dual, dual, other.dual, MPFR_RNDN);
        return result;
    }

    mpfr_DualNumber operator*(const mpfr_DualNumber& other) const {
        mpfr_DualNumber result;
        mpfr_t temp;
        mpfr_init2(temp, precision);
        mpfr_mul(result.real, real, other.real, MPFR_RNDN);
        mpfr_mul(result.dual, real, other.dual, MPFR_RNDN);
        mpfr_mul(temp, dual, other.real, MPFR_RNDN);
        mpfr_add(result.dual, result.dual, temp, MPFR_RNDN);
        mpfr_clear(temp);
        return result;
    }

    mpfr_DualNumber& operator=(const mpfr_DualNumber& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        // Copy the real and dual values from 'other'
        mpfr_set(real, other.real, MPFR_RNDN);
        mpfr_set(dual, other.dual, MPFR_RNDN);

        return *this;
    }

    // Print function
    void print() const {
        std::cout << "Real: " << mpfrToString(real, precision) << std::endl;
        std::cout << "Dual: " << mpfrToString(dual, precision) << std::endl;
    }
};
mpfr_DualNumber exp(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t exp_real, exp_dual;
    mpfr_init2(exp_real, precision);
    mpfr_init2(exp_dual, precision);

    mpfr_exp(exp_real, x.getReal(), MPFR_RNDN);
    mpfr_mul(exp_dual, x.getDual(), exp_real, MPFR_RNDN);

    result.setReal(exp_real);
    result.setDual(exp_dual);

    mpfr_clear(exp_real);
    mpfr_clear(exp_dual);

    return result;
}

mpfr_DualNumber log(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t one, temp;
    mpfr_init2(one, precision);
    mpfr_init2(temp, precision);

    mpfr_set_si(one, 1, MPFR_RNDN);
    mpfr_div(temp, one, x.getReal(), MPFR_RNDN);
    mpfr_mul(temp, x.getDual(), temp, MPFR_RNDN);
    mpfr_log(result.getReal(), x.getReal(), MPFR_RNDN);

    result.setDual(temp);

    mpfr_clear(one);
    mpfr_clear(temp);

    return result;
}

mpfr_DualNumber pow(mpfr_DualNumber& x, mpfr_t& y) {
    mpfr_DualNumber result;

    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_pow(real_result, x.getReal(), y, MPFR_RNDN);
    mpfr_mul(dual_result, x.getDual(), y, MPFR_RNDN);
    mpfr_div(real_result, real_result, y, MPFR_RNDN);
    mpfr_mul(dual_result, dual_result, real_result, MPFR_RNDN);

    result.setReal(real_result);
    result.setDual(dual_result);

    mpfr_clear(real_result);
    mpfr_clear(dual_result);

    return result;
}

mpfr_DualNumber cos(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t cos_result, sin_result;
    mpfr_init2(cos_result, precision);
    mpfr_init2(sin_result, precision);

    mpfr_sin_cos(sin_result, cos_result, x.getReal(), MPFR_RNDN);
    mpfr_mul(sin_result, sin_result, x.getDual(), MPFR_RNDN);
    mpfr_neg(sin_result, sin_result, MPFR_RNDN);

    result.setReal(cos_result);
    result.setDual(sin_result);

    mpfr_clear(cos_result);
    mpfr_clear(sin_result);

    return result;
}

mpfr_DualNumber sin(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t sin_result, cos_result;
    mpfr_init2(sin_result, precision);
    mpfr_init2(cos_result, precision);

    mpfr_sin_cos(sin_result, cos_result, x.getReal(), MPFR_RNDN);
    mpfr_mul(cos_result, cos_result, x.getDual(), MPFR_RNDN);

    result.setReal(sin_result);
    result.setDual(cos_result);

    mpfr_clear(sin_result);
    mpfr_clear(cos_result);

    return result;
}

mpfr_DualNumber tan(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_tan(real_result, x.getReal(), MPFR_RNDN);
    mpfr_sec(dual_result, x.getReal(), MPFR_RNDN);
    mpfr_pow_si(dual_result, dual_result, 2, MPFR_RNDN);
    mpfr_mul(dual_result, dual_result, x.getDual(), MPFR_RNDN);

    result.setReal(real_result);
    result.setDual(dual_result);

    mpfr_clear(real_result);
    mpfr_clear(dual_result);

    return result;
}

mpfr_DualNumber acos(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t real_result, dual_result, one_minus_x_squared;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);
    mpfr_init2(one_minus_x_squared, precision);

    mpfr_acos(real_result, x.getReal(), MPFR_RNDN);

    mpfr_pow_si(one_minus_x_squared, x.getReal(), 2, MPFR_RNDN);
    mpfr_neg(one_minus_x_squared, one_minus_x_squared, MPFR_RNDN);
    mpfr_add_si(one_minus_x_squared, one_minus_x_squared, 1, MPFR_RNDN);
    mpfr_sqrt(one_minus_x_squared, one_minus_x_squared, MPFR_RNDN);

    mpfr_div(dual_result, x.getDual(), one_minus_x_squared, MPFR_RNDN);
    mpfr_neg(dual_result, dual_result, MPFR_RNDN);

    result.setReal(real_result);
    result.setDual(dual_result);

    mpfr_clear(real_result);
    mpfr_clear(one_minus_x_squared);
    mpfr_clear(dual_result);

    return result;
}

mpfr_DualNumber asin(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t real_result, dual_result, one_minus_x_squared;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);
    mpfr_init2(one_minus_x_squared, precision);

    mpfr_asin(real_result, x.getReal(), MPFR_RNDN);

    mpfr_pow_si(one_minus_x_squared, x.getReal(), 2, MPFR_RNDN);
    mpfr_neg(one_minus_x_squared, one_minus_x_squared, MPFR_RNDN);
    mpfr_add_si(one_minus_x_squared, one_minus_x_squared, 1, MPFR_RNDN);
    mpfr_sqrt(one_minus_x_squared, one_minus_x_squared, MPFR_RNDN);

    mpfr_div(dual_result, x.getDual(), one_minus_x_squared, MPFR_RNDN);

    result.setReal(real_result);
    result.setDual(dual_result);

    mpfr_clear(one_minus_x_squared);
    mpfr_clear(real_result);
    mpfr_clear(dual_result);

    return result;
}

mpfr_DualNumber atan(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t real_result, dual_result, one_plus_x_squared;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);
    mpfr_init2(one_plus_x_squared, precision);

    mpfr_atan(real_result, x.getReal(), MPFR_RNDN);

    mpfr_pow_si(one_plus_x_squared, x.getReal(), 2, MPFR_RNDN);
    mpfr_add_si(one_plus_x_squared, one_plus_x_squared, 1, MPFR_RNDN);
    mpfr_sqrt(one_plus_x_squared, one_plus_x_squared, MPFR_RNDN);
    mpfr_div(dual_result, x.getDual(), one_plus_x_squared, MPFR_RNDN);

    result.setReal(real_result);
    result.setDual(dual_result);

    mpfr_clear(one_plus_x_squared);
    mpfr_clear(real_result);
    mpfr_clear(dual_result);

    return result;
}

mpfr_DualNumber sqrt(mpfr_DualNumber& x) {
    mpfr_DualNumber result;

    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_sqrt(real_result, x.getReal(), MPFR_RNDN);
    mpfr_div(dual_result, x.getDual(), real_result, MPFR_RNDN);
    mpfr_div_si(dual_result, dual_result, 2, MPFR_RNDN);
    mpfr_mul(dual_result, dual_result, x.getDual(), MPFR_RNDN);

    result.setReal(real_result);
    result.setDual(dual_result);

    mpfr_clear(real_result);
    mpfr_clear(dual_result);

    return result;
}



#endif