#ifndef DUALNUMBER_H
#define DUALNUMBER_H

#include <iostream>
#include "mpfr_helpers.hpp"
#include <mpfr.h>

#define precision 12

class DualNumber {
private:
    mpfr_t real;
    mpfr_t dual;

public:
    // Constructors
    DualNumber() {
        mpfr_init(real);
        mpfr_init(dual);
    }

    DualNumber(const mpfr_t& real_value, const mpfr_t& dual_value) {
        mpfr_init_set(real, real_value, MPFR_RNDN);
        mpfr_init_set(dual, dual_value, MPFR_RNDN);
    }

    // Copy constructor
    DualNumber(const DualNumber& other) {
        mpfr_init_set(real, other.real, MPFR_RNDN);
        mpfr_init_set(dual, other.dual, MPFR_RNDN);
    }

    // Destructor
    ~DualNumber() {
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
    DualNumber operator+(const DualNumber& other) const {
        DualNumber result;
        mpfr_add(result.real, real, other.real, MPFR_RNDN);
        mpfr_add(result.dual, dual, other.dual, MPFR_RNDN);
        return result;
    }

    DualNumber operator-(const DualNumber& other) const {
        DualNumber result;
        mpfr_sub(result.real, real, other.real, MPFR_RNDN);
        mpfr_sub(result.dual, dual, other.dual, MPFR_RNDN);
        return result;
    }

    DualNumber operator*(const DualNumber& other) const {
        DualNumber result;
        mpfr_t temp;
        mpfr_init2(temp, precision);
        mpfr_mul(result.real, real, other.real, MPFR_RNDN);
        mpfr_mul(result.dual, real, other.dual, MPFR_RNDN);
        mpfr_mul(temp, dual, other.real, MPFR_RNDN);
        mpfr_add(result.dual, result.dual, temp, MPFR_RNDN);
        mpfr_clear(temp);
        return result;
    }

    DualNumber& operator=(const DualNumber& other) {
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
DualNumber exp(DualNumber& x) {
    DualNumber result;

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

DualNumber log(DualNumber& x) {
    DualNumber result;

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

DualNumber pow(DualNumber& x, mpfr_t& y) {
    DualNumber result;

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

DualNumber cos(DualNumber& x) {
    DualNumber result;

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

DualNumber sin(DualNumber& x) {
    DualNumber result;

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

DualNumber tan(DualNumber& x) {
    DualNumber result;

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

DualNumber acos(DualNumber& x) {
    DualNumber result;

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

DualNumber asin(DualNumber& x) {
    DualNumber result;

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

DualNumber atan(DualNumber& x) {
    DualNumber result;

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

DualNumber sqrt(DualNumber& x) {
    DualNumber result;

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