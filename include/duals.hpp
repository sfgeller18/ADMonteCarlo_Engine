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

    void setReal(const mpfr_t& _real) {
        mpfr_set(real, _real, MPFR_RNDN);
    }
    
    void setDual(const mpfr_t& _dual) {
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

    // You can implement other operators (e.g., division) similarly

    // Print function
    void print() const {
        std::cout << "Real: " << mpfrToString(real, precision)<< std::endl;
        std::cout << "Dual: " << mpfrToString(dual, precision)<< std::endl;
    }

void exp(DualNumber& x) {
    mpfr_t exp;
    mpfr_init2(exp, precision);
    mpfr_exp(exp, x.getReal(), MPFR_RNDN); // Derivative of exp(x) is exp(x)
    x.setReal(exp);
    mpfr_mul(exp, x.getDual(), exp, MPFR_RNDN);
    x.setDual(exp);
    mpfr_clear(exp);
}

void log(DualNumber& x) {
    mpfr_t one;
    mpfr_t temp;
    mpfr_init2(one, precision);
    mpfr_set_si(one, 1, MPFR_RNDN);
    mpfr_div(temp, one, x.getReal(), MPFR_RNDN);
    mpfr_mul(dual, dual, temp, MPFR_RNDN);
    mpfr_log(x.getReal(), x.getReal(), MPFR_RNDN);
    mpfr_clear(one);
    mpfr_clear(temp);
}

#include <mpfr.h>

// Assuming precision is a variable that specifies the precision of the mpfr_t variables.

void pow(DualNumber& x, const mpfr_t& y) {
    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_pow(real_result, x.getReal(), y, MPFR_RNDN); // f(x) = x^y
    mpfr_mul(dual_result, x.getDual(), y, MPFR_RNDN); 
    mpfr_div(real_result, real_result, y, MPFR_RNDN);
    mpfr_mul(dual_result, dual_result, real_result, MPFR_RNDN); // f'(x) = y * x^(y-1) * x_dual

    x.setReal(real_result);
    x.setDual(dual_result);

    mpfr_clear(real_result);
    mpfr_clear(dual_result);
}

void cos(DualNumber& x) {
    mpfr_t cos_result, sin_result;
    mpfr_init2(cos_result, precision);
    mpfr_init2(sin_result, precision);

    mpfr_sin_cos( sin_result, cos_result, x.getReal(), MPFR_RNDN);
    mpfr_mul(sin_result, sin_result, x.getDual(), MPFR_RNDN); // f'(x) = -sin(x) * x_dual
    mpfr_neg(sin_result, sin_result, MPFR_RNDN);

    x.setReal(cos_result);
    x.setDual(sin_result);

    mpfr_clear(cos_result);
    mpfr_clear(sin_result);
}

void sin(DualNumber& x) {
    mpfr_t sin_result, cos_result;
    mpfr_init2(sin_result, precision);
    mpfr_init2(cos_result, precision);

    mpfr_sin_cos(sin_result, cos_result, x.getReal(), MPFR_RNDN); // f(x) = sin(x)
    mpfr_mul(cos_result, cos_result, x.getDual(), MPFR_RNDN); // f'(x) = cos(x) * x_dual

    x.setReal(sin_result);
    x.setDual(cos_result);

    mpfr_clear(sin_result);
    mpfr_clear(cos_result);
}

void tan(DualNumber& x) {
    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_tan(real_result, x.getReal(), MPFR_RNDN); // f(x) = tan(x)
    mpfr_sec(dual_result, x.getReal(), MPFR_RNDN);
    mpfr_pow_si(dual_result, dual_result, 2, MPFR_RNDN);
    mpfr_mul(dual_result, dual_result, x.getDual(), MPFR_RNDN); // f'(x) = sec^2(x) * x_dual

    x.setReal(real_result);
    x.setDual(dual_result);

    mpfr_clear(real_result);
    mpfr_clear(dual_result);
}

void acos(DualNumber& x) {
    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_acos(real_result, x.getReal(), MPFR_RNDN); // f(x) = acos(x)
    
    mpfr_t one_minus_x_squared;
    mpfr_init2(one_minus_x_squared, precision);
    mpfr_pow_si(one_minus_x_squared, x.getReal(), 2, MPFR_RNDN);
    mpfr_neg(one_minus_x_squared, one_minus_x_squared, MPFR_RNDN);
    mpfr_add_si(one_minus_x_squared,  one_minus_x_squared, 1, MPFR_RNDN); // 1 - x^2
    mpfr_sqrt(one_minus_x_squared, one_minus_x_squared, MPFR_RNDN); // sqrt(1 - x^2)
    mpfr_div(dual_result, x.getDual(), one_minus_x_squared, MPFR_RNDN); // f'(x) = -x_dual / sqrt(1 - x^2)
    mpfr_neg(dual_result, dual_result, MPFR_RNDN);

    x.setReal(real_result);
    x.setDual(dual_result);

    mpfr_clear(real_result);
    mpfr_clear(one_minus_x_squared);
    mpfr_clear(dual_result);
}


void asin(DualNumber& x) {
    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_asin(real_result, x.getReal(), MPFR_RNDN); // f(x) = asin(x)
    mpfr_t one_minus_x_squared;
    mpfr_init2(one_minus_x_squared, precision);
    mpfr_pow_si(one_minus_x_squared, x.getReal(), 2, MPFR_RNDN);
    mpfr_neg(one_minus_x_squared, one_minus_x_squared, MPFR_RNDN);
    mpfr_add_si(one_minus_x_squared,  one_minus_x_squared, 1, MPFR_RNDN); // 1 - x^2
    mpfr_sqrt(one_minus_x_squared, one_minus_x_squared, MPFR_RNDN); // sqrt(1 - x^2)
    mpfr_div(dual_result, x.getDual(), one_minus_x_squared, MPFR_RNDN); // f'(x) = 1 / sqrt(1 - x^2) * x_dual

    x.setReal(real_result);
    x.setDual(dual_result);

    mpfr_clear(one_minus_x_squared);
    mpfr_clear(real_result);
    mpfr_clear(dual_result);
}

void atan(DualNumber& x) {
    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_atan(real_result, x.getReal(), MPFR_RNDN); // f(x) = atan(x)
    mpfr_t one_plus_x_squared;
    mpfr_init2(one_plus_x_squared, precision);
    mpfr_pow_si(one_plus_x_squared, x.getReal(), 2, MPFR_RNDN);
    mpfr_add_si(one_plus_x_squared,  one_plus_x_squared, 1, MPFR_RNDN); // 1 + x^2
    mpfr_sqrt(one_plus_x_squared, one_plus_x_squared, MPFR_RNDN); // sqrt(1 + x^2)
    mpfr_div(dual_result, x.getDual(), one_plus_x_squared, MPFR_RNDN); // f'(x) = 1 / sqrt(1 + x^2) * x_dual

    x.setReal(real_result);
    x.setDual(dual_result);

    mpfr_clear(one_plus_x_squared);
    mpfr_clear(real_result);
    mpfr_clear(dual_result);
}

void sqrt(DualNumber& x) {
    mpfr_t real_result, dual_result;
    mpfr_init2(real_result, precision);
    mpfr_init2(dual_result, precision);

    mpfr_sqrt(real_result, x.getReal(), MPFR_RNDN); // f(x) = sqrt(x)
    mpfr_div(dual_result, x.getDual(), real_result, MPFR_RNDN); // f'(x) = 1 / (2 * sqrt(x)) * x_dual
    mpfr_div_si(dual_result, dual_result, 2, MPFR_RNDN);
    mpfr_mul(dual_result, dual_result, x.getDual(), MPFR_RNDN);

    x.setReal(real_result);
    x.setDual(dual_result);

    mpfr_clear(real_result);
    mpfr_clear(dual_result);
}


};