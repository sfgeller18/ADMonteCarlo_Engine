#ifndef DUALNUMBER_H
#define DUALNUMBER_H

#include <iostream>
#include <cuda_runtime.h>



//Need to add extra stochastic dual variable (i.e. x^3 == 0, E[x] = 0, E[x^2] = dual)
template <typename T>
struct DualNumber {
    T real;
    T dual;
    // Constructors
    DualNumber() : real(0.0), dual(0.0) {}

    DualNumber(const T& real_value, const T& dual_value) : real(real_value), dual(dual_value) {}

    // Copy constructor
    DualNumber(const DualNumber<T>& other) : real(other.real), dual(other.dual) {}

    // Accessor functions
    T getReal() const {
        return real;
    }

    T getDual() const {
        return dual;
    }

    void setReal(const T& _real) {
        real = _real;
    }

    void setDual(const T& _dual) {
        dual = _dual;
    }

    // Overloaded operators for basic arithmetic
    DualNumber<T> operator+(const DualNumber<T>& other) const {
        return DualNumber<T>(real + other.real, dual + other.dual);
    }

    DualNumber<T> operator-(const DualNumber<T>& other) const {
        return DualNumber<T>(real - other.real, dual - other.dual);
    }

    DualNumber<T> operator*(const DualNumber<T>& other) const {
        return DualNumber<T>(real * other.real, real * other.dual + dual * other.real);
    }
    DualNumber<T> operator*(const T& other) const {
        return DualNumber<T>(real * other, dual * other);
    }

    DualNumber<T>& operator=(const DualNumber<T>& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }
        real = other.real;
        dual = other.dual;

        return *this;
    }

       DualNumber<T> operator/(const T& other) const {
        return DualNumber<T>(real/other, dual/other);
    }

    // Print function
    void print() const {
        std::cout << "Real: " << real << std::endl;
        std::cout << "Dual: " << dual << std::endl;
    }

    // Additional mathematical functions for DualNumber
    static DualNumber<T> exp(const DualNumber<T>& x) {
        return DualNumber<T>(std::exp(x.getReal()), x.getDual() * std::exp(x.getReal()));
    }

    static DualNumber<T> log(const DualNumber<T>& x) {
        return DualNumber<T>(std::log(x.getReal()), x.getDual() / x.getReal());
    }

    static DualNumber<T> pow(const DualNumber<T>& x, const T& y) {
        return DualNumber<T>(std::pow(x.real, y), y * std::pow(x.real, y - 1) * x.dual);
    }

    static DualNumber<T> cos(const DualNumber<T>& x) {
        return DualNumber<T>(std::cos(x.getReal()), -std::sin(x.getReal()) * x.getDual());
    }

    static DualNumber<T> sin(const DualNumber<T>& x) {
        return DualNumber<T>(std::sin(x.getReal()), std::cos(x.getReal()) * x.getDual());
    }

    static DualNumber<T> tan(const DualNumber<T>& x) {
        T cos_x = std::cos(x.getReal());
        return DualNumber<T>(std::tan(x.getReal()), x.getDual() / (cos_x * cos_x));
    }

    static DualNumber<T> acos(const DualNumber<T>& x) {
        return DualNumber<T>(std::acos(x.getReal()), -x.getDual() / std::sqrt(1.0 - x.getReal() * x.getReal()));
    }

    static DualNumber<T> asin(const DualNumber<T>& x) {
        return DualNumber<T>(std::asin(x.getReal()), x.getDual() / std::sqrt(1.0 - x.getReal() * x.getReal()));
    }

    static DualNumber<T> atan(const DualNumber<T>& x) {
        return DualNumber<T>(std::atan(x.getReal()), x.getDual() / (1.0 + x.getReal() * x.getReal()));
    }
};








#endif