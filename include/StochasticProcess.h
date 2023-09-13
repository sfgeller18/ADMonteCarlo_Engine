#pragma once
#include <iostream>
#include <random>
#include <mpfr.h>
#include <random>
#include <chrono>

#define precision 12

class StochasticProcess {
public:
    StochasticProcess(const mpfr_t initialPosition, const mpfr_t drift, const mpfr_t variance);
    StochasticProcess(const mpfr_t initialPosition, const mpfr_t drift, const mpfr_t variance, double seed);
    StochasticProcess(const StochasticProcess& other);
    StochasticProcess();
    void step(const mpfr_t stepValue);

    const mpfr_t* getPosition() const;
    void setPosition(const mpfr_t value);
    const mpfr_t* getDrift() const;
    const mpfr_t* getVariance() const;
    void setVariance(const mpfr_t value);
    mpfr_t* getRandomStep(const mpfr_t timeStep);
    void printProperties() const;
    double getRand();


protected:
    mpfr_t currentPosition;
    mpfr_t drift;
    mpfr_t variance;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
};
