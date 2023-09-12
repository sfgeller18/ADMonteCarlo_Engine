#pragma once

#include "StochasticProcess.h"

#include "StochasticProcess.h"

class HestonProcess : public StochasticProcess {
public:
    HestonProcess(const mpfr_t _currentPosition, const mpfr_t _mu, const mpfr_t _init_vol, const mpfr_t _kappa, const mpfr_t _theta, const mpfr_t _zeta);
    HestonProcess(const mpfr_t _currentPosition, const mpfr_t _mu, const mpfr_t _init_vol);
    void volStep(const mpfr_t timeStep);
    void HestonStep(const mpfr_t timeStep);
    void printProperties() const;
    StochasticProcess getVolatility() const;

private:
    mpfr_t kappa;
    mpfr_t sigma;
    mpfr_t theta;
    mpfr_t mu;
    StochasticProcess volatility;
};



