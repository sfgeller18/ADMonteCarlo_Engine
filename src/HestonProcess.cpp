#include "HestonProcess.h"
#include <iostream>
#include <cmath>
#include <random>


HestonProcess::HestonProcess(const mpfr_t _currentPosition, const mpfr_t _mu, const mpfr_t _init_vol, const mpfr_t _kappa, const mpfr_t _theta, const mpfr_t _zeta) {
    mpfr_init_set(currentPosition, _currentPosition, MPFR_RNDN);
    mpfr_init_set(drift, _mu, MPFR_RNDN);
    mpfr_init_set(sigma, _init_vol, MPFR_RNDN);
    mpfr_init_set(kappa, _kappa, MPFR_RNDN);
    mpfr_init_set(theta, _theta, MPFR_RNDN);
    mpfr_t zero;
    mpfr_init2(zero, precision);
    mpfr_set_d(zero, 0.0, MPFR_RNDN);
    StochasticProcess _volatility(_init_vol, zero, _zeta);
    volatility = _volatility;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    distribution = std::normal_distribution<double>(0.0, 1.0);
    generator = std::default_random_engine(seed);
    mpfr_clear(zero);
}

HestonProcess::HestonProcess(const mpfr_t _currentPosition, const mpfr_t _mu, const mpfr_t _init_vol, const mpfr_t _kappa, const mpfr_t _theta, const mpfr_t _zeta, double seed) {
    mpfr_init_set(currentPosition, _currentPosition, MPFR_RNDN);
    mpfr_init_set(drift, _mu, MPFR_RNDN);
    mpfr_init_set(sigma, _init_vol, MPFR_RNDN);
    mpfr_init_set(kappa, _kappa, MPFR_RNDN);
    mpfr_init_set(theta, _theta, MPFR_RNDN);
    mpfr_t zero;
    mpfr_init2(zero, precision);
    mpfr_set_d(zero, 0.0, MPFR_RNDN);
    StochasticProcess _volatility(_init_vol, zero, _zeta);
    volatility = _volatility;
    distribution = std::normal_distribution<double>(0.0, 1.0);
    generator = std::default_random_engine(seed);
    mpfr_clear(zero);
}

HestonProcess::HestonProcess(const mpfr_t _currentPosition, const mpfr_t _mu, const mpfr_t _init_vol) {
    mpfr_init_set(currentPosition, _currentPosition, MPFR_RNDN);
    mpfr_init_set(drift, _mu, MPFR_RNDN);
    mpfr_init_set(sigma, _init_vol, MPFR_RNDN);
    mpfr_t zero;
    mpfr_init2(zero, precision);
    mpfr_set_d(zero, 0.0, MPFR_RNDN);
    mpfr_init_set(kappa, zero, MPFR_RNDN);
    mpfr_init_set(theta, zero, MPFR_RNDN);
    StochasticProcess _volatility(_init_vol, zero, zero);
    volatility = _volatility;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    distribution = std::normal_distribution<double>(0.0, 1.0);
    generator = std::default_random_engine(seed);
    mpfr_clear(zero);
}


StochasticProcess HestonProcess::getVolatility() const {
    return volatility;
}


void HestonProcess::volStep(const mpfr_t timeStep) {
    double vol_sample = volatility.getRand();

    mpfr_t vol_sample_mpfr;
    mpfr_t vol_drift;
    mpfr_t vol_sigma;
    mpfr_t sqrtTimeStep;
    mpfr_t sqrt_vol;

    mpfr_init2(vol_sample_mpfr, precision);
    mpfr_init2(sqrtTimeStep, precision);
    mpfr_init2(vol_drift, precision);
    mpfr_init2(vol_sigma, precision);
    mpfr_init2(sqrt_vol, precision);

    mpfr_set_d(vol_sample_mpfr, vol_sample, MPFR_RNDN);
    mpfr_sqrt(sqrtTimeStep, timeStep, MPFR_RNDN);
    mpfr_sqrt(sqrt_vol, *(volatility.getPosition()), MPFR_RNDN);

    mpfr_sub(vol_drift, kappa, *(volatility.getPosition()), MPFR_RNDN);
    mpfr_mul(vol_drift, kappa, vol_drift, MPFR_RNDN);
    mpfr_mul(vol_drift, vol_drift, timeStep, MPFR_RNDN);

    mpfr_mul(sqrt_vol, sqrt_vol, vol_sample_mpfr, MPFR_RNDN);
    mpfr_mul(vol_sigma, *(volatility.getVariance()), sqrt_vol, MPFR_RNDN);

    mpfr_add(vol_drift, vol_drift, vol_sigma, MPFR_RNDN);
    volatility.step(vol_drift);

    mpfr_clear(vol_sample_mpfr);
    mpfr_clear(sqrtTimeStep);
    mpfr_clear(vol_drift);
    mpfr_clear(vol_sigma);
    mpfr_clear(sqrt_vol);
}
void HestonProcess::HestonStep(const mpfr_t timeStep) {
    double asset_sample = (this->distribution)(this->generator);

    mpfr_t asset_sample_mpfr;
    mpfr_t sqrtTimeStep;
    mpfr_t sqrt_vol;
    mpfr_t stepValue;

    mpfr_init2(asset_sample_mpfr, precision);
    mpfr_init2(sqrtTimeStep, precision);
    mpfr_init2(sqrt_vol, precision);
    mpfr_init2(stepValue, precision);

    mpfr_set_d(asset_sample_mpfr, asset_sample, MPFR_RNDN);
    mpfr_sqrt(sqrtTimeStep, timeStep, MPFR_RNDN);
    mpfr_sqrt(sqrt_vol, *volatility.getPosition(), MPFR_RNDN);
    mpfr_mul(sqrt_vol, sqrt_vol, asset_sample_mpfr, MPFR_RNDN);
    mpfr_mul(sqrt_vol, sqrt_vol, sqrtTimeStep, MPFR_RNDN);

    mpfr_mul(stepValue, drift, timeStep, MPFR_RNDN);
    mpfr_add(stepValue, stepValue, sqrt_vol, MPFR_RNDN);
    this->step(stepValue);

    mpfr_clear(asset_sample_mpfr);
    mpfr_clear(sqrt_vol);
    mpfr_clear(stepValue);
    mpfr_clear(sqrtTimeStep);
}


void HestonProcess::printProperties() const {
    std::cout << "HestonProcess Properties:" << std::endl;
    std::cout << "Current Position: " << mpfr_get_d(currentPosition, MPFR_RNDN) << std::endl;
    std::cout << "Drift: " << mpfr_get_d(drift, MPFR_RNDN) << std::endl;
    std::cout << "Kappa: " << mpfr_get_d(kappa, MPFR_RNDN) << std::endl;
    std::cout << "Theta: " << mpfr_get_d(theta, MPFR_RNDN) << std::endl;
    std::cout << "Sigma: " << mpfr_get_d(sigma, MPFR_RNDN) << std::endl;
    std::cout << "Zeta: " << mpfr_get_d(*volatility.getVariance(), MPFR_RNDN) << std::endl;
    std::cout << "Volatility: " << mpfr_get_d(*volatility.getPosition(), MPFR_RNDN) << std::endl;
}
