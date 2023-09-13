#include "StochasticProcess.h"



StochasticProcess::StochasticProcess(const mpfr_t _initialPosition, const mpfr_t _drift, const mpfr_t _variance) {
    mpfr_init_set(currentPosition, _initialPosition, MPFR_RNDN);
    mpfr_init_set(drift, _drift, MPFR_RNDN);
    mpfr_init_set(variance, _variance, MPFR_RNDN);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    this->distribution = std::normal_distribution<double>(0.0, mpfr_get_d(_variance, MPFR_RNDN));
    this->generator = std::default_random_engine(seed);
}

StochasticProcess::StochasticProcess(const mpfr_t _initialPosition, const mpfr_t _drift, const mpfr_t _variance, double seed) {
    mpfr_init_set(currentPosition, _initialPosition, MPFR_RNDN);
    mpfr_init_set(drift, _drift, MPFR_RNDN);
    mpfr_init_set(variance, _variance, MPFR_RNDN);
    this->distribution = std::normal_distribution<double>(0.0, mpfr_get_d(_variance, MPFR_RNDN));
    this->generator = std::default_random_engine(seed);
}

StochasticProcess::StochasticProcess(const StochasticProcess& other) {
    mpfr_init_set(currentPosition, other.currentPosition, MPFR_RNDN);
    mpfr_init_set(drift, other.drift, MPFR_RNDN);
    mpfr_init_set(variance, other.variance, MPFR_RNDN);
    this->distribution = other.distribution;
    this->generator = other.generator;
}

StochasticProcess::StochasticProcess() {
    mpfr_t zero;
    mpfr_init2(zero, precision);
    mpfr_set_d(zero, 0.0, MPFR_RNDN);
    mpfr_t one;
    mpfr_init2(one, precision);
    mpfr_set_d(one, 0.0, MPFR_RNDN);
    mpfr_init_set(currentPosition, zero, MPFR_RNDN);
    mpfr_init_set(drift, zero, MPFR_RNDN);
    mpfr_init_set(variance, one, MPFR_RNDN);
    mpfr_clear(zero);
    mpfr_clear(one);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    this->distribution = std::normal_distribution<double>(0.0, mpfr_get_d(variance, MPFR_RNDN));
    this->generator = std::default_random_engine(seed);
}

void StochasticProcess::step(const mpfr_t stepValue) {
    mpfr_t temp;
    mpfr_init2(temp, precision);
    mpfr_exp(temp, stepValue, MPFR_RNDN);
    mpfr_mul(currentPosition, currentPosition, temp, MPFR_RNDN);
    mpfr_clear(temp);
}

const mpfr_t* StochasticProcess::getPosition() const {
    return &currentPosition;
}

const mpfr_t* StochasticProcess::getDrift() const {
    return &drift;
}

const mpfr_t* StochasticProcess::getVariance() const {
    return &variance;
}

void StochasticProcess::setPosition(const mpfr_t value) {
    mpfr_set(currentPosition, value, MPFR_RNDN);
}

void StochasticProcess::setVariance(const mpfr_t value) {
    mpfr_set(variance, value, MPFR_RNDN);
    // Update the distribution with the new variance
    this->distribution = std::normal_distribution<double>(0.0, mpfr_get_d(value, MPFR_RNDN));
}

double StochasticProcess::getRand() {
    double sample = distribution(generator);
    return sample;
}

mpfr_t* StochasticProcess::getRandomStep(const mpfr_t timeStep) {
    double sample = this->getRand();
    mpfr_t sample_mpfr;
    mpfr_init2(sample_mpfr, precision);
    mpfr_set_d(sample_mpfr, sample, MPFR_RNDN);
    mpfr_t randomStep;
    mpfr_init2(randomStep, precision);
    mpfr_t sqrt_vol;
    mpfr_init2(sqrt_vol, precision);
    mpfr_sqrt(sqrt_vol, variance, MPFR_RNDN);
    
    mpfr_mul_d(randomStep, sqrt_vol, std::sqrt(mpfr_get_d(timeStep, MPFR_RNDN)), MPFR_RNDN);
    mpfr_mul(randomStep, randomStep, sample_mpfr, MPFR_RNDN);
    mpfr_mul(drift, drift, timeStep, MPFR_RNDN);
    mpfr_add(randomStep, randomStep, drift, MPFR_RNDN);

    mpfr_clear(sample_mpfr);
    mpfr_clear(sqrt_vol);
    
    return &randomStep;
}

void StochasticProcess::printProperties() const {
    std::cout << "StochasticProcess Properties:" << std::endl;
    std::cout << "Current Position: " << mpfr_get_d(currentPosition, MPFR_RNDN) << std::endl;
    std::cout << "Drift: " << mpfr_get_d(drift, MPFR_RNDN) << std::endl;
    std::cout << "Variance: " << mpfr_get_d(variance, MPFR_RNDN) << std::endl;
}
