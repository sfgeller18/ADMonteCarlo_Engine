#include "StochasticProcess.h"
#include <iostream>
#include <random>

StochasticProcess::StochasticProcess(double _initialPosition, double _drift, double _variance) {
    this->currentPosition = _initialPosition;
    this->drift = _drift;
    this->variance = _variance;
    this->distribution = std::normal_distribution<double> (0.0, _variance);
    this->generator = std::default_random_engine();
}

void StochasticProcess::step(double stepValue) {
    currentPosition *= std::exp(stepValue);
}

double StochasticProcess::getPosition() const {
    return currentPosition;
}

void StochasticProcess::setPosition(double value) {
    this->currentPosition = value;
}

double StochasticProcess::getDrift() const {
    return drift;
}

double StochasticProcess::getVariance() const {
    return variance;
}

void StochasticProcess::setVariance(double value) {
    this->variance = value;
}

double StochasticProcess::getRandomStep(double timeStep) {
    double sample = distribution(generator);
    return drift*timeStep+sample*std::sqrt(timeStep); // Return a random step value
}

void StochasticProcess::printProperties() const {
    std::cout << "StochasticProcess Properties:" << std::endl;
    std::cout << "Current Position: " << this->currentPosition << std::endl;
    std::cout << "Drift: " << this->drift << std::endl;
    std::cout << "Variance: " << this->variance << std::endl;

}


