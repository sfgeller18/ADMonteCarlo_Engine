#include "HestonProcess.h"
#include <cmath>
#include <random>
#include <iostream>

HestonProcess::HestonProcess(double _currentPosition, double _mu, double _init_vol, double _kappa, double _theta, double _zeta) {
    StochasticProcess _volatility(_init_vol, 0.0, _zeta);
    this->currentPosition = _currentPosition;
    this->drift = _mu;
    this->distribution = std::normal_distribution<double>(0.0, 1.0);
    this->generator = std::default_random_engine();
     this->kappa = _kappa;
     this->theta = _theta;
     this->volatility = _volatility; 
}

HestonProcess::HestonProcess(double _currentPosition, double _mu, double _init_vol) {
    this->currentPosition = _currentPosition;
    this->drift = _mu;
    StochasticProcess _volatility(_init_vol, 0.0, 0.0);
    this-> volatility = _volatility; 
}


StochasticProcess HestonProcess::getVolatility() {
    return volatility;
}

void HestonProcess::HestonStep(double timeStep) {
    double sample = (this->distribution)(this->generator);
    double sqrtTimeStep = std::sqrt(timeStep);
    double value = kappa * (theta - (this->volatility).getPosition()) * timeStep + volatility.getVariance() * std::sqrt(volatility.getPosition() * timeStep) * sample;
    (this->volatility).setPosition((this->volatility).getPosition()+value);
    double randomStep = std::sqrt((this->volatility).getPosition()) * sqrtTimeStep * sample + this->getDrift()*timeStep;
    this->currentPosition*=std::exp(randomStep);
}

void HestonProcess::printProperties() const {
    std::cout << "HestonProcess Properties:" << std::endl;
    std::cout << "Current Position: " << currentPosition << std::endl;
    std::cout << "Drift: " << drift << std::endl;
    std::cout << "Kappa: " << kappa << std::endl;
    std::cout << "Theta: " << theta << std::endl;
    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Zeta: " << volatility.getVariance() << std::endl;
    std::cout << "Volatility: " << volatility.getPosition() << std::endl;
}

