#include "HestonProcess.h"
#include <cmath>
#include <random>
#include <iostream>

HestonProcess::HestonProcess(double _currentPosition, double _kappa, double _theta, double _init_vol, double _zeta, double _mu) {
    StochasticProcess _volatility(_init_vol, 0.0, _zeta);
    this->currentPosition = _currentPosition;
    this->drift = _mu;
    this->distribution = std::normal_distribution<double>(0.0, 1.0);
    this->generator = std::default_random_engine();
     this->kappa = _kappa;
     this->theta = _theta;
     this->volatility = _volatility; 
}

void HestonProcess::step(double timeStep) {
    double epsilon = distribution(generator);
    double sqrtTimeStep = std::sqrt(timeStep);
    double value = std::max(kappa * (theta - volatility.getPosition()) * timeStep + volatility.getVariance() * std::sqrt(volatility.getPosition() * timeStep) * epsilon, 0.0);
    volatility.setPosition(value);
    double randomStep = volatility.getPosition() * sqrtTimeStep * epsilon + getDrift();
    StochasticProcess::step(randomStep);
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


