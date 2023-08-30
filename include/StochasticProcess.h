#pragma once
#include <iostream>
#include <random>

class StochasticProcess {
public:
    StochasticProcess(double initialPosition = 0, double drift = 0.0, double variance = 1.0);
    StochasticProcess(const StochasticProcess& other);
    void step(double stepValue);

    double getPosition() const;
    void setPosition(double value);
    double getDrift() const;
    double getVariance() const;
    void setVariance(double value);
    double getRandomStep(double timeStep);
    void printProperties() const;


protected:
    double currentPosition = 0.0;
    double drift = 0.0;
    double variance = 0.0;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
};
