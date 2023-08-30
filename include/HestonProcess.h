#pragma once

#include "StochasticProcess.h"

class HestonProcess : public StochasticProcess {
public:
    HestonProcess(double _currentPosition, double _mu, double _init_vol, double kappa, double theta, double _zeta);
    HestonProcess(double _currentPosition, double _mu, double _init_vol);
    void HestonStep(double timeStep);
    void printProperties() const;
    StochasticProcess getVolatility();

private:
    double kappa = 0.0;
    double sigma = 0.0;
    double theta = 0.0;
    double mu = 0.0;
    StochasticProcess volatility;
};


