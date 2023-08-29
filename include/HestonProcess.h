#pragma once

#include "StochasticProcess.h"

class HestonProcess : public StochasticProcess {
public:
    HestonProcess(double _currentPosition, double kappa, double theta, double _init_vol, double _zeta, double _mu);
    void HestonStep(double timeStep);
    void printProperties() const;
    StochasticProcess getVolatility();

private:
    double kappa;
    double sigma;
    double theta;
    double mu;
    StochasticProcess volatility;
};


