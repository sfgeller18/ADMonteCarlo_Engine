#pragma once

#include "StochasticProcess.h"

class HestonProcess : public StochasticProcess {
public:
    HestonProcess(double _currentPosition, double kappa, double theta, double _init_vol, double _zeta, double _mu);

    void step(double timeStep) ;
    void printProperties() const;

private:
    double kappa;
    double sigma;
    double theta;
    double mu;
    StochasticProcess volatility;
};


