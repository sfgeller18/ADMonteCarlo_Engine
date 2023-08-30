#ifndef ASSETS_H
#define ASSETS_H

#include "HestonProcess.h"
#include <vector>

template<class T>
class asset {
    public:
        asset(double _S0, double _mu, double _sigma, double _carry, std::vector<double> _extra_params) {
            this->S0 = _S0;
            this->mu = _mu;
            this->sigma = _sigma;
            this->carry = _carry;
            double implied_rate = mu - carry - pow(sigma, 2);
            //Write an  = operator overload for both of your process classes
            this->forecast_object = T::T(_S0, implied_rate, _sigma, std::vector<double> (_extra_params)...);
        }
    private:
        double S0 = 0.0;
        double mu = 0.0;
        double sigma = 0.0;
        double carry = 0.0;
        std::vector<double> extra_params;
        T forecast_object;
}

#endif