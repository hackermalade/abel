// abel/brain/IzhikevichNeuron.h
#pragma once
#include "Core.h"

namespace abel {

class IzhikevichNeuron {
public:
    IzhikevichNeuron(double a, double b, double c, double d,
                     double v_init = -65.0, double u_init = -1e9);

    bool update(double I, double dt = 0.5);

    double a, b, c, d;
    double v, u;
    bool fired;
    double energy;
};

} // namespace abel
