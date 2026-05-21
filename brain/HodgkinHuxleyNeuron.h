// abel/brain/HodgkinHuxleyNeuron.h
#pragma once
#include "Core.h"

namespace abel {

class HodgkinHuxleyNeuron {
public:
    HodgkinHuxleyNeuron(double C_m = 1.0, double g_Na = 120, double g_K = 36, double g_L = 0.3,
                        double E_Na = 50, double E_K = -77, double E_L = -54.387, double dt = 0.025);

    bool update(double I_ext, double dt_override = -1.0);

    double alpha_m(double V) const;
    double beta_m (double V) const;
    double alpha_h(double V) const;
    double beta_h (double V) const;
    double alpha_n(double V) const;
    double beta_n (double V) const;

    double V;
    double m, h, n;
    bool fired;
    double energy;

private:
    double C_m, g_Na, g_K, g_L;
    double E_Na, E_K, E_L;
    double dt;
};

} // namespace abel
