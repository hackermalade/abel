#pragma once

#include "Core.h"
#include <string>
#include <map>

namespace abel {

// ═══════════════════════════════════════════════════════════════════════════
//  Hodgkin‑Huxley neuron
// ═══════════════════════════════════════════════════════════════════════════
class HodgkinHuxleyNeuron {
public:
    HodgkinHuxleyNeuron(double C_m = 1.0, double g_Na = 120, double g_K = 36, double g_L = 0.3,
                        double E_Na = 50, double E_K = -77, double E_L = -54.387, double dt = 0.025);

    // Update membrane potential; returns true if neuron fired
    bool update(double I_ext, double dt_override = -1.0);

    // Gating functions
    double alpha_m(double V) const;
    double beta_m (double V) const;
    double alpha_h(double V) const;
    double beta_h (double V) const;
    double alpha_n(double V) const;
    double beta_n (double V) const;

    double V;           // membrane potential (mV)
    double m, h, n;     // gating variables
    bool fired;
    double energy;      // local ATP (0..1)

private:
    double C_m, g_Na, g_K, g_L;
    double E_Na, E_K, E_L;
    double dt;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Izhikevich neuron
// ═══════════════════════════════════════════════════════════════════════════
class IzhikevichNeuron {
public:
    IzhikevichNeuron(double a, double b, double c, double d,
                     double v_init = -65.0, double u_init = -1e9);

    // Update membrane potential; returns true if neuron fired
    bool update(double I, double dt = 0.5);

    double a, b, c, d;
    double v, u;        // membrane potential and recovery variable
    bool fired;
    double energy;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Factory functions
// ═══════════════════════════════════════════════════════════════════════════
NeuronParams getNeuronParams(const std::string& type);
HodgkinHuxleyNeuron createHodgkinHuxleyNeuron(double dt);
IzhikevichNeuron createIzhikevichNeuron(double a, double b, double c, double d, double dt);

} // namespace abel
