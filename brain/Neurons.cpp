#include "Neurons.h"
#include "Core.h"

#include <cmath>
#include <map>
#include <string>

namespace abel {

// ═══════════════════════════════════════════════════════════════════════════
//  Hodgkin‑Huxley neuron
// ═══════════════════════════════════════════════════════════════════════════
HodgkinHuxleyNeuron::HodgkinHuxleyNeuron(double C_m, double g_Na, double g_K, double g_L,
                                         double E_Na, double E_K, double E_L, double dt)
    : C_m(C_m), g_Na(g_Na), g_K(g_K), g_L(g_L),
      E_Na(E_Na), E_K(E_K), E_L(E_L), dt(dt),
      V(-65.0), m(0.05), h(0.6), n(0.32),
      fired(false), energy(1.0)
{}

double HodgkinHuxleyNeuron::alpha_m(double V) const {
    return 0.1 * (V + 40.0) / (1.0 - std::exp(-(V + 40.0) / 10.0));
}
double HodgkinHuxleyNeuron::beta_m(double V) const {
    return 4.0 * std::exp(-(V + 65.0) / 18.0);
}
double HodgkinHuxleyNeuron::alpha_h(double V) const {
    return 0.07 * std::exp(-(V + 65.0) / 20.0);
}
double HodgkinHuxleyNeuron::beta_h(double V) const {
    return 1.0 / (1.0 + std::exp(-(V + 35.0) / 10.0));
}
double HodgkinHuxleyNeuron::alpha_n(double V) const {
    return 0.01 * (V + 55.0) / (1.0 - std::exp(-(V + 55.0) / 10.0));
}
double HodgkinHuxleyNeuron::beta_n(double V) const {
    return 0.125 * std::exp(-(V + 65.0) / 80.0);
}

bool HodgkinHuxleyNeuron::update(double I_ext, double dt_override) {
    double dt_val = (dt_override > 0.0) ? dt_override : dt;
    double V_m = V;
    double m_val = m, h_val = h, n_val = n;

    double dm = (alpha_m(V_m) * (1.0 - m_val) - beta_m(V_m) * m_val) * dt_val;
    double dh = (alpha_h(V_m) * (1.0 - h_val) - beta_h(V_m) * h_val) * dt_val;
    double dn = (alpha_n(V_m) * (1.0 - n_val) - beta_n(V_m) * n_val) * dt_val;

    double I_Na = g_Na * (m_val * m_val * m_val) * h_val * (V_m - E_Na);
    double I_K  = g_K  * (n_val * n_val * n_val * n_val) * (V_m - E_K);
    double I_L  = g_L  * (V_m - E_L);

    double dV = (I_ext - I_Na - I_K - I_L) / C_m * dt_val;

    m += dm; h += dh; n += dn; V += dV;

    fired = false;
    if (V >= -20.0) {
        V = -65.0; m = 0.05; h = 0.6; n = 0.32;
        fired = true;
        energy -= 0.01;
        if (energy < 0.0) energy = 0.0;
    }
    energy = std::min(1.0, energy + dt_val / 1000.0 * 0.01);
    return fired;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Izhikevich neuron
// ═══════════════════════════════════════════════════════════════════════════
IzhikevichNeuron::IzhikevichNeuron(double a, double b, double c, double d,
                                   double v_init, double u_init)
    : a(a), b(b), c(c), d(d),
      v(v_init), u(u_init == -1e9 ? b * v_init : u_init),  // sentinel for auto‑init
      fired(false), energy(1.0)
{}

bool IzhikevichNeuron::update(double I, double dt_val) {
    double dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * dt_val;
    double du = (a * (b * v - u)) * dt_val;
    v += dv;
    u += du;
    fired = false;
    if (v >= 30.0) {
        v = c;
        u += d;
        fired = true;
        energy -= 0.005;
        if (energy < 0.0) energy = 0.0;
    }
    energy = std::min(1.0, energy + dt_val / 1000.0 * 0.01);
    return fired;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Comprehensive neuron type dictionary (same as Python NEURON_TYPES)
// ═══════════════════════════════════════════════════════════════════════════
static const std::map<std::string, NeuronParams> neuron_type_map = {
    // Excitatory pyramidal & spiny stellate cells
    {"RS",     {false, 0.02, 0.2, -65, 8}},
    {"IB",     {false, 0.02, 0.2, -55, 4}},
    {"CH",     {false, 0.02, 0.2, -50, 2}},
    {"RZ",     {false, 0.1,  0.26,-60,-1}},
    {"LTS",    {false, 0.02, 0.25,-65, 2}},
    {"TC",     {false, 0.02, 0.25,-65, 0.05}},
    {"SPINY",  {false, 0.02, 0.2, -65, 8}},

    // Inhibitory cortical interneurons
    {"FS",     {false, 0.1,  0.2, -65, 2}},
    {"LTSi",   {false, 0.02, 0.25,-65, 2}},
    {"RSi",    {false, 0.02, 0.2, -65, 8}},
    {"RZi",    {false, 0.1,  0.26,-60,-1}},
    {"MART",   {false, 0.02, 0.2, -65, 2}},
    {"NGC",    {false, 0.02, 0.2, -65, 8}},
    {"DBC",    {false, 0.02, 0.25,-65, 2}},
    {"BPC",    {false, 0.1,  0.2, -65, 2}},
    {"CHANDEL",{false, 0.1,  0.2, -65, 2}},
    {"VIP",    {false, 0.02, 0.2, -55, 4}},

    // Striatum
    {"MSN",    {false, 0.02, 0.2, -65, 8}},
    {"FSI",    {false, 0.1,  0.2, -65, 2}},

    // Cerebellum
    {"PC",     {true,  0, 0, 0, 0}},     // Hodgkin-Huxley
    {"GC",     {false, 0.02, 0.2, -65, 8}},
    {"GO",     {false, 0.1,  0.2, -65, 2}},
    {"SC",     {false, 0.02, 0.25,-65, 2}},
    {"BC",     {false, 0.1,  0.2, -65, 2}},
    {"CN",     {false, 0.02, 0.25,-50, 2}},

    // Special human cells (use HH)
    {"L5P",    {true,  0, 0, 0, 0}},
    {"BETZ",   {true,  0, 0, 0, 0}},
    {"SPINDLE",{true,  0, 0, 0, 0}},
    {"MARTINOTTI",{true, 0, 0, 0, 0}},

    // Hippocampus
    {"CA1PYR", {false, 0.02, 0.2, -65, 8}},
    {"CA3PYR", {false, 0.02, 0.2, -55, 4}},
    {"DGGC",   {false, 0.02, 0.2, -65, 8}},

    // Thalamus
    {"TCR",    {false, 0.02, 0.25,-65, 0.05}},
    {"TRN",    {false, 0.1,  0.25,-65, 2}},

    // Amygdala
    {"LA",     {false, 0.02, 0.2, -65, 8}},
    {"BA",     {false, 0.02, 0.2, -55, 4}},
    {"CE",     {false, 0.02, 0.2, -65, 2}},

    // Neuromodulatory (unsimplified)
    {"DA",     {false, 0.02, 0.25,-60, 2}},
    {"NE",     {false, 0.02, 0.2, -65, 4}},
    {"5HT",    {false, 0.02, 0.15,-65, 1}},
    {"ACh",    {false, 0.1,  0.2, -55, 0.5}}
};

// ═══════════════════════════════════════════════════════════════════════════
//  Public helpers
// ═══════════════════════════════════════════════════════════════════════════
NeuronParams getNeuronParams(const std::string& type) {
    auto it = neuron_type_map.find(type);
    if (it != neuron_type_map.end()) {
        return it->second;
    }
    // default: Regular Spiking
    return {false, 0.02, 0.2, -65.0, 8.0};
}

HodgkinHuxleyNeuron createHodgkinHuxleyNeuron(double dt) {
    return HodgkinHuxleyNeuron(1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.387, dt);
}

IzhikevichNeuron createIzhikevichNeuron(double a, double b, double c, double d, double dt) {
    return IzhikevichNeuron(a, b, c, d);
}

} // namespace abel
