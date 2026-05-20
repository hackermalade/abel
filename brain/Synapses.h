#pragma once

#include <string>
#include <vector>
#include <memory>

namespace abel {

class Synapse {
public:
    Synapse(int pre_idx, int post_idx, double weight = 0.5,
            const std::string& type = "AMPA",
            double U = 0.5, double tau_d = 200.0, double tau_f = 100.0,
            double delay = 0.0);

    // Magnesium block for NMDA receptors
    double mgBlock(double V_post) const;

    // Presynaptic spike – returns conductance increment
    double preSpike(double t);

    // Postsynaptic spike – updates trace and timestamp
    void postSpike(double t);

    // Decay of conductance and traces, recovery of short‑term plasticity
    void update(double dt, double t);

    // STDP + BCM metaplasticity
    void stdpUpdate(double dt, double t);

    // ---- public members ----
    int pre, post;            // neuron indices
    double w;                 // synaptic weight (scaled conductance)
    std::string type;         // "AMPA", "NMDA", "GABAa", "GABAb"
    double E;                 // reversal potential (mV)
    double tau;               // conductance decay time constant (ms)

    // Short‑term plasticity parameters
    double U;                 // baseline utilisation
    double x;                 // available vesicles
    double y;                 // release fraction (active)
    double tau_d, tau_f;      // recovery and facilitation time constants (ms)

    // State variables
    double g;                 // conductance (nS)
    double pre_trace, post_trace;
    double last_pre, last_post;  // spike times (ms)
    double cumulative_trace;  // for microglial pruning
    bool pruned;              // mark for removal

    // BCM metaplasticity
    double theta_m;           // modification threshold
    double tau_theta;         // adaptation time constant (ms)

    double delay;             // conduction delay (ms), used by long‑range connections
};

} // namespace abel
