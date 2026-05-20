#include "Glia.h"
#include "Core.h"
#include "Synapses.h"   // needed for Microglia::pruneSynapses

#include <cmath>
#include <random>

using namespace abel;

// ──────────────────────────────────────────────────────────────────────────
//  Astrocyte
// ──────────────────────────────────────────────────────────────────────────
Astrocyte::Astrocyte(int column_id, double dt)
    : id(column_id), dt(dt), ca(0.05), IP3(0.01), glio(0.0), lactate(0.5)
{}

void Astrocyte::update(int spike_count, double dt_override) {
    double dt = dt_override > 0.0 ? dt_override : this->dt;

    // IP3 production driven by local spike count
    IP3 += dt * (0.01 * spike_count - 0.1 * IP3);
    // Calcium increase
    ca  += dt * (0.5 * IP3 - 0.05 * ca);
    // Gliotransmitter release probability (sigmoidal)
    glio = 1.0 / (1.0 + std::exp(-10.0 * (ca - 0.3)));
    // Lactate production proportional to metabolic demand
    lactate = 0.5 + 0.5 * std::min(1.0, spike_count / 10.0);
}

void Astrocyte::modulateSynapse(std::shared_ptr<Synapse> syn) {
    if (glio > 0.1) {
        syn->w += 0.001 * glio * (1.0 - syn->w);
        syn->w = std::max(0.0, std::min(5.0, syn->w));
    }
}

// ──────────────────────────────────────────────────────────────────────────
//  Oligodendrocyte
// ──────────────────────────────────────────────────────────────────────────
Oligodendrocyte::Oligodendrocyte(int column_id)
    : id(column_id), myelin(0.75)   // start with moderate myelination
{
    // randomise slightly
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> dist(0.5, 1.0);
    myelin = dist(rng);
}

double Oligodendrocyte::delayFactor() const {
    return 1.0 / myelin;
}

void Oligodendrocyte::updateMyelin(double activity) {
    // Activity-dependent myelination (slow)
    myelin = std::min(1.0, myelin + 0.001 * activity);
}

// ──────────────────────────────────────────────────────────────────────────
//  Microglia
// ──────────────────────────────────────────────────────────────────────────
Microglia::Microglia(int column_id)
    : id(column_id), state("surveillance")
{}

void Microglia::update(double inflammation_signal, double dt) {
    // Activation dynamics: triggered by high inflammation, slowly returns to surveillance
    if (inflammation_signal > 0.5) {
        state = "activated";
    } else if (state == "activated" && inflammation_signal < 0.2) {
        state = "surveillance";
    }
}

void Microglia::pruneSynapses(std::vector<std::shared_ptr<Synapse>>& synapses) {
    if (state == "activated") {
        for (auto& syn : synapses) {
            if (syn->cumulative_trace < 0.01) {
                syn->pruned = true;
            }
        }
    }
}
