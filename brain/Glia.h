#pragma once

#include "Core.h"
#include <vector>
#include <memory>
#include <string>

// Forward declaration for synapse pointer used in modulation and pruning
#include "Synapses.h"   // for Synapse definition (needed by pruneSynapses parameter)

namespace abel {

class Astrocyte {
public:
    Astrocyte(int column_id, double dt = DEFAULT_BRAIN_DT);

    void update(int spike_count, double dt_override = -1.0);
    void modulateSynapse(std::shared_ptr<Synapse> syn);

    int id;
    double dt;
    double ca;          // internal calcium (arb. units)
    double IP3;         // inositol trisphosphate
    double glio;        // gliotransmitter release probability
    double lactate;     // energy supply level
};

class Oligodendrocyte {
public:
    Oligodendrocyte(int column_id);

    double delayFactor() const;
    void updateMyelin(double activity);

    int id;
    double myelin;      // degree of myelination (0.5‑1.0)
};

class Microglia {
public:
    Microglia(int column_id);

    void update(double inflammation_signal, double dt = 0.5);
    void pruneSynapses(std::vector<std::shared_ptr<Synapse>>& synapses);

    int id;
    std::string state;   // "surveillance" or "activated"
};

} // namespace abel
