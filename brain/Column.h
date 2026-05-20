#pragma once

#include "Core.h"
#include "Neurons.h"        // for HodgkinHuxleyNeuron, IzhikevichNeuron, getNeuronParams
#include "Glia.h"           // for Astrocyte, Oligodendrocyte, Microglia
#include "Synapses.h"       // for Synapse
#include "PredictiveCoder.h"
#include "spatial/Coords.h"

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <random>
#include <algorithm>

namespace abel {

// ── Neuron data structures ───────────────────────────────────────────────
struct NeuronRecord {
    bool isHH;
    std::shared_ptr<HodgkinHuxleyNeuron> hh;
    std::shared_ptr<IzhikevichNeuron> izh;
};

struct NeuronEntry {
    NeuronRecord neuron;
    std::string type;
    std::string layer;
    int idx;              // global column neuron index
    Coord3D rel_pos;      // relative to column centre
    Coord3D abs_pos;      // absolute position in brain
};

// ── Cortical column class ────────────────────────────────────────────────
class CorticalColumn {
public:
    CorticalColumn(int col_id, const std::string& region,
                   int neurons_per_layer = 30, double dt = 0.5,
                   const Coord3D& position = Coord3D(0,0,0));

    // Neural dynamics
    std::vector<int> updateNeurons(double dt, const std::vector<double>& external_currents);
    void propagateSpikes(const std::vector<int>& fires, double t);
    std::vector<double> collectCurrents() const;

    // Glial maintenance
    void updateGlia(int spike_count, double dt);

    // Predictive coding
    double predictiveStep(const std::vector<double>& activity);

    // Member access
    int id;
    std::string region;
    double dt;
    Coord3D position;

    std::vector<NeuronEntry> neurons;
    std::vector<Astrocyte> astrocytes;
    std::vector<Oligodendrocyte> oligodendrocytes;
    std::vector<Microglia> microglia;
    std::vector<std::shared_ptr<Synapse>> synapses;
    std::shared_ptr<PredictiveCoder> predictor;

private:
    void buildNeurons(int npl);
    void buildGlia();
    void buildInternalConnections();
};

} // namespace abel
