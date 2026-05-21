// abel/brain/Column.h
#pragma once
#include "Core.h"
#include "Neurons.h"            // provides NeuronRecord, getNeuronParams
#include "Glia.h"
#include "Synapses.h"
#include "PredictiveCoder.h"
#include "spatial/Coords.h"
#include <vector>
#include <memory>
#include <string>

namespace abel {

struct NeuronEntry {
    NeuronRecord neuron;
    std::string type;
    std::string layer;
    int idx;
    Coord3D rel_pos;
    Coord3D abs_pos;
};

class CorticalColumn {
public:
    CorticalColumn(int col_id, const std::string& region,
                   int neurons_per_layer = 30, double dt = 0.5,
                   const Coord3D& position = Coord3D(0,0,0));

    std::vector<int> updateNeurons(double dt, const std::vector<double>& external_currents);
    void propagateSpikes(const std::vector<int>& fires, double t);
    std::vector<double> collectCurrents() const;
    void updateGlia(int spike_count, double dt);
    double predictiveStep(const std::vector<double>& activity);

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
