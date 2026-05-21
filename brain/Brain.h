// abel/brain/Brain.h
#pragma once
#include "Core.h"
#include "Column.h"
#include "Synapses.h"
#include "Neurons.h"
#include "Glia.h"
#include "PredictiveCoder.h"
#include "Quantum.h"
#include "spatial/Coords.h"
#include "spatial/Topology.h"
#include "spatial/Wiring.h"

#include <vector>
#include <map>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <fstream>
#include <numeric>
#include <algorithm>

namespace abel {

struct LongRangeSyn {
    std::shared_ptr<Synapse> syn;
    std::shared_ptr<CorticalColumn> src_col;
    std::shared_ptr<CorticalColumn> tgt_col;
    double delay;   // ms
};

struct WorkspaceNeuron {
    std::shared_ptr<CorticalColumn> col;
    int idx;
};

struct DelayedSpike {
    std::shared_ptr<Synapse> syn;
    std::shared_ptr<CorticalColumn> tgt_col;
    double g_inc;
};

class Brain {
public:
    Brain(int embedding_dim = 64, double dt = 0.5);

    void processSensory(const std::vector<double>& embedding);
    void step();
    bool isWorkspaceIgnited() const;
    std::vector<double> getWorkspaceVector() const;
    std::vector<double> getContextVector() const;
    std::vector<double> getMotorOutput() const;
    double getRecentReward() const;

    // Mood & logit bias (for LLM)
    std::tuple<double,double,double,double> getMoodVector() const;
    std::vector<float> getLogitBias(int vocab_size) const;

    // Persistence
    void saveState(const std::string& path) const;
    void loadState(const std::string& path);

    // Background thread
    void startBackgroundUpdates(int dt_ms = 50);
    void stopBackground();

    // Accessors
    const std::map<std::string, std::vector<std::shared_ptr<CorticalColumn>>>& getRegions() const { return regions; }
    double getDA() const { return DA; }
    double getNE() const { return NE; }
    double getACh() const { return ACh; }
    double getSerotonin() const { return serotonin; }

private:
    void buildLongRange(WiringRules& wiring);
    void deliverDelayedSpikes(double t);

    // ---------- members ----------
    double dt;
    int embedding_dim;
    double time;

    std::map<std::string, std::vector<std::shared_ptr<CorticalColumn>>> regions;
    std::vector<std::shared_ptr<CorticalColumn>> columns;
    std::map<int, Coord3D> column_positions;
    std::map<std::string, std::vector<Coord3D>> region_col_positions;

    std::vector<LongRangeSyn> long_range_synapses;

    std::map<int, std::vector<double>> external_currents;

    std::vector<WorkspaceNeuron> workspace_neurons;
    bool ws_ignited;

    double wm_field = 0.0;

    double DA, NE, ACh, serotonin;

    std::deque<double> surprise_history;

    QuantumDecoherenceModule quantum_module;

    std::map<double, std::vector<DelayedSpike>> delay_buffer;

    std::thread bg_thread;
    bool running;
};

} // namespace abel
