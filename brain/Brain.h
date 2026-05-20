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
    // Constructor
    Brain(int embedding_dim = 64, double dt = 0.5);

    // Public interface
    void processSensory(const std::vector<double>& embedding);
    void step();
    bool isWorkspaceIgnited() const;
    std::vector<double> getWorkspaceVector() const;
    std::vector<double> getContextVector() const;
    std::vector<double> getMotorOutput() const;
    double getRecentReward() const;

    // Persistence
    void saveState(const std::string& path) const;
    void loadState(const std::string& path);

    // Background thread
    void startBackground(int dt_ms = 50);
    void stopBackground();

    // Accessors (for external interaction)
    const std::map<std::string, std::vector<std::shared_ptr<CorticalColumn>>>& getRegions() const { return regions; }
    double getDA() const { return DA; }
    double getNE() const { return NE; }
    double getACh() const { return ACh; }
    double getSerotonin() const { return serotonin; }

private:
    // Construction helpers
    void buildLongRange(WiringRules& wiring);
    void deliverDelayedSpikes(double t);

    // ---------- members ----------
    double dt;
    int embedding_dim;
    double time;

    // Regions and columns
    std::map<std::string, std::vector<std::shared_ptr<CorticalColumn>>> regions;
    std::vector<std::shared_ptr<CorticalColumn>> columns;
    std::map<int, Coord3D> column_positions;
    std::map<std::string, std::vector<Coord3D>> region_col_positions;

    // Long‑range connectivity
    std::vector<LongRangeSyn> long_range_synapses;

    // External currents per column (column.id -> vector of currents)
    std::map<int, std::vector<double>> external_currents;

    // Global workspace (L5 neurons in PFC/ACC/PCC/THAL)
    std::vector<WorkspaceNeuron> workspace_neurons;
    bool ws_ignited;

    // Working memory (PFC activity scalar)
    double wm_field = 0.0;

    // Neuromodulators
    double DA, NE, ACh, serotonin;

    // Surprise history (for predictive coding)
    std::deque<double> surprise_history;

    // Quantum module
    QuantumDecoherenceModule quantum_module;

    // Delay buffer (arrival time -> list of delayed spikes)
    std::map<double, std::vector<DelayedSpike>> delay_buffer;

    // Background thread
    std::thread bg_thread;
    bool running;
};

} // namespace abel
