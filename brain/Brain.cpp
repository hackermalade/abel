#include "Brain.h"
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

#include <cmath>
#include <random>
#include <algorithm>
#include <thread>
#include <chrono>

using namespace abel;

// ─── Constructor ──────────────────────────────────────────────────────
Brain::Brain(int embedding_dim, double dt)
    : dt(dt),
      embedding_dim(embedding_dim),
      time(0.0),
      DA(0.3), NE(0.4), ACh(0.6), serotonin(0.5),
      ws_ignited(false),
      quantum_module(10, dt),
      running(false)
{
    // Region definitions (number of columns per region)
    std::map<std::string, int> region_ncols = {
        {"PFC",5}, {"ACC",3}, {"INS",2}, {"THAL",4}, {"V1",4}, {"A1",2},
        {"HIP",3}, {"AMY",2}, {"STR",3}, {"CBL",5}, {"PCC",2}, {"PREMOTOR",2}
    };

    // Generate 3‑D positions for columns
    SpatialTopology topology;
    region_col_positions = topology.generateBrainGeometry(region_ncols);

    WiringRules wiring;
    int col_id = 0;
    for (const auto& [region, ncol] : region_ncols) {
        auto positions = region_col_positions[region];
        for (int i = 0; i < ncol; ++i) {
            Coord3D pos = positions[i];
            auto col = std::make_shared<CorticalColumn>(col_id, region, 30, dt, pos);
            columns.push_back(col);
            regions[region].push_back(col);
            column_positions[col_id] = pos;
            ++col_id;
        }
    }

    // Build long‑range connections using distance‑based rules
    buildLongRange(wiring);

    // Allocate external currents per column
    for (auto& col : columns) {
        external_currents[col->id] = std::vector<double>(col->neurons.size(), 0.0);
    }

    // Build workspace list (all L5 neurons in PFC, ACC, PCC, THAL)
    for (auto& reg_name : {"PFC", "ACC", "PCC", "THAL"}) {
        for (auto& col : regions[reg_name]) {
            for (int i = 0; i < col->neurons.size(); ++i) {
                if (col->neurons[i].layer == "L5") {
                    workspace_neurons.push_back({col, i});
                }
            }
        }
    }

    // Initialise the background thread later via startBackground
}

// ─── Long‑range wiring ────────────────────────────────────────────────
void Brain::buildLongRange(WiringRules& wiring) {
    // Projection list: (src, tgt, base_prob)
    std::vector<std::tuple<std::string, std::string, double>> proj_list = {
        {"THAL","PFC",0.3},{"THAL","ACC",0.2},{"THAL","V1",0.4},{"THAL","A1",0.4},
        {"PFC","THAL",0.2},{"V1","THAL",0.3},{"PFC","ACC",0.3},{"ACC","PFC",0.3},
        {"HIP","PFC",0.2},{"PFC","HIP",0.2},{"AMY","PFC",0.4},{"PFC","AMY",0.2},
        {"STR","THAL",0.2},{"PFC","STR",0.2},{"CBL","THAL",0.2},{"PFC","CBL",0.1},
        {"PREMOTOR","PFC",0.2},{"PFC","PREMOTOR",0.2},{"PCC","PFC",0.2},{"PFC","PCC",0.2}
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (auto& [src_reg, tgt_reg, base_prob] : proj_list) {
        for (auto& src_col : regions[src_reg]) {
            Coord3D src_pos = column_positions[src_col->id];
            for (auto& tgt_col : regions[tgt_reg]) {
                Coord3D tgt_pos = column_positions[tgt_col->id];
                double dist = src_pos.distanceTo(tgt_pos);
                double prob = wiring.connectProbability(dist, src_reg, tgt_reg, base_prob);
                if (prob <= 0) continue;

                for (int pre_i = 0; pre_i < src_col->neurons.size(); ++pre_i) {
                    for (int post_j = 0; post_j < tgt_col->neurons.size(); ++post_j) {
                        if (uniform(rng) < prob) {
                            double w = uniform(rng) * 0.9 + 0.1; // 0.1..1.0
                            auto syn = std::make_shared<Synapse>(pre_i, post_j, w, "AMPA");
                            double delay = wiring.conductionDelay(dist);
                            long_range_synapses.push_back({syn, src_col, tgt_col, delay});
                        }
                    }
                }
            }
        }
    }
}

// ─── Sensory input processing ─────────────────────────────────────────
void Brain::processSensory(const std::vector<double>& embedding) {
    // Project onto V1, A1
    for (auto& reg_name : {"V1", "A1"}) {
        for (auto& col : regions[reg_name]) {
            auto& ext = external_currents[col->id];
            for (int i = 0; i < col->neurons.size(); ++i) {
                ext[i] = 5.0 * embedding[i % embedding.size()];
            }
        }
    }
    // Run 20 steps (20 ms)
    for (int i = 0; i < 20; ++i) {
        step();
    }
}

// ─── Main simulation step ─────────────────────────────────────────────
void Brain::step() {
    double t = time;
    double dt = this->dt;
    std::map<int, std::vector<int>> all_fires;

    // Deliver delayed spikes
    deliverDelayedSpikes(t);

    // 1. Predictive surprise per column
    for (auto& col : columns) {
        std::vector<double> rate(col->neurons.size(), 0.0);
        for (int i = 0; i < col->neurons.size(); ++i) {
            rate[i] = col->neurons[i].neuron.fired ? 1.0 : 0.0;
        }
        double surp = col->predictiveStep(rate);
        surprise_history.push_back(surp);
        if (surprise_history.size() > 100) surprise_history.pop_front();
    }

    // 2. Update each column
    for (auto& col : columns) {
        auto I_syn = col->collectCurrents();
        auto& ext = external_currents[col->id];
        std::vector<double> I_total(col->neurons.size(), 0.0);
        for (int i = 0; i < I_syn.size(); ++i) {
            I_total[i] = I_syn[i] + ext[i];
        }
        auto fires = col->updateNeurons(dt, I_total);
        all_fires[col->id] = fires;
        col->propagateSpikes(fires, t);
        // Reset external currents
        std::fill(ext.begin(), ext.end(), 0.0);
    }

    // 3. Long‑range spike propagation (with delay)
    for (auto& lr : long_range_synapses) {
        if (std::find(all_fires[lr.src_col->id].begin(),
                      all_fires[lr.src_col->id].end(),
                      lr.syn->pre) != all_fires[lr.src_col->id].end()) {
            double g_inc = lr.syn->preSpike(t);
            double arrival = t + lr.delay;
            delay_buffer[arrival].push_back({lr.syn, lr.tgt_col, g_inc});
        }
    }

    // 4. Plasticity & glia
    for (auto& col : columns) {
        for (auto& syn : col->synapses) {
            syn->stdpUpdate(dt, t);
            for (auto& ast : col->astrocytes) {
                if (ast.glio > 0.2) ast.modulateSynapse(syn);
            }
        }
        for (auto& oligo : col->oligodendrocytes) {
            double act = (double)all_fires[col->id].size() / col->neurons.size();
            oligo.updateMyelin(act);
        }
        int spike_count = (int)all_fires[col->id].size();
        col->updateGlia(spike_count, dt);
    }
    for (auto& lr : long_range_synapses) {
        lr.syn->stdpUpdate(dt, t);
    }

    // 5. Energy metabolism
    for (auto& col : columns) {
        for (auto& nrn : col->neurons) {
            for (auto& ast : col->astrocytes) {
                nrn.neuron.energy = std::min(1.0, nrn.neuron.energy + 0.01 * ast.lactate);
            }
        }
    }

    // 6. Neuromodulator dynamics
    double avg_surprise = 0.0;
    if (!surprise_history.empty()) {
        avg_surprise = std::accumulate(surprise_history.begin(), surprise_history.end(), 0.0) / surprise_history.size();
    }
    int total_spikes = 0;
    int total_neurons = 0;
    for (auto& col : columns) {
        total_spikes += (int)all_fires[col->id].size();
        total_neurons += (int)col->neurons.size();
    }
    double avg_rate = total_spikes / std::max(1.0, (double)total_neurons) / dt * 1000.0;
    DA = 0.95 * DA + 0.05 * std::max(0.0, (avg_rate - 1.0) / 10.0);
    NE = 0.95 * NE + 0.05 * std::abs(avg_rate - 2.0) / 5.0;
    ACh = 0.95 * ACh + 0.05 * avg_rate / 5.0;
    serotonin = 0.95 * serotonin + 0.05 * (1.0 - avg_rate / 5.0);

    // 7. Quantum decoherence
    std::vector<double> thalamic_rate;
    for (auto& col : regions["THAL"]) {
        for (int i = 0; i < std::min(10, (int)col->neurons.size()); ++i) {
            thalamic_rate.push_back(col->neurons[i].neuron.fired ? 1.0 : 0.0);
        }
    }
    if (thalamic_rate.size() >= 10) {
        std::vector<double> qprobs = quantum_module.evolve(thalamic_rate);
        for (int i = 0; i < std::min(10, (int)workspace_neurons.size()); ++i) {
            auto& nrn = workspace_neurons[i].first->neurons[workspace_neurons[i].second].neuron;
            nrn.V += qprobs[i] * 2.0 - 1.0; // tiny perturbation
        }
    }

    // 8. Workspace ignition
    int ws_spikes = 0;
    for (auto& [col, idx] : workspace_neurons) {
        if (std::find(all_fires[col->id].begin(), all_fires[col->id].end(), idx) != all_fires[col->id].end()) {
            ++ws_spikes;
        }
    }
    double ws_frac = (double)ws_spikes / std::max(1.0, (double)workspace_neurons.size());
    ws_ignited = ws_frac > 0.3;

    // 9. Working memory (PFC activity)
    double pfc_activity = 0.0;
    for (auto& col : regions["PFC"]) {
        pfc_activity += (double)all_fires[col->id].size() / col->neurons.size();
    }
    if (!regions["PFC"].empty()) {
        pfc_activity /= regions["PFC"].size();
    }
    wm_field = 0.9 * wm_field + 0.1 * pfc_activity;

    time += dt;
}

// ─── Deliver delayed spikes ───────────────────────────────────────────
void Brain::deliverDelayedSpikes(double t) {
    auto it = delay_buffer.find(t);
    if (it != delay_buffer.end()) {
        for (auto& entry : it->second) {
            entry.syn->g += entry.g_inc;
        }
        delay_buffer.erase(it);
    }
}

// ─── Public accessors ─────────────────────────────────────────────────
bool Brain::isWorkspaceIgnited() const { return ws_ignited; }

std::vector<double> Brain::getWorkspaceVector() const {
    std::vector<double> vec(64, 0.0);
    vec[0] = ws_ignited ? 1.0 : 0.0;
    for (int i = 0; i < 32; ++i) vec[1 + i] = wm_field;
    double avg_surprise = 0.0;
    if (!surprise_history.empty())
        avg_surprise = std::accumulate(surprise_history.begin(), surprise_history.end(), 0.0) / surprise_history.size();
    vec[33] = avg_surprise * 0.1;
    return vec;
}

std::vector<double> Brain::getContextVector() const {
    // Could add more context (e.g., last sensory embedding)
    return std::vector<double>(64, 0.0); // placeholder
}

std::vector<double> Brain::getMotorOutput() const {
    std::vector<double> motor(64, 0.0);
    for (auto& col : regions.at("PREMOTOR")) {
        for (auto& nrn : col->neurons) {
            if (nrn.neuron.fired) motor[0] += 0.1;
            else motor[0] -= 0.05;
        }
    }
    return motor;
}

double Brain::getRecentReward() const {
    // Reward = surprise reduction (negative mean of recent surprises)
    if (surprise_history.empty()) return 0.0;
    double avg = std::accumulate(surprise_history.begin(), surprise_history.end(), 0.0) / surprise_history.size();
    return -avg;
}

// ─── Persistence ──────────────────────────────────────────────────────
void Brain::saveState(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&time), sizeof(time));
    ofs.write(reinterpret_cast<const char*>(&DA), sizeof(DA));
    ofs.write(reinterpret_cast<const char*>(&NE), sizeof(NE));
    ofs.write(reinterpret_cast<const char*>(&ACh), sizeof(ACh));
    ofs.write(reinterpret_cast<const char*>(&serotonin), sizeof(serotonin));
    // Add more detailed state if needed
}

void Brain::loadState(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return;
    ifs.read(reinterpret_cast<char*>(&time), sizeof(time));
    ifs.read(reinterpret_cast<char*>(&DA), sizeof(DA));
    ifs.read(reinterpret_cast<char*>(&NE), sizeof(NE));
    ifs.read(reinterpret_cast<char*>(&ACh), sizeof(ACh));
    ifs.read(reinterpret_cast<char*>(&serotonin), sizeof(serotonin));
}

// ─── Background thread ────────────────────────────────────────────────
void Brain::startBackground(int dt_ms) {
    running = true;
    bg_thread = std::thread([this, dt_ms]() {
        while (running) {
            this->step();
            std::this_thread::sleep_for(std::chrono::milliseconds(dt_ms));
        }
    });
}

void Brain::stopBackground() {
    running = false;
    if (bg_thread.joinable()) bg_thread.join();
}
