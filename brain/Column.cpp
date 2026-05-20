#include "Column.h"
#include "Neurons.h"
#include "Glia.h"
#include "Synapses.h"
#include "PredictiveCoder.h"
#include "spatial/Coords.h"

#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace abel;

// ─────────────────────── Helper: random number gen ───────────────────────
static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uniform(0.0, 1.0);
static std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

// ─────────────────────── Construction ────────────────────────────────────
CorticalColumn::CorticalColumn(int col_id, const std::string& region,
                               int neurons_per_layer, double dt,
                               const Coord3D& position)
    : id(col_id), region(region), dt(dt), position(position)
{
    buildNeurons(neurons_per_layer);
    buildGlia();
    buildInternalConnections();
    predictor = std::make_shared<PredictiveCoder>((int)neurons.size(), 32);
}

// ─────────────────────── Neuron population ───────────────────────────────
void CorticalColumn::buildNeurons(int npl) {
    // Layer specifications with proportions
    struct LayerSpec { std::string name; std::vector<std::pair<std::string, double>> types; };
    std::vector<LayerSpec> layer_specs = {
        {"L1",   {{"RSi",0.5}, {"LTSi",0.5}}},
        {"L2/3", {{"RS",0.4}, {"IB",0.1}, {"FS",0.2},
                  {"LTSi",0.1}, {"CD",0.1}, {"SPINDLE",0.1}}},
        {"L4",   {{"RS",0.3}, {"FS",0.4}, {"LTS",0.3}}},
        {"L5",   {{"L5P",0.5}, {"BETZ",0.05}, {"FS",0.2},
                  {"LTSi",0.1}, {"RS",0.15}}},
        {"L6",   {{"RSi",0.3}, {"LTSi",0.2}, {"FS",0.2}, {"RS",0.3}}}
    };

    // Depths of each layer (normalised 0‑1 within column)
    std::map<std::string, double> layer_depth = {
        {"L1", 0.0}, {"L2/3", 0.2}, {"L4", 0.4}, {"L5", 0.6}, {"L6", 0.8}
    };
    const double column_radius = 0.2;   // mm

    for (auto& layer : layer_specs) {
        int total = npl;
        std::map<std::string, double> probs;
        for (auto& t : layer.types) probs[t.first] = t.second;
        double sum_p = 0.0;
        for (auto& p : probs) sum_p += p.second;
        std::map<std::string, int> counts;
        for (auto& p : probs) counts[p.first] = (int)(total * p.second / sum_p);
        int diff = total - std::accumulate(counts.begin(), counts.end(), 0,
            [](int a, const std::pair<std::string,int>& b){ return a + b.second; });
        counts[layer.types.front().first] += diff;

        for (auto& [ntype, cnt] : counts) {
            NeuronParams params = getNeuronParams(ntype);   // from Neurons.h
            double depth = layer_depth[layer.name];
            for (int j = 0; j < cnt; ++j) {
                Neuron neur;
                if (params.isHH) {
                    neur = createHodgkinHuxleyNeuron(dt);   // factory
                } else {
                    neur = createIzhikevichNeuron(params.a, params.b, params.c, params.d, dt);
                }

                // Random position within the layer
                double angle = angle_dist(rng);
                double r = column_radius * std::sqrt(uniform(rng));
                double x_off = r * std::cos(angle);
                double y_off = r * std::sin(angle);
                double z_off = depth * uniform(rng) * 0.2 + depth;  // slight jitter around depth

                Coord3D rel_pos(x_off, y_off, z_off);
                Coord3D abs_pos = position + rel_pos;

                neurons.push_back({neur, ntype, layer.name, (int)neurons.size(), rel_pos, abs_pos});
            }
        }
    }
}

// ─────────────────────── Glial population ─────────────────────────────────
void CorticalColumn::buildGlia() {
    int n = (int)neurons.size();
    for (int i = 0; i < std::max(1, n/10); ++i)
        astrocytes.push_back(Astrocyte(id, dt));
    for (int i = 0; i < std::max(1, n/20); ++i)
        oligodendrocytes.push_back(Oligodendrocyte(id));
    for (int i = 0; i < std::max(1, n/50); ++i)
        microglia.push_back(Microglia(id));
}

// ─────────────────────── Internal synaptic wiring ─────────────────────────
void CorticalColumn::buildInternalConnections() {
    std::map<std::pair<std::string, std::string>, double> conn_prob = {
        {{"L2/3","L5"}, 0.1}, {{"L4","L2/3"}, 0.15}, {{"L5","L2/3"}, 0.05},
        {{"L6","L4"}, 0.1},   {{"L1","L2/3"}, 0.2},  {{"L5","L6"}, 0.1},
        {{"L6","L1"}, 0.05}
    };

    for (auto& pre : neurons) {
        bool is_E = !(pre.type == "FS" || pre.type == "LTSi" || pre.type == "RSi" || pre.type == "RZi");
        for (auto& post : neurons) {
            if (pre.idx == post.idx) continue;
            double prob = 0.0;
            if (pre.layer == post.layer) prob = 0.15;
            else {
                auto key = std::make_pair(pre.layer, post.layer);
                auto it = conn_prob.find(key);
                prob = (it != conn_prob.end()) ? it->second : 0.05;
            }
            if (!is_E) prob *= 1.5;

            if (uniform(rng) < prob) {
                std::string stype = is_E ? "AMPA" : "GABAa";
                double w = uniform(rng) * 0.9 + 0.1; // 0.1‑1.0
                auto syn = std::make_shared<Synapse>(pre.idx, post.idx, w, stype);
                synapses.push_back(syn);
            }
        }
    }
}

// ─────────────────────── Neural update ────────────────────────────────────
std::vector<int> CorticalColumn::updateNeurons(double dt,
                                               const std::vector<double>& external_currents) {
    std::vector<int> fires;
    for (int i = 0; i < (int)neurons.size(); ++i) {
        double I = external_currents[i];
        bool fired = false;
        if (neurons[i].neuron.isHH) {
            fired = neurons[i].neuron.hh->update(I, dt);
        } else {
            fired = neurons[i].neuron.izh->update(I, dt);
        }
        if (fired) fires.push_back(i);
    }
    return fires;
}

// ─────────────────────── Spike propagation ───────────────────────────────
void CorticalColumn::propagateSpikes(const std::vector<int>& fires, double t) {
    for (auto& syn : synapses) {
        if (std::find(fires.begin(), fires.end(), syn->pre) != fires.end()) {
            double dg = syn->preSpike(t);
            syn->g += dg;
        }
    }
}

// ─────────────────────── Collect synaptic currents ────────────────────────
std::vector<double> CorticalColumn::collectCurrents() const {
    std::vector<double> I(neurons.size(), 0.0);
    for (auto& syn : synapses) {
        double V_post = neurons[syn->post].neuron.isHH
                            ? neurons[syn->post].neuron.hh->V
                            : neurons[syn->post].neuron.izh->v;
        I[syn->post] += syn->g * (syn->E - V_post);
    }
    return I;
}

// ─────────────────────── Glial update & pruning ──────────────────────────
void CorticalColumn::updateGlia(int spike_count, double dt) {
    for (auto& ast : astrocytes) ast.update(spike_count, dt);
    double inflammation = spike_count / (double)neurons.size() * 5.0;
    for (auto& mic : microglia) {
        mic.update(inflammation, dt);
        if (mic.state == "activated") {
            mic.pruneSynapses(synapses);
        }
    }
    // Remove pruned synapses
    synapses.erase(std::remove_if(synapses.begin(), synapses.end(),
                                  [](const auto& s) { return s->pruned; }),
                   synapses.end());
}

// ─────────────────────── Predictive coding step ──────────────────────────
double CorticalColumn::predictiveStep(const std::vector<double>& activity) {
    return predictor->forward(activity);
}
