#include "Wiring.h"
#include "Coords.h"
#include "../Core.h"

#include <random>
#include <cmath>
#include <map>

namespace abel {

// ── Local random number generator ──────────────────────────────────────
static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uniform(0.0, 1.0);

// ── Internal helper functions ──────────────────────────────────────────
static double gaussianDecay(double x, double sigma) {
    return std::exp(-0.5 * (x / sigma) * (x / sigma));
}

static double exponentialDecay(double x, double lambda) {
    return std::exp(-lambda * x);
}

// ── Constructor ────────────────────────────────────────────────────────
WiringRules::WiringRules()
    : local_sigma(1.5),
      medium_sigma(8.0),
      long_lambda(0.02),
      transition_dist(15.0),
      speed_unmyelinated(0.5),
      speed_thin_myel(3.0),
      speed_heavy_myel(50.0)
{
    // Initialise region‑pair bias factors
    region_bias = {
        {"V1,THAL", 1.2}, {"THAL,V1", 1.2},
        {"A1,THAL", 1.1}, {"THAL,A1", 1.1},
        {"PFC,ACC", 1.3}, {"ACC,PFC", 1.3},
        {"HIP,PFC", 1.15},
        {"AMY,PFC", 1.25}, {"PFC,AMY", 1.0},
        {"STR,THAL", 1.1}
    };
}

// ── Connection probability ─────────────────────────────────────────────
double WiringRules::connectProbability(double distance_mm,
                                       const std::string& src_region,
                                       const std::string& tgt_region,
                                       double base_prob) const {
    if (distance_mm < 0.0) return 0.0;

    double p_dist = 0.0;
    if (distance_mm < 5.0) {
        p_dist = gaussianDecay(distance_mm, local_sigma);
    } else if (distance_mm < transition_dist) {
        p_dist = gaussianDecay(distance_mm, medium_sigma);
    } else {
        p_dist = exponentialDecay(distance_mm - transition_dist, long_lambda);
    }

    // Region‑pair specific bias
    std::string key = src_region + "," + tgt_region;
    auto it = region_bias.find(key);
    double bias = (it != region_bias.end()) ? it->second : 1.0;

    double prob = base_prob * p_dist * bias;
    return std::max(0.0, std::min(1.0, prob));
}

// ── Conduction delay ───────────────────────────────────────────────────
double WiringRules::conductionDelay(double distance_mm) const {
    double r = uniform(rng);
    double speed = 0.0;
    if (r < 0.6) {
        speed = speed_thin_myel;       // 60% thinly myelinated
    } else if (r < 0.9) {
        speed = speed_heavy_myel;      // 30% heavily myelinated
    } else {
        speed = speed_unmyelinated;    // 10% unmyelinated
    }
    return distance_mm / speed;
}

// ── Topographic maps (convenience) ─────────────────────────────────────
std::vector<double> WiringRules::retinotopicMap(const Coord3D& col_pos,
                                                int out_dim) const {
    std::vector<double> vec(out_dim, 0.0);
    double nx = col_pos.x / 20.0;
    double ny = col_pos.y / 15.0;
    double nz = col_pos.z / 15.0;

    for (int i = 0; i < out_dim; ++i) {
        double freq_x = (i + 1) * 0.7;
        double freq_y = (i * 1.3 + 1) * 0.6;
        double phase = i * 0.4;
        vec[i] = std::sin(freq_x * nx + freq_y * ny + phase) *
                 std::cos(freq_x * nz + phase);
    }
    return vec;
}

std::vector<double> WiringRules::tonotopicMap(const Coord3D& col_pos,
                                              int out_dim) const {
    std::vector<double> vec(out_dim, 0.0);
    double freq_axis = (col_pos.y + 10.0) / 20.0;
    freq_axis = std::max(0.0, std::min(1.0, freq_axis));

    for (int i = 0; i < out_dim; ++i) {
        double centre = freq_axis * out_dim;
        double dist = (i - centre) / (out_dim * 0.3);
        vec[i] = std::exp(-dist * dist);
    }
    return vec;
}

} // namespace abel
