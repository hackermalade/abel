#include "Topology.h"
#include "Coords.h"
#include "../Core.h"

#include <random>
#include <map>
#include <vector>
#include <string>
#include <cmath>

namespace abel {

// ── Local random number generator ──────────────────────────────────────
static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uniform(-1.0, 1.0);

// ── Static region definitions ──────────────────────────────────────────
static const std::map<std::string, RegionBBox> region_bboxes = {
    {"PFC",       { 0.0,  65.0, 55.0,  25.0, 20.0, 15.0}},
    {"ACC",       { 0.0,  40.0, 50.0,  10.0, 15.0, 10.0}},
    {"INS",       {35.0,  30.0, 25.0,  10.0, 10.0, 10.0}},
    {"V1",        { 0.0, -60.0, 40.0,  20.0, 15.0, 15.0}},
    {"A1",        {55.0,   0.0, 25.0,  10.0, 15.0, 10.0}},
    {"THAL",      { 0.0,   5.0, 10.0,  15.0, 10.0, 10.0}},
    {"HIP",       {25.0,   0.0,  0.0,  10.0, 20.0,  5.0}},
    {"AMY",       {20.0,  25.0,  0.0,  10.0, 10.0,  8.0}},
    {"STR",       {20.0,  20.0, 10.0,  15.0, 15.0, 10.0}},
    {"CBL",       { 0.0, -70.0,-20.0,  30.0, 20.0, 15.0}},
    {"PCC",       { 0.0, -30.0, 50.0,  10.0, 15.0, 10.0}},
    {"PREMOTOR",  { 0.0,  50.0, 60.0,  25.0, 10.0, 10.0}}
};

// ── Public methods ─────────────────────────────────────────────────────
SpatialTopology::SpatialTopology() {}

std::map<std::string, std::vector<Coord3D>>
SpatialTopology::generateBrainGeometry(
        const std::map<std::string, int>& region_ncols) const {
    std::map<std::string, std::vector<Coord3D>> geometry;

    for (const auto& [region, ncol] : region_ncols) {
        auto it = region_bboxes.find(region);
        if (it == region_bboxes.end())
            throw std::runtime_error("Unknown brain region: " + region);

        const RegionBBox& bbox = it->second;
        std::vector<Coord3D> positions;
        positions.reserve(ncol);

        for (int i = 0; i < ncol; ++i) {
            double x = bbox.cx + uniform(rng) * bbox.sx;
            double y = bbox.cy + uniform(rng) * bbox.sy;
            double z = bbox.cz + uniform(rng) * bbox.sz;
            positions.emplace_back(x, y, z);
        }
        geometry[region] = std::move(positions);
    }
    return geometry;
}

std::vector<double> SpatialTopology::retinotopicMap(const Coord3D& col_pos,
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

std::vector<double> SpatialTopology::tonotopicMap(const Coord3D& col_pos,
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
