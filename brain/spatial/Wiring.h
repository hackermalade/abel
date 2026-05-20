#pragma once

#include "Coords.h"
#include <string>
#include <vector>
#include <map>

namespace abel {

class WiringRules {
public:
    WiringRules();

    double connectProbability(double distance_mm,
                              const std::string& src_region,
                              const std::string& tgt_region,
                              double base_prob) const;

    double conductionDelay(double distance_mm) const;

    std::vector<double> retinotopicMap(const Coord3D& col_pos, int out_dim = 64) const;
    std::vector<double> tonotopicMap(const Coord3D& col_pos, int out_dim = 64) const;

private:
    double local_sigma;
    double medium_sigma;
    double long_lambda;
    double transition_dist;

    double speed_unmyelinated;
    double speed_thin_myel;
    double speed_heavy_myel;

    std::map<std::string, double> region_bias;
};

} // namespace abel
