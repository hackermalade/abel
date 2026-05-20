#pragma once
#include "Coords.h"
#include <map>
#include <vector>
#include <string>

namespace abel {

struct RegionBBox {
    double cx, cy, cz;   // centre
    double sx, sy, sz;   // half‑widths
};

class SpatialTopology {
public:
    SpatialTopology();
    std::map<std::string, std::vector<Coord3D>>
    generateBrainGeometry(const std::map<std::string, int>& region_ncols) const;

    std::vector<double> retinotopicMap(const Coord3D& col_pos, int out_dim = 64) const;
    std::vector<double> tonotopicMap(const Coord3D& col_pos, int out_dim = 64) const;
};

} // namespace abel
