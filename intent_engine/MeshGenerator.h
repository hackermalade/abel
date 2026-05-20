#pragma once

#include "../render/Mesh.h"
#include <vector>

namespace abel {

class MeshGenerator {
public:
    MeshGenerator();

    // Generate a 3D mesh from a latent vector (size 128 typical)
    Mesh generate(const std::vector<double>& latent);
};

} // namespace abel
