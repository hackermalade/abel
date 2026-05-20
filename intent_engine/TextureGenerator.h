#pragma once

#include "../render/Texture.h"
#include <vector>

namespace abel {

class TextureGenerator {
public:
    TextureGenerator();

    /// Generate a procedural texture from a latent vector (size 128 typical).
    Texture generate(const std::vector<double>& latent);
};

} // namespace abel
