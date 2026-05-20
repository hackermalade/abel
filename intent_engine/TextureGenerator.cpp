#include "TextureGenerator.h"
#include "../render/Texture.h"   // Texture class (must provide create, setPixel, setFiltering)
#include "../brain/Core.h"      // for clamp

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace abel {

// ── Local RNG for procedural noise ───────────────────────────────────────
static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uniform(-1.0, 1.0);

// ── Helper to safely read a latent dimension ────────────────────────────
static double getVal(const std::vector<double>& latent, size_t idx, double def = 0.0) {
    return (idx < latent.size()) ? latent[idx] : def;
}

// ── Simple Perlin‑like noise function (gradient noise) ──────────────────
static double fade(double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
static double lerp(double a, double b, double t) { return a + (b - a) * t; }

static double grad(int hash, double x, double y) {
    int h = hash & 3;
    double u = (h < 2) ? x : y;
    double v = (h < 2) ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

static double noise2D(double x, double y, const std::vector<int>& perm) {
    int X = (int)std::floor(x) & 255;
    int Y = (int)std::floor(y) & 255;
    x -= std::floor(x);
    y -= std::floor(y);
    double u = fade(x);
    double v = fade(y);
    int a = perm[X] + Y;
    int b = perm[X + 1] + Y;
    return lerp(lerp(grad(perm[a], x, y), grad(perm[b], x - 1, y), u),
                lerp(grad(perm[a + 1], x, y - 1), grad(perm[b + 1], x - 1, y - 1), u), v);
}

TextureGenerator::TextureGenerator() {}

Texture TextureGenerator::generate(const std::vector<double>& latent) {
    // Default texture size
    const int width = 256;
    const int height = 256;
    Texture tex;
    tex.create(width, height, 4);   // RGBA

    // Decode texture type from first latent dimension (0‑4 mapped to categories)
    double type_raw = getVal(latent, 0, 0.0);
    int tex_type = (int)((type_raw + 1.0) * 2.5); // 0..4
    tex_type = std::clamp(tex_type, 0, 4);

    // Base colour
    double baseR = std::clamp((getVal(latent, 1, 0.5) + 1.0) * 0.5, 0.0, 1.0);
    double baseG = std::clamp((getVal(latent, 2, 0.5) + 1.0) * 0.5, 0.0, 1.0);
    double baseB = std::clamp((getVal(latent, 3, 0.5) + 1.0) * 0.5, 0.0, 1.0);

    // Secondary colour (for patterns)
    double secR = std::clamp((getVal(latent, 4, -0.5) + 1.0) * 0.5, 0.0, 1.0);
    double secG = std::clamp((getVal(latent, 5, -0.5) + 1.0) * 0.5, 0.0, 1.0);
    double secB = std::clamp((getVal(latent, 6, -0.5) + 1.0) * 0.5, 0.0, 1.0);

    // Pattern frequency / scale
    double freq = std::abs(getVal(latent, 7, 0.5)) * 20.0 + 1.0; // 1‑21

    // For noise: build a permutation table seeded by latent values
    std::vector<int> perm(512);
    std::iota(perm.begin(), perm.begin() + 256, 0);
    // Shuffle using latent as seed (simplistic)
    unsigned seed = (unsigned)(std::abs(getVal(latent, 8, 0.0)) * 100000.0);
    std::mt19937 permRng(seed);
    std::shuffle(perm.begin(), perm.begin() + 256, permRng);
    for (int i = 0; i < 256; ++i) perm[256 + i] = perm[i];

    // Fill pixels
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double r = baseR, g = baseG, b = baseB, a = 1.0;
            double fx = (double)x / width;
            double fy = (double)y / height;

            switch (tex_type) {
                case 0: // solid color
                    break;
                case 1: // horizontal gradient
                    r = lerp(baseR, secR, fy);
                    g = lerp(baseG, secG, fy);
                    b = lerp(baseB, secB, fy);
                    break;
                case 2: // checkerboard
                    {
                        int squares = (int)(freq);
                        if (squares < 2) squares = 2;
                        int sx = (int)(fx * squares) % 2;
                        int sy = (int)(fy * squares) % 2;
                        if (sx == sy) { r = secR; g = secG; b = secB; }
                    }
                    break;
                case 3: // stripes
                    {
                        double stripe = std::sin(fx * freq * 2.0 * M_PI);
                        double t = (stripe + 1.0) * 0.5;
                        r = lerp(baseR, secR, t);
                        g = lerp(baseG, secG, t);
                        b = lerp(baseB, secB, t);
                    }
                    break;
                case 4: // noise cloud
                    {
                        double n = noise2D(fx * freq, fy * freq, perm);
                        n = (n + 1.0) * 0.5; // 0‑1
                        r = lerp(baseR, secR, n);
                        g = lerp(baseG, secG, n);
                        b = lerp(baseB, secB, n);
                    }
                    break;
                default: break;
            }

            tex.setPixel(x, y, (unsigned char)(r * 255),
                                (unsigned char)(g * 255),
                                (unsigned char)(b * 255),
                                (unsigned char)(a * 255));
        }
    }

    // Set default filtering (linear)
    tex.setFiltering(true);

    return tex;
}

} // namespace abel
