#include "MeshGenerator.h"
#include "../render/Mesh.h"
#include "../brain/Core.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace abel {

static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uniform(-1.0, 1.0);

static inline double lerp(double a, double b, double t) { return a + (b - a) * t; }

// Helper to safely read a latent dimension
static double getVal(const std::vector<double>& latent, size_t idx, double def = 0.0) {
    return (idx < latent.size()) ? latent[idx] : def;
}

// ── Build a sphere (procedural, uses Mesh methods) ─────────────────
static void buildSphere(Mesh& mesh, double radius, int segments,
                        const std::vector<double>& latent, int& idx) {
    int rings = segments, sectors = segments;
    const float pi = 3.14159265358979323846f;
    std::vector<Vertex> verts;
    std::vector<unsigned int> indices;
    verts.reserve((rings+1)*(sectors+1));
    for (int r = 0; r <= rings; ++r) {
        float phi = pi * static_cast<float>(r) / rings;
        float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
        for (int s = 0; s <= sectors; ++s) {
            float theta = 2.0f * pi * static_cast<float>(s) / sectors;
            float sinTheta = std::sin(theta), cosTheta = std::cos(theta);
            float x = static_cast<float>(radius) * sinPhi * cosTheta;
            float y = static_cast<float>(radius) * sinPhi * sinTheta;
            float z = static_cast<float>(radius) * cosPhi;
            if (!latent.empty() && idx+1 < static_cast<int>(latent.size())) {
                double def = latent[idx++] * 0.2;
                x += static_cast<float>(def * std::sin(phi * 3.0));
                y += static_cast<float>(def * std::cos(theta * 2.0));
                z += static_cast<float>(def * std::sin(theta * 2.0));
            }
            verts.push_back({x, y, z,
                             sinPhi*cosTheta, sinPhi*sinTheta, cosPhi,
                             static_cast<float>(s)/sectors,
                             static_cast<float>(r)/rings,
                             0.8f, 0.8f, 0.8f, 1.0f});
        }
    }
    for (int r = 0; r < rings; ++r)
        for (int s = 0; s < sectors; ++s) {
            unsigned int a = r*(sectors+1)+s, b = a+1, c = a+(sectors+1), d = c+1;
            indices.insert(indices.end(), {a,b,d,a,d,c});
        }
    mesh.addVerticesIndices(std::move(verts), std::move(indices));
}

// ── Build a cylinder (procedural) ──────────────────────────────────
static void buildCylinder(Mesh& mesh, double radius, double height,
                          int segments, bool capTop, bool capBottom,
                          const std::vector<double>& latent, int& idx) {
    int sectors = segments;
    const float pi = 3.14159265358979323846f;
    std::vector<Vertex> verts;
    std::vector<unsigned int> indices;
    float halfH = static_cast<float>(height * 0.5);
    // body
    for (int i = 0; i <= 1; ++i) {
        float z = i ? halfH : -halfH;
        for (int s = 0; s <= sectors; ++s) {
            float theta = 2.0f * pi * static_cast<float>(s) / sectors;
            float cosT = std::cos(theta), sinT = std::sin(theta);
            float x = static_cast<float>(radius) * cosT;
            float y = static_cast<float>(radius) * sinT;
            if (!latent.empty() && idx < static_cast<int>(latent.size())) {
                double def = latent[idx++] * 0.1;
                x += static_cast<float>(def * std::sin(theta * 3.0));
                y += static_cast<float>(def * std::cos(theta * 2.0));
            }
            verts.push_back({x, y, z,
                             cosT, sinT, 0.0f,
                             static_cast<float>(s)/sectors,
                             static_cast<float>(i),
                             0.8f, 0.8f, 0.8f, 1.0f});
        }
    }
    for (int s = 0; s < sectors; ++s) {
        unsigned int a = s, b = s+1, c = s+sectors+1, d = s+sectors+2;
        indices.insert(indices.end(), {a,b,d,a,d,c});
    }
    // caps
    if (capBottom) {
        unsigned int baseIdx = static_cast<unsigned int>(verts.size());
        verts.push_back({0,0,-halfH, 0,0,-1, 0.5f,0.5f, 0.8f,0.8f,0.8f,1.0f});
        for (int s = 0; s <= sectors; ++s) {
            float theta = 2.0f * pi * static_cast<float>(s) / sectors;
            float x = static_cast<float>(radius) * std::cos(theta);
            float y = static_cast<float>(radius) * std::sin(theta);
            verts.push_back({x,y,-halfH, 0,0,-1,
                             (std::cos(theta)+1.0f)*0.5f,
                             (std::sin(theta)+1.0f)*0.5f,
                             0.8f,0.8f,0.8f,1.0f});
        }
        for (int s = 0; s < sectors; ++s)
            indices.insert(indices.end(), {baseIdx, baseIdx+s+1, baseIdx+s+2});
    }
    if (capTop) {
        unsigned int baseIdx = static_cast<unsigned int>(verts.size());
        verts.push_back({0,0,halfH, 0,0,1, 0.5f,0.5f, 0.8f,0.8f,0.8f,1.0f});
        for (int s = 0; s <= sectors; ++s) {
            float theta = 2.0f * pi * static_cast<float>(s) / sectors;
            float x = static_cast<float>(radius) * std::cos(theta);
            float y = static_cast<float>(radius) * std::sin(theta);
            verts.push_back({x,y,halfH, 0,0,1,
                             (std::cos(theta)+1.0f)*0.5f,
                             (std::sin(theta)+1.0f)*0.5f,
                             0.8f,0.8f,0.8f,1.0f});
        }
        for (int s = 0; s < sectors; ++s)
            indices.insert(indices.end(), {baseIdx, baseIdx+s+2, baseIdx+s+1});
    }
    mesh.addVerticesIndices(std::move(verts), std::move(indices));
}

// ── Constructor ────────────────────────────────────────────────────
MeshGenerator::MeshGenerator() {}

// ── Generate mesh from latent vector ───────────────────────────────
Mesh MeshGenerator::generate(const std::vector<double>& latent) {
    Mesh mesh;
    if (latent.empty()) {
        int dummyIdx = 0;
        buildSphere(mesh, 1.0, 16, latent, dummyIdx);
        return mesh;
    }

    double shape_selector = latent[0];
    int idx = 1;
    int shape_type = static_cast<int>(shape_selector * 5.0);
    shape_type = std::clamp(shape_type, 0, 4);

    auto getValLocal = [&](int pos, double def = 0.0) -> double {
        return (pos < static_cast<int>(latent.size())) ? latent[pos] : uniform(rng) * 0.5;
    };

    switch (shape_type) {
        case 0: {
            double radius = 0.5 + getValLocal(idx++) * 2.0;
            int segments = 8 + static_cast<int>(std::abs(getValLocal(idx++)) * 24);
            buildSphere(mesh, radius, segments, latent, idx);
            break;
        }
        case 1: {
            double radius = 0.2 + getValLocal(idx++) * 1.0;
            double height = 0.5 + getValLocal(idx++) * 2.0;
            int segments = 8 + static_cast<int>(std::abs(getValLocal(idx++)) * 16);
            buildCylinder(mesh, radius, height, segments, true, true, latent, idx);
            break;
        }
        case 2: {
            double base_radius = 0.3 + getValLocal(idx++) * 0.8;
            int num_spikes = 3 + static_cast<int>(std::abs(getValLocal(idx++)) * 10);
            double spike_len = 0.5 + getValLocal(idx++) * 2.0;
            double spike_radius = 0.05 + getValLocal(idx++) * 0.2;
            buildSphere(mesh, base_radius, 12, latent, idx);
            for (int i = 0; i < num_spikes; ++i) {
                double angle = 2.0 * PI * i / num_spikes + getValLocal(idx++) * 0.5;
                double tilt = (getValLocal(idx++) * 0.5) * PI;
                double cx = base_radius * std::cos(angle) * std::cos(tilt);
                double cy = base_radius * std::sin(angle) * std::cos(tilt);
                double cz = base_radius * std::sin(tilt);
                Mesh spike;
                buildCylinder(spike, spike_radius, spike_len, 6, false, false, latent, idx);
                spike.translate(static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz));
                mesh.merge(spike);
            }
            break;
        }
        case 3: {
            double major_radius = 0.5 + getValLocal(idx++) * 1.5;
            double minor_radius = 0.1 + getValLocal(idx++) * 0.4;
            int major_segments = 12 + static_cast<int>(std::abs(getValLocal(idx++)) * 16);
            int minor_segments = 6 + static_cast<int>(std::abs(getValLocal(idx++)) * 8);
            std::vector<Vertex> verts;
            for (int i = 0; i <= major_segments; ++i) {
                double theta = 2.0 * PI * i / major_segments;
                double ct = std::cos(theta), st = std::sin(theta);
                for (int j = 0; j <= minor_segments; ++j) {
                    double phi = 2.0 * PI * j / minor_segments;
                    double cp = std::cos(phi), sp = std::sin(phi);
                    float x = static_cast<float>((major_radius + minor_radius * cp) * ct);
                    float y = static_cast<float>((major_radius + minor_radius * cp) * st);
                    float z = static_cast<float>(minor_radius * sp);
                    if (idx < static_cast<int>(latent.size())) {
                        double def = latent[idx++] * 0.1;
                        x += static_cast<float>(def * std::sin(theta * 3.0));
                        y += static_cast<float>(def * std::cos(phi * 2.0));
                        z += static_cast<float>(def * std::cos(theta));
                    }
                    verts.push_back({x, y, z,
                                     static_cast<float>(x - major_radius*ct),
                                     static_cast<float>(y - major_radius*st),
                                     z,
                                     static_cast<float>(i)/major_segments,
                                     static_cast<float>(j)/minor_segments,
                                     0.8f,0.8f,0.8f,1.0f});
                }
            }
            std::vector<unsigned int> inds;
            for (int i = 0; i < major_segments; ++i)
                for (int j = 0; j < minor_segments; ++j) {
                    unsigned int a = i*(minor_segments+1)+j, b = a+1, c = a+(minor_segments+1), d = c+1;
                    inds.insert(inds.end(), {a,b,d,a,d,c});
                }
            mesh.addVerticesIndices(std::move(verts), std::move(inds));
            break;
        }
        case 4: {
            Mesh body;
            buildSphere(body, 0.4 + getValLocal(idx++)*0.6, 12, latent, idx);
            body.translate(0, 0, 0.5f);
            mesh.merge(body);
            int num_legs = 2 + static_cast<int>(std::abs(getValLocal(idx++)) * 4);
            for (int i = 0; i < num_legs; ++i) {
                double angle = 2.0 * PI * i / num_legs + getValLocal(idx++) * 0.3;
                double len = 0.5 + getValLocal(idx++) * 1.0;
                Mesh leg;
                buildCylinder(leg, 0.05, len, 6, true, false, latent, idx);
                leg.translate(static_cast<float>(0.2*std::cos(angle)),
                              static_cast<float>(0.2*std::sin(angle)), -0.3f);
                mesh.merge(leg);
            }
            Mesh head;
            buildSphere(head, 0.2 + getValLocal(idx++)*0.2, 10, latent, idx);
            head.translate(0, 0, 1.0f);
            mesh.merge(head);
            if (getValLocal(idx++) > 0.3) {
                Mesh tail;
                buildCylinder(tail, 0.04, 0.7, 6, false, false, latent, idx);
                tail.translate(0, 0, -0.2f);
                mesh.merge(tail);
            }
            break;
        }
        default: break;
    }

    // Set base color from remaining latent dimensions
    float r = 0.5f + 0.5f * static_cast<float>(getValLocal(idx++, 0.0));
    float g = 0.5f + 0.5f * static_cast<float>(getValLocal(idx++, 0.0));
    float b = 0.5f + 0.5f * static_cast<float>(getValLocal(idx++, 0.0));
    mesh.setBaseColor(r, g, b);

    return mesh;
}

} // namespace abel
