#include "MeshGenerator.h"
#include "../render/Mesh.h"
#include "../brain/Core.h"

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace abel {

// ── Local RNG for procedural noise ──────────────────────────────────────
static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uniform(-1.0, 1.0);

// ── Helper: lerp ────────────────────────────────────────────────────────
static inline double lerp(double a, double b, double t) {
    return a + (b - a) * t;
}

// ── Helper: 3D vertex with normal and UV ─────────────────────────────────
struct Vertex {
    double x, y, z;
    double nx, ny, nz;
    double u, v;
};

// ── Helper: build a sphere mesh (smooth) ─────────────────────────────────
static void buildSphere(Mesh& mesh, double radius, int segments, const std::vector<double>& latent, int& idx) {
    int rings = segments;
    int sectors = segments;
    std::vector<Vertex> verts;
    verts.reserve((rings+1)*(sectors+1));
    for (int r = 0; r <= rings; ++r) {
        double phi = PI * (double)r / rings;
        for (int s = 0; s <= sectors; ++s) {
            double theta = 2.0 * PI * (double)s / sectors;
            double x = radius * sin(phi) * cos(theta);
            double y = radius * sin(phi) * sin(theta);
            double z = radius * cos(phi);
            double nx = sin(phi) * cos(theta);
            double ny = sin(phi) * sin(theta);
            double nz = cos(phi);
            double u = (double)s / sectors;
            double v = (double)r / rings;
            // optional deformation from latent
            if (!latent.empty() && idx+1 < latent.size()) {
                double def = latent[idx++] * 0.2;
                x += def * sin(phi * 3.0);
                y += def * cos(theta * 2.0);
                z += def * sin(theta * 2.0);
            }
            verts.push_back({x, y, z, nx, ny, nz, u, v});
        }
    }
    std::vector<unsigned int> indices;
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            unsigned int a = r * (sectors+1) + s;
            unsigned int b = a + 1;
            unsigned int c = a + (sectors+1);
            unsigned int d = c + 1;
            indices.push_back(a); indices.push_back(b); indices.push_back(d);
            indices.push_back(a); indices.push_back(d); indices.push_back(c);
        }
    }
    mesh.addTriangles(verts, indices);
}

// ── Helper: build a cylinder/cone mesh ──────────────────────────────────
static void buildCylinder(Mesh& mesh, double radius, double height, int segments, bool capTop, bool capBottom, const std::vector<double>& latent, int& idx) {
    int sectors = segments;
    // body
    std::vector<Vertex> verts;
    for (int i = 0; i <= 1; ++i) {
        double z = i * height;
        for (int s = 0; s <= sectors; ++s) {
            double theta = 2.0 * PI * (double)s / sectors;
            double x = radius * cos(theta);
            double y = radius * sin(theta);
            double nx = cos(theta), ny = sin(theta), nz = 0;
            if (!latent.empty() && idx < latent.size()) {
                double def = latent[idx++] * 0.1;
                x += def * sin(theta * 3.0);
                y += def * cos(theta * 2.0);
            }
            verts.push_back({x, y, z, nx, ny, nz, (double)s/sectors, (double)i});
        }
    }
    std::vector<unsigned int> indices;
    for (int s = 0; s < sectors; ++s) {
        int a = s, b = s+1, c = s + sectors + 1, d = s + sectors + 2;
        indices.push_back(a); indices.push_back(b); indices.push_back(d);
        indices.push_back(a); indices.push_back(d); indices.push_back(c);
    }
    // caps
    if (capBottom) {
        // bottom
        for (int s = 0; s < sectors; ++s) {
            verts.push_back({0,0,0, 0,0,-1, 0.5,0.5}); // center
            double theta = 2.0 * PI * (double)s / sectors;
            double x = radius * cos(theta), y = radius * sin(theta);
            verts.push_back({x,y,0, 0,0,-1, (double)s/sectors,0});
        }
        int base = verts.size() - 2*sectors;
        for (int s = 0; s < sectors; ++s) {
            indices.push_back(base);
            indices.push_back(base + 2*s + 1);
            indices.push_back(base + 2*((s+1)%sectors) + 1);
        }
    }
    if (capTop) {
        // top similar
        for (int s = 0; s < sectors; ++s) {
            verts.push_back({0,0,height, 0,0,1, 0.5,0.5});
            double theta = 2.0 * PI * (double)s / sectors;
            double x = radius * cos(theta), y = radius * sin(theta);
            verts.push_back({x,y,height, 0,0,1, (double)s/sectors,1});
        }
        int base = verts.size() - 2*sectors;
        for (int s = 0; s < sectors; ++s) {
            indices.push_back(base);
            indices.push_back(base + 2*((s+1)%sectors) + 1);
            indices.push_back(base + 2*s + 1);
        }
    }
    mesh.addTriangles(verts, indices);
}

// ── Constructor ─────────────────────────────────────────────────────────
MeshGenerator::MeshGenerator() {}

// ── Generate mesh from latent vector ────────────────────────────────────
Mesh MeshGenerator::generate(const std::vector<double>& latent) {
    Mesh mesh;

    // Default latent size: 128
    if (latent.empty()) {
        // create a default sphere
        buildSphere(mesh, 1.0, 16, latent, int());
        return mesh;
    }

    // The first dimension determines shape type (0..1 mapped to categories)
    double shape_selector = latent[0];
    int idx = 1; // current position in latent vector

    // Map to shape categories: 0.0‑0.2 -> sphere, 0.2‑0.4 -> cylinder, 0.4‑0.6 -> star, 0.6‑0.8 -> torus, 0.8‑1.0 -> custom/complex
    int shape_type = (int)(shape_selector * 5.0);
    if (shape_type < 0) shape_type = 0;
    if (shape_type > 4) shape_type = 4;

    // Use further latent dimensions for parameters
    // Ensure we have enough dimensions, otherwise pad with random
    auto getVal = [&](int pos, double def = 0.0) -> double {
        return (pos < latent.size()) ? latent[pos] : uniform(rng)*0.5;
    };

    switch (shape_type) {
        case 0: { // Sphere (or elongated spheroid)
            double radius = 0.5 + getVal(idx++) * 2.0; // 0.5‑2.5
            int segments = 8 + (int)(std::abs(getVal(idx++)) * 24); // 8‑32
            buildSphere(mesh, radius, segments, latent, idx);
            break;
        }
        case 1: { // Cylinder / limb
            double radius = 0.2 + getVal(idx++) * 1.0;
            double height = 0.5 + getVal(idx++) * 2.0;
            int segments = 8 + (int)(std::abs(getVal(idx++)) * 16);
            buildCylinder(mesh, radius, height, segments, true, true, latent, idx);
            break;
        }
        case 2: { // Star shape: multiple cylinders as spikes from a sphere
            double base_radius = 0.3 + getVal(idx++) * 0.8;
            int num_spikes = 3 + (int)(std::abs(getVal(idx++)) * 10); // 3‑13
            double spike_len = 0.5 + getVal(idx++) * 2.0;
            double spike_radius = 0.05 + getVal(idx++) * 0.2;
            // central sphere
            buildSphere(mesh, base_radius, 12, latent, idx);
            // spikes as cylinders
            for (int i = 0; i < num_spikes; ++i) {
                double angle = 2.0 * PI * i / num_spikes + getVal(idx++) * 0.5;
                double tilt = (getVal(idx++) * 0.5) * PI;
                double cx = base_radius * cos(angle) * cos(tilt);
                double cy = base_radius * sin(angle) * cos(tilt);
                double cz = base_radius * sin(tilt);
                // create a cylinder pointing along spike direction, then translate
                Mesh spike;
                buildCylinder(spike, spike_radius, spike_len, 6, false, false, latent, idx);
                spike.translate(cx, cy, cz); // need transform method in Mesh
                mesh.merge(spike);
            }
            break;
        }
        case 3: { // Torus
            double major_radius = 0.5 + getVal(idx++) * 1.5;
            double minor_radius = 0.1 + getVal(idx++) * 0.4;
            int major_segments = 12 + (int)(std::abs(getVal(idx++)) * 16);
            int minor_segments = 6 + (int)(std::abs(getVal(idx++)) * 8);
            // procedural torus
            std::vector<Vertex> verts;
            for (int i = 0; i <= major_segments; ++i) {
                double theta = 2.0 * PI * i / major_segments;
                double ct = cos(theta), st = sin(theta);
                for (int j = 0; j <= minor_segments; ++j) {
                    double phi = 2.0 * PI * j / minor_segments;
                    double cp = cos(phi), sp = sin(phi);
                    double x = (major_radius + minor_radius * cp) * ct;
                    double y = (major_radius + minor_radius * cp) * st;
                    double z = minor_radius * sp;
                    // deformation
                    if (idx < latent.size()) {
                        double def = latent[idx++] * 0.1;
                        x += def * sin(theta * 3.0);
                        y += def * cos(phi * 2.0);
                        z += def * cos(theta);
                    }
                    verts.push_back({x, y, z, x-major_radius*ct, y-major_radius*st, z, (double)i/major_segments, (double)j/minor_segments});
                }
            }
            std::vector<unsigned int> indices;
            for (int i = 0; i < major_segments; ++i) {
                for (int j = 0; j < minor_segments; ++j) {
                    int a = i * (minor_segments+1) + j;
                    int b = a + 1;
                    int c = a + (minor_segments+1);
                    int d = c + 1;
                    indices.push_back(a); indices.push_back(b); indices.push_back(d);
                    indices.push_back(a); indices.push_back(d); indices.push_back(c);
                }
            }
            mesh.addTriangles(verts, indices);
            break;
        }
        case 4: { // Complex: animal/creature by combining primitives
            // Use latent dimensions to build a body (sphere), legs (cylinders), head, tail
            Mesh body;
            buildSphere(body, 0.4 + getVal(idx++)*0.6, 12, latent, idx);
            body.translate(0, 0, 0.5);
            mesh.merge(body);

            int num_legs = 2 + (int)(std::abs(getVal(idx++)) * 4); // 2‑6
            for (int i = 0; i < num_legs; ++i) {
                double angle = 2.0 * PI * i / num_legs + getVal(idx++) * 0.3;
                double len = 0.5 + getVal(idx++) * 1.0;
                Mesh leg;
                buildCylinder(leg, 0.05, len, 6, true, false, latent, idx);
                leg.translate(0.2*cos(angle), 0.2*sin(angle), -0.3);
                mesh.merge(leg);
            }
            // head
            Mesh head;
            buildSphere(head, 0.2 + getVal(idx++)*0.2, 10, latent, idx);
            head.translate(0, 0, 1.0);
            mesh.merge(head);
            // tail
            if (getVal(idx++) > 0.3) {
                Mesh tail;
                buildCylinder(tail, 0.04, 0.7, 6, false, false, latent, idx);
                tail.translate(0, 0, -0.2);
                mesh.merge(tail);
            }
            break;
        }
        default: break;
    }

    // Apply overall color from remaining latent dimensions (embed into mesh material)
    double r = 0.5 + 0.5 * getVal(idx++);
    double g = 0.5 + 0.5 * getVal(idx++);
    double b = 0.5 + 0.5 * getVal(idx++);
    mesh.setBaseColor(r, g, b);

    return mesh;
}

} // namespace abel
