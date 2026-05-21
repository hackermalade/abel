// abel/render/Mesh.h
#pragma once

#include <vector>
#include <array>
#include "../brain/spatial/Coords.h"   // for Coord3D

namespace abel {

// Forward declaration
class Texture;

// ── Vertex structure ────────────────────────────────────────────────────
struct Vertex {
    float x, y, z;        // position
    float nx, ny, nz;     // normal
    float u, v;           // texcoord
    float r, g, b, a;     // colour
};

// ── Mesh class ──────────────────────────────────────────────────────────
class Mesh {
public:
    Mesh();
    ~Mesh();

    // Move‑only (no copying)
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;

    // ── Raw vertex/index insertion (used by GLB loader) ──────────────
    void addVerticesIndices(std::vector<Vertex>&& verts,
                            std::vector<unsigned int>&& inds);

    // ── Procedural primitives ────────────────────────────────────────
    void addSphere(float radius, const Coord3D& center, int segments = 16);
    void addCylinder(float radius, float height, const Coord3D& center,
                     int segments = 12, bool capTop = true,
                     bool capBottom = true);

    // ── Transformations ──────────────────────────────────────────────
    void translate(float dx, float dy, float dz);

    // ── Appearance ──────────────────────────────────────────────────
    void setBaseColor(float r, float g, float b, float a = 1.0f);
    void applyNoiseToLast(float deformAmount, float noiseScale = 0.1f);

    // ── Merging ─────────────────────────────────────────────────────
    void merge(const Mesh& other);

    // ── Texture ─────────────────────────────────────────────────────
    void setTexture(const Texture* tex);
    const Texture* getTexture() const;

    // ── Accessors (for renderer) ────────────────────────────────────
    const std::vector<Vertex>& getVertices() const { return vertices; }
    const std::vector<unsigned int>& getIndices() const { return indices; }

private:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    float baseColor[4] = {0.8f, 0.8f, 0.8f, 1.0f};   // default light gray

    // Tracking of the last added primitive (for applyNoiseToLast)
    int lastPrimitiveStart = 0;
    int lastPrimitiveVertexCount = 0;

    // Optional texture
    const Texture* texture_ = nullptr;
};

} // namespace abel
