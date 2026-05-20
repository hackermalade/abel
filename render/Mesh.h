// Mesh.h
#pragma once
#include <vector>
#include "../brain/spatial/Coords.h"   // for Coord3D

namespace abel {

struct Vertex {
    float x, y, z;        // position
    float nx, ny, nz;     // normal
    float u, v;           // texcoord
    float r, g, b, a;     // colour
};

class Mesh {
public:
    Mesh();
    ~Mesh();
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;

    void addSphere(float radius, const Coord3D& center, int segments = 16);
    void addCylinder(float radius, float height, const Coord3D& center,
                     int segments = 12, bool capTop = true, bool capBottom = true);
    void translate(float dx, float dy, float dz);
    void setBaseColor(float r, float g, float b, float a = 1.0f);
    void applyNoiseToLast(float deformAmount, float noiseScale = 0.1f);
    void merge(const Mesh& other);

    // Access for renderer
    const std::vector<Vertex>& getVertices() const { return vertices; }
    const std::vector<unsigned int>& getIndices() const { return indices; }

private:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    float baseColor[4] = {0.8f, 0.8f, 0.8f, 1.0f};  // default light gray

    int lastPrimitiveStart;
    int lastPrimitiveVertexCount;
};

} // namespace abel
