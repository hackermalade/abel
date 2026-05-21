// abel/render/Mesh.cpp
#include "Mesh.h"
#include "../brain/Core.h"
#include "../brain/spatial/Coords.h"

#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>

namespace abel {

static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<float> noiseDist(-1.0f, 1.0f);

Mesh::Mesh()
    : lastPrimitiveStart(0),
      lastPrimitiveVertexCount(0),
      texture_(nullptr)
{
    baseColor[0] = 0.8f;
    baseColor[1] = 0.8f;
    baseColor[2] = 0.8f;
    baseColor[3] = 1.0f;
}

Mesh::~Mesh() {}

Mesh::Mesh(Mesh&& other) noexcept
    : vertices(std::move(other.vertices)),
      indices(std::move(other.indices)),
      lastPrimitiveStart(other.lastPrimitiveStart),
      lastPrimitiveVertexCount(other.lastPrimitiveVertexCount),
      texture_(other.texture_)
{
    std::memcpy(baseColor, other.baseColor, sizeof(baseColor));
    other.lastPrimitiveStart = 0;
    other.lastPrimitiveVertexCount = 0;
    other.texture_ = nullptr;
}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if (this != &other) {
        vertices = std::move(other.vertices);
        indices = std::move(other.indices);
        std::memcpy(baseColor, other.baseColor, sizeof(baseColor));
        lastPrimitiveStart = other.lastPrimitiveStart;
        lastPrimitiveVertexCount = other.lastPrimitiveVertexCount;
        texture_ = other.texture_;
        other.lastPrimitiveStart = 0;
        other.lastPrimitiveVertexCount = 0;
        other.texture_ = nullptr;
    }
    return *this;
}

void Mesh::addVerticesIndices(std::vector<Vertex>&& verts,
                              std::vector<unsigned int>&& inds) {
    unsigned int vertexOffset = static_cast<unsigned int>(vertices.size());
    vertices.insert(vertices.end(), verts.begin(), verts.end());
    for (auto idx : inds) {
        indices.push_back(idx + vertexOffset);
    }
}

void Mesh::addSphere(float radius, const Coord3D& center, int segments) {
    const int rings = segments;
    const int sectors = segments;
    const float pi = static_cast<float>(PI);

    lastPrimitiveStart = static_cast<int>(vertices.size());
    lastPrimitiveVertexCount = 0;

    std::vector<Vertex> newVerts;
    newVerts.reserve((rings + 1) * (sectors + 1));

    for (int r = 0; r <= rings; ++r) {
        float phi = pi * static_cast<float>(r) / rings;
        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);
        for (int s = 0; s <= sectors; ++s) {
            float theta = 2.0f * pi * static_cast<float>(s) / sectors;
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);

            float x = radius * sinPhi * cosTheta;
            float y = radius * sinPhi * sinTheta;
            float z = radius * cosPhi;

            float nx = sinPhi * cosTheta;
            float ny = sinPhi * sinTheta;
            float nz = cosPhi;

            float u = static_cast<float>(s) / sectors;
            float v = static_cast<float>(r) / rings;

            newVerts.push_back({
                static_cast<float>(center.x) + x,
                static_cast<float>(center.y) + y,
                static_cast<float>(center.z) + z,
                nx, ny, nz, u, v,
                baseColor[0], baseColor[1], baseColor[2], baseColor[3]
            });
        }
    }

    std::vector<unsigned int> newIndices;
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            unsigned int a = r * (sectors + 1) + s;
            unsigned int b = a + 1;
            unsigned int c = a + (sectors + 1);
            unsigned int d = c + 1;
            newIndices.push_back(a);
            newIndices.push_back(b);
            newIndices.push_back(d);
            newIndices.push_back(a);
            newIndices.push_back(d);
            newIndices.push_back(c);
        }
    }

    unsigned int vertexOffset = static_cast<unsigned int>(vertices.size());
    for (auto& idx : newIndices)
        idx += vertexOffset;

    vertices.insert(vertices.end(), newVerts.begin(), newVerts.end());
    indices.insert(indices.end(), newIndices.begin(), newIndices.end());

    lastPrimitiveVertexCount = static_cast<int>(newVerts.size());
}

void Mesh::addCylinder(float radius, float height, const Coord3D& center,
                       int segments, bool capTop, bool capBottom) {
    const float pi = static_cast<float>(PI);
    const int sectors = segments;

    lastPrimitiveStart = static_cast<int>(vertices.size());
    lastPrimitiveVertexCount = 0;

    std::vector<Vertex> newVerts;
    newVerts.reserve((sectors + 1) * 2 + (capBottom ? sectors + 1 : 0) + (capTop ? sectors + 1 : 0));

    float halfH = height * 0.5f;
    float topY = static_cast<float>(center.z) + halfH;
    float bottomY = static_cast<float>(center.z) - halfH;

    // Side vertices (two rings)
    for (int i = 0; i <= sectors; ++i) {
        float theta = 2.0f * pi * static_cast<float>(i) / sectors;
        float cosT = std::cos(theta);
        float sinT = std::sin(theta);

        float x = radius * cosT;
        float y = radius * sinT;

        // Bottom ring
        newVerts.push_back({
            static_cast<float>(center.x) + x,
            static_cast<float>(center.y) + y,
            bottomY,
            cosT, sinT, 0.0f,
            static_cast<float>(i) / sectors, 0.0f,
            baseColor[0], baseColor[1], baseColor[2], baseColor[3]
        });
        // Top ring
        newVerts.push_back({
            static_cast<float>(center.x) + x,
            static_cast<float>(center.y) + y,
            topY,
            cosT, sinT, 0.0f,
            static_cast<float>(i) / sectors, 1.0f,
            baseColor[0], baseColor[1], baseColor[2], baseColor[3]
        });
    }

    // Side indices
    std::vector<unsigned int> newIndices;
    for (int i = 0; i < sectors; ++i) {
        unsigned int bl = i * 2;       // bottom left
        unsigned int tl = bl + 1;      // top left
        unsigned int br = (i + 1) * 2; // bottom right
        unsigned int tr = br + 1;      // top right
        newIndices.push_back(bl); newIndices.push_back(br); newIndices.push_back(tl);
        newIndices.push_back(tl); newIndices.push_back(br); newIndices.push_back(tr);
    }

    // Cap bottom (centre + ring)
    if (capBottom) {
        unsigned int centerIdx = static_cast<unsigned int>(newVerts.size());
        newVerts.push_back({
            static_cast<float>(center.x),
            static_cast<float>(center.y),
            bottomY,
            0.0f, 0.0f, -1.0f,
            0.5f, 0.5f,
            baseColor[0], baseColor[1], baseColor[2], baseColor[3]
        });
        for (int i = 0; i <= sectors; ++i) {
            float theta = 2.0f * pi * static_cast<float>(i) / sectors;
            float x = radius * std::cos(theta);
            float y = radius * std::sin(theta);
            newVerts.push_back({
                static_cast<float>(center.x) + x,
                static_cast<float>(center.y) + y,
                bottomY,
                0.0f, 0.0f, -1.0f,
                (std::cos(theta) + 1.0f) * 0.5f,
                (std::sin(theta) + 1.0f) * 0.5f,
                baseColor[0], baseColor[1], baseColor[2], baseColor[3]
            });
        }
        for (int i = 0; i < sectors; ++i) {
            newIndices.push_back(centerIdx);
            newIndices.push_back(centerIdx + i + 2);
            newIndices.push_back(centerIdx + i + 1);
        }
    }

    // Cap top (centre + ring)
    if (capTop) {
        unsigned int centerIdx = static_cast<unsigned int>(newVerts.size());
        newVerts.push_back({
            static_cast<float>(center.x),
            static_cast<float>(center.y),
            topY,
            0.0f, 0.0f, 1.0f,
            0.5f, 0.5f,
            baseColor[0], baseColor[1], baseColor[2], baseColor[3]
        });
        for (int i = 0; i <= sectors; ++i) {
            float theta = 2.0f * pi * static_cast<float>(i) / sectors;
            float x = radius * std::cos(theta);
            float y = radius * std::sin(theta);
            newVerts.push_back({
                static_cast<float>(center.x) + x,
                static_cast<float>(center.y) + y,
                topY,
                0.0f, 0.0f, 1.0f,
                (std::cos(theta) + 1.0f) * 0.5f,
                (std::sin(theta) + 1.0f) * 0.5f,
                baseColor[0], baseColor[1], baseColor[2], baseColor[3]
            });
        }
        for (int i = 0; i < sectors; ++i) {
            newIndices.push_back(centerIdx);
            newIndices.push_back(centerIdx + i + 1);
            newIndices.push_back(centerIdx + i + 2);
        }
    }

    unsigned int vertexOffset = static_cast<unsigned int>(vertices.size());
    for (auto& idx : newIndices)
        idx += vertexOffset;

    vertices.insert(vertices.end(), newVerts.begin(), newVerts.end());
    indices.insert(indices.end(), newIndices.begin(), newIndices.end());

    lastPrimitiveVertexCount = static_cast<int>(newVerts.size());
}

void Mesh::translate(float dx, float dy, float dz) {
    for (auto& v : vertices) {
        v.x += dx;
        v.y += dy;
        v.z += dz;
    }
}

void Mesh::setBaseColor(float r, float g, float b, float a) {
    baseColor[0] = r;
    baseColor[1] = g;
    baseColor[2] = b;
    baseColor[3] = a;
    for (auto& v : vertices) {
        v.r = r;
        v.g = g;
        v.b = b;
        v.a = a;
    }
}

void Mesh::applyNoiseToLast(float deformAmount, float noiseScale) {
    if (lastPrimitiveVertexCount <= 0) return;

    int start = lastPrimitiveStart;
    int end = start + lastPrimitiveVertexCount;

    for (int i = start; i < end; ++i) {
        float dx = noiseDist(rng) * noiseScale * deformAmount;
        float dy = noiseDist(rng) * noiseScale * deformAmount;
        float dz = noiseDist(rng) * noiseScale * deformAmount;
        vertices[i].x += dx;
        vertices[i].y += dy;
        vertices[i].z += dz;
    }
}

void Mesh::merge(const Mesh& other) {
    unsigned int offset = static_cast<unsigned int>(vertices.size());
    vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
    for (auto idx : other.indices) {
        indices.push_back(idx + offset);
    }
    lastPrimitiveStart = 0;
    lastPrimitiveVertexCount = static_cast<int>(vertices.size());
}

void Mesh::setTexture(const Texture* tex) {
    texture_ = tex;
}

const Texture* Mesh::getTexture() const {
    return texture_;
}

} // namespace abel
