#include "World.h"
#include "Entity.h"
#include "Player.h"
#include "Camera.h"
#include "PhysicsEngine.h"
#include "../render/Mesh.h"
#include "../render/Texture.h"
#include "../intent_engine/MeshGenerator.h"
#include "../intent_engine/TextureGenerator.h"
#include "../intent_engine/PhysicsGenerator.h"
#include "../brain/Brain.h"
#include "../brain/spatial/Coords.h"
#include "../brain/Core.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace abel {

// ---------------------------------------------------------------------------
//  Minimal GLB / glTF 2.0 loader (binary GLB)
// ---------------------------------------------------------------------------
static std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

static void loadGLB(const std::string& path, World& world) {
    std::vector<char> data = readFile(path);
    if (data.size() < 12) throw std::runtime_error("Invalid GLB file");

    uint32_t magic = *reinterpret_cast<uint32_t*>(data.data());
    if (magic != 0x46546C67) throw std::runtime_error("Not a GLB file (magic mismatch)");

    uint32_t jsonLength = *reinterpret_cast<uint32_t*>(data.data() + 12);
    std::string jsonStr(data.data() + 16, jsonLength);
    nlohmann::json gltf = nlohmann::json::parse(jsonStr);

    size_t binOffset = 16 + jsonLength;
    if (binOffset + 8 > data.size()) throw std::runtime_error("GLB: missing binary chunk");
    uint32_t binLength = *reinterpret_cast<uint32_t*>(data.data() + binOffset);
    const uint8_t* binData = reinterpret_cast<const uint8_t*>(data.data() + binOffset + 8);

    auto getAccessorData = [&](int accIdx) -> const float* {
        if (accIdx < 0 || accIdx >= gltf["accessors"].size()) return nullptr;
        auto& acc = gltf["accessors"][accIdx];
        int viewIdx = acc["bufferView"];
        auto& view = gltf["bufferViews"][viewIdx];
        int byteOffset = acc.value("byteOffset", 0) + view["byteOffset"].get<int>();
        int componentType = acc["componentType"];
        if (componentType != 5126) return nullptr;
        return reinterpret_cast<const float*>(binData + byteOffset);
    };

    for (auto& meshJson : gltf["meshes"]) {
        for (auto& prim : meshJson["primitives"]) {
            Mesh mesh;
            int posAcc = prim["attributes"]["POSITION"];
            const float* posData = getAccessorData(posAcc);
            if (!posData) continue;
            int posCount = gltf["accessors"][posAcc]["count"];

            const float* normData = nullptr;
            if (prim["attributes"].contains("NORMAL")) {
                int normAcc = prim["attributes"]["NORMAL"];
                normData = getAccessorData(normAcc);
            }

            const float* uvData = nullptr;
            if (prim["attributes"].contains("TEXCOORD_0")) {
                int uvAcc = prim["attributes"]["TEXCOORD_0"];
                uvData = getAccessorData(uvAcc);
            }

            const uint16_t* idx16 = nullptr;
            const uint32_t* idx32 = nullptr;
            int idxCount = 0;
            if (prim.contains("indices")) {
                int idxAcc = prim["indices"];
                auto& idxAccessor = gltf["accessors"][idxAcc];
                idxCount = idxAccessor["count"];
                int viewIdx = idxAccessor["bufferView"];
                auto& view = gltf["bufferViews"][viewIdx];
                int byteOffset = idxAccessor.value("byteOffset", 0) + view["byteOffset"].get<int>();
                int compType = idxAccessor["componentType"];
                if (compType == 5123) idx16 = reinterpret_cast<const uint16_t*>(binData + byteOffset);
                else if (compType == 5125) idx32 = reinterpret_cast<const uint32_t*>(binData + byteOffset);
            }

            std::vector<Vertex> verts;
            verts.reserve(posCount);
            std::vector<unsigned int> inds;
            for (int i = 0; i < posCount; i++) {
                Vertex v;
                v.x = posData[i*3];  v.y = posData[i*3+1];  v.z = posData[i*3+2];
                if (normData) { v.nx = normData[i*3]; v.ny = normData[i*3+1]; v.nz = normData[i*3+2]; }
                else { v.nx = v.ny = v.nz = 0; }
                if (uvData) { v.u = uvData[i*2]; v.v = uvData[i*2+1]; }
                else { v.u = v.v = 0; }
                v.r = v.g = v.b = 0.8f; v.a = 1.0f;
                verts.push_back(v);
            }
            if (idxCount > 0) {
                for (int i = 0; i < idxCount; i++) {
                    unsigned int idx = idx16 ? idx16[i] : idx32[i];
                    inds.push_back(idx);
                }
            } else {
                for (int i = 0; i < posCount; i++) inds.push_back(i);
            }
            mesh.addVerticesIndices(std::move(verts), std::move(inds));

            auto ent = std::make_unique<Entity>();
            ent->setMesh(std::move(mesh));
            world.addEntity(std::move(ent));
        }
    }
}

// ---------------------------------------------------------------------------
//  World implementation
// ---------------------------------------------------------------------------
World::World()
    : running_(true)
{
}

void World::loadFromFile(const std::string& path) {
    loadGLB(path, *this);
}

void World::addEntity(std::unique_ptr<Entity> entity) {
    entities_.push_back(std::move(entity));
    physics_.addEntity(entities_.back().get());
}

void World::spawnEntity(const std::vector<double>& latent) {
    MeshGenerator meshGen;
    TextureGenerator texGen;
    PhysicsGenerator physGen;

    Mesh mesh = meshGen.generate(latent);
    Texture tex = texGen.generate(latent);
    PhysicsParams phys = physGen.generate(latent);

    auto ent = std::make_unique<Entity>();
    ent->setMesh(std::move(mesh));
    ent->setTexture(std::move(tex));
    ent->applyPhysicsParams(phys);
    ent->setLatentVector(latent);
    addEntity(std::move(ent));
}

Entity* World::getEntity(int id) {
    for (auto& e : entities_) {
        if (e->getId() == id) return e.get();
    }
    return nullptr;
}

const Entity* World::getEntity(int id) const {
    for (auto& e : entities_) {
        if (e->getId() == id) return e.get();
    }
    return nullptr;
}

const std::vector<std::unique_ptr<Entity>>& World::getAllEntities() const {
    return entities_;
}

void World::removeEntity(int id) {
    auto it = std::find_if(entities_.begin(), entities_.end(),
                           [id](const std::unique_ptr<Entity>& e) { return e->getId() == id; });
    if (it != entities_.end()) {
        physics_.removeEntity(it->get());
        entities_.erase(it);
    }
}

void World::markEntityDirty(int id) {
    // No explicit action needed; physics and renderer automatically sync
}

void World::setPlayer(Player* player) {
    player_ = player;
}

Player* World::getPlayer() {
    return player_;
}

const Player* World::getPlayer() const {
    return player_;
}

// ---------------------------------------------------------------------------
//  Sensory extraction – builds a 64‑element vector that the brain consumes
// ---------------------------------------------------------------------------
SensoryVector World::getSensoryState(const Player& player) const {
    SensoryVector sensory(64, 0.0);

    glm::vec3 ppos = player.getPosition();
    sensory[0] = ppos.x / 100.0f;
    sensory[1] = ppos.y / 100.0f;
    sensory[2] = ppos.z / 100.0f;

    // Player velocity placeholder – could be fetched from physics body
    sensory[3] = 0.0f;
    sensory[4] = 0.0f;
    sensory[5] = 0.0f;

    // Closest entity distance and direction
    float minDist = 1000.0f;
    Coord3D nearestPos(0,0,0);
    for (auto& e : entities_) {
        glm::vec3 epos = e->getPosition();
        float dx = ppos.x - epos.x;
        float dy = ppos.y - epos.y;
        float dz = ppos.z - epos.z;
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < minDist) {
            minDist = dist;
            nearestPos = Coord3D(epos.x, epos.y, epos.z);
        }
    }
    if (minDist < 1000.0f) {
        sensory[6] = minDist / 50.0f;
        sensory[7] = (nearestPos.x - ppos.x) / minDist;
        sensory[8] = (nearestPos.y - ppos.y) / minDist;
        sensory[9] = (nearestPos.z - ppos.z) / minDist;
    }

    // World metrics
    sensory[10] = (float)entities_.size() / 100.0f;
    sensory[11] = 0.0f;   // gravity x
    sensory[12] = 0.0f;   // gravity y
    sensory[13] = -1.0f;  // gravity z (down)

    // Remaining dimensions reserved for future sensory modalities
    return sensory;
}

void World::update(float deltaTime) {
    physics_.step(deltaTime);
}

bool World::isRunning() const { return running_; }
void World::shutdown() { running_ = false; }
PhysicsEngine& World::getPhysicsEngine() {
    return physics_;
}

const PhysicsEngine& World::getPhysicsEngine() const {
    return physics_;
}
} // namespace abel
