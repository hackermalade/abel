#include "EvolutionManager.h"
#include "../world/World.h"
#include "../world/Entity.h"
#include "MeshGenerator.h"
#include "TextureGenerator.h"
#include "PhysicsGenerator.h"

#include <random>
#include <algorithm>
#include <cmath>

namespace abel {

static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uniform(-1.0, 1.0);
static std::normal_distribution<double> noise(0.0, 0.05);

EvolutionManager::EvolutionManager() {}

void EvolutionManager::mutateExistingEntity(World& world,
                                            int entity_id,
                                            const std::vector<double>& mutation) {
    auto* entity = world.getEntity(entity_id);
    if (!entity) return;

    std::vector<double> latent = entity->getLatentVector();
    if (latent.empty()) {
        latent = mutation;
        if (latent.empty()) latent.resize(128, 0.0);
    }

    const double mutation_strength = 0.3;
    for (size_t i = 0; i < latent.size() && i < mutation.size(); ++i) {
        latent[i] = (1.0 - mutation_strength) * latent[i] + mutation_strength * mutation[i];
    }
    if (mutation.size() > latent.size()) {
        latent.insert(latent.end(), mutation.begin() + latent.size(), mutation.end());
    }

    for (auto& val : latent) {
        val += noise(rng);
        val = std::max(-1.0, std::min(1.0, val));
    }

    MeshGenerator meshGen;
    TextureGenerator texGen;
    PhysicsGenerator physGen;

    Mesh mesh = meshGen.generate(latent);
    Texture texture = texGen.generate(latent);
    PhysicsParams physics = physGen.generate(latent);

    entity->setMesh(std::move(mesh));
    entity->setTexture(std::move(texture));
    entity->applyPhysicsParams(physics);
    entity->setLatentVector(latent);

    world.markEntityDirty(entity_id);
}

void EvolutionManager::randomMutation(World& world) {
    const auto& entities = world.getAllEntities();
    if (entities.empty()) return;

    std::uniform_int_distribution<int> pick(0, (int)entities.size() - 1);
    int idx = pick(rng);
    int entity_id = entities[idx]->getId();

    std::vector<double> mutation(128);
    for (auto& val : mutation) val = uniform(rng) * 0.2;

    mutateExistingEntity(world, entity_id, mutation);
}

} // namespace abel
