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

// ── Local RNG ───────────────────────────────────────────────────────────
static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> uniform(-1.0, 1.0);
static std::normal_distribution<double> noise(0.0, 0.05);

// ── Constructor ─────────────────────────────────────────────────────────
EvolutionManager::EvolutionManager() {}

// ── Mutate an existing entity ───────────────────────────────────────────
void EvolutionManager::mutateExistingEntity(World& world,
                                            int entity_id,
                                            const std::vector<double>& mutation) {
    auto* entity = world.getEntity(entity_id);
    if (!entity) return;

    // Current latent vector of the entity
    std::vector<double> latent = entity->getLatentVector();
    if (latent.empty()) {
        // If no latent vector exists, create one from the mutation
        latent = mutation;
        if (latent.empty()) latent.resize(128, 0.0); // default size
    }

    // Apply mutation: blend the current latent with the mutation vector
    // mutation strength = 0.3 (so the entity doesn't change drastically)
    const double mutation_strength = 0.3;
    for (size_t i = 0; i < latent.size() && i < mutation.size(); ++i) {
        latent[i] = (1.0 - mutation_strength) * latent[i] + mutation_strength * mutation[i];
    }
    // If mutation vector is longer, append the rest
    if (mutation.size() > latent.size()) {
        latent.insert(latent.end(), mutation.begin() + latent.size(), mutation.end());
    }

    // Add small Gaussian noise to encourage exploration
    for (auto& val : latent) {
        val += noise(rng);
        val = std::max(-1.0, std::min(1.0, val)); // keep bounded
    }

    // Regenerate the entity's mesh, texture, and physics from the new latent
    MeshGenerator meshGen;
    TextureGenerator texGen;
    PhysicsGenerator physGen;

    auto mesh = meshGen.generate(latent);
    auto texture = texGen.generate(latent);
    auto physics = physGen.generate(latent);

    entity->setMesh(mesh);
    entity->setTexture(texture);
    entity->applyPhysicsParams(physics);
    entity->setLatentVector(latent);

    // Optionally, trigger an event that the entity changed
    world.markEntityDirty(entity_id);
}

// ── Randomly mutate a random entity (called by Intent Engine) ──────────
void EvolutionManager::randomMutation(World& world) {
    const auto& entities = world.getAllEntities();
    if (entities.empty()) return;

    // Pick a random entity
    std::uniform_int_distribution<int> pick(0, (int)entities.size() - 1);
    int idx = pick(rng);
    int entity_id = entities[idx]->getId();

    // Create a small random mutation vector (size 128)
    std::vector<double> mutation(128);
    for (auto& val : mutation) {
        val = uniform(rng) * 0.2; // small amplitude
    }

    mutateExistingEntity(world, entity_id, mutation);
}

} // namespace abel
