#pragma once

#include <vector>
#include <random>
#include <cmath>

namespace abel {

// Forward declarations (full includes are in the .cpp)
class World;
class Entity;

class EvolutionManager {
public:
    EvolutionManager();

    // Mutate a specific entity using a mutation vector
    void mutateExistingEntity(World& world,
                              int entity_id,
                              const std::vector<double>& mutation);

    // Randomly pick an entity and apply a small random mutation
    void randomMutation(World& world);
};

} // namespace abel
