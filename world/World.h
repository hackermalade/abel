#pragma once

#include <vector>
#include <memory>
#include <string>

#include "../brain/Core.h"                // for SensoryVector
#include "../brain/spatial/Coords.h"     // for Coord3D
#include "PhysicsEngine.h"               // full definition (was forward declaration)
#include "Entity.h" 

namespace abel {

class Entity;
class Player;

class World {
public:
    World();

    // Load an initial 3D scene from a GLB file
    void loadFromFile(const std::string& path);

    // Entity management
    void addEntity(std::unique_ptr<Entity> entity);
    void spawnEntity(const std::vector<double>& latent);
    Entity* getEntity(int id);
    const Entity* getEntity(int id) const;
    const std::vector<std::unique_ptr<Entity>>& getAllEntities() const;
    void removeEntity(int id);
    void markEntityDirty(int id);

    // Player
    void setPlayer(Player* player);
    Player* getPlayer();
    const Player* getPlayer() const;

    // Physics access (for rule changes, gravity, etc.)
    PhysicsEngine& getPhysicsEngine();
    const PhysicsEngine& getPhysicsEngine() const;

    // Sensory feedback for the brain
    SensoryVector getSensoryState(const Player& player) const;

    // Main update (physics, etc.)
    void update(float deltaTime);

    // Run state
    bool isRunning() const;
    void shutdown();

private:
    std::vector<std::unique_ptr<Entity>> entities_;
    PhysicsEngine physics_;
    Player* player_ = nullptr;
    bool running_;
};

} // namespace abel
