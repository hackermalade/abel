// abel/world/PhysicsEngine.h
#pragma once
#include <unordered_map>

class btCollisionConfiguration;
class btCollisionDispatcher;
class btBroadphaseInterface;
class btSequentialImpulseConstraintSolver;
class btDiscreteDynamicsWorld;
class btRigidBody;

namespace abel {

class Entity;
class Player;
struct Coord3D;

class PhysicsEngine {
public:
    PhysicsEngine();
    ~PhysicsEngine();

    // Entity physics
    void addEntity(Entity* entity);
    void removeEntity(Entity* entity);
    btRigidBody* getRigidBody(const Entity* entity) const;

    // Player physics
    void addPlayer(Player* player);
    void removePlayer(Player* player);
    btRigidBody* getPlayerRigidBody(const Player* player) const;

    // World step & gravity
    void step(float deltaTime);
    void setGravity(const Coord3D& gravity);

    btDiscreteDynamicsWorld* getWorld() const;

private:
    btCollisionConfiguration* collisionConfig_ = nullptr;
    btCollisionDispatcher* dispatcher_ = nullptr;
    btBroadphaseInterface* broadphase_ = nullptr;
    btSequentialImpulseConstraintSolver* solver_ = nullptr;
    btDiscreteDynamicsWorld* dynamicsWorld_ = nullptr;

    std::unordered_map<int, btRigidBody*> rigidBodies_;          // entity ID → body
    std::unordered_map<const Player*, btRigidBody*> playerBodies_; // player → body
};

} // namespace abel
