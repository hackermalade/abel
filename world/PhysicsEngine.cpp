#include "PhysicsEngine.h"
#include "../world/Entity.h"
#include "../world/Player.h"
#include "../render/Mesh.h"
#include "../intent_engine/PhysicsGenerator.h"   // for PhysicsParams
#include "../brain/spatial/Coords.h"
#include "../brain/Core.h"

#include <btBulletDynamicsCommon.h>
#include <btBulletCollisionCommon.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <unordered_map>
#include <cassert>
#include <algorithm>
#include <memory>

namespace abel {

// ---------------------------------------------------------------------------
//  EntityMotionState – synchronises Bullet transform with an Entity/Player
// ---------------------------------------------------------------------------
class EntityMotionState : public btMotionState {
public:
    explicit EntityMotionState(Entity* entity)
        : entity_(entity), player_(nullptr)
    {
        assert(entity_);
    }

    explicit EntityMotionState(Player* player)
        : entity_(nullptr), player_(player)
    {
        assert(player_);
    }

    void getWorldTransform(btTransform& worldTrans) const override {
        if (entity_) {
            glm::vec3 pos = entity_->getPosition();
            glm::quat rot = entity_->getRotation();
            worldTrans.setOrigin(btVector3(pos.x, pos.y, pos.z));
            worldTrans.setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));
        } else if (player_) {
            glm::vec3 pos = player_->getPosition();
            glm::quat rot = player_->getRotation();
            worldTrans.setOrigin(btVector3(pos.x, pos.y, pos.z));
            worldTrans.setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));
        }
    }

    void setWorldTransform(const btTransform& worldTrans) override {
        btVector3  origin = worldTrans.getOrigin();
        btQuaternion q    = worldTrans.getRotation();
        glm::vec3 newPos(origin.x(), origin.y(), origin.z());
        glm::quat newRot(q.w(), q.x(), q.y(), q.z());

        if (entity_) {
            entity_->setPosition(newPos);
            entity_->setRotation(newRot);
        } else if (player_) {
            player_->setPosition(newPos);
            player_->setRotation(newRot);
        }
    }

    Entity* getEntity() const { return entity_; }
    Player* getPlayer() const { return player_; }

private:
    Entity* entity_;
    Player* player_;
};

// ---------------------------------------------------------------------------
//  Helper: create a collision shape from a Mesh (compound of convex hulls)
// ---------------------------------------------------------------------------
static btCollisionShape* createCollisionShapeFromMesh(const Mesh* mesh) {
    if (!mesh || mesh->getVertices().empty())
        return new btEmptyShape();

    const auto& verts = mesh->getVertices();
    const auto& indices = mesh->getIndices();

    auto* compound = new btCompoundShape();
    const size_t chunkSize = 64;
    for (size_t start = 0; start < indices.size(); start += chunkSize * 3) {
        btConvexHullShape* hull = new btConvexHullShape();
        for (size_t i = start; i < start + chunkSize * 3 && i + 2 < indices.size(); i += 3) {
            const Vertex& v0 = verts[indices[i]];
            const Vertex& v1 = verts[indices[i + 1]];
            const Vertex& v2 = verts[indices[i + 2]];
            hull->addPoint(btVector3(v0.x, v0.y, v0.z), false);
            hull->addPoint(btVector3(v1.x, v1.y, v1.z), false);
            hull->addPoint(btVector3(v2.x, v2.y, v2.z), false);
        }
        hull->recalcLocalAabb();
        btTransform ident;
        ident.setIdentity();
        compound->addChildShape(ident, hull);
    }
    return compound;
}

static btCollisionShape* createDefaultShape() {
    return new btBoxShape(btVector3(0.5f, 0.5f, 0.5f));
}

static btCollisionShape* createPlayerShape() {
    // human‑sized capsule
    return new btCapsuleShape(0.3f, 1.8f);
}

// ---------------------------------------------------------------------------
//  PhysicsEngine implementation
// ---------------------------------------------------------------------------
PhysicsEngine::PhysicsEngine() {
    collisionConfig_   = new btDefaultCollisionConfiguration();
    dispatcher_        = new btCollisionDispatcher(collisionConfig_);
    broadphase_        = new btDbvtBroadphase();
    solver_            = new btSequentialImpulseConstraintSolver();

    dynamicsWorld_ = new btDiscreteDynamicsWorld(dispatcher_, broadphase_,
                                                 solver_, collisionConfig_);
    dynamicsWorld_->setGravity(btVector3(0.0f, 0.0f, -9.81f));   // Z‑up
}

PhysicsEngine::~PhysicsEngine() {
    // Remove all entities and players
    for (auto& pair : rigidBodies_) {
        dynamicsWorld_->removeRigidBody(pair.second);
        delete pair.second->getMotionState();
        delete pair.second->getCollisionShape();
        delete pair.second;
    }
    for (auto& pair : playerBodies_) {
        dynamicsWorld_->removeRigidBody(pair.second);
        delete pair.second->getMotionState();
        delete pair.second->getCollisionShape();
        delete pair.second;
    }
    rigidBodies_.clear();
    playerBodies_.clear();

    delete dynamicsWorld_;
    delete solver_;
    delete broadphase_;
    delete dispatcher_;
    delete collisionConfig_;
}

// ── Entity physics ──────────────────────────────────────────────────────
void PhysicsEngine::addEntity(Entity* entity) {
    if (!entity) return;

    int entityId = entity->getId();
    auto it = rigidBodies_.find(entityId);
    if (it != rigidBodies_.end()) {
        removeEntity(entity);
    }

    btCollisionShape* shape = nullptr;
    const Mesh* mesh = entity->getMesh();
    if (mesh && !mesh->getVertices().empty()) {
        shape = createCollisionShapeFromMesh(mesh);
    } else {
        shape = createDefaultShape();
    }

    glm::vec3 pos   = entity->getPosition();
    glm::quat rot   = entity->getRotation();
    PhysicsParams p = entity->getPhysicsParams();

    btTransform startTransform;
    startTransform.setIdentity();
    startTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
    startTransform.setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));

    auto* motionState = new EntityMotionState(entity);

    btVector3 localInertia(0.0f, 0.0f, 0.0f);
    if (p.mass > 0.0f && !p.isKinematic) {
        shape->calculateLocalInertia(p.mass, localInertia);
    }

    btRigidBody::btRigidBodyConstructionInfo rbInfo(
        p.mass, motionState, shape, localInertia);
    rbInfo.m_friction        = p.friction;
    rbInfo.m_restitution     = p.restitution;
    rbInfo.m_linearDamping   = p.linearDamping;
    rbInfo.m_angularDamping  = p.angularDamping;
    rbInfo.m_rollingFriction = 0.01f;

    btRigidBody* body = new btRigidBody(rbInfo);
    body->setUserPointer(static_cast<void*>(entity));   // store direct pointer

    if (p.isKinematic) {
        body->setCollisionFlags(body->getCollisionFlags()
                                | btCollisionObject::CF_KINEMATIC_OBJECT);
        body->setActivationState(DISABLE_DEACTIVATION);
    }
    if (p.isTrigger) {
        body->setCollisionFlags(body->getCollisionFlags()
                                | btCollisionObject::CF_NO_CONTACT_RESPONSE);
    }

    dynamicsWorld_->addRigidBody(body);
    rigidBodies_[entityId] = body;
    entity->physics_ = body;   // store raw pointer in entity
}

void PhysicsEngine::removeEntity(Entity* entity) {
    if (!entity) return;

    int entityId = entity->getId();
    auto it = rigidBodies_.find(entityId);
    if (it != rigidBodies_.end()) {
        btRigidBody* body = it->second;
        dynamicsWorld_->removeRigidBody(body);
        delete body->getMotionState();
        delete body->getCollisionShape();
        delete body;
        rigidBodies_.erase(it);
        entity->physics_ = nullptr;
    }
}

btRigidBody* PhysicsEngine::getRigidBody(const Entity* entity) const {
    auto it = rigidBodies_.find(entity->getId());
    return (it != rigidBodies_.end()) ? it->second : nullptr;
}

// ── Player physics ──────────────────────────────────────────────────────
void PhysicsEngine::addPlayer(Player* player) {
    if (!player) return;

    auto it = playerBodies_.find(player);
    if (it != playerBodies_.end()) {
        removePlayer(player);
    }

    btCollisionShape* shape = createPlayerShape();

    glm::vec3 pos = player->getPosition();
    glm::quat rot = player->getRotation();

    btTransform startTransform;
    startTransform.setIdentity();
    startTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
    startTransform.setRotation(btQuaternion(rot.x, rot.y, rot.z, rot.w));

    auto* motionState = new EntityMotionState(player);

    btVector3 localInertia(0.0f, 0.0f, 0.0f);
    btRigidBody::btRigidBodyConstructionInfo rbInfo(
        0.0f, motionState, shape, localInertia);
    rbInfo.m_friction    = 0.8f;
    rbInfo.m_restitution = 0.1f;
    rbInfo.m_linearDamping  = 0.1f;
    rbInfo.m_angularDamping = 0.2f;
    rbInfo.m_rollingFriction = 0.01f;

    btRigidBody* body = new btRigidBody(rbInfo);
    body->setCollisionFlags(body->getCollisionFlags()
                            | btCollisionObject::CF_KINEMATIC_OBJECT);
    body->setActivationState(DISABLE_DEACTIVATION);
    body->setUserPointer(static_cast<void*>(player));

    dynamicsWorld_->addRigidBody(body);
    playerBodies_[player] = body;
}

void PhysicsEngine::removePlayer(Player* player) {
    if (!player) return;

    auto it = playerBodies_.find(player);
    if (it != playerBodies_.end()) {
        btRigidBody* body = it->second;
        dynamicsWorld_->removeRigidBody(body);
        delete body->getMotionState();
        delete body->getCollisionShape();
        delete body;
        playerBodies_.erase(it);
    }
}

btRigidBody* PhysicsEngine::getPlayerRigidBody(const Player* player) const {
    auto it = playerBodies_.find(player);
    return (it != playerBodies_.end()) ? it->second : nullptr;
}

// ── World step & gravity ────────────────────────────────────────────────
void PhysicsEngine::step(float deltaTime) {
    dynamicsWorld_->stepSimulation(deltaTime, 3, 1.0f / 60.0f);

    // Per‑entity gravity factor (Bullet uses a single global gravity)
    for (auto& pair : rigidBodies_) {
        btRigidBody* body = pair.second;
        if (!body->isActive()) continue;
        Entity* entity = static_cast<Entity*>(body->getUserPointer());
        if (entity) {
            float gf = entity->getPhysicsParams().gravityFactor;
            if (gf != 1.0f) {
                btVector3 grav = dynamicsWorld_->getGravity();
                btVector3 force = grav * (gf - 1.0f) * body->getMass();
                body->applyCentralForce(force);
            }
        }
    }
}

void PhysicsEngine::setGravity(const Coord3D& gravity) {
    dynamicsWorld_->setGravity(btVector3(gravity.x, gravity.y, gravity.z));
}

btDiscreteDynamicsWorld* PhysicsEngine::getWorld() const {
    return dynamicsWorld_;
}

} // namespace abel
