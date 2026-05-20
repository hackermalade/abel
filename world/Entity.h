#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>
#include "../render/Mesh.h"
#include "../render/Texture.h"
#include "../intent_engine/PhysicsGenerator.h"   // for PhysicsParams
#include "../brain/spatial/Coords.h"

namespace abel {

class Entity {
public:
    Entity();
    ~Entity();

    // Move only (no copying of unique resources)
    Entity(Entity&& other) noexcept;
    Entity& operator=(Entity&& other) noexcept;

    // Identity
    int getId() const;
    const std::string& getName() const;
    void setName(const std::string& name);

    // Visuals
    void setMesh(Mesh&& mesh);
    const Mesh* getMesh() const;
    Mesh* getMesh();

    void setTexture(Texture&& tex);
    const Texture* getTexture() const;
    Texture* getTexture();

    // AI‑driven latent representation
    void setLatentVector(const std::vector<double>& latent);
    const std::vector<double>& getLatentVector() const;

    // Physics
    void applyPhysicsParams(const PhysicsParams& p);
    PhysicsParams getPhysicsParams() const;

    // Transform
    glm::mat4 getTransform() const;
    void setPosition(const glm::vec3& pos);
    void setRotation(const glm::quat& rot);
    void setScale(const glm::vec3& scl);

    glm::vec3 getPosition() const;
    glm::quat getRotation() const;
    glm::vec3 getScale() const;

    void translate(const glm::vec3& delta);
    void rotate(float angle, const glm::vec3& axis);

    // Visibility & activity
    void setVisible(bool v);
    bool isVisible() const;
    void setActive(bool a);
    bool isActive() const;

    // Abel coordinate helpers
    Coord3D getAbelPosition() const;
    void setAbelPosition(const Coord3D& pos);

    // Cloning (for AI self‑replication)
    Entity clone() const;

private:
    int id_;
    std::string name_;

    glm::vec3 position_;
    glm::quat rotation_;
    glm::vec3 scale_;

    Mesh* mesh_;
    Texture* texture_;
    // Physics body pointer (if any) – omitted for now, but physics params stored
    void* physics_;   // opaque pointer to Bullet rigid body, managed externally

    std::vector<double> latent_vector_;

    bool visible_;
    bool active_;

    // Cached physics parameters
    float mass_;
    float friction_;
    float restitution_;
    float linearDamping_;
    float angularDamping_;
    float gravityFactor_;
    bool isTrigger_;
    bool isKinematic_;

    mutable bool transformDirty_;
    mutable glm::mat4 cachedTransform_;
};

} // namespace abel
