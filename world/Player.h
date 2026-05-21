#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>

// Include the full PhysicsParams definition (fixes incomplete type error)
#include "../intent_engine/PhysicsGenerator.h"

namespace abel {

class Mesh;

class Player {
public:
    Player();
    ~Player();

    // Move‑only
    Player(Player&& other) noexcept;
    Player& operator=(Player&& other) noexcept;

    // Mesh (visual representation)
    void setMesh(Mesh&& mesh);
    const Mesh* getMesh() const;
    Mesh* getMesh();

    // Physics properties
    void applyPhysics(const PhysicsParams& params);
    PhysicsParams getPhysics() const;

    // Movement parameters (set by AI or defaults)
    void setMovement(double walkSpeed, double runSpeed, double jumpImpulse, bool solid);
    double getWalkSpeed() const;
    double getRunSpeed() const;
    double getJumpImpulse() const;

    // Transform
    glm::vec3 getPosition() const;
    void setPosition(const glm::vec3& pos);

    glm::quat getRotation() const;
    void setRotation(const glm::quat& rot);

    glm::mat4 getTransform() const;

    // Input (to be driven by external input or AI)
    void setMoveForward(bool on);
    void setMoveBackward(bool on);
    void setMoveLeft(bool on);
    void setMoveRight(bool on);
    void setMoveUp(bool on);
    void setMoveDown(bool on);
    void setSprint(bool on);
    void setJumpRequested(bool on);

    bool isMoveForward() const;
    bool isMoveBackward() const;
    bool isMoveLeft() const;
    bool isMoveRight() const;
    bool isMoveUp() const;
    bool isMoveDown() const;
    bool isSprint() const;
    bool isJumpRequested() const;

    // Camera (for human spectator)
    void setLookOrientation(float yaw, float pitch);
    float getYaw() const;
    float getPitch() const;

    // Collision flag (can be set by AI)
    bool isSolid() const;

private:
    glm::vec3 position_;
    glm::quat rotation_;
    Mesh* mesh_;                    // owned

    PhysicsParams phys_;            // current physics parameters
    double walkSpeed_;
    double runSpeed_;
    double jumpImpulse_;
    bool solid_;

    // Input flags
    bool moveForward_;
    bool moveBackward_;
    bool moveLeft_;
    bool moveRight_;
    bool moveUp_;
    bool moveDown_;
    bool sprint_;
    bool jumpRequested_;

    // View orientation (for human player)
    float yaw_;
    float pitch_;

    mutable bool transformDirty_;
    mutable glm::mat4 cachedTransform_;
};

} // namespace abel
