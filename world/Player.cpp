#include "Player.h"
#include "../render/Mesh.h"
#include "../intent_engine/PhysicsGenerator.h"   // for PhysicsParams
#include "../brain/spatial/Coords.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace abel {

Player::Player()
    : position_(0.0f, 0.0f, 1.0f),
      rotation_(1.0f, 0.0f, 0.0f, 0.0f),   // identity quaternion
      mesh_(nullptr),
      walkSpeed_(1.5),
      runSpeed_(4.0),
      jumpImpulse_(7.0),
      solid_(true),
      moveForward_(false),
      moveBackward_(false),
      moveLeft_(false),
      moveRight_(false),
      moveUp_(false),
      moveDown_(false),
      sprint_(false),
      jumpRequested_(false),
      yaw_(0.0f),
      pitch_(0.0f),
      transformDirty_(true)
{
}

Player::~Player() {
    delete mesh_;
}

Player::Player(Player&& other) noexcept
    : position_(other.position_),
      rotation_(other.rotation_),
      mesh_(other.mesh_),
      phys_(other.phys_),
      walkSpeed_(other.walkSpeed_),
      runSpeed_(other.runSpeed_),
      jumpImpulse_(other.jumpImpulse_),
      solid_(other.solid_),
      moveForward_(other.moveForward_),
      moveBackward_(other.moveBackward_),
      moveLeft_(other.moveLeft_),
      moveRight_(other.moveRight_),
      moveUp_(other.moveUp_),
      moveDown_(other.moveDown_),
      sprint_(other.sprint_),
      jumpRequested_(other.jumpRequested_),
      yaw_(other.yaw_),
      pitch_(other.pitch_),
      transformDirty_(true)
{
    other.mesh_ = nullptr;
}

Player& Player::operator=(Player&& other) noexcept {
    if (this != &other) {
        delete mesh_;
        mesh_ = other.mesh_;
        other.mesh_ = nullptr;

        position_ = other.position_;
        rotation_ = other.rotation_;
        phys_ = other.phys_;
        walkSpeed_ = other.walkSpeed_;
        runSpeed_ = other.runSpeed_;
        jumpImpulse_ = other.jumpImpulse_;
        solid_ = other.solid_;
        moveForward_ = other.moveForward_;
        moveBackward_ = other.moveBackward_;
        moveLeft_ = other.moveLeft_;
        moveRight_ = other.moveRight_;
        moveUp_ = other.moveUp_;
        moveDown_ = other.moveDown_;
        sprint_ = other.sprint_;
        jumpRequested_ = other.jumpRequested_;
        yaw_ = other.yaw_;
        pitch_ = other.pitch_;
        transformDirty_ = true;
    }
    return *this;
}

// Mesh
void Player::setMesh(Mesh&& mesh) {
    delete mesh_;
    mesh_ = new Mesh(std::move(mesh));
}

const Mesh* Player::getMesh() const { return mesh_; }
Mesh* Player::getMesh() { return mesh_; }

// Physics
void Player::applyPhysics(const PhysicsParams& params) {
    phys_ = params;
}

PhysicsParams Player::getPhysics() const {
    return phys_;
}

// Movement parameters (set by AI or default)
void Player::setMovement(double walkSpeed, double runSpeed, double jumpImpulse, bool solid) {
    walkSpeed_ = walkSpeed;
    runSpeed_ = runSpeed;
    jumpImpulse_ = jumpImpulse;
    solid_ = solid;
}

double Player::getWalkSpeed() const { return walkSpeed_; }
double Player::getRunSpeed() const { return runSpeed_; }
double Player::getJumpImpulse() const { return jumpImpulse_; }

// Transform
glm::vec3 Player::getPosition() const { return position_; }
void Player::setPosition(const glm::vec3& pos) { position_ = pos; transformDirty_ = true; }

glm::quat Player::getRotation() const { return rotation_; }
void Player::setRotation(const glm::quat& rot) { rotation_ = rot; transformDirty_ = true; }

glm::mat4 Player::getTransform() const {
    if (transformDirty_) {
        glm::mat4 trans = glm::translate(glm::mat4(1.0f), position_);
        glm::mat4 rot = glm::mat4_cast(rotation_);
        cachedTransform_ = trans * rot;
        transformDirty_ = false;
    }
    return cachedTransform_;
}

// Input handling
void Player::setMoveForward(bool on)  { moveForward_ = on; }
void Player::setMoveBackward(bool on) { moveBackward_ = on; }
void Player::setMoveLeft(bool on)     { moveLeft_ = on; }
void Player::setMoveRight(bool on)    { moveRight_ = on; }
void Player::setMoveUp(bool on)       { moveUp_ = on; }
void Player::setMoveDown(bool on)     { moveDown_ = on; }
void Player::setSprint(bool on)       { sprint_ = on; }
void Player::setJumpRequested(bool on){ jumpRequested_ = on; }

bool Player::isMoveForward() const  { return moveForward_; }
bool Player::isMoveBackward() const { return moveBackward_; }
bool Player::isMoveLeft() const     { return moveLeft_; }
bool Player::isMoveRight() const    { return moveRight_; }
bool Player::isMoveUp() const       { return moveUp_; }
bool Player::isMoveDown() const     { return moveDown_; }
bool Player::isSprint() const       { return sprint_; }
bool Player::isJumpRequested() const { return jumpRequested_; }

// Camera orientation
void Player::setLookOrientation(float yaw, float pitch) {
    yaw_ = yaw;
    pitch_ = pitch;
}

float Player::getYaw() const   { return yaw_; }
float Player::getPitch() const { return pitch_; }

// Solid flag
bool Player::isSolid() const { return solid_; }

} // namespace abel
