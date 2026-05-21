#include "Camera.h"
#include "../brain/Core.h"            // for PI, RAD2DEG, DEG2RAD
#include "../brain/spatial/Coords.h"  // for Coord3D

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>

namespace abel {

// ── Constructor ─────────────────────────────────────────────────────────
Camera::Camera()
    : position_(0.0f, 0.0f, 5.0f),
      yaw_(-90.0f),
      pitch_(0.0f),
      roll_(0.0f),
      fov_(60.0f),
      aspectRatio_(16.0f / 9.0f),
      nearPlane_(0.1f),
      farPlane_(1000.0f),
      movementSpeed_(5.0f),
      sprintMultiplier_(2.5f),
      mouseSensitivity_(0.1f),
      scrollSensitivity_(1.5f),
      active_(true),
      viewMatrixDirty_(true),
      projectionMatrixDirty_(true),
      worldUp_(0.0f, 0.0f, 1.0f)
{
    updateVectors();
}

// ── View matrix ─────────────────────────────────────────────────────────
glm::mat4 Camera::getView() const {
    if (viewMatrixDirty_) {
        viewMatrix_ = glm::lookAt(position_, position_ + forward_, up_);
        viewMatrixDirty_ = false;
    }
    return viewMatrix_;
}

// ── Projection matrix ───────────────────────────────────────────────────
glm::mat4 Camera::getProjection() const {
    if (projectionMatrixDirty_) {
        projectionMatrix_ = glm::perspective(
            glm::radians(fov_), aspectRatio_, nearPlane_, farPlane_);
        projectionMatrixDirty_ = false;
    }
    return projectionMatrix_;
}

// ── Combined view‑projection ────────────────────────────────────────────
glm::mat4 Camera::getViewProjection() const {
    return getProjection() * getView();
}

// ── Position access ─────────────────────────────────────────────────────
glm::vec3 Camera::getPosition() const {
    return position_;
}

void Camera::setPosition(const glm::vec3& pos) {
    position_ = pos;
    viewMatrixDirty_ = true;
}

// ── Movement input ──────────────────────────────────────────────────────
void Camera::moveForward(float deltaTime, bool sprint) {
    float speed = movementSpeed_ * (sprint ? sprintMultiplier_ : 1.0f) * deltaTime;
    position_ += forward_ * speed;
    viewMatrixDirty_ = true;
}

void Camera::moveBackward(float deltaTime, bool sprint) {
    float speed = movementSpeed_ * (sprint ? sprintMultiplier_ : 1.0f) * deltaTime;
    position_ -= forward_ * speed;
    viewMatrixDirty_ = true;
}

void Camera::moveLeft(float deltaTime, bool sprint) {
    float speed = movementSpeed_ * (sprint ? sprintMultiplier_ : 1.0f) * deltaTime;
    position_ -= right_ * speed;
    viewMatrixDirty_ = true;
}

void Camera::moveRight(float deltaTime, bool sprint) {
    float speed = movementSpeed_ * (sprint ? sprintMultiplier_ : 1.0f) * deltaTime;
    position_ += right_ * speed;
    viewMatrixDirty_ = true;
}

void Camera::moveUp(float deltaTime) {
    position_ += worldUp_ * movementSpeed_ * deltaTime;
    viewMatrixDirty_ = true;
}

void Camera::moveDown(float deltaTime) {
    position_ -= worldUp_ * movementSpeed_ * deltaTime;
    viewMatrixDirty_ = true;
}

// ── Mouse look ──────────────────────────────────────────────────────────
void Camera::rotate(float deltaYaw, float deltaPitch) {
    yaw_   += deltaYaw   * mouseSensitivity_;
    pitch_ += deltaPitch * mouseSensitivity_;

    // Clamp pitch to avoid gimbal lock and flipping
    pitch_ = std::clamp(pitch_, -89.0f, 89.0f);

    updateVectors();
}

// ── Scroll zoom ─────────────────────────────────────────────────────────
void Camera::zoom(float scrollDelta) {
    fov_ -= scrollDelta * scrollSensitivity_;
    fov_ = std::clamp(fov_, 1.0f, 120.0f);
    projectionMatrixDirty_ = true;
}

// ── Direct orientation setter ───────────────────────────────────────────
void Camera::setOrientation(float yaw, float pitch) {
    yaw_   = yaw;
    pitch_ = std::clamp(pitch, -89.0f, 89.0f);
    updateVectors();
}

// ── Projection control ──────────────────────────────────────────────────
void Camera::setFOV(float fov) {
    fov_ = std::clamp(fov, 1.0f, 120.0f);
    projectionMatrixDirty_ = true;
}

void Camera::setAspectRatio(float ratio) {
    aspectRatio_ = ratio;
    projectionMatrixDirty_ = true;
}

void Camera::setNearFar(float near, float far) {
    nearPlane_ = near;
    farPlane_  = far;
    projectionMatrixDirty_ = true;
}

// ── Ray casting (screen to world) ───────────────────────────────────────
glm::vec3 Camera::screenToWorld(float screenX, float screenY,
                                float screenWidth, float screenHeight) const {
    float ndcX = (2.0f * screenX) / screenWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * screenY) / screenHeight;

    glm::mat4 invProj = glm::inverse(getProjection());
    glm::mat4 invView = glm::inverse(getView());

    glm::vec4 rayClip(ndcX, ndcY, 1.0f, 1.0f);
    glm::vec4 rayEye = invProj * rayClip;
    rayEye.z = 1.0f;
    rayEye.w = 0.0f;

    glm::vec3 rayWorld = glm::vec3(invView * rayEye);
    return glm::normalize(rayWorld);
}

// ── Abelian coordinate helpers ──────────────────────────────────────────
Coord3D Camera::getAbelPosition() const {
    return Coord3D(position_.x, position_.y, position_.z);
}

glm::vec3 Camera::getForward() const {
    return forward_;
}

glm::vec3 Camera::getRight() const {
    return right_;
}

glm::vec3 Camera::getUp() const {
    return up_;
}

// ── Toggle camera active state ──────────────────────────────────────────
void Camera::setActive(bool active) {
    active_ = active;
}

bool Camera::isActive() const {
    return active_;
}

// ── Internal vector recomputation ───────────────────────────────────────
void Camera::updateVectors() {
    glm::vec3 front;
    front.x = std::cos(glm::radians(yaw_)) * std::cos(glm::radians(pitch_));
    front.y = std::sin(glm::radians(pitch_));
    front.z = std::sin(glm::radians(yaw_)) * std::cos(glm::radians(pitch_));
    forward_ = glm::normalize(front);

    right_ = glm::normalize(glm::cross(forward_, worldUp_));
    up_    = glm::normalize(glm::cross(right_, forward_));

    viewMatrixDirty_ = true;
}

} // namespace abel
