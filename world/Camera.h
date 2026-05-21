// abel/world/Camera.h
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../brain/spatial/Coords.h"

namespace abel {

class Camera {
public:
    Camera();

    // ── Matrices ──────────────────────────────────────────────────────
    glm::mat4 getView() const;
    glm::mat4 getProjection() const;
    glm::mat4 getViewProjection() const;

    // ── Position ──────────────────────────────────────────────────────
    glm::vec3 getPosition() const;
    void setPosition(const glm::vec3& pos);

    // ── Movement (WASD + space / ctrl) ────────────────────────────────
    void moveForward(float deltaTime, bool sprint = false);
    void moveBackward(float deltaTime, bool sprint = false);
    void moveLeft(float deltaTime, bool sprint = false);
    void moveRight(float deltaTime, bool sprint = false);
    void moveUp(float deltaTime);
    void moveDown(float deltaTime);

    // ── Look / zoom ───────────────────────────────────────────────────
    void rotate(float deltaYaw, float deltaPitch);
    void zoom(float scrollDelta);
    void setOrientation(float yaw, float pitch);

    // ── Projection settings ───────────────────────────────────────────
    void setFOV(float fov);
    void setAspectRatio(float ratio);
    void setNearFar(float near, float far);

    // ── Screen‑to‑world ray (for mouse picking) ───────────────────────
    glm::vec3 screenToWorld(float screenX, float screenY,
                            float screenWidth, float screenHeight) const;

    // ── Abel coordinate helpers ───────────────────────────────────────
    Coord3D getAbelPosition() const;
    glm::vec3 getForward() const;
    glm::vec3 getRight() const;
    glm::vec3 getUp() const;

    // ── Orientation getters (needed by main) ──────────────────────────
    float getYaw() const   { return yaw_; }
    float getPitch() const { return pitch_; }

    // ── Active toggle (e.g. disable when UI focused) ──────────────────
    void setActive(bool active);
    bool isActive() const;

private:
    void updateVectors();

    // ── State ────────────────────────────────────────────────────────
    glm::vec3 position_;
    float yaw_, pitch_, roll_;
    float fov_, aspectRatio_, nearPlane_, farPlane_;

    // ── Speed / sensitivity ──────────────────────────────────────────
    float movementSpeed_       = 5.0f;
    float sprintMultiplier_    = 2.5f;
    float mouseSensitivity_    = 0.1f;
    float scrollSensitivity_   = 1.5f;

    bool active_ = true;

    // ── Cached matrices (dirty‑flag pattern) ──────────────────────────
    mutable bool viewMatrixDirty_       = true;
    mutable bool projectionMatrixDirty_ = true;
    mutable glm::mat4 viewMatrix_;
    mutable glm::mat4 projectionMatrix_;

    // ── Local axes ───────────────────────────────────────────────────
    glm::vec3 forward_;
    glm::vec3 right_;
    glm::vec3 up_;

    const glm::vec3 worldUp_ = glm::vec3(0.0f, 1.0f, 0.0f);
};

} // namespace abel
