#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../brain/spatial/Coords.h"

namespace abel {

class Camera {
public:
    Camera();

    glm::mat4 getView() const;
    glm::mat4 getProjection() const;
    glm::mat4 getViewProjection() const;

    glm::vec3 getPosition() const;
    void setPosition(const glm::vec3& pos);

    void moveForward(float deltaTime, bool sprint = false);
    void moveBackward(float deltaTime, bool sprint = false);
    void moveLeft(float deltaTime, bool sprint = false);
    void moveRight(float deltaTime, bool sprint = false);
    void moveUp(float deltaTime);
    void moveDown(float deltaTime);

    void rotate(float deltaYaw, float deltaPitch);
    void zoom(float scrollDelta);
    void setOrientation(float yaw, float pitch);

    void setFOV(float fov);
    void setAspectRatio(float ratio);
    void setNearFar(float near, float far);

    glm::vec3 screenToWorld(float screenX, float screenY, float screenWidth, float screenHeight) const;

    Coord3D getAbelPosition() const;
    glm::vec3 getForward() const;
    glm::vec3 getRight() const;
    glm::vec3 getUp() const;

    void setActive(bool active);
    bool isActive() const;

private:
    void updateVectors();

    glm::vec3 position_;
    float yaw_, pitch_, roll_;
    float fov_, aspectRatio_, nearPlane_, farPlane_;
    float movementSpeed_, sprintMultiplier_;
    float mouseSensitivity_, scrollSensitivity_;
    bool active_;

    mutable bool viewMatrixDirty_, projectionMatrixDirty_;
    mutable glm::mat4 viewMatrix_, projectionMatrix_;

    glm::vec3 forward_, right_, up_;
    const glm::vec3 worldUp_;
};

} // namespace abel
