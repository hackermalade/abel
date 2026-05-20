#include "Entity.h"
#include "../render/Mesh.h"
#include "../render/Texture.h"
#include "../intent_engine/PhysicsGenerator.h"
#include "../brain/spatial/Coords.h"
#include "../brain/Core.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cstring>
#include <algorithm>

namespace abel {

// ── Static ID counter ────────────────────────────────────────────────────
static int nextId = 0;

// ── Constructor ──────────────────────────────────────────────────────────
Entity::Entity()
    : id_(nextId++),
      name_("entity_" + std::to_string(id_)),
      position_(0.0f, 0.0f, 0.0f),
      rotation_(1.0f, 0.0f, 0.0f, 0.0f),   // identity quaternion
      scale_(1.0f, 1.0f, 1.0f),
      mesh_(nullptr),
      texture_(nullptr),
      physics_(nullptr),
      latent_vector_(128, 0.0),
      visible_(true),
      active_(true),
      mass_(1.0f),
      friction_(0.5f),
      restitution_(0.0f),
      linearDamping_(0.1f),
      angularDamping_(0.1f),
      gravityFactor_(1.0f),
      isTrigger_(false),
      isKinematic_(false),
      transformDirty_(true)
{
}

// ── Destructor ───────────────────────────────────────────────────────────
Entity::~Entity() {
    delete mesh_;
    delete texture_;
    delete physics_;
}

// ── Copy disabled, move enabled ──────────────────────────────────────────
Entity::Entity(Entity&& other) noexcept
    : id_(other.id_),
      name_(std::move(other.name_)),
      position_(other.position_),
      rotation_(other.rotation_),
      scale_(other.scale_),
      mesh_(other.mesh_),
      texture_(other.texture_),
      physics_(other.physics_),
      latent_vector_(std::move(other.latent_vector_)),
      visible_(other.visible_),
      active_(other.active_),
      mass_(other.mass_),
      friction_(other.friction_),
      restitution_(other.restitution_),
      linearDamping_(other.linearDamping_),
      angularDamping_(other.angularDamping_),
      gravityFactor_(other.gravityFactor_),
      isTrigger_(other.isTrigger_),
      isKinematic_(other.isKinematic_),
      transformDirty_(true)
{
    other.mesh_ = nullptr;
    other.texture_ = nullptr;
    other.physics_ = nullptr;
}

Entity& Entity::operator=(Entity&& other) noexcept {
    if (this != &other) {
        delete mesh_; mesh_ = other.mesh_; other.mesh_ = nullptr;
        delete texture_; texture_ = other.texture_; other.texture_ = nullptr;
        delete physics_; physics_ = other.physics_; other.physics_ = nullptr;

        id_ = other.id_;
        name_ = std::move(other.name_);
        position_ = other.position_;
        rotation_ = other.rotation_;
        scale_ = other.scale_;
        latent_vector_ = std::move(other.latent_vector_);
        visible_ = other.visible_;
        active_ = other.active_;
        mass_ = other.mass_;
        friction_ = other.friction_;
        restitution_ = other.restitution_;
        linearDamping_ = other.linearDamping_;
        angularDamping_ = other.angularDamping_;
        gravityFactor_ = other.gravityFactor_;
        isTrigger_ = other.isTrigger_;
        isKinematic_ = other.isKinematic_;
        transformDirty_ = true;
    }
    return *this;
}

// ── ID and name ──────────────────────────────────────────────────────────
int Entity::getId() const { return id_; }
const std::string& Entity::getName() const { return name_; }
void Entity::setName(const std::string& name) { name_ = name; }

// ── Mesh ─────────────────────────────────────────────────────────────────
void Entity::setMesh(Mesh&& mesh) {
    delete mesh_;
    mesh_ = new Mesh(std::move(mesh));
}

const Mesh* Entity::getMesh() const { return mesh_; }
Mesh* Entity::getMesh() { return mesh_; }

// ── Texture ──────────────────────────────────────────────────────────────
void Entity::setTexture(Texture&& tex) {
    delete texture_;
    texture_ = new Texture(std::move(tex));
}

const Texture* Entity::getTexture() const { return texture_; }
Texture* Entity::getTexture() { return texture_; }

// ── Latent vector ────────────────────────────────────────────────────────
void Entity::setLatentVector(const std::vector<double>& latent) {
    latent_vector_ = latent;
}

const std::vector<double>& Entity::getLatentVector() const {
    return latent_vector_;
}

// ── Physics params ───────────────────────────────────────────────────────
void Entity::applyPhysicsParams(const PhysicsParams& p) {
    mass_           = p.mass;
    friction_       = p.friction;
    restitution_    = p.restitution;
    linearDamping_  = p.linearDamping;
    angularDamping_ = p.angularDamping;
    gravityFactor_  = p.gravityFactor;
    isTrigger_      = p.isTrigger;
    isKinematic_    = p.isKinematic;
}

PhysicsParams Entity::getPhysicsParams() const {
    PhysicsParams p;
    p.mass           = mass_;
    p.friction       = friction_;
    p.restitution    = restitution_;
    p.linearDamping  = linearDamping_;
    p.angularDamping = angularDamping_;
    p.gravityFactor  = gravityFactor_;
    p.isTrigger      = isTrigger_;
    p.isKinematic    = isKinematic_;
    return p;
}

// ── Transform ────────────────────────────────────────────────────────────
glm::mat4 Entity::getTransform() const {
    if (transformDirty_) {
        glm::mat4 trans = glm::translate(glm::mat4(1.0f), position_);
        glm::mat4 rot   = glm::mat4_cast(rotation_);
        glm::mat4 scl   = glm::scale(glm::mat4(1.0f), scale_);
        cachedTransform_ = trans * rot * scl;
        transformDirty_ = false;
    }
    return cachedTransform_;
}

void Entity::setPosition(const glm::vec3& pos) {
    position_ = pos;
    transformDirty_ = true;
}

void Entity::setRotation(const glm::quat& rot) {
    rotation_ = rot;
    transformDirty_ = true;
}

void Entity::setScale(const glm::vec3& scl) {
    scale_ = scl;
    transformDirty_ = true;
}

glm::vec3 Entity::getPosition() const { return position_; }
glm::quat Entity::getRotation() const { return rotation_; }
glm::vec3 Entity::getScale() const    { return scale_; }

void Entity::translate(const glm::vec3& delta) {
    position_ += delta;
    transformDirty_ = true;
}

void Entity::rotate(float angle, const glm::vec3& axis) {
    rotation_ = glm::rotate(rotation_, angle, axis);
    transformDirty_ = true;
}

// ── Visibility / activity ────────────────────────────────────────────────
void Entity::setVisible(bool v) { visible_ = v; }
bool Entity::isVisible() const { return visible_; }
void Entity::setActive(bool a) { active_ = a; }
bool Entity::isActive() const { return active_; }

// ── Abelian coordinate helpers ───────────────────────────────────────────
Coord3D Entity::getAbelPosition() const {
    return Coord3D(position_.x, position_.y, position_.z);
}

void Entity::setAbelPosition(const Coord3D& pos) {
    position_ = glm::vec3(pos.x, pos.y, pos.z);
    transformDirty_ = true;
}

// ── Clone ────────────────────────────────────────────────────────────────
Entity Entity::clone() const {
    Entity copy;
    copy.name_ = name_ + "_clone";
    copy.position_ = position_;
    copy.rotation_ = rotation_;
    copy.scale_ = scale_;
    if (mesh_)   copy.mesh_   = new Mesh(*mesh_);
    if (texture_) copy.texture_ = new Texture(*texture_);
    copy.latent_vector_ = latent_vector_;
    copy.visible_ = visible_;
    copy.active_ = active_;
    copy.mass_ = mass_;
    copy.friction_ = friction_;
    copy.restitution_ = restitution_;
    copy.linearDamping_ = linearDamping_;
    copy.angularDamping_ = angularDamping_;
    copy.gravityFactor_ = gravityFactor_;
    copy.isTrigger_ = isTrigger_;
    copy.isKinematic_ = isKinematic_;
    return copy;
}

} // namespace abel
