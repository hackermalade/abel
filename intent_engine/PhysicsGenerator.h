#pragma once

#include "../brain/spatial/Coords.h"
#include <vector>

namespace abel {

// ── Physics parameters for an entity ────────────────────────────────────
struct PhysicsParams {
    double mass = 1.0;
    double friction = 0.5;
    double restitution = 0.0;      // bounciness
    double linearDamping = 0.1;
    double angularDamping = 0.1;
    double gravityFactor = 1.0;    // multiplier for world gravity
    bool isTrigger = false;        // no collision response, only events
    bool isKinematic = false;      // moved by code, not physics forces

    bool hasJoint = false;
    Coord3D jointAxis = Coord3D(0, 1, 0);
    double jointLowerLimit = -3.1415926535;
    double jointUpperLimit =  3.1415926535;
};

// ── Physics generator from latent vector ────────────────────────────────
class PhysicsGenerator {
public:
    PhysicsGenerator();

    // Generate physics parameters from a latent vector (size 128 typical)
    PhysicsParams generate(const std::vector<double>& latent) const;
};

} // namespace abel
