#include "PhysicsGenerator.h"
#include "../brain/Core.h"

#include <vector>
#include <cmath>
#include <algorithm>

namespace abel {

// ── Constructor ─────────────────────────────────────────────────────────
PhysicsGenerator::PhysicsGenerator() {}

// ── Generate physics parameters from latent vector ──────────────────────
PhysicsParams PhysicsGenerator::generate(const std::vector<double>& latent) const {
    PhysicsParams params;

    // Helper: safely fetch a latent dimension or use a default
    auto getVal = [&](size_t index, double def = 0.0) -> double {
        return (index < latent.size()) ? latent[index] : def;
    };

    // Mass: map [-1,1] to range [0.1, 10.0] (heavy objects up to 10 units)
    double mass_raw = getVal(0, 0.0);
    params.mass = 0.1 + (mass_raw + 1.0) * 4.95; // 0.1 .. 10.0

    // Friction coefficient (0.0 - 1.0)
    double fric_raw = getVal(1, 0.0);
    params.friction = std::max(0.0, std::min(1.0, (fric_raw + 1.0) * 0.5));

    // Restitution (bounciness) (0.0 - 1.0)
    double rest_raw = getVal(2, 0.0);
    params.restitution = std::max(0.0, std::min(1.0, (rest_raw + 1.0) * 0.5));

    // Linear damping (0.0 - 1.0)
    double lindamp_raw = getVal(3, 0.0);
    params.linearDamping = std::max(0.0, std::min(1.0, (lindamp_raw + 1.0) * 0.5));

    // Angular damping (0.0 - 1.0)
    double angdamp_raw = getVal(4, 0.0);
    params.angularDamping = std::max(0.0, std::min(1.0, (angdamp_raw + 1.0) * 0.5));

    // Gravity factor (0.5 - 2.0) – can make object lighter or heavier than world gravity
    double grav_raw = getVal(5, 0.0);
    params.gravityFactor = 0.5 + (grav_raw + 1.0) * 0.75;

    // Is the object a trigger (no collision response) ? threshold > 0.5
    double trigger_raw = getVal(6, -1.0);
    params.isTrigger = (trigger_raw > 0.5);

    // Is the object kinematic (moved by code, not physics) ?
    double kinem_raw = getVal(7, -1.0);
    params.isKinematic = (kinem_raw > 0.5);

    // Optional: joint constraints (if needed) – for now placeholder
    // Could encode joint type (0=free, 1=hinge, 2=slider) and axis/limits from later dimensions
    double joint_type_raw = getVal(8, -1.0);
    if (joint_type_raw > 0.5) {
        // interpret joint type
        params.hasJoint = true;
        double joint_axis_x = getVal(9, 0.0);
        double joint_axis_y = getVal(10, 1.0);
        double joint_axis_z = getVal(11, 0.0);
        params.jointAxis = Coord3D(joint_axis_x, joint_axis_y, joint_axis_z);
        // joint limits
        params.jointLowerLimit = getVal(12, -1.0) * PI;   // -PI .. PI
        params.jointUpperLimit = getVal(13, 1.0) * PI;
    } else {
        params.hasJoint = false;
    }

    return params;
}

} // namespace abel
