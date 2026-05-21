// abel/intent_engine/PlayerGenerator.cpp
#include "PlayerGenerator.h"
#include "../world/Player.h"            // Player class (must provide setMesh, applyPhysics, setMovement)
#include "../render/Mesh.h"
#include "../intent_engine/PhysicsGenerator.h"   // for PhysicsParams
#include "../brain/Core.h"              // for clamp, PI

#include <cmath>
#include <algorithm>

namespace abel {

// ── Helper to safely read a latent dimension ──────────────────────────────
static double getLatent(const std::vector<double>& latent, size_t idx, double def = 0.0) {
    return (idx < latent.size()) ? latent[idx] : def;
}

PlayerGenerator::PlayerGenerator() {}

void PlayerGenerator::generate(Player& player, const std::vector<double>& latent) {
    // ── Decode appearance parameters ────────────────────────────────────
    const double heightNorm   = getLatent(latent, 0, 0.0);
    const double buildNorm    = getLatent(latent, 1, 0.0);
    const double skinR        = std::clamp((getLatent(latent, 2, 0.0) + 1.0) * 0.5, 0.0, 1.0);
    const double skinG        = std::clamp((getLatent(latent, 3, 0.0) + 1.0) * 0.5, 0.0, 1.0);
    const double skinB        = std::clamp((getLatent(latent, 4, 0.0) + 1.0) * 0.5, 0.0, 1.0);
    const double uglyNorm     = getLatent(latent, 5, 0.0);

    const double walkSpeedNorm  = getLatent(latent, 6, 0.0);
    const double runSpeedNorm   = getLatent(latent, 7, 0.0);
    const double jumpNorm       = getLatent(latent, 8, 0.0);
    const double solidRaw       = getLatent(latent, 9, 0.0);

    // ── Compute concrete values ─────────────────────────────────────────
    double height    = 1.5 + heightNorm * 0.7;                // 0.8 .. 2.2 m
    double bodyRadius= 0.2 + (buildNorm + 1.0) * 0.15;       // 0.2 .. 0.5
    double limbRadius= 0.06+ (buildNorm + 1.0) * 0.04;       // 0.06..0.14

    double ugly      = std::clamp((uglyNorm + 1.0) * 0.5, 0.0, 1.0);
    double headDeform = ugly * 0.3;
    double bodyDeform = ugly * 0.2;

    double walkSpeed   = 1.2 + walkSpeedNorm * 0.8;           // 0.4 – 2.0
    double runSpeed    = 3.0 + runSpeedNorm * 2.0;            // 1.0 – 5.0
    double jumpImpulse = 5.0 + jumpNorm * 3.0;               // 2.0 – 8.0
    bool solid         = (solidRaw > 0.3);

    // ── Build player mesh procedurally ──────────────────────────────────
    Mesh mesh;

    // Head (sphere)
    float headRadius = static_cast<float>(height * 0.12);
    mesh.addSphere(headRadius, Coord3D(0.0, 0.0, height * 0.85), 16);
    if (ugly > 0.3) mesh.applyNoiseToLast(headDeform, 3);

    // Torso (cylinder)
    float torsoHeight = static_cast<float>(height * 0.4);
    float torsoRadius = static_cast<float>(bodyRadius);
    mesh.addCylinder(torsoRadius, torsoHeight, Coord3D(0.0, 0.0, height * 0.45), 12);
    if (ugly > 0.4) mesh.applyNoiseToLast(bodyDeform, 2);

    // Arms (cylinders)
    float armLength = static_cast<float>(height * 0.35);
    float armRadius = static_cast<float>(limbRadius);
    mesh.addCylinder(armRadius, armLength, Coord3D(-bodyRadius * 1.2f, 0.0, height * 0.6), 8);
    mesh.addCylinder(armRadius, armLength, Coord3D( bodyRadius * 1.2f, 0.0, height * 0.6), 8);

    // Legs (cylinders)
    float legLength = static_cast<float>(height * 0.5);
    float legRadius = static_cast<float>(limbRadius * 1.2f);
    mesh.addCylinder(legRadius, legLength, Coord3D(-bodyRadius * 0.5f, 0.0, height * 0.15), 8);
    mesh.addCylinder(legRadius, legLength, Coord3D( bodyRadius * 0.5f, 0.0, height * 0.15), 8);

    // Skin colour
    mesh.setBaseColor(static_cast<float>(skinR),
                      static_cast<float>(skinG),
                      static_cast<float>(skinB));

    player.setMesh(std::move(mesh));

    // ── Physics parameters ──────────────────────────────────────────────
    PhysicsParams phys;
    phys.mass           = 40.0 + (buildNorm + 1.0) * 40.0;   // 40‑120 kg
    phys.friction       = 0.8f;
    phys.restitution    = 0.1f;
    phys.linearDamping  = 0.1f;
    phys.angularDamping = 0.2f;
    phys.gravityFactor  = 1.0f;
    phys.isTrigger      = false;
    phys.isKinematic    = false;
    player.applyPhysics(phys);

    // ── Movement abilities ──────────────────────────────────────────────
    player.setMovement(walkSpeed, runSpeed, jumpImpulse, solid);
}

} // namespace abel
