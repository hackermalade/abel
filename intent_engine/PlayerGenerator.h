// world/Player.h (stub)
#pragma once
#include "../render/Mesh.h"
#include "../intent_engine/PhysicsGenerator.h"  // for PhysicsParams

namespace abel {

class Player {
public:
    void setMesh(Mesh&& mesh);
    void applyPhysics(const PhysicsParams& phys);
    void setMovement(double walkSpeed, double runSpeed, double jumpImpulse, bool solid);
};

} // namespace abel
