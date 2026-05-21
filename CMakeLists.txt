cmake_minimum_required(VERSION 3.20)
project(abel CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(Bullet REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(nlohmann_json REQUIRED)

# ── Core sources (transformer excluded) ─────────────────────────────
add_executable(abel
    main.cpp
    world/World.cpp world/Entity.cpp world/Player.cpp world/Camera.cpp world/PhysicsEngine.cpp
    render/Renderer.cpp render/ShaderManager.cpp render/Mesh.cpp render/Texture.cpp
    brain/Neurons.cpp brain/Glia.cpp brain/Synapses.cpp brain/Column.cpp brain/Brain.cpp
    brain/PredictiveCoder.cpp brain/Quantum.cpp
    brain/spatial/Coords.cpp brain/spatial/Topology.cpp brain/spatial/Wiring.cpp
    intent_engine/WorldDecoder.cpp intent_engine/MeshGenerator.cpp intent_engine/TextureGenerator.cpp
    intent_engine/PhysicsGenerator.cpp intent_engine/PlayerGenerator.cpp intent_engine/EvolutionManager.cpp
    online/ExperienceBuffer.cpp online/OnlineTrainer.cpp
)

target_include_directories(abel PRIVATE
    .
    third_party
    third_party/stb
    ${BULLET_INCLUDE_DIRS}
    /usr/include/bullet
    ${GLM_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
)

target_link_libraries(abel
    OpenGL::GL
    glfw
    glm::glm
    BulletDynamics
    BulletCollision
    LinearMath
    nlohmann_json::nlohmann_json
)
