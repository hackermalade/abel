#include "world/World.h"
#include "world/Entity.h"
#include "world/Player.h"
#include "world/Camera.h"
#include "world/PhysicsEngine.h"
#include "render/Renderer.h"
#include "brain/Brain.h"
#include "intent_engine/WorldDecoder.h"
#include "intent_engine/MeshGenerator.h"
#include "intent_engine/TextureGenerator.h"
#include "intent_engine/PhysicsGenerator.h"
#include "intent_engine/PlayerGenerator.h"
#include "intent_engine/EvolutionManager.h"
#include "online/ExperienceBuffer.h"
#include "online/OnlineTrainer.h"

#include <GLFW/glfw3.h>
#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <cmath>
#include <algorithm>

using namespace abel;

// ---------------------------------------------------------------------------
//  Global input state (updated by GLFW callbacks)
// ---------------------------------------------------------------------------
struct InputState {
    bool forward       = false;
    bool backward      = false;
    bool left          = false;
    bool right         = false;
    bool up            = false;
    bool down          = false;
    bool sprint        = false;
    bool jump          = false;

    double mouseX      = 0.0;
    double mouseY      = 0.0;
    double lastMouseX  = 0.0;
    double lastMouseY  = 0.0;
    bool firstMouse    = true;
} input;

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    switch (key) {
        case GLFW_KEY_W:            input.forward  = pressed; break;
        case GLFW_KEY_S:            input.backward = pressed; break;
        case GLFW_KEY_A:            input.left     = pressed; break;
        case GLFW_KEY_D:            input.right    = pressed; break;
        case GLFW_KEY_SPACE:        input.up       = pressed; break;
        case GLFW_KEY_LEFT_CONTROL: input.down     = pressed; break;
        case GLFW_KEY_LEFT_SHIFT:   input.sprint   = pressed; break;
        case GLFW_KEY_E:            input.jump     = pressed; break;
        default: break;
    }
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    input.mouseX = xpos;
    input.mouseY = ypos;
    if (input.firstMouse) {
        input.lastMouseX = xpos;
        input.lastMouseY = ypos;
        input.firstMouse = false;
    }
}

static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    // can be used for zoom
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------
int main() {
    // ---- 1. Create renderer and window ----
    Renderer renderer;
    if (!renderer.init()) {
        std::cerr << "Failed to initialise renderer (window)" << std::endl;
        return -1;
    }
    GLFWwindow* window = renderer.getWindow();
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // capture mouse

    // ---- 2. Core systems (world owns the physics engine) ----
    World world;
    Player player;
    Camera camera;

    // Player initial state
    player.setPosition(glm::vec3(0.0f, 0.0f, 3.0f));
    player.setMovement(1.5, 4.0, 7.0, true);
    world.getPhysicsEngine().addPlayer(&player);   // add player to physics

    // Camera starts behind the player
    camera.setPosition(player.getPosition() + glm::vec3(0.0f, 0.0f, 5.0f));
    camera.setOrientation(-90.0f, -10.0f);

    // ---- 3. AI modules ----
    Brain brain(64, 0.5);
    brain.loadState("assets/pretrained_models/brain_state.bin");

    WorldDecoder decoder(128, 256, 0.001f);
    MeshGenerator meshGen;
    TextureGenerator texGen;
    PhysicsGenerator physGen;
    PlayerGenerator playerGen;
    EvolutionManager evoMgr;

    ExperienceBuffer expBuffer(10000);
    OnlineTrainer onlineTrainer(&decoder, &expBuffer);

    // ---- 4. Load world and initial entities ----
    world.loadFromFile("assets/base_world.glb");
    world.setPlayer(&player);

    // Spawn a few random entities so the world is not empty
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < 5; i++) {
        std::vector<double> latent(128);
        for (auto& v : latent) v = dist(rng);
        world.spawnEntity(latent);
    }

    // ---- 5. Timing ----
    const double targetFrameTime = 1.0 / 60.0;
    auto lastFrame = std::chrono::high_resolution_clock::now();

    // ---- 6. Main loop ----
    while (!glfwWindowShouldClose(window) && world.isRunning()) {
        auto frameStart = std::chrono::high_resolution_clock::now();
        double deltaTime = std::chrono::duration<double>(frameStart - lastFrame).count();
        if (deltaTime <= 0.0) deltaTime = targetFrameTime;
        lastFrame = frameStart;

        // --- 6a. Input processing ---
        glfwPollEvents();

        // Camera rotation from mouse
        double mouseDX = input.mouseX - input.lastMouseX;
        double mouseDY = input.mouseY - input.lastMouseY;
        input.lastMouseX = input.mouseX;
        input.lastMouseY = input.mouseY;
        camera.rotate(static_cast<float>(mouseDX), static_cast<float>(mouseDY));

        // Transfer input flags to player
        player.setMoveForward(input.forward);
        player.setMoveBackward(input.backward);
        player.setMoveLeft(input.left);
        player.setMoveRight(input.right);
        player.setMoveUp(input.up);
        player.setMoveDown(input.down);
        player.setSprint(input.sprint);
        player.setJumpRequested(input.jump);

        // Compute desired velocity from camera orientation
        glm::vec3 desiredVelocity(0.0f);
        float speed = input.sprint ? player.getRunSpeed() : player.getWalkSpeed();
        glm::vec3 camForward = camera.getForward();
        glm::vec3 camRight = camera.getRight();
        // Flatten forward and right to horizontal plane (Z is up in Abel)
        glm::vec3 fwd = glm::normalize(glm::vec3(camForward.x, camForward.y, 0.0f));
        glm::vec3 rgt = glm::normalize(glm::vec3(camRight.x, camRight.y, 0.0f));

        if (input.forward)  desiredVelocity += fwd * speed;
        if (input.backward) desiredVelocity -= fwd * speed;
        if (input.right)    desiredVelocity += rgt * speed;
        if (input.left)     desiredVelocity -= rgt * speed;
        if (input.up)       desiredVelocity += glm::vec3(0.0f, 0.0f, 1.0f) * speed;
        if (input.down)     desiredVelocity -= glm::vec3(0.0f, 0.0f, 1.0f) * speed;

        // Apply velocity to player's rigid body
        btRigidBody* playerBody = world.getPhysicsEngine().getPlayerRigidBody(&player);
        if (playerBody) {
            playerBody->setLinearVelocity(btVector3(desiredVelocity.x, desiredVelocity.y, desiredVelocity.z));
            if (input.jump && playerBody->getLinearVelocity().z() < 0.01f) {
                playerBody->applyCentralImpulse(btVector3(0.0f, 0.0f, player.getJumpImpulse()));
            }
        }

        // Update camera to follow player (third person)
        glm::vec3 playerPos = player.getPosition();
        float camYaw = camera.getYaw();
        float camPitch = camera.getPitch();
        float camDist = 5.0f;
        glm::vec3 camOffset(
            camDist * std::sin(glm::radians(camYaw)) * std::cos(glm::radians(camPitch)),
            camDist * std::sin(glm::radians(camPitch)),
            camDist * std::cos(glm::radians(camYaw)) * std::cos(glm::radians(camPitch))
        );
        camera.setPosition(playerPos - camOffset);

        // --- 6b. AI sensory processing ---
        SensoryVector sensory = world.getSensoryState(player);
        brain.processSensory(sensory);
        for (int s = 0; s < 20; ++s) brain.step();

        // --- 6c. Intent generation (when workspace ignited) ---
        if (brain.isWorkspaceIgnited()) {
            std::vector<double> state = brain.getWorkspaceVector();
            std::vector<double> context = brain.getContextVector();
            state.insert(state.end(), context.begin(), context.end());
            if (state.size() < 128) state.resize(128, 0.0);

            std::vector<double> action = decoder.sampleAction(state, 0.1);

            double actionTypeVal = action[0];
            int actionType = static_cast<int>((actionTypeVal + 1.0) * 2.5);
            actionType = std::clamp(actionType, 0, 4);

            double reward = 0.0;

            switch (static_cast<ActionType>(actionType)) {
                case ActionType::SPAWN_ENTITY: {
                    std::vector<double> latent(action.begin()+1, action.end());
                    if (latent.size() < 128) latent.resize(128, 0.0);
                    world.spawnEntity(latent);
                    reward = brain.getRecentReward();
                    break;
                }
                case ActionType::MUTATE_ENTITY: {
                    const auto& entities = world.getAllEntities();
                    if (!entities.empty()) {
                        int idx = rand() % entities.size();
                        int entityId = entities[idx]->getId();
                        std::vector<double> mutation(action.begin()+1, action.end());
                        if (mutation.size() < 128) mutation.resize(128, 0.0);
                        evoMgr.mutateExistingEntity(world, entityId, mutation);
                    }
                    reward = brain.getRecentReward();
                    break;
                }
                case ActionType::CHANGE_PLAYER: {
                    std::vector<double> latent(action.begin()+1, action.end());
                    if (latent.size() < 128) latent.resize(128, 0.0);
                    playerGen.generate(player, latent);
                    reward = brain.getRecentReward();
                    break;
                }
                case ActionType::ALTER_WORLD_RULES: {
                    double gx = action[1] * 10.0;
                    double gy = action[2] * 10.0;
                    double gz = -9.81 + action[3] * 10.0;
                    world.getPhysicsEngine().setGravity(Coord3D(gx, gy, gz));
                    reward = brain.getRecentReward();
                    break;
                }
                default: break;
            }

            expBuffer.add(state, action, reward);
        }

        // --- 6d. Online training ---
        if (expBuffer.isFull()) {
            onlineTrainer.trainFromBuffer(32);
            expBuffer.clear();
        }

        // --- 6e. Step physics and world ---
        world.getPhysicsEngine().step(static_cast<float>(deltaTime));
        world.update(static_cast<float>(deltaTime));

        // --- 6f. Render ---
        renderer.draw(world, camera);

        // --- 6g. Frame pacing ---
        auto frameEnd = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(frameEnd - frameStart).count();
        if (elapsed < targetFrameTime) {
            std::this_thread::sleep_for(std::chrono::duration<double>(targetFrameTime - elapsed));
        }
    }

    // ---- 7. Shutdown ----
    brain.saveState("assets/pretrained_models/brain_state.bin");
    world.getPhysicsEngine().removePlayer(&player);
    renderer.shutdown();
    return 0;
}
