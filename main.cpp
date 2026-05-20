// Abel – God‑AI main loop
// The AI continuously senses its world, updates its biological brain,
// decodes intent into world mutations, steps physics, renders, and learns.

#include "world/World.h"
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

#include <iostream>
#include <chrono>
#include <thread>

int main() {
    // ---- 1. Create core systems ----
    World world;
    Player player;
    Camera camera(player);
    PhysicsEngine physics;
    Renderer renderer;

    Brain brain;
    WorldDecoder decoder;
    MeshGenerator meshGen;
    TextureGenerator texGen;
    PhysicsGenerator physGen;
    PlayerGenerator playerGen;
    EvolutionManager evoMgr;

    ExperienceBuffer expBuffer;
    OnlineTrainer onlineTrainer;

    // ---- 2. Initialise subsystems ----
    world.loadFromFile("assets/base_world.glb");
    renderer.init();
    brain.loadState("assets/pretrained_models/brain_state.bin");

    std::cout << "Abel is awake...\n";

    // ---- 3. Main loop ----
    while (world.isRunning()) {
        auto frameStart = std::chrono::high_resolution_clock::now();

        // --- 3a. Gather sensory input ---
        // The agent (and player) provide a combined sensory state vector
        SensoryVector sensory = world.getSensoryState(player);

        // --- 3b. Feed sensory vector to brain ---
        brain.processSensory(sensory);
        brain.step(20);   // simulate 20 ms of neural activity

        // --- 3c. If workspace ignited, decode intent ---
        if (brain.isWorkspaceIgnited()) {
            IntentVector intent = decoder.decode(brain.getWorkspaceVector(),
                                                 brain.getContextVector());

            // Apply intent to world – spawn, mutate, modify
            switch (intent.actionType) {
                case ActionType::SPAWN_ENTITY:
                    meshGen.generate(intent.meshParams);
                    texGen.generate(intent.textureParams);
                    physGen.apply(intent.physicsParams);
                    world.spawnEntity(intent);
                    break;
                case ActionType::MUTATE_ENTITY:
                    evoMgr.mutateExistingEntity(world, intent.targetEntity, intent.mutation);
                    break;
                case ActionType::CHANGE_PLAYER:
                    playerGen.apply(player, intent.playerParams);
                    break;
                case ActionType::ALTER_WORLD_RULES:
                    physics.applyGlobalRule(intent.ruleChange);
                    break;
                default: break;
            }
        }

        // --- 3d. Step physics ---
        physics.step(1.0 / 60.0);
        world.update(1.0 / 60.0);

        // --- 3e. Render scene ---
        renderer.draw(world, camera);

        // --- 3f. Log experience for online learning ---
        expBuffer.add(sensory, brain.getMotorOutput(), brain.getRecentReward());
        if (expBuffer.isFull()) {
            onlineTrainer.trainStep(expBuffer.getBatch());
            expBuffer.clear();
        }

        // --- 3g. Maintain frame rate ---
        auto frameEnd = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<float>(frameEnd - frameStart).count();
        if (elapsed < 1.0f / 60.0f) {
            std::this_thread::sleep_for(
                std::chrono::duration<float>(1.0f / 60.0f - elapsed));
        }
    }

    // ---- 4. Shutdown ----
    brain.saveState("assets/pretrained_models/brain_state.bin");
    renderer.shutdown();
    return 0;
}
