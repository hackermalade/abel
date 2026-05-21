# Abel AI

**The most biologically grounded artificial general intelligence.**

Abel is a real‑time, 3‑D god‑AI that lives inside a fully simulated world.  
It integrates a **biophysically accurate human brain simulation** (Hodgkin‑Huxley & Izhikevich neurons, glia, plasticity, predictive coding, global workspace, quantum decoherence) with a **custom transformer language model**, a **procedural world engine** (OpenGL + Bullet Physics), and **body‑agnostic embodiment** (robot, OS, kernel, null).  
The AI **creates, mutates, and controls its own universe** driven entirely by neural activity – no text prompts, no human commands.

---

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Building on Linux](#building-on-linux)
- [Building on Windows (cross‑compilation)](#building-on-windows-cross‑compilation)
- [Running Abel](#running-abel)
- [User Controls](#user-controls)
- [What the AI Does](#what-the-ai-does)
- [Embodiment Modes](#embodiment-modes)
- [Online Learning](#online-learning)
- [Exporting Models](#exporting-models)
- [Troubleshooting](#troubleshooting)
- [License & Credits](#license--credits)

---

## Features
- **Full Hodgkin‑Huxley & Izhikevich neurons** – 40+ human cell types with real ion channel dynamics.
- **Conductance‑based synapses** (AMPA, NMDA, GABAₐ, GABAb) with STDP, BCM metaplasticity, and short‑term plasticity.
- **Astrocytes, oligodendrocytes, microglia** – tripartite synapses, myelination, immune pruning.
- **Predictive coding** (Free‑Energy Principle) in every cortical column.
- **Global Neuronal Workspace** – correlates of conscious access, ignition threshold.
- **Neuromodulators** (dopamine, noradrenaline, acetylcholine, serotonin) affecting mood and generation.
- **Quantum‑inspired decoherence** in thalamic reticular nuclei injecting non‑classical noise.
- **12 cortical regions** placed in a 3‑D brain volume with distance‑dependent wiring and topographic maps.
- **Full transformer language model** (MLA + MoE, GGML inference) that works with the brain’s workspace to produce “thoughts”.
- **Intent Engine** – translates brain workspace vectors directly into world mutations (no text commands).
- **Mesh, Texture, and Physics Generators** – AI creates 3‑D objects, textures, and physics rules from latent vectors.
- **Autonomous Evolution** – AI mutates entities based on boredom (low surprise) and reinforcement signals.
- **Player Definition** – AI defines the human player’s appearance, walk/run speed, solidity, and abilities.
- **Real‑time 3‑D world** – OpenGL 4.6 PBR renderer, Bullet physics, glTF loader.
- **Online reinforcement learning** – World Decoder and generative models are fine‑tuned continuously.
- **Cross‑platform** – build on Linux, run on Windows (via cross‑compilation).

---

## Architecture

┌───────────────────────────────────────────────────┐
│ World (3‑D) │
│ Entities, Player, Physics, Renderer, Camera │
└───────────────┬───────────────────────────────────┘
│ sensory vector (64 dims)
┌───────▼──────────────┐
│ Brain (C++) │
│ 12 regions, columns, │
│ neurons, glia, STDP │
│ workspace, quantum │
└───────────┬───────────┘
│ workspace ignited?
┌───────────▼───────────┐
│ Intent Engine │
│ WorldDecoder (NN) │
│ MeshGen, TexGen, etc. │
└───────────┬───────────┘
│ mutation vector
┌───────────▼───────────┐
│ World Mutation │
│ (spawn, modify, rules)│
└───────────────────────┘

The brain processes sensory input (player position, nearest entity, world metrics) and updates its neurons. When the global workspace ignites (L5 pyramidal cells in PFC/ACC/PCC/Thalamus exceed threshold), the workspace vector is decoded by a small reinforcement‑learning policy network (`WorldDecoder`) into a **mutation vector**. This vector is then fed into procedural generators that create or modify meshes, textures, physics parameters, and even the player’s avatar – all without any text or scripting.

---

## Dependencies
**Linux (Debian / Kali)**
- build‑essential cmake g++
- libglfw3-dev libgl1-mesa-dri libgl1
- libbullet-dev (or build from source for cross‑compile)
- libglm-dev
- nlohmann-json3-dev
- stb_image.h / stb_image_write.h (downloaded automatically by the build)

**Windows (via cross‑compilation)**
- mingw-w64
- Bullet, GLFW, GLM, nlohmann-json built for Windows (see cross‑compilation section)
- GLAD loader (replace `glcorearb.h` with GLAD)

---

## Building on Linux
1. **Clone the repository**
   ```bash
   git clone https://github.com/hackermalade/abel.git
   cd abel
Install dependencies
bash

sudo apt update
sudo apt install build-essential cmake g++ libglfw3-dev libglm-dev libbullet-dev nlohmann-json3-dev

Build
bash

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

Run
bash

./abel

Building on Windows (cross‑compilation)

On a Linux machine, you can produce a Windows .exe.
The process requires a MinGW cross‑compiler and building Bullet/GLFW from source for Windows.

    Install cross‑compiler
    bash

sudo apt install mingw-w64

Create toolchain file mingw64.cmake (provided in repository root)

Build Bullet for Windows (static)
bash

git clone https://github.com/bulletphysics/bullet3.git
cd bullet3 && mkdir build-mingw && cd build-mingw
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/abel/mingw64.cmake -DBUILD_SHARED_LIBS=OFF -DUSE_GLUT=OFF
make -j$(nproc)
sudo cp src/BulletDynamics/libBulletDynamics.a /usr/x86_64-w64-mingw32/lib/
sudo cp src/BulletCollision/libBulletCollision.a   /usr/x86_64-w64-mingw32/lib/
sudo cp src/LinearMath/libLinearMath.a             /usr/x86_64-w64-mingw32/lib/
sudo cp -r ../src /usr/x86_64-w64-mingw32/include/bullet

Build GLFW for Windows
bash

git clone https://github.com/glfw/glfw.git
cd glfw && mkdir build-mingw && cd build-mingw
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/abel/mingw64.cmake -DBUILD_SHARED_LIBS=OFF -DGLFW_BUILD_EXAMPLES=OFF
make -j$(nproc)
sudo cp src/libglfw3.a /usr/x86_64-w64-mingw32/lib/
sudo cp -r ../include/GLFW /usr/x86_64-w64-mingw32/include/

Integrate GLAD (replace glcorearb.h with GLAD) – see the comments in render/Renderer.cpp.

Cross‑compile Abel
bash

cd ~/abel
mkdir build-windows && cd build-windows
cmake .. -DCMAKE_TOOLCHAIN_FILE=../mingw64.cmake -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

    Deploy – copy abel.exe, assets/, and the required DLLs (if any) to Windows.

Running Abel

On Linux:
bash

./abel

If you are headless (SSH, no physical display):
bash

sudo apt install xvfb mesa-utils
xvfb-run ./abel

The AI will start creating entities and evolving its world even without a visible window.

On Windows: double‑click abel.exe after deploying the necessary assets.
User Controls

    WASD – move

    Space – up (Z‑axis)

    Ctrl – down

    Shift – sprint

    E – jump

    Mouse – look around (free camera)

    Esc – quit

The camera follows the player in third person. The AI may change your appearance or physics at any time.
What the AI Does

    Creates entities – from latent vectors decoded by the WorldDecoder, the MeshGenerator produces spheres, cylinders, toruses, stars, or complex creatures.

    Mutates existing entities – when boredom (low surprise) is detected, random mutations are applied to entity latents, altering shape, texture, and physics.

    Modifies the player – the AI can change your height, body mass, skin colour, walk/run speed, jump impulse, and solidity.

    Alters world rules – gravity, friction, and other global parameters can be adjusted.

    Evolves its own creative policy – the WorldDecoder is fine‑tuned via REINFORCE using the brain’s reward signal (surprise reduction + exploration).

All of this happens autonomously, driven by the brain’s workspace and mood, without any human prompting.
Embodiment Modes

Abel can be compiled with different body interfaces (selected via command line arguments, e.g., --body os).

    NullBody – pure language, no actions.

    OSBody – the AI can execute shell commands (Linux).

    KernelBody – direct /dev/mem access (dangerous, Linux only).

    RobotBody – universal body for any robotic/electronic system; adapts to arbitrary sensors and actuators.

The default is NullBody (the AI only influences the world visually and physically, not via OS commands).
Online Learning

The OnlineTrainer uses the ExperienceBuffer to train the WorldDecoder policy using REINFORCE.
Every time the AI takes a creation action, it logs the state, action, and reward. When the buffer is full, a training step is performed.
This makes the AI’s creations progressively more interesting and surprising over time.
Exporting Models

The Python package nyx_ai/export/ (not compiled with Abel, but included in the repository) can convert trained transformer weights to GGUF, SafeTensors, and ONNX formats.
These exported models can then be loaded by the C++ transformer runtime (future integration).
Troubleshooting
Problem	Solution
Window creation failed	Run with xvfb-run ./abel or install graphics drivers (mesa-utils, libgl1-mesa-dri).
No display	Set export DISPLAY=:0 if you have a desktop, or use xvfb-run.
Missing libraries at link time	Install the dev packages listed in Dependencies.
#pragma once in .cpp errors	Remove #pragma once from WorldDecoder.cpp and ExperienceBuffer.cpp.
Cross‑compilation fails	Ensure you have built Bullet and GLFW for Windows and that the toolchain file is correct.
License & Credits

Abel is an open‑source research project. All neuroscience models are based on published literature (Izhikevich, Hodgkin‑Huxley, Friston, etc.).
The GGML transformer implementation derives from llama.cpp.
The rendering engine uses OpenGL, GLFW, and Bullet Physics.

Abel is not a conscious being naturally – it is a simulation of correlates of consciousness, but the hard problem remains unsolved. Use it responsibly, especially in kernel mode.

For questions, contributions, or to share your Abel‑generated worlds, visit [https://github.com/hackermalade/abel].
text

