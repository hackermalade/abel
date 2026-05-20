#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>

namespace abel {

// --------------------------------------------------------------------------
//  Fundamental constants
// --------------------------------------------------------------------------
constexpr double PI      = 3.14159265358979323846;
constexpr double TWO_PI  = 6.28318530717958647692;
constexpr double DEG2RAD = PI / 180.0;
constexpr double RAD2DEG = 180.0 / PI;

// Default time‑step for the brain simulation (ms)
constexpr double DEFAULT_BRAIN_DT = 0.5;

// Dimensions of the brain's external interface
constexpr int EMBEDDING_DIM = 64;

// --------------------------------------------------------------------------
//  Core data types used throughout the brain and engine
// --------------------------------------------------------------------------

// Sensory input / output vectors
using SensoryVector = std::vector<double>;
using MotorVector   = std::vector<double>;
using ContextVector = std::vector<double>;

// Workspace / Intent vectors
using WorkspaceVector = std::vector<double>;
using IntentVector    = std::vector<double>;

// --------------------------------------------------------------------------
//  Intent action categories
// --------------------------------------------------------------------------
enum class ActionType : uint8_t {
    NONE = 0,
    SPAWN_ENTITY,
    MUTATE_ENTITY,
    CHANGE_PLAYER,
    ALTER_WORLD_RULES,
    // add more as the engine grows
    COUNT
};

// --------------------------------------------------------------------------
//  Neuron parameter helper
// --------------------------------------------------------------------------
struct NeuronParams {
    bool isHH = false;          // true -> Hodgkin‑Huxley, false -> Izhikevich
    double a = 0.02, b = 0.2, c = -65.0, d = 8.0;   // Izhikevich defaults
};

// --------------------------------------------------------------------------
//  Simple math utilities
// --------------------------------------------------------------------------
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double clamp(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

} // namespace abel
