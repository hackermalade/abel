#include "Quantum.h"
#include "Core.h"

#include <complex>
#include <cmath>
#include <random>
#include <vector>

using namespace abel;

// ── Local random number generator ────────────────────────────────────────
static std::mt19937 rng(std::random_device{}());
static std::normal_distribution<double> normal(0.0, 1.0);
static std::uniform_real_distribution<double> uniform(0.0, 1.0);

// ──────────────────────────────────────────────────────────────────────────
QuantumDecoherenceModule::QuantumDecoherenceModule(int n_neurons, double dt)
    : n(n_neurons), dt(dt)
{
    // Initial random pure state (complex amplitudes)
    amplitudes.resize(n);
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double real_part = uniform(rng) - 0.5;
        double imag_part = uniform(rng) - 0.5;
        amplitudes[i] = std::complex<double>(real_part, imag_part);
        norm += std::norm(amplitudes[i]);
    }
    // Normalise
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (int i = 0; i < n; ++i)
            amplitudes[i] /= norm;
    }
}

// ──────────────────────────────────────────────────────────────────────────
std::vector<double> QuantumDecoherenceModule::evolve(const std::vector<double>& activity_vector) {
    if (activity_vector.size() < n) return std::vector<double>(n, 0.0);

    // Environmental decoherence: amplitudes are pulled toward eigenstates
    for (int i = 0; i < n; ++i) {
        double weight = std::abs(activity_vector[i]) + 1e-6;
        amplitudes[i] *= (1.0 - 0.01 * weight);
    }

    // Add small complex noise (quantum fluctuations)
    for (int i = 0; i < n; ++i) {
        double noise_real = normal(rng) * 0.01;
        double noise_imag = normal(rng) * 0.01;
        amplitudes[i] += std::complex<double>(noise_real, noise_imag) * 0.001;
    }

    // Renormalise to conserve probability
    double norm = 0.0;
    for (int i = 0; i < n; ++i)
        norm += std::norm(amplitudes[i]);
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (int i = 0; i < n; ++i)
            amplitudes[i] /= norm;
    }

    // Compute classical probabilities (squared magnitudes)
    std::vector<double> probs(n, 0.0);
    for (int i = 0; i < n; ++i)
        probs[i] = std::norm(amplitudes[i]);

    // Spontaneous collapse (objective reduction) with small probability per step
    if (uniform(rng) < 0.1) {
        // Pick a state according to probabilities
        double rand = uniform(rng);
        double cumulative = 0.0;
        int collapse_idx = 0;
        for (int i = 0; i < n; ++i) {
            cumulative += probs[i];
            if (rand < cumulative) {
                collapse_idx = i;
                break;
            }
        }
        // Collapse to that pure state
        for (int i = 0; i < n; ++i)
            amplitudes[i] = (i == collapse_idx) ? 1.0 : 0.0;
        // Update probabilities accordingly
        for (int i = 0; i < n; ++i)
            probs[i] = (i == collapse_idx) ? 1.0 : 0.0;
    }

    return probs;
}
