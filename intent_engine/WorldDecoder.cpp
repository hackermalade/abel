#pragma once
#include <vector>
#include <random>

namespace abel {

class WorldDecoder {
public:
    WorldDecoder(int inputDim = 128, int outputDim = 256, double lr = 0.001);

    // Deterministic forward pass – returns the policy mean μ
    std::vector<double> decode(const std::vector<double>& state) const;

    // Sample an action by adding Gaussian noise to the mean
    std::vector<double> sampleAction(const std::vector<double>& state, double noiseStd = 0.1);

    // Update the policy using the REINFORCE rule (reward * (action - mean) * state)
    void reinforce(const std::vector<double>& state, const std::vector<double>& action, double reward);

private:
    int inputDim_, outputDim_;
    double learningRate_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> bias_;
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<double> noiseDist_;
};

} // namespace abel
