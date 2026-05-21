#include "WorldDecoder.h"
#include "../brain/Core.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace abel {

WorldDecoder::WorldDecoder(int inputDim, int outputDim, double lr)
    : inputDim_(inputDim), outputDim_(outputDim), learningRate_(lr),
      rng_(std::random_device{}()),
      noiseDist_(0.0, 0.1)
{
    weights_.resize(outputDim_, std::vector<double>(inputDim_, 0.0));
    bias_.resize(outputDim_, 0.0);
    double limit = std::sqrt(2.0 / inputDim_);
    std::uniform_real_distribution<double> initDist(-limit, limit);
    for (auto& row : weights_)
        for (auto& w : row) w = initDist(rng_);
    for (auto& b : bias_) b = initDist(rng_);
}

std::vector<double> WorldDecoder::decode(const std::vector<double>& state) const {
    std::vector<double> output(outputDim_, 0.0);
    for (int i = 0; i < outputDim_; ++i) {
        double z = bias_[i];
        for (int j = 0; j < inputDim_; ++j) z += weights_[i][j] * state[j];
        output[i] = std::tanh(z);
    }
    return output;
}

std::vector<double> WorldDecoder::sampleAction(const std::vector<double>& state, double noiseStd) {
    std::vector<double> mean = decode(state);
    for (int i = 0; i < outputDim_; ++i) {
        double noise = noiseDist_(rng_) * noiseStd;
        mean[i] += noise;
        mean[i] = clamp(mean[i], -1.0, 1.0);
    }
    return mean;
}

void WorldDecoder::reinforce(const std::vector<double>& state,
                             const std::vector<double>& action,
                             double reward) {
    std::vector<double> mean = decode(state);
    for (int i = 0; i < outputDim_; ++i) {
        double diff = action[i] - mean[i];
        bias_[i] += learningRate_ * reward * diff;
        for (int j = 0; j < inputDim_; ++j)
            weights_[i][j] += learningRate_ * reward * diff * state[j];
    }
}

} // namespace abel
