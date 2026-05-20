#pragma once
#include <vector>
#include "ExperienceBuffer.h"

namespace abel {

class WorldDecoder;   // forward declaration

class OnlineTrainer {
public:
    OnlineTrainer(WorldDecoder* decoder, ExperienceBuffer* buffer);

    // Train on a specific batch of experiences (sensory = decoder state, motor = action)
    void trainStep(const std::vector<Experience>& batch);

    // Convenience: pull a batch from the buffer and train on it
    void trainFromBuffer(int batchSize = 32);

private:
    WorldDecoder* decoder_;
    ExperienceBuffer* buffer_;
};

} // namespace abel
