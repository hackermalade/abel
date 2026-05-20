#include "OnlineTrainer.h"
#include "ExperienceBuffer.h"
#include "../intent_engine/WorldDecoder.h"

#include <algorithm>
#include <vector>

namespace abel {

OnlineTrainer::OnlineTrainer(WorldDecoder* decoder, ExperienceBuffer* buffer)
    : decoder_(decoder), buffer_(buffer)
{}

void OnlineTrainer::trainStep(const std::vector<Experience>& batch) {
    if (!decoder_) return;

    // Each experience should contain:
    //   sensory : the state vector (workspace + context) fed to the decoder
    //   motor   : the action vector (intent) sampled by the decoder
    //   reward  : scalar reward obtained after applying the action
    for (const auto& exp : batch) {
        decoder_->reinforce(exp.sensory, exp.motor, exp.reward);
    }
}

void OnlineTrainer::trainFromBuffer(int batchSize) {
    if (!buffer_ || !decoder_) return;

    auto batch = buffer_->getBatch(batchSize);
    trainStep(batch);
}

} // namespace abel
