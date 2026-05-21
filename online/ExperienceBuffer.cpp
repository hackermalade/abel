#include "ExperienceBuffer.h"
#include <algorithm>
#include <random>

namespace abel {

static std::mt19937 rng(std::random_device{}());

ExperienceBuffer::ExperienceBuffer(int capacity)
    : capacity_(capacity > 0 ? capacity : 10000),
      nextIdx_(0), size_(0)
{
    buffer_.resize(capacity_);
}

void ExperienceBuffer::add(const SensoryVector& sensory,
                           const MotorVector& motor,
                           double reward) {
    Experience exp;
    exp.sensory = sensory;
    exp.motor   = motor;
    exp.reward  = reward;
    buffer_[nextIdx_] = std::move(exp);
    nextIdx_ = (nextIdx_ + 1) % capacity_;
    if (size_ < capacity_) ++size_;
}

bool ExperienceBuffer::isFull() const { return size_ >= capacity_; }
int ExperienceBuffer::size() const { return size_; }

std::vector<Experience> ExperienceBuffer::getBatch(int batchSize) const {
    if (batchSize <= 0 || size_ == 0) return {};
    batchSize = std::min(batchSize, size_);
    std::vector<Experience> batch;
    batch.reserve(batchSize);
    std::uniform_int_distribution<int> dist(0, size_ - 1);
    for (int i = 0; i < batchSize; ++i)
        batch.push_back(buffer_[dist(rng)]);
    return batch;
}

void ExperienceBuffer::clear() {
    nextIdx_ = 0;
    size_ = 0;
}

} // namespace abel
