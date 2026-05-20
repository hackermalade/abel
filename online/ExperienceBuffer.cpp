#pragma once

#include "../brain/Core.h"   // for SensoryVector, MotorVector
#include <vector>

namespace abel {

struct Experience {
    SensoryVector sensory;
    MotorVector   motor;
    double        reward = 0.0;
};

class ExperienceBuffer {
public:
    explicit ExperienceBuffer(int capacity = 10000);

    void add(const SensoryVector& sensory, const MotorVector& motor, double reward);
    bool isFull() const;
    int  size() const;

    std::vector<Experience> getBatch(int batchSize) const;
    void clear();

private:
    int capacity_;
    std::vector<Experience> buffer_;
    int nextIdx_;
    int size_;
};

} // namespace abel
