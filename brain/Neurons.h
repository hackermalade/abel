// abel/brain/Neurons.h
#pragma once
#include "Core.h"
#include <string>
#include <map>
#include <memory>

// Full class definitions – the files you just created
#include "HodgkinHuxleyNeuron.h"
#include "IzhikevichNeuron.h"

namespace abel {

struct NeuronRecord {
    std::shared_ptr<HodgkinHuxleyNeuron> hh;
    std::shared_ptr<IzhikevichNeuron> izh;
    bool fired = false;
    double energy = 1.0;
};

// Factory functions
HodgkinHuxleyNeuron createHodgkinHuxleyNeuron(double dt);
IzhikevichNeuron createIzhikevichNeuron(double a, double b, double c, double d, double dt);
NeuronParams getNeuronParams(const std::string& type);

} // namespace abel
