// abel/intent_engine/PlayerGenerator.h
#pragma once
#include <vector>

namespace abel {
class Player;   // forward declaration – real class is in world/Player.h

class PlayerGenerator {
public:
    PlayerGenerator();
    void generate(Player& player, const std::vector<double>& latent);
};
} // namespace abel
