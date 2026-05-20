#pragma once
#include <vector>

namespace abel {
class PredictiveCoder {
public:
    PredictiveCoder(int input_size, int hidden_size = 16);
    double forward(const std::vector<double>& x);

private:
    int input_size_, hidden_size_;
    std::vector<double> hidden_;
    // weight matrices
    std::vector<std::vector<double>> W_ir_, W_hr_, W_iz_, W_hz_, W_in_, W_hn_;
    std::vector<std::vector<double>> W_pred_;
};
}
