#include "PredictiveCoder.h"
#include <cmath>
#include <random>

using namespace abel;

// ── local random number generators ──────────────────────────────────────
static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<double> rand_w(-0.1, 0.1);   // small init

// helper: sigmoid
static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ──────────────────────────────────────────────────────────────────────────
PredictiveCoder::PredictiveCoder(int input_size, int hidden_size)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      hidden_(hidden_size, 0.0)
{
    // Weight matrices: input->hidden and hidden->hidden for GRU gates
    auto initMat = [](int rows, int cols) {
        std::vector<std::vector<double>> mat(rows, std::vector<double>(cols));
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                mat[r][c] = rand_w(rng);
        return mat;
    };

    W_ir_ = initMat(hidden_size, input_size);   // input -> reset gate
    W_hr_ = initMat(hidden_size, hidden_size);  // hidden -> reset gate

    W_iz_ = initMat(hidden_size, input_size);   // input -> update gate
    W_hz_ = initMat(hidden_size, hidden_size);  // hidden -> update gate

    W_in_ = initMat(hidden_size, input_size);   // input -> new gate
    W_hn_ = initMat(hidden_size, hidden_size);  // hidden -> new gate

    // Linear layer: hidden -> input (prediction)
    W_pred_ = initMat(input_size, hidden_size);
}

// ──────────────────────────────────────────────────────────────────────────
double PredictiveCoder::forward(const std::vector<double>& x) {
    // x size must equal input_size_
    const int H = hidden_size_;

    // Compute gates
    auto matVecMul = [](const std::vector<std::vector<double>>& mat,
                        const std::vector<double>& vec) {
        std::vector<double> res(mat.size(), 0.0);
        for (size_t r = 0; r < mat.size(); ++r)
            for (size_t c = 0; c < vec.size(); ++c)
                res[r] += mat[r][c] * vec[c];
        return res;
    };

    auto addVecs = [](const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
        return res;
    };

    auto hadamard = [](const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] * b[i];
        return res;
    };

    // r_t = sigmoid( W_ir * x + W_hr * h_prev )
    std::vector<double> xW_ir = matVecMul(W_ir_, x);
    std::vector<double> hW_hr = matVecMul(W_hr_, hidden_);
    std::vector<double> r_t(H);
    for (int i = 0; i < H; ++i) r_t[i] = sigmoid(xW_ir[i] + hW_hr[i]);

    // z_t = sigmoid( W_iz * x + W_hz * h_prev )
    std::vector<double> xW_iz = matVecMul(W_iz_, x);
    std::vector<double> hW_hz = matVecMul(W_hz_, hidden_);
    std::vector<double> z_t(H);
    for (int i = 0; i < H; ++i) z_t[i] = sigmoid(xW_iz[i] + hW_hz[i]);

    // n_t = tanh( W_in * x + r_t * (W_hn * h_prev) )
    std::vector<double> xW_in = matVecMul(W_in_, x);
    std::vector<double> hW_hn = matVecMul(W_hn_, hidden_);
    std::vector<double> r_had = hadamard(r_t, hW_hn);
    std::vector<double> n_t(H);
    for (int i = 0; i < H; ++i) n_t[i] = std::tanh(xW_in[i] + r_had[i]);

    // h_new = (1 - z_t) * n_t + z_t * h_prev
    std::vector<double> h_new(H);
    for (int i = 0; i < H; ++i)
        h_new[i] = (1.0 - z_t[i]) * n_t[i] + z_t[i] * hidden_[i];

    // Predict output
    std::vector<double> pred = matVecMul(W_pred_, h_new);

    // Compute MSE surprise
    double mse = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double diff = pred[i] - x[i];
        mse += diff * diff;
    }
    mse /= x.size();

    // Update hidden state
    hidden_ = h_new;

    return mse;
}
