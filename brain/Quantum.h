#pragma once
#include <complex>
#include <vector>

namespace abel {
class QuantumDecoherenceModule {
public:
    QuantumDecoherenceModule(int n_neurons = 10, double dt = 0.5);
    std::vector<double> evolve(const std::vector<double>& activity_vector);

private:
    int n;
    double dt;
    std::vector<std::complex<double>> amplitudes;
};
}
