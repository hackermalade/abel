#include "Synapses.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace abel;

// ── Constructor ──────────────────────────────────────────────────────────
Synapse::Synapse(int pre_idx, int post_idx, double weight, const std::string& type,
                 double U, double tau_d, double tau_f, double delay)
    : pre(pre_idx), post(post_idx), w(weight), type(type),
      U(U), x(1.0), y(0.0), tau_d(tau_d), tau_f(tau_f),
      g(0.0), pre_trace(0.0), post_trace(0.0),
      last_pre(-1e9), last_post(-1e9),
      cumulative_trace(0.0), pruned(false),
      theta_m(0.5), tau_theta(1000.0), delay(delay)
{
    // Set reversal potential and decay time constant based on type
    if (type == "AMPA") {
        E = 0.0;
        tau = 5.0;
    } else if (type == "NMDA") {
        E = 0.0;
        tau = 150.0;
    } else if (type == "GABAa") {
        E = -70.0;
        tau = 10.0;
    } else if (type == "GABAb") {
        E = -90.0;
        tau = 150.0;
    } else {
        throw std::invalid_argument("Unknown synapse type: " + type);
    }
}

// ── Magnesium block for NMDA ─────────────────────────────────────────────
double Synapse::mgBlock(double V_post) const {
    if (type != "NMDA") return 1.0;
    const double mg_conc = 1.0;    // mM
    const double kd = 3.57;        // mM
    const double gamma = 0.062;    // 1/mV
    return 1.0 / (1.0 + (mg_conc / kd) * std::exp(-gamma * V_post));
}

// ── Presynaptic spike ────────────────────────────────────────────────────
double Synapse::preSpike(double t) {
    // Short‑term plasticity: release from the active zone
    y += x * U * (1.0 - y);
    double g_inc = y * w;
    x -= g_inc;                  // consume vesicles
    pre_trace = 1.0;             // reset presynaptic trace
    last_pre = t;
    return g_inc;
}

// ── Postsynaptic spike ───────────────────────────────────────────────────
void Synapse::postSpike(double t) {
    post_trace = 1.0;
    last_post = t;
}

// ── Decay and recovery ───────────────────────────────────────────────────
void Synapse::update(double dt, double t) {
    // Conductance decay
    g -= g * dt / tau;
    if (g < 0.0) g = 0.0;

    // Trace decay
    pre_trace  *= std::exp(-dt / 20.0);
    post_trace *= std::exp(-dt / 20.0);

    // Vesicle recovery (depression) and facilitation decay
    x += (1.0 - x) * dt / tau_d;
    y -= y * dt / tau_f;
    if (y < 0.0) y = 0.0;

    // Cumulative trace for microglial pruning (slow integration)
    cumulative_trace = 0.99 * cumulative_trace + 0.01 * (pre_trace * post_trace);
}

// ── STDP + BCM plasticity ────────────────────────────────────────────────
void Synapse::stdpUpdate(double dt, double t) {
    if (pre_trace > 0.01 && post_trace > 0.01) {
        const double A_plus  = 0.005;
        const double A_minus = 0.003;
        const double tau_plus  = 20.0;
        const double tau_minus = 20.0;

        double dt_pre  = t - last_pre;
        double dt_post = t - last_post;

        double dw = 0.0;
        if (dt_pre > 0)  dw += A_plus  * std::exp(-dt_pre  / tau_plus);
        if (dt_post > 0) dw -= A_minus * std::exp(-dt_post / tau_minus);

        w += dw * dt;

        // BCM metaplasticity: sliding threshold
        theta_m += dt * (post_trace * post_trace - theta_m) / tau_theta;
        w += dt * 0.001 * post_trace * (post_trace - theta_m);

        // Clamp weight
        w = std::max(0.0, std::min(5.0, w));
    }
}
