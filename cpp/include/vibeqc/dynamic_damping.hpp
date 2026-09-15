// Dynamic damping — adaptive density-mixing α (Zerner-Hehenberger 1979).
//
// Reference:
//   * M. C. Zerner, M. Hehenberger, "A dynamical damping scheme for
//     converging molecular SCF calculations", Chem. Phys. Lett. 62, 550
//     (1979). The original paper's heuristic: when the SCF energy is
//     monotonically decreasing the damping α is reduced toward 0
//     (less mixing → faster progress); when the energy oscillates or
//     increases the α is increased toward 1 (more mixing → stabilises
//     the residual).
//
// The static-damping case (``damping`` constant) is the special case
// ``dynamic_damping = false``. When true, α is adjusted iteration-by-
// iteration via ``update_dynamic_damping(...)``; the driver keeps the
// previous E and α as local state and feeds them in.
//
// Distinct from the C1c quadratic-fallback / D2c Newton phases:
//   * Dynamic damping is a *first-order* mixing aid that modulates the
//     density before the next Fock build. Always cheap (one float
//     update per iteration).
//   * Quadratic / Newton are *second-order* SCF steps that replace the
//     diagonalisation entirely. Expensive per step but quadratic
//     convergence near the fixed point.
//
// Composes freely with FMIXING and the SCF-accelerator family (DIIS /
// KDIIS / EDIIS / EDIIS+DIIS) — those operate on the Fock matrix
// downstream of the density-mixing step and don't see the α directly.

#pragma once

#include <algorithm>

namespace vibeqc {

// Update the dynamic-damping α for the next iteration given the latest
// SCF energy. Caller maintains ``alpha`` and ``e_prev`` as local state
// across iterations.
//
// Heuristic:
//   * First call (``have_prev == false``): no update; alpha unchanged.
//   * dE = E_new − E_prev. If dE > 0 (energy increased — SCF is
//     diverging or oscillating), bump α toward ``alpha_max`` to damp
//     more aggressively. If dE < threshold (substantial decrease),
//     ease α toward ``alpha_min`` to let the SCF make faster progress.
//     Otherwise leave α unchanged (small fluctuations don't move it).
//
// Returns the updated α, clamped to [alpha_min, alpha_max].
inline double update_dynamic_damping(double alpha,
                                      double e_new,
                                      double e_prev,
                                      bool have_prev,
                                      double alpha_min = 0.0,
                                      double alpha_max = 0.95,
                                      double step = 0.1,
                                      double decrease_threshold = 1.0e-4)
{
    if (!have_prev) {
        return std::clamp(alpha, alpha_min, alpha_max);
    }
    const double dE = e_new - e_prev;
    if (dE > 0.0) {
        // Energy went up — damp harder.
        alpha = std::min(alpha_max, alpha + step);
    } else if (dE < -decrease_threshold) {
        // Substantial decrease — ease the damping back toward 0 so the
        // SCF makes more aggressive progress next iter.
        alpha = std::max(alpha_min, alpha - 0.5 * step);
    }
    // Small fluctuations (|dE| < decrease_threshold and >= 0) leave α
    // unchanged — avoids pumping α up and down on near-converged
    // iterations where the energy is stationary to noise.
    return std::clamp(alpha, alpha_min, alpha_max);
}

}  // namespace vibeqc
