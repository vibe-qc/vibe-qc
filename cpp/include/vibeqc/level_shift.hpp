// Unified Saunders-Hillier level-shift policy, shared by the molecular
// and periodic SCF drivers.
//
// The Saunders-Hillier form adds
//
//     F_shifted = F + b · (S − w · S · D · S)
//
// to the Fock matrix before diagonalisation, raising the virtual orbital
// eigenvalues by ``b`` while leaving the occupied ones fixed (``w = ½``
// for a closed-shell density with occupations in {0, 2}; ``w = 1`` for a
// single-spin density with occupations in {0, 1}). The shift is inert at
// the converged density — ``S · D_∞ · S`` projects onto the occupied
// subspace, so at the fixed point the added term vanishes for the occupied
// block and the SCF fixed point is unchanged. Only the iteration path is
// damped.
//
// This header is the single source of truth for *how much* shift applies
// at a given SCF iteration. Both the auto/warm-up step function (start
// shifted, release after a few startup cycles) and an explicit
// per-iteration decay schedule (CRYSTAL ``LEVSHIFT B IRESET`` style) are
// resolved here. The Python ``LevelShiftSchedule`` helper lowers to the
// ``level_shift_schedule`` vector consumed below, and the periodic Python
// drivers call the bound version of ``level_shift_at_iter`` so molecular
// and periodic share identical semantics.
//
// CRYSTAL reference: Dovesi, Pisani, Roetti, Saunders, Phys. Rev. B 28,
// 5781 (1983); Pisani, Dovesi, Roetti, Hartree-Fock Ab Initio Treatment
// of Crystalline Systems, Lecture Notes in Chemistry vol 48 (Springer,
// 1988), Sec. 3.7.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace vibeqc {

// Resolve the Saunders-Hillier shift magnitude ``b`` for a single SCF
// iteration. O(1), allocation-free — safe to call inside the SCF hot loop.
//
//   level_shift            base shift magnitude (Hartree); 0 disables.
//   level_shift_warmup      -1 = auto (up to five shifted startup cycles,
//     _cycles               always leaving ≥1 unshifted tail cycle);
//                           0 = persistent (shift held every iteration);
//                           N > 0 = explicit warm-up length, capped to
//                           ``max_iter - 1`` so a tail cycle survives.
//   level_shift_schedule   explicit per-iteration curve. Empty ⇒ derive
//                          from the warm-up logic above. Non-empty ⇒ use
//                          it directly: ``schedule[iter-1]`` with the last
//                          entry reused for iterations past its length
//                          (set the last entry to 0.0 to release).
//   max_iter               SCF iteration cap (drives the auto warm-up
//                          length and the tail-cycle guarantee).
//   iter                   current SCF iteration, 1-indexed.
inline double level_shift_at_iter(double level_shift,
                                  int level_shift_warmup_cycles,
                                  const std::vector<double>& level_shift_schedule,
                                  int max_iter,
                                  int iter) {
    // An explicit schedule takes precedence over the warm-up logic.
    if (!level_shift_schedule.empty()) {
        int idx = iter - 1;
        if (idx < 0) idx = 0;
        const int last = static_cast<int>(level_shift_schedule.size()) - 1;
        if (idx > last) idx = last;
        return level_shift_schedule[static_cast<std::size_t>(idx)];
    }

    // No base shift, or too few iterations to warm up — nothing to do.
    if (level_shift == 0.0 || max_iter <= 1) return 0.0;

    // Resolve the warm-up length (mirror of the Python
    // ``_resolve_level_shift_warmup_cycles`` helper).
    int raw = level_shift_warmup_cycles;
    if (raw < -1) raw = -1;  // defensive; the bind layer validates the input
    int warmup;
    if (raw < 0) {
        warmup = std::min(5, max_iter - 1);   // auto
    } else if (raw == 0) {
        warmup = 0;                           // persistent
    } else {
        warmup = std::min(raw, max_iter - 1); // explicit, tail-cycle capped
    }

    // Persistent: hold the shift at every iteration.
    if (warmup == 0) return level_shift;

    // Warm-up-then-release step function: shifted for iters 1..warmup,
    // released (0) thereafter.
    return (iter <= warmup) ? level_shift : 0.0;
}

// Convenience overload for any options struct exposing ``level_shift``,
// ``level_shift_warmup_cycles``, ``level_shift_schedule`` and ``max_iter``.
template <typename Opts>
inline double level_shift_at_iter(const Opts& opts, int iter) {
    return level_shift_at_iter(opts.level_shift,
                               opts.level_shift_warmup_cycles,
                               opts.level_shift_schedule,
                               opts.max_iter,
                               iter);
}

// Which density the caller is handing to ``apply_level_shift``. This picks
// the weight ``w`` in ``F + b·(S − w·S·D·S)`` and is NOT a matter of taste:
//
//   SPIN   a single-spin density D_σ, occupations in {0, 1}. It is
//          idempotent in the overlap metric (D_σ S D_σ = D_σ), so S·D_σ·S is
//          exactly the projector onto the occupied manifold and ``w = 1``.
//   TOTAL  a closed-shell total density D = 2P, occupations in {0, 2}, with
//          P S P = P. Then S·D·S = 2·S·P·S and ``w = ½`` recovers the same
//          projector.
//
// Either way the occupied eigenvalues do not move and every virtual rises by
// exactly ``b``. Any other weight lifts the occupied block too, so the gap
// opens by less than ``b``: the SCF still reaches the same fixed point (a
// uniform shift inside the occupied subspace cannot rotate occupied
// orbitals), which is exactly why such a mistake is invisible to every
// energy assertion. ``run_uks_periodic_gamma_ewald3d`` carried ``w = ½`` on
// spin densities until 2026-07-10 for that reason. Pinned by
// ``tests/test_periodic_level_shift.py::test_level_shift_operator_convention``.
enum class LevelShiftDensity { SPIN, TOTAL };

namespace detail {
inline double level_shift_weight(LevelShiftDensity convention) {
    return convention == LevelShiftDensity::TOTAL ? 0.5 : 1.0;
}
}  // namespace detail

// Real (molecular, or Γ-point periodic) Saunders-Hillier shift.
// Returns ``fock`` untouched when ``b == 0``, so the S·D·S products are
// skipped entirely on any cycle whose resolved shift has been released.
inline Eigen::MatrixXd apply_level_shift(const Eigen::MatrixXd& fock,
                                         const Eigen::MatrixXd& overlap,
                                         const Eigen::MatrixXd& density,
                                         double b,
                                         LevelShiftDensity convention) {
    if (b == 0.0) {
        return fock;
    }
    const double w = detail::level_shift_weight(convention);
    Eigen::MatrixXd shifted = fock + b * overlap
        - (b * w) * (overlap * density * overlap);
    return 0.5 * (shifted + shifted.transpose().eval());
}

// Per-k Hermitian analogue: F(k) + b·(S(k) − w·S(k)·P(k)·S(k)).
inline Eigen::MatrixXcd apply_level_shift_k(const Eigen::MatrixXcd& fock_k,
                                            const Eigen::MatrixXcd& overlap_k,
                                            const Eigen::MatrixXcd& density_k,
                                            double b,
                                            LevelShiftDensity convention) {
    if (b == 0.0) {
        return fock_k;
    }
    const double w = detail::level_shift_weight(convention);
    Eigen::MatrixXcd shifted = fock_k + b * overlap_k
        - (b * w) * (overlap_k * density_k * overlap_k);
    return 0.5 * (shifted + shifted.adjoint().eval());
}

}  // namespace vibeqc
