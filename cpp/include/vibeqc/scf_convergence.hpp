// SCF convergence helpers — shared by every molecular and periodic SCF
// driver. Lifts the duplicated convergence-check logic out of each
// driver's inner loop.
//
// Convention: convergence is declared once iter > 1 (so the first
// iteration's ``dE`` against the zero-initialised ``E_prev`` doesn't
// trigger), the absolute energy change is below ``conv_tol_energy``,
// AND the orbital-gradient (or per-spin max) is below ``conv_tol_grad``.
// Both must hold. Used identically by run_rhf / run_uhf / run_rks /
// run_uks (molecular) and the periodic Γ-only + multi-k drivers.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace vibeqc {

inline void validate_scf_max_iter(int max_iter, const char* context) {
    if (max_iter < 0) {
        throw std::invalid_argument(
            std::string(context) + ": max_iter must be non-negative");
    }
}

// Standard SCF convergence test.
//
// Parameters
// ----------
// iter            : SCF iteration index (1-based). Iter 1 always
//                   returns false because the energy-change reference
//                   is meaningless on the first pass.
// delta_e         : E[iter] − E[iter − 1] (Hartree).
// grad_norm       : Orbital-gradient norm (closed-shell:
//                   ``‖F D S − S D F‖_F``; open-shell: max over
//                   spins). The KDIIS / orbital-rotation-gradient
//                   metric is NOT what's checked here — convergence
//                   is always on the commutator metric, regardless
//                   of which DIIS variant is driving extrapolation.
// conv_tol_energy : Energy change tolerance.
// conv_tol_grad   : Gradient norm tolerance.
inline bool is_scf_converged(int iter,
                              double delta_e,
                              double grad_norm,
                              double conv_tol_energy,
                              double conv_tol_grad) {
    return iter > 1
        && std::abs(delta_e) < conv_tol_energy
        && grad_norm < conv_tol_grad;
}

// Decide whether a molecular direct-SCF run must leave its coarse Fock map
// before convergence can be certified.  Pulay's commutator is
// the stationarity condition (J. Comput. Chem. 3, 556 (1982), Eq. 4).
// Almlöf, Fægri & Korsell give the difference-Fock recurrence in Eq. 20 and
// recommend a final energy-oriented iteration after a non-energy monitor
// passes (J. Comput. Chem. 3, 385 (1982), p. 389).  Häser & Ahlrichs show
// that the current difference-Fock error contains errors from prior screened
// increments (J. Comput. Chem. 10, 104 (1989), Eqs. 30-35).  We therefore
// treat dE across the incremental/nonincremental boundary as a change between
// different numerical maps, not as a convergence observation.
//
// Molecular direct drivers consequently use incremental and loose-screened
// builds only as coarse phases.  Once the orbital gradient first becomes
// eligible for convergence, they disable the cache, force the configured
// tight Schwarz threshold, reject the first cross-map dE, and require the next
// same-map row to pass both ordinary tolerances.  No switch is needed on iter
// 1 because convergence cannot yet be accepted against a meaningful
// predecessor.  Periodic drivers deliberately do not use this helper; their
// maps and builders are separate.
inline bool needs_full_fock_refinement(int iter,
                                       double grad_norm,
                                       double conv_tol_grad,
                                       bool coarse_fock_active) {
    return coarse_fock_active
        && iter > 1
        && grad_norm < conv_tol_grad;
}

// SCF oscillation detector — shared by every molecular SCF driver.
//
// An oscillating SCF is one where significant energy changes flip sign
// frequently while the orbital-gradient envelope is not contracting.
// Sign flips alone are insufficient: a healthy DIIS tail can alternate
// as both dE and the commutator norm shrink toward their tolerances.
//
// Changes within two orders of ``conv_tol_energy`` are treated as a
// converging tail, not oscillation. Since a sign-flipping limit cycle is
// nominally period two, each commutator norm is compared with the same
// phase two iterations earlier. Any contraction leaves DIIS intact.
// A DIIS-history reset also starts a new detection epoch: samples on the
// two sides of a restart do not describe one limit cycle.
//
// When oscillation is detected, the SCF drivers engage a persistent
// Saunders-Hillier level shift (auto_level_shift_value), clear the
// DIIS history, and continue — this damps the virtual block and steers
// the SCF into the correct basin. The shift is held persistently;
// oscillation is a mid-SCF phenomenon, not a startup transient.
//
template <typename SCFIter>
inline bool detect_scf_oscillation(const std::vector<SCFIter>& trace,
                                    int window,
                                    int min_sign_flips,
                                    double conv_tol_energy,
                                    double noise_floor = 1e-10) {
    const auto n = static_cast<int>(trace.size());
    if (window < 3 || min_sign_flips < 1 || min_sign_flips >= window
        || n < window || !std::isfinite(conv_tol_energy)
        || !std::isfinite(noise_floor) || conv_tol_energy < 0.0
        || noise_floor < 0.0) {
        return false;
    }

    const int start = n - window;
    const double significance_floor = std::max(
        noise_floor, 100.0 * conv_tol_energy);

    for (int i = start; i < n; ++i) {
        const auto& row = trace[static_cast<std::size_t>(i)];
        if (!std::isfinite(row.delta_e) || !std::isfinite(row.grad_norm)
            || row.grad_norm < 0.0) {
            return false;
        }
        if (i > start) {
            const auto& previous = trace[static_cast<std::size_t>(i - 1)];
            if (previous.diis_subspace > 0
                && row.diis_subspace < previous.diis_subspace) {
                return false;
            }
        }
        if (i >= start + 2) {
            const auto& same_phase = trace[static_cast<std::size_t>(i - 2)];
            if (row.grad_norm < same_phase.grad_norm) return false;
        }
    }

    if (std::abs(trace.back().delta_e) <= significance_floor) return false;

    int flips = 0;
    int prev_sign = 0;
    for (int i = start; i < n; ++i) {
        const double de = trace[static_cast<std::size_t>(i)].delta_e;
        if (std::abs(de) <= significance_floor) {
            prev_sign = 0;
            continue;
        }
        const int sign = (de > 0.0) ? 1 : -1;
        if (prev_sign != 0 && sign != prev_sign) ++flips;
        prev_sign = sign;
    }
    return flips >= min_sign_flips;
}

}  // namespace vibeqc
