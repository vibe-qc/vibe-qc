// Physical-basin gate for GFN2 shell-charge fixed points, shared by every
// GFN2 SCC driver (issue #411).
//
// The GFN2 third-order term admits a spurious over-polarized
// intra-atomic shell-transfer fixed point.  A converged state that leaves
// it unfiltered is a wrong answer, never a result, so each driver tests its
// converged shell charges against one set of bounds before returning them.
//
// These bounds were originally calibrated in the SECCM adapter
// (seccm/gfn2.cpp) and were copied verbatim into the periodic Gamma driver
// by the 1fc1dc73b port (IID 300).  They MUST live here, in one place: a
// recalibration that touches one copy and not the other silently diverges
// the two routes' fail-closed behaviour.
#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"

namespace vibeqc {
namespace semiempirical {

// Shell population tolerance. A converged state whose Mulliken shell
// population leaves [0, 2*n_funcs] by more than this many electrons is
// unphysical: the over-polarized intra-atomic shell-transfer fixed point
// that bulk-metal supercells can visit at compressed volumes (fcc Cu 2x2x2
// at a = 3.434 A reaches d-shell populations around 16 and negative s-shell
// populations while its atomic charges stay near zero). The tolerance is
// loose enough that ordinary Mulliken overshoot at sane fixed points
// (observed up to about one electron in the same cells) never trips it.
inline constexpr double kShellPopulationViolationTolerance = 1.5;

// True when every shell Mulliken population lies inside
// [-tol, 2*n_funcs + tol]; false at the over-polarized fixed point.
//
// ``n0_shell`` is the neutral reference occupation per shell (the shell
// populations are dq_shell + n0_shell). ``shell_info`` must be parallel to
// both vectors.
inline bool shell_populations_within_basin(
    const std::vector<GFN2ShellInfo>& shell_info,
    const Eigen::VectorXd& n0_shell,
    const Eigen::VectorXd& dq_shell) {
    if (dq_shell.size() != n0_shell.size()
        || dq_shell.size() != static_cast<Eigen::Index>(shell_info.size())) {
        return false;
    }
    for (std::size_t si = 0; si < shell_info.size(); ++si) {
        const double population = dq_shell(si) + n0_shell(si);
        const double capacity =
            2.0 * static_cast<double>(shell_info[si].n_funcs);
        if (population < -kShellPopulationViolationTolerance
            || population > capacity + kShellPopulationViolationTolerance) {
            return false;
        }
    }
    return true;
}

// Over-polarization gate: the spurious SCC basin observed in bulk-metal
// supercells moves electrons between shells on a scale of several
// electrons per shell (fcc Cu 2x2x2 at a = 3.434 A: shell dq rms 4.8 e;
// 4x4x4: 5.8-6.2 e; the embedded map's charge-separated variant: 3.8 e
// with atomic rms 3.3 e). Physical states never reach this scale: the
// on-site shell hardness makes shell fluctuations of several electrons
// cost order 1 Ha per atom, and the observed legitimate ceiling across
// the ladder (diamond, MgO, corundum, Cu sane branch) is a shell rms
// of about 0.73 e with single-shell peaks up to about 1.24 e (madelung
// MgO 2x2x2, re-measured 2026-08-26).
// The bound sits at 2.5 e with wide margins on both sides. The earlier
// atomic-mismatch ratio condition was dropped because the embedded map
// can reach the basin WITH large atomic charges (charge separation),
// which the ratio tolerated.
inline constexpr double kShellRmsPolarizationLimit = 2.5;  // electrons

// The self-consistent-kernel (ewald_gamma) map reaches a second spurious
// fixed point with the opposite signature: a charge-density-wave with
// atomic charges of a few electrons and shell fluctuations just under
// the shell bound (fcc Cu 2x2x2 ewald_gamma: atomic q_rms 2.7 e, shell
// rms 2.2 e, ~0.8 Ha/atom below xtb). No physical GFN2 state has atomic
// Mulliken charges near this scale (the most ionic ladder members stay
// below 1 e rms), so an absolute atomic-rms bound fails that basin
// closed without touching legitimate results.
inline constexpr double kAtomicRmsPolarizationLimit = 2.0;  // electrons

// The RMS checks above are intentionally system-size diagnostics, so they
// must be paired with local bounds: otherwise one pathological atom or shell
// can be diluted by arbitrarily many ordinary sites.  The conservative 4 e
// ceiling remains more than three times the largest local fluctuation
// observed in the validated molecular/bulk ladder (1.24 e, madelung MgO
// 2x2x2) while rejecting the 11.5 e localized counterexample from the
// independent audit. States with per-site amplitudes inside the
// (1.24, 4.0] e window would still pass; no such state has been observed
// on a converged physical run.
inline constexpr double kLocalShellPolarizationLimit = 4.0;  // electrons
inline constexpr double kLocalAtomicPolarizationLimit = 4.0;  // electrons

// True when both the shell and the atomic charge fluctuations sit inside
// the calibrated RMS and per-site bounds above. An empty dq_shell vector
// is a no-op (true): nothing has been polarized.
inline bool charge_state_within_basin(
    const Eigen::VectorXd& dq_atom, const Eigen::VectorXd& dq_shell) {
    if (dq_shell.size() == 0) return true;
    const double shell_rms = std::sqrt(
        dq_shell.squaredNorm() / static_cast<double>(dq_shell.size()));
    const double atom_rms = dq_atom.size() == 0
        ? 0.0
        : std::sqrt(
              dq_atom.squaredNorm() / static_cast<double>(dq_atom.size()));
    const double shell_max = dq_shell.cwiseAbs().maxCoeff();
    const double atom_max = dq_atom.size() == 0
        ? 0.0
        : dq_atom.cwiseAbs().maxCoeff();
    return shell_rms <= kShellRmsPolarizationLimit
        && atom_rms <= kAtomicRmsPolarizationLimit
        && shell_max <= kLocalShellPolarizationLimit
        && atom_max <= kLocalAtomicPolarizationLimit;
}

}  // namespace semiempirical
}  // namespace vibeqc
