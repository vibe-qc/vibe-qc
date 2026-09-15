// Translation-invariant enumeration of the images a two-centre lattice sum
// needs.
//
// A periodic two-centre matrix element is a sum over lattice translations
// of the ket centre,
//
//     <a|K|b>^per = sum_P <a|K|b_P> ,
//
// and the lattice-dependent factor depends on P ONLY through the physical
// separation of the two centres, T(P) = rho * ||A - B - P||^2 (Sharma &
// Beylkin, J. Chem. Theory Comput. (2021),
// doi:10.1021/acs.jctc.0c01195, Eqs. 17-18 and the definition of T(P)
// beneath Eq. 18; they prescribe accumulating over increasing P until that
// sum converges). The translations that contribute therefore form a ball of
// radius R_cut centred on the intra-cell offset A - B, NOT on the origin.
//
// Enumerating |g| <= R_cut and then computing every shell pair at that g --
// which is what vibe-qc's periodic kernels did before 2026-08-03 -- both
// misses translations that bring a distant pair into range and visits ones
// that do not, asymmetrically between the (P,Q) and (Q,P) orderings. The
// resulting matrix is not translation invariant: re-describing the same
// crystal by moving an atom one lattice vector changes it. Measured
// consequences of the same defect elsewhere in the tree:
//
//   * the compensated 2c Coulomb metric on MgO primitive FCC / def2-svp-jk
//     (offset 6.89 bohr) had a -5.4e-05 eigenvalue although it is a Gram
//     matrix, and moved by 7.2e-02 under a lattice translation; MDF's
//     orthogonalisation divides by its square root, which is how MgO
//     reached +5.3e+05 Ha (fixed in aux_eri.cpp, 2026-08-03);
//   * the Gamma overlap of LiH rocksalt / def2-svp at the shipped
//     cutoff_bohr = 15 moved by 7.2e-01 -- an overlap element is O(1) --
//     under the same relabelling. STILL OPEN as of 2026-08-06: the fix
//     needs images beyond the |g| ball, so the one-electron cell list
//     grows ~2.7x and a family of consumers that pair it index-for-index
//     with the plain ball has to move with it. A prototype was built and
//     measured (invariance 7.1e-15; Si/def2-SVP becomes positive definite
//     at the shipped cutoff). See handovers/HANDOVER_OPEN_BUGS_V015.md for
//     the measurements, the consumer inventory, and which sums must NOT
//     get this treatment.
//
// A single-atom cell has zero intra-cell offset, so the |g| ball IS the
// correct ball there; every vacuum-box fixture is structurally blind to
// this class.
//
// The fix pattern these helpers implement: pad the enumeration radius by
// the largest intra-cell offset so the cell list is a superset of what any
// pair needs, then filter each (pair, g) on the true separation. The set of
// (pair, g) terms is then exactly {(p, q, g) : ||O_p - O_q - g|| <= R_cut},
// which is translation invariant by construction, and the engine-call count
// stays at the physically required number rather than growing with the
// padding.
//
// Kernels that share a consistency contract must move together: an energy
// sum with its gradient partner (the gradient must differentiate the sum
// the energy builds) and with any cell-resolved *_blocks variant (summing
// the blocks must reproduce the Gamma result).

#pragma once

#include <cmath>
#include <vector>

#include <libint2/basis.h>
#include <libint2/shell.h>

#include <stdexcept>
#include <string>
#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

// Largest intra-cell separation between a bra-shell origin and a
// ket-shell origin, in bohr. This is how far the ball of contributing
// translations can be displaced from the origin.
inline double max_shell_offset(const libint2::BasisSet& bra,
                               const libint2::BasisSet& ket) {
    double d2_max = 0.0;
    for (const auto& p : bra) {
        for (const auto& q : ket) {
            const double dx = p.O[0] - q.O[0];
            const double dy = p.O[1] - q.O[1];
            const double dz = p.O[2] - q.O[2];
            d2_max = std::max(d2_max, dx * dx + dy * dy + dz * dz);
        }
    }
    return std::sqrt(d2_max);
}

// Lattice translations covering every (bra, ket) shell pair whose physical
// separation is within ``cutoff``. A superset: pair_in_range restores the
// exact physical set per pair.
inline std::vector<LatticeCell> pair_complete_cells(
    const PeriodicSystem& system, double cutoff,
    const libint2::BasisSet& bra, const libint2::BasisSet& ket) {
    return direct_lattice_cells(system,
                                cutoff + max_shell_offset(bra, ket));
}

// Is the physical separation of this shell pair within ``cutoff``? ``q`` is
// expected to be the ket shell already translated by the lattice vector, so
// the test is on |O_p - (O_q + g)|.
inline bool pair_in_range(const libint2::Shell& p, const libint2::Shell& q,
                          double cutoff) {
    const double dx = p.O[0] - q.O[0];
    const double dy = p.O[1] - q.O[1];
    const double dz = p.O[2] - q.O[2];
    return dx * dx + dy * dy + dz * dz <= cutoff * cutoff;
}

// As above, but taking the untranslated ket origin and the translation
// explicitly, so a caller can screen before materialising shifted shells.
inline bool pair_in_range(const libint2::Shell& p, const libint2::Shell& q,
                          const Eigen::Vector3d& g, double cutoff) {
    const double dx = p.O[0] - q.O[0] - g[0];
    const double dy = p.O[1] - q.O[1] - g[1];
    const double dz = p.O[2] - q.O[2] - g[2];
    return dx * dx + dy * dy + dz * dz <= cutoff * cutoff;
}

// #429 (stage 2): the Ewald-split nuclear family -- compute_nuclear_erfc_lattice,
// compute_nuclear_lattice_ewald, compute_vsap_lattice and the erfc gradient
// partner -- enumerates the plain |g| ball on purpose: its erfc half and its
// reciprocal / grid half must cover the same term set, and only the erfc half
// has a pair-resolved enumeration today. Under pair_complete_1e the overlap,
// kinetic and direct-nuclear sums ride the longer pair-complete list, so an
// Hcore assembled from them would keep T(g) and S(g) on the tail cells while
// V_ne(g) is absent there: a different operator, not a tighter truncation of
// the same one. Moving both halves together is a gauge decision owned by the
// route that builds the reciprocal half; until it lands, refuse the switch on
// this family instead of assembling that Hcore.
inline void require_plain_ball_for_ewald_nuclear(const LatticeSumOptions& opts,
                                                 const char* who) {
    if (!opts.pair_complete_1e) return;
    throw std::invalid_argument(
        std::string(who) +
        ": LatticeSumOptions.pair_complete_1e is not supported on the "
        "Ewald-split nuclear family yet (the erfc half and the reciprocal / "
        "grid half must move to the pair-complete cell list together; #429 "
        "stage 2). Leave pair_complete_1e off on this route, or use the "
        "direct nuclear sum (compute_nuclear_lattice).");
}

}  // namespace vibeqc
