#pragma once

// Deterministic occupation of an exactly degenerate SCF frontier (#210).
//
// Aufbau says "fill the lowest nocc orbitals". When eps(nocc-1) == eps(nocc)
// that instruction does not name a density: the occupied set is a point of the
// Grassmannian Gr(m, k) of the k-fold degenerate block, a continuum, and
// `C.leftCols(nocc)` resolves it with whatever basis LAPACK happened to return
// for the degenerate subspace. That basis is not a physical quantity -- it
// moves with the orientation of the molecule and with the LAPACK build -- so
// the converged energy moved with it: 90-degree twisted ethylene / STO-3G
// returned either -76.8553699161 or -76.7972798964, 58.090 mHa apart, both
// converged, depending on how the molecule was rotated in space.
//
// The tie is broken here by choosing the most DELOCALIZED occupied subspace,
// i.e. the one minimising the Loewdin atomic inverse participation ratio
//
//     IPR(U) = sum_A ( sum_{mu in A} [W U U^T W^T]_{mu mu} )^2 ,   W = S^{1/2} C_block
//
// over orthonormal k x m matrices U. Three properties make that the right
// tie-break rather than merely a reproducible one:
//
//   * IPR depends on the SUBSPACE, not on the basis of it (U -> U O leaves it
//     invariant), so it is a genuine function on Gr(m, k);
//   * it is invariant under a rigid rotation of the molecule and under atom
//     reordering, which is exactly the invariance that was broken;
//   * it selects the symmetry-preserving member of the degenerate manifold.
//     For twisted ethylene the two candidates are a delocalized solution with
//     Loewdin populations (0.444, 0.444) on the two carbons and a
//     charge-localized one at (0.000, 0.887); the 58 mHa penalty of the latter
//     is a spurious C+/C- separation. Breaking spatial symmetry is the
//     stability check's job, deliberately, not the eigensolver's by accident.
//
// Canonicalising the block with a secondary operator (Hcore, the previous
// density) cannot work: the degenerate plane carries an irreducible
// representation of the molecular point group, so by Schur's lemma any
// operator sharing that symmetry is proportional to the identity on it.
// Measured splittings are 1e-16 to 1e-14 -- pure noise -- and the 58 mHa
// spread survives. Minimising the energy freely over the block is not an
// occupation rule either; it is a stability search, and it returns the
// broken-symmetry biradical.
//
// The whole thing is gated on the frontier gap and is a no-op above it. The
// gap distribution is sharply bimodal: 1e-16..1e-14 Ha when the frontier is
// genuinely degenerate, 0.2..0.8 Ha otherwise, so any tolerance in
// 1e-10..1e-4 selects identically. Ordinary molecules never enter the branch.

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace vibeqc {

// Absolute frontier-gap tolerance below which the occupied set is ambiguous.
inline constexpr double kDegenerateFrontierTol = 1e-8;

// Symmetric orthogonaliser S^{1/2}, for Loewdin populations.
Eigen::MatrixXd symmetric_sqrt(const Eigen::MatrixXd& S);

// Occupied coefficient block for `nocc` electrons in `C` with eigenvalues
// `eps`, resolving an exactly degenerate frontier deterministically.
//
// `s_sqrt` is S^{1/2} and `ao_atom` maps each basis function to its atom. When
// the frontier gap exceeds `tol`, or the inputs do not support the tie-break,
// this returns `C.leftCols(nocc)` unchanged -- the historical behaviour.
Eigen::MatrixXd occupied_block_deterministic(
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& eps,
    int nocc,
    const Eigen::MatrixXd& s_sqrt,
    const std::vector<int>& ao_atom,
    double tol = kDegenerateFrontierTol);

}  // namespace vibeqc
