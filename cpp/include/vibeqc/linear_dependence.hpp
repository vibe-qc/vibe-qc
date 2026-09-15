// Canonical orthogonalization of a basis.
//
// Given an overlap matrix S (n x n, SPD or semi-definite), produce a
// rectangular transformation X (n x m, m <= n) such that
//
//     X^T S X = I_m
//
// by diagonalizing S = V diag(lambda) V^T, dropping eigenvectors with
// lambda < threshold (the "near-null" subspace — the directions
// responsible for linear dependence), and setting
//
//     X = V_kept · diag(1 / sqrt(lambda_kept))
//
// Using X to orthogonalize the Fock problem — F' = X^T F X is m x m,
// diagonalize, then MO coefficients C = X C' come back as an (n x m)
// matrix — keeps the SCF numerically stable even when S itself is
// near-singular. For well-conditioned bases (all eigenvalues above
// threshold) m = n and the resulting SCF is equivalent to plain
// symmetric orthogonalization.
//
// Reference: Löwdin 1970; Szabo & Ostlund §3.4.5.

#pragma once

#include <Eigen/Dense>

namespace vibeqc {

struct CanonicalOrthogonalizer {
    // X : n_basis x n_kept rectangular transformation.
    Eigen::MatrixXd X;

    // n_kept = m, the dimension of the orthogonalized space.
    // n_dropped = n_basis - n_kept = number of near-null directions
    // projected out.
    int n_kept = 0;
    int n_dropped = 0;

    // Smallest eigenvalue of S retained (>= threshold by construction).
    double min_kept_eigenvalue = 0.0;

    // Largest eigenvalue of S that was dropped (< threshold). NaN if
    // no eigenvalue was dropped.
    double max_dropped_eigenvalue = 0.0;
};

// Build the canonical orthogonalizer for a real, symmetric, positive
// semi-definite S. ``threshold`` controls which eigenvalues count as
// near-null and get projected out; typical values:
//
//   * 1e-7  -- conservative, lets through bases with marginal
//              dependence (aug-cc-pVXZ on some geometries). Default.
//   * 1e-8  -- strictly matches the pre-canonical throw threshold.
//   * 1e-5 / 1e-6 -- aggressive; used when the SCF otherwise oscillates
//              on strongly dependent bases.
//
// Throws std::runtime_error if S is not square or not symmetric to
// within 1e-10, or if the diagonalizer fails.
CanonicalOrthogonalizer canonical_orthogonalizer(
    const Eigen::MatrixXd& S, double threshold);

}  // namespace vibeqc
