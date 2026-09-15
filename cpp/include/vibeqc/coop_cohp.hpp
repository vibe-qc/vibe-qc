#pragma once
/// COOP/COHP bonding-analysis kernels.
///
/// These replace the Python-side triple loops over pairs × k-points × bands
/// that currently dominate the COOP/COHP compute time. Every function
/// accepts numpy arrays (via pybind11) and returns per-pair results at once,
/// eliminating O(n_pairs · n_k · n_bands · n_bf²) CPython crossings.
///
/// References
/// ----------
/// * Hughbanks & Hoffmann, JACS 105, 3528 (1983) — original COOP
/// * Dronskowski & Blöchl, J. Phys. Chem. 97, 8617 (1993) — COHP

#include <Eigen/Core>
#include <complex>
#include <utility>
#include <vector>

namespace vibeqc {

using ComplexMatrix = Eigen::MatrixXcd;

// ---------------------------------------------------------------------------
// Per-k COOP weight
// ---------------------------------------------------------------------------

/// Compute COOP weights for all atom-pairs at a single k-point.
///
///   w_{AB,n} = Re[ Σ_{μ∈A, ν∈B}  C*_{μn} · S_{μν} · C_{νn} ]
///
/// @param C_k      Eigenvector matrix, (n_bf, n_bands) complex.
/// @param S_k      Overlap matrix at k, (n_bf, n_bf) complex.
/// @param pairs_ao Vector of (ao_indices_A, ao_indices_B) pairs.
/// @returns        Real (n_pairs, n_bands) matrix — COOP weight per
///                 pair per band at this k-point.
Eigen::MatrixXd coop_weights_k(
    const ComplexMatrix& C_k,
    const ComplexMatrix& S_k,
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>&
        pairs_ao);

// ---------------------------------------------------------------------------
// Per-k COHP weight
// ---------------------------------------------------------------------------

/// Compute COHP weights for all atom-pairs at a single k-point.
///
///   w_{AB,n} = Re[ Σ_{μ∈A, ν∈B}  C*_{μn} · H_{μν} · C_{νn} ]
///
/// @param C_k      Eigenvector matrix, (n_bf, n_bands) complex.
/// @param H_k      Hamiltonian matrix at k, (n_bf, n_bf) complex.
/// @param pairs_ao Vector of (ao_indices_A, ao_indices_B) pairs.
/// @returns        Real (n_pairs, n_bands) matrix — COHP weight per
///                 pair per band at this k-point.
Eigen::MatrixXd cohp_weights_k(
    const ComplexMatrix& C_k,
    const ComplexMatrix& H_k,
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>&
        pairs_ao);

// ---------------------------------------------------------------------------
// Gaussian broadening accumulator
// ---------------------------------------------------------------------------

/// Gaussian-broaden per-pair orbital weights onto an energy grid.
///
/// For each pair p, k-point k, band n:
///   contrib(p, E) = k_weight[k] · weight(p, k, n) · norm · exp(-(E-ε)²/2σ²)
///
/// @param energies_per_k  Eigenvalues at each k, shape (n_k, n_bands).
/// @param weights_per_k   Per-pair COOP/COHP weights, shape
///                        (n_pairs, n_k, n_bands).  Stored row-major so
///                        pair is the slowest index.
/// @param k_weights       k-point weights, length n_k.
/// @param energy_grid     Target energy grid, length n_e.
/// @param sigma           Gaussian width (same units as energies).
/// @returns               (n_pairs, n_e) matrix of broadened projections.
Eigen::MatrixXd gaussian_broaden_projected(
    const Eigen::MatrixXd& energies_per_k,
    const Eigen::MatrixXd& weights_per_k,
    const Eigen::VectorXd& k_weights,
    const Eigen::VectorXd& energy_grid,
    double sigma);

}  // namespace vibeqc
