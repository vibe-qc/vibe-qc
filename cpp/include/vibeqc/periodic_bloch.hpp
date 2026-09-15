#pragma once
/// Fused Bloch-sum kernels for multi-k periodic SCF.
///
/// These replace the Python-side loops over k-points and lattice cells
/// that currently dominate per-SCF-iteration wall time for N_k > 1.
/// Every function accepts all data in one call and returns all results
/// at once, eliminating O(N_k · n_cells) CPython crossings.

#include <Eigen/Core>
#include <complex>
#include <vector>

namespace vibeqc {

/// Bloch sum over real-space blocks.
///
///   F_k = Σ_g  exp(+i k · R_g) · block[g]
///
/// @param blocks    Real-space cell blocks, each (n_bf, n_bf).
/// @param cell_r    Cell origins R_g in bohr, same order as blocks.
/// @param k_vectors Cartesian k-vectors in bohr⁻¹, shape (N_k, 3).
/// @returns         Per-k Fock matrices F(k), complex (n_bf, n_bf).
std::vector<Eigen::MatrixXcd> bloch_sum_multi_k(
    const std::vector<Eigen::MatrixXd>& blocks,
    const std::vector<Eigen::Vector3d>& cell_r,
    const std::vector<Eigen::Vector3d>& k_vectors);

/// Full per-k Fock assembly: Bloch sum + optional Hcore.
///
///   F(k) = Σ_g exp(+i k·R_g) · F_2e(g)  +  H_core(k)
///
/// @param f2e_blocks  Two-electron Fock blocks in real space.
/// @param cell_r      Cell origins.
/// @param k_vectors   k-point list.
/// @param hcore_k     Per-k core Hamiltonian (same length as k_vectors).
///                    If empty, Hcore is not added.
/// @returns           Per-k total Fock matrices.
std::vector<Eigen::MatrixXcd> assemble_fock_multi_k(
    const std::vector<Eigen::MatrixXd>& f2e_blocks,
    const std::vector<Eigen::Vector3d>& cell_r,
    const std::vector<Eigen::Vector3d>& k_vectors,
    const std::vector<Eigen::MatrixXcd>& hcore_k);

/// Inverse Bloch transform (BvK torus) over all k-points.
///
///   D(k) = Σ_{g∈torus} exp(+i k·R_g) · D(g)
///
/// then Hermitised: D(k) = ½[D(k) + D(k)†].
///
/// @param blocks    Real-space density blocks (one per torus cell).
/// @param cell_r    Torus cell origins.
/// @param k_vectors k-point list.
/// @returns         Per-k complex density matrices.
std::vector<Eigen::MatrixXcd> inverse_bloch_multi_k(
    const std::vector<Eigen::MatrixXd>& blocks,
    const std::vector<Eigen::Vector3d>& cell_r,
    const std::vector<Eigen::Vector3d>& k_vectors);

}  // namespace vibeqc
