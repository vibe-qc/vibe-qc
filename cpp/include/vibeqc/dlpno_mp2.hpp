#pragma once
/// Fused DLPNO-MP2 LMP2 iteration kernel.
///
/// Replaces the Python triple loop (pairs × occupied × matrix multiplies)
/// that dominates per-iteration wall time in :mod:`vibeqc.dlpno.mp2`.
///
/// A single C++ call performs one full LMP2 iteration: residual build,
/// Fock coupling, damping, amplitude update, and energy evaluation.

#include <Eigen/Core>
#include <utility>
#include <vector>

namespace vibeqc {

/// Result of one LMP2 amplitude iteration.
struct DLPNOMP2IterResult {
    /// Updated T amplitudes, one (n_pno, n_pno) matrix per pair.
    std::vector<Eigen::MatrixXd> T_new;
    /// Maximum absolute residual across all pairs.
    double max_residual;
    /// Total correlation energy this iteration.
    double e_corr;
};

/// Perform one LMP2 amplitude iteration.
///
/// For each pair (i,j):
///   R = K + (eps_i + eps_j) ∘ T
///   for each k in occupied:
///     R -= F_ik · P(T_kj)  (P projects into pair (i,j)'s PNO basis)
///     R -= F_kj · P(T_ik)
///   T_new = T + damping · R / denom
///
/// @param pair_i      i indices for each pair (length N_pairs).
/// @param pair_j      j indices for each pair.
/// @param K_blocks    K integrals per pair, each (n_pno, n_pno).
/// @param eps_blocks  PNO energies per pair, each length n_pno.
/// @param T_blocks    Current T amplitudes, each (n_pno, n_pno).
/// @param V_blocks    PNO coefficient matrices, each (n_pao, n_pno).
/// @param F_oo        Occupied-occupied Fock matrix (n_act, n_act).
/// @param S_ao        AO overlap matrix (nbf, nbf).
/// @param damping     Damping factor (typically 0.5–0.8).
/// @returns           Updated amplitudes, max residual, correlation energy.
DLPNOMP2IterResult dlpno_mp2_iterate(
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    const std::vector<Eigen::MatrixXd>& K_blocks,
    const std::vector<Eigen::VectorXd>& eps_blocks,
    const std::vector<Eigen::MatrixXd>& T_blocks,
    const std::vector<Eigen::MatrixXd>& V_blocks,
    const Eigen::MatrixXd& F_oo,
    const Eigen::MatrixXd& S_ao,
    double damping);

}  // namespace vibeqc
