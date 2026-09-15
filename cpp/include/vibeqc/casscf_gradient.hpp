#pragma once
/// C++ CASSCF orbital gradient + generalized Fock kernel.
///
/// Replaces the Python ``_orbital_gradient_and_fock`` + ``_generalized_fock``
/// calls in :mod:`vibeqc.solvers._casscf`, which currently dominate
/// per-macro-iteration wall time through 4D einsum intermediates and
/// Python-level core/active loops.
///
/// The kernel accepts the integrals in chemist notation g[p,q,r,s] = (pq|rs),
/// matching vibe-qc's internal convention.

#include <Eigen/Core>
#include <utility>
#include <vector>

namespace vibeqc {

/// Result of one CASSCF orbital gradient evaluation.
struct CasscfGradientResult {
    /// Orbital gradient g[pq] = 2(F_qp - F_pq) for the rotation pairs.
    Eigen::VectorXd gradient;
    /// Generalized Fock matrix F (norb × norb).
    Eigen::MatrixXd F;
    /// Symmetric-averaged Fock Favg (norb × norb).
    Eigen::MatrixXd Favg;
    /// Diagonal Super-CI Hessian H_diag[pq] = Favg_pp - Favg_qq.
    Eigen::VectorXd H_diag;
};

/// Compute the CASSCF orbital gradient, generalized Fock, and diagonal Hessian.
///
/// @param h1          One-electron Hamiltonian (norb, norb).
/// @param eri_chem    Two-electron integrals in chemist notation (pq|rs),
///                    flattened row-major: eri_flat[p*norb³ + q*norb² + r*norb + s].
/// @param dm1         Active-space 1-RDM (n_act, n_act), unpacked in orbital basis
///                    (i.e. dm1[t,u] = active-block of the spin-summed 1-RDM).
/// @param dm2         Active-space 2-RDM (n_act, n_act, n_act, n_act) packed as
///                    dm2_flat[t*n_act³ + u*n_act² + v*n_act + w].
/// @param n_core      Number of doubly-occupied (inactive) orbitals.
/// @param n_act       Number of active orbitals.
/// @param norb        Total number of spatial orbitals (n_core + n_act + n_vir).
/// @param pairs       List of (p, q) rotation-pair indices (0-based).
/// @returns           Gradient, F, Favg, and diagonal Hessian.
CasscfGradientResult casscf_orbital_gradient(
    const Eigen::MatrixXd& h1,
    const Eigen::VectorXd& eri_chem_flat,
    const Eigen::MatrixXd& dm1,
    const Eigen::VectorXd& dm2_flat,
    int n_core,
    int n_act,
    int norb,
    const std::vector<std::pair<int, int>>& pairs);

}  // namespace vibeqc
