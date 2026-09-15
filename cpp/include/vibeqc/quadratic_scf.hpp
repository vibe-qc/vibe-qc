// Phase C1c-3 — second-order ("quadratic") SCF fallback, C++ side.
//
// Mirrors the Python ``vibeqc.quadratic_scf`` module that ships the
// fallback for the eight periodic Ewald drivers (Phase C1c-1 / C1c-2);
// this header is the Eigen-based equivalent for the molecular C++
// drivers (run_rhf, run_uhf, run_rks, run_uks).
//
// The fallback replaces the standard "diagonalize F" step of the SCF
// with a Newton step in MO space:
//
//     κ_{ai}  =  -F_{ai}^{MO} / (ε_a - ε_i + λ)
//     C_new   =  C_prev · exp(κ)            (skew-symmetric κ)
//     D_new   =  2 · C_new[:, :n_occ] · C_new[:, :n_occ]^T   (RHF / RKS)
//
// where the denominator is the diagonal of the orbital-rotation Hessian
// and ``λ > 0`` regularises near-degenerate occ-vir pairs. The step is
// trust-region capped at ``max_step`` to keep the matrix exponential
// well-conditioned (Taylor + scaling-and-squaring inside expm_skew).
//
// When this helps. Insulators with HOMO-LUMO gap ≲ 0.1 eV; open-shell
// systems with near-degenerate ground / excited states; high-symmetry
// molecules where many MO pairs are close. Activated explicitly via
// ``opts.quadratic_fallback_iter > 0`` (default 0 = disabled — keeps
// pre-C1c behavior bit-identical).
//
// References. The diagonal-Hessian / orbital-energy-difference
// preconditioner is the standard "augmented Hessian" approach used by
// PySCF (``pyscf.scf.newton``), ORCA (``! NRSCF``), Q-Chem (``GDM``).
// Vanilla Newton is unstable far from convergence; with the energy-
// difference denominator it becomes a damped quasi-Newton method that's
// robust on small-gap insulators where DIIS oscillates.

#pragma once

#include <Eigen/Dense>

namespace vibeqc {

// Matrix exponential of a real skew-symmetric matrix.
//
// Implementation: scaling-and-squaring + truncated Taylor series. For
// ``‖K‖ ≤ max_norm_per_step``, Taylor at order 12 is accurate to ~10⁻¹⁵
// at ‖K‖ = 0.5 (the order-12 term is bounded by 0.5^12 / 12! ≈ 6e-13);
// outside, K is repeatedly halved until the tail is small enough, then
// the result is squared back.
//
// The function does NOT verify that K is genuinely skew-symmetric — it
// just exponentiates whatever you give it. The name reflects the use
// case (we feed it the κ matrix from the orbital-rotation Newton step,
// which is skew-symmetric by construction); callers can skip the
// redundant symmetry check on hot paths.
//
// Avoids a hard scipy dependency on the Python side; on the C++ side
// it's just one Eigen matrix multiply per Taylor term + s squarings.
Eigen::MatrixXd expm_skew(const Eigen::MatrixXd& K,
                          double max_norm_per_step = 0.5);

// Result of one C1c Newton step. New MO coefficients + MO energies;
// SCF driver updates its density and result.* fields from these.
struct QuadraticStepResult {
    Eigen::MatrixXd C;     // (n_bf, n_kept) rotated MO coefficients
    Eigen::VectorXd eps;   // (n_kept,)     new MO energies
};

// One C1c Newton step in MO space. Drop-in replacement for the
// ``diagonalize(F)`` step of the SCF driver when the fallback is
// active.
//
// Parameters
// ----------
// F           : (n_bf, n_bf) AO-basis Fock matrix. Symmetric (caller's
//               responsibility — typically F = ½(F + F^T) before this).
// C_prev      : (n_bf, n_kept) current MO coefficients in the
//               canonical-orth or full-rank basis. Rotation stays
//               inside the kept-direction subspace.
// eps_prev    : (n_kept,) current MO energies. Used as the diagonal
//               Hessian preconditioner.
// n_occ       : Number of doubly-occupied MOs (RHF / RKS convention).
//               For UHF / UKS, call once per spin with the per-spin
//               n_occ_σ.
// shift       : Damping λ in (ε_a − ε_i + λ). Default 0.1 Hartree.
// max_step    : Trust-region cap on ‖κ‖_F. Default 0.1.
//
// Returns
// -------
// QuadraticStepResult{C_new, eps_new}.
//
// Notes
// -----
// Robust to the canonical-orth case where n_kept < n_bf: the rotation
// lives entirely in the kept-direction subspace, so the (n_bf, n_kept)
// shape of C_new is preserved.
//
// For systems where n_occ == n_kept (no virtual subspace), the rotation
// is the identity — function returns C unchanged with refreshed ε.
QuadraticStepResult quadratic_step(const Eigen::MatrixXd& F,
                                   const Eigen::MatrixXd& C_prev,
                                   const Eigen::VectorXd& eps_prev,
                                   int n_occ,
                                   double shift = 0.1,
                                   double max_step = 0.1);

}  // namespace vibeqc
