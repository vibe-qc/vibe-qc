// Phase D2e — TRAH (Trust-Region Augmented Hessian) SCF step.
//
// This implementation targets the orbital trust-region model
//   minimise m(kappa) = g^T kappa + 1/2 kappa^T A kappa,
//   subject to ||kappa|| <= Delta,
// using shifted preconditioned CG and a More-Sorensen secular search.
// The reference below instead solves a scaled augmented-Hessian eigenproblem;
// this implementation does not reproduce that inner eigensolver or all of its
// guarantees. UHF/UKS retain the occupied/virtual Fock blocks in A; restricted
// TRAH still uses the diagonal-gap approximation for that term. Both include
// JKBuilder (and, when supplied, XC-kernel) response products.
//
// An unconstrained Newton step inside the radius is accepted directly.
// Otherwise the shift lambda is searched so that
//   (A + lambda I) kappa = -g, ||kappa|| = Delta.
// Detected negative curvature increases the shift. PCG can miss curvature
// orthogonal to the gradient: the hard case can miss the model optimum and
// misreport on_boundary (#424). A hard-case minimizer need not be unique.
//
// The caller adapts the radius using the ratio of actual to model-predicted
// decrease. That outer-controller validation is separate from the Hessian
// action and inner-solve regressions.
//
// Reference:
//   * B. Helmich-Paris, "A trust-region augmented Hessian implementation
//     for restricted and unrestricted Hartree-Fock and Kohn-Sham
//     methods", J. Chem. Phys. 154, 164104 (2021), doi:10.1063/5.0040798.
//
// Activation policy mirrors Newton / SOSCF: the SCF driver swaps
// "diagonalize F" for ``trah_step`` once
//   ‖F D S − S D F‖_F < trah_threshold.
//
// Mutual exclusion: when more than one of newton_threshold /
// soscf_threshold / trah_threshold is set, the priority is
//   quadratic_fallback > Newton > TRAH > SOSCF.
// (Newton wins over TRAH because Newton's fixed trust-region cap is
// more predictable; TRAH wins over SOSCF because the full Hessian is
// more accurate than the diagonal-dominant one.)

#pragma once

#include <Eigen/Dense>

namespace vibeqc {

class JKBuilder;
class XCKernelBuilder;
class UHFXCKernelBuilder;

struct TRAHOptions {
    // CG cap on the inner Newton matvec — same default as NewtonOptions.
    int cg_max_iter = 50;
    // CG residual tolerance.
    double cg_tol = 1e-4;

    // Trust-region schedule (Powell-style).
    double initial_trust_radius = 0.3;   // same starting point as Newton
    double max_trust_radius = 0.7;
    double min_trust_radius = 1e-4;
    double rho_shrink = 0.25;   // if ρ < rho_shrink ⇒ halve trust radius
    double rho_expand = 0.75;   // if ρ > rho_expand AND ‖κ‖ near radius ⇒ double
    double trust_shrink_factor = 0.5;
    double trust_expand_factor = 2.0;

    // More-Sorensen secular root-find for the boundary level shift λ.
    // ``ms_max_iter`` caps the secular iterations; ``ms_tol`` is the
    // relative tolerance on ‖κ(λ)‖ = Δ (the step lands within
    // ms_tol·Δ of the trust-region boundary). The secular function is
    // near-linear so the default 12 / 1e-3 converges in 2-4 iters on
    // typical SCF Hessians.
    int ms_max_iter = 12;
    double ms_tol = 1e-3;
};

struct TRAHStepResult {
    Eigen::MatrixXd C;     // (n_bf, n_kept) rotated MO coefficients
    Eigen::VectorXd eps;   // (n_kept,)     refreshed MO energies
    int cg_iter = 0;
    bool cg_converged = false;
    double kappa_norm = 0.0;          // ‖κ‖_F of the accepted step
    double trust_radius_used = 0.0;   // radius applied this step
    double level_shift = 0.0;         // λ from the More-Sorensen root-find
                                       // (0 ⇒ interior unconstrained step;
                                       // > 0 ⇒ boundary trust-region step)
    bool on_boundary = false;         // true ⇒ the step landed on ‖κ‖ = Δ
    double predicted_decrease = 0.0;  // model-predicted ΔE
                                       // = −gᵀκ − ½ κᵀAκ
                                       // (used by the driver to compute ρ
                                       // on the next iteration's actual ΔE)
};

// One Trust-Region Augmented Hessian step on the closed-shell RHF
// orbital-rotation manifold. Drop-in replacement for ``diagonalize(F)``
// when the SCF driver enters its second-order phase.
//
// ``trust_radius`` is in/out — passed by reference so the driver can
// keep a single per-SCF-run trust-radius state and let TRAH adapt it
// step-by-step.
//
// The driver is responsible for the Powell ρ test on the NEXT
// iteration: ρ = (E_prev − E_this) / step.predicted_decrease. Based on
// ρ the driver updates ``trust_radius`` for the following call (shrink
// when ρ small or negative, expand when ρ near 1 and the previous
// step was near the radius cap).
//
// Parameters
// ----------
// F           : (n_bf, n_bf) AO-basis Fock matrix (symmetric).
// C_prev      : (n_bf, n_kept) current MO coefficients.
// eps_prev    : (n_kept,) current MO energies.
// n_occ       : Number of doubly-occupied MOs.
// jk          : JKBuilder used to compute G[D^κ] = J[D^κ] − ½ K[D^κ]
//               for each CG matvec.
// opts        : TRAHOptions (CG knobs + trust-region schedule).
// trust_radius: current radius (in/out). Modified by the caller after
//               the ρ test on the next iteration.
//
// **RKS extension (Phase D2c-KS).** Same plumbing as ``newton_step`` —
// ``xc_kernel`` defaults to nullptr (HF behaviour bit-for-bit); RKS
// callers pass a non-null kernel pinned at the current density and the
// matvec adds ``xc_kernel->apply(D^κ)`` to G[D^κ]. ``alpha_hf`` scales
// the HF exchange piece for hybrids.
TRAHStepResult trah_step(const Eigen::MatrixXd& F,
                          const Eigen::MatrixXd& C_prev,
                          const Eigen::VectorXd& eps_prev,
                          int n_occ,
                          const JKBuilder& jk,
                          const TRAHOptions& opts,
                          double trust_radius,
                          const XCKernelBuilder* xc_kernel = nullptr,
                          double alpha_hf = 1.0);

// Open-shell extension: one TRAH step on the coupled UHF orbital-
// rotation manifold. Mirrors uhf_newton_step internally (one coupled
// CG iterates on the stacked (κ_α, κ_β); per-spin K, shared J on
// D^κ_α + D^κ_β), with the same predicted_decrease + adaptive trust
// radius the closed-shell variant uses for Powell's ρ update.
struct UHFTRAHStepResult {
    Eigen::MatrixXd C_alpha;
    Eigen::MatrixXd C_beta;
    Eigen::VectorXd eps_alpha;
    Eigen::VectorXd eps_beta;
    int cg_iter = 0;
    bool cg_converged = false;
    double kappa_norm = 0.0;          // ‖(κ_α, κ_β)‖_F of the accepted step
    double trust_radius_used = 0.0;
    double level_shift = 0.0;         // λ from the More-Sorensen root-find
    bool on_boundary = false;         // true ⇒ step landed on ‖κ‖ = Δ
    double predicted_decrease = 0.0;  // model ΔE = −gᵀκ − ½ κᵀAκ
};

// **UKS extension (Phase D2c-KS-UHF).** Same as ``uhf_newton_step``.
UHFTRAHStepResult uhf_trah_step(const Eigen::MatrixXd& F_alpha,
                                 const Eigen::MatrixXd& F_beta,
                                 const Eigen::MatrixXd& C_alpha_prev,
                                 const Eigen::MatrixXd& C_beta_prev,
                                 const Eigen::VectorXd& eps_alpha_prev,
                                 const Eigen::VectorXd& eps_beta_prev,
                                 int n_alpha,
                                 int n_beta,
                                 const JKBuilder& jk,
                                 const TRAHOptions& opts,
                                 double trust_radius,
                                 const UHFXCKernelBuilder* xc_kernel = nullptr,
                                 double alpha_hf = 1.0);

// Helper for the driver: update the trust radius from the actual /
// predicted ΔE ratio (Powell's ρ). The schedule:
//   * ρ < rho_shrink:                 trust_radius *= trust_shrink_factor
//   * ρ > rho_expand AND κ near cap:  trust_radius *= trust_expand_factor
//   * else:                            unchanged
// Always clamped to [min_trust_radius, max_trust_radius].
//
// Returns the updated radius. Pass the latest ``kappa_norm`` from the
// previous TRAH step + the current ``trust_radius`` + the predicted
// + actual energy decreases.
double update_trust_radius(double trust_radius,
                            double actual_decrease,
                            double predicted_decrease,
                            double kappa_norm,
                            const TRAHOptions& opts);

}  // namespace vibeqc
