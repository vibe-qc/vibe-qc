// Phase D2d — SOSCF (approximate second-order SCF) via augmented Hessian
// with a diagonal-dominant orbital Hessian + adaptive λ-shift.
//
// Distinct from D2c (Newton, vibeqc/newton.hpp) which uses the FULL orbital
// Hessian via preconditioned CG with a JKBuilder for matvecs, and from
// D2e (TRAH, future vibeqc/trah.hpp) which uses the full AH with adaptive
// trust radius. SOSCF (D2d) drops the off-diagonal two-electron coupling
// entirely — the orbital Hessian is approximated as diagonal
//   A_{ai,bj} ≈ (ε_a − ε_i) δ_{ab} δ_{ij}
// and the optimal step is obtained from the augmented-Hessian eigenvalue
// problem
//   [diag(d) g] [κ]    [κ]
//   [   g^T  0] [α]  = λ [α]
// taking the lowest eigenvalue. The eigenvector's last component
// normalises to 1; the leading n_vir·n_occ components are κ.
//
// Reference:
//   * F. Neese, "Approximate second-order SCF method for large
//     configuration interaction expansions of the diradical type",
//     Chem. Phys. Lett. 325, 93 (2000). ORCA's standard SOSCF.
//
// Properties:
//   * No Fock build inside the step — only AH eigsolve on a small
//     ((n_vir·n_occ + 1) × (n_vir·n_occ + 1)) matrix. Cheap per step
//     (microseconds for n_pairs < 100; ~1 ms for n_pairs ~ 1000).
//   * λ is automatic (lowest eigenvalue of AH = optimal shift).
//   * Trust-region cap on ‖κ‖_F is the same simple step-size cap used
//     by quadratic_step / newton_step.
//   * Linearly convergent in the asymptotic regime (vs quadratic for
//     full Newton). Robust where DIIS oscillates and Newton is too
//     expensive (no Fock rebuild needed per step).
//
// Activation policy lives in the SCF driver — once
// ‖F D S − S D F‖_F drops below ``soscf_threshold`` (molecular default
// 1e-3), the driver swaps "diagonalize F" for the stateful L-BFGS
// ``SOSCF::step`` (see class SOSCF below).

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <deque>

namespace vibeqc {

struct SOSCFOptions {
    // Trust-region cap on ‖κ‖_F. Same default as Newton (vibeqc/newton.hpp).
    // Larger ⇒ more aggressive steps; smaller ⇒ more conservative. 0.3 is
    // a moderate default that keeps the matrix exponential well-conditioned.
    double trust_radius = 0.3;

    // L-BFGS history length. The stateful SOSCF accelerator (see class
    // SOSCF below) uses the diagonal orbital Hessian (ε_a − ε_i) only as
    // the initial inverse-Hessian preconditioner H0; the true curvature
    // (including the two-electron / XC-kernel coupling that the diagonal
    // model omits) is built up from the last ``lbfgs_history`` pairs of
    // (orbital-rotation step, gradient difference). This is what makes
    // SOSCF actually accelerate — a memoryless diagonal step (history 0)
    // has no curvature information and, for Kohn-Sham DFT, oscillates
    // because the diagonal Hessian ignores the large XC response.
    std::size_t lbfgs_history = 10;
};

struct SOSCFStepResult {
    Eigen::MatrixXd C;     // (n_bf, n_kept) rotated MO coefficients
    Eigen::VectorXd eps;   // (n_kept,)     refreshed MO energies
    double kappa_norm = 0.0;   // ‖κ‖_F before trust-region capping
    double lambda_shift = 0.0; // Lowest AH eigenvalue (the "optimal λ")
};

// One approximate-second-order SCF step on the closed-shell RHF orbital-
// rotation manifold via the augmented-Hessian formulation with diagonal
// orbital Hessian. Drop-in replacement for ``diagonalize(F)`` when the
// SCF driver enters its second-order phase; also a drop-in alternative
// to ``newton_step`` when the JKBuilder-based full-Hessian matvec is
// too expensive (large basis, no DF).
//
// Parameters
// ----------
// F           : (n_bf, n_bf) AO-basis Fock matrix (symmetric).
// C_prev      : (n_bf, n_kept) current MO coefficients in the
//               canonical-orth basis.
// eps_prev    : (n_kept,) current MO energies. Used as the diagonal
//               orbital Hessian (A_{ai} = ε_a − ε_i).
// n_occ       : Number of doubly-occupied MOs.
// opts        : SOSCFOptions (trust radius).
//
// Returns
// -------
// SOSCFStepResult{C_new, eps_new, kappa_norm, lambda_shift}.
//
// Notes
// -----
// * For ``n_occ == n_kept`` (no virtual subspace) the rotation is the
//   identity; returns C_prev with refreshed ε.
// * Throws if the AH eigenvector's last component is too small
//   (~1e-12) — that means the gradient is already at convergence and
//   the AH formulation degenerates. Caller should check ``grad_norm``
//   before calling soscf_step.
SOSCFStepResult soscf_step(const Eigen::MatrixXd& F,
                            const Eigen::MatrixXd& C_prev,
                            const Eigen::VectorXd& eps_prev,
                            int n_occ,
                            const SOSCFOptions& opts = {});

// Open-shell extension: one D2d SOSCF step on the UHF orbital-rotation
// manifold. The diagonal-dominant Hessian approximation drops the
// cross-spin two-electron coupling entirely, so the per-spin orbital
// rotations decouple — we solve two independent AH eigvalue problems,
// one per spin, with the SAME trust-region budget for each.
//
// Returns the rotated MOs + refreshed energies for both spins, plus
// the larger of the two ‖κ‖_F norms and the more-negative of the two
// AH eigenvalues (so a single scalar represents the worst-case shift).
struct UHFSOSCFStepResult {
    Eigen::MatrixXd C_alpha;
    Eigen::MatrixXd C_beta;
    Eigen::VectorXd eps_alpha;
    Eigen::VectorXd eps_beta;
    double kappa_norm = 0.0;        // max(‖κ_α‖_F, ‖κ_β‖_F) before clip
    double lambda_shift = 0.0;      // min(λ_α, λ_β) — most-negative
};

UHFSOSCFStepResult uhf_soscf_step(
    const Eigen::MatrixXd& F_alpha,
    const Eigen::MatrixXd& F_beta,
    const Eigen::MatrixXd& C_alpha_prev,
    const Eigen::MatrixXd& C_beta_prev,
    const Eigen::VectorXd& eps_alpha_prev,
    const Eigen::VectorXd& eps_beta_prev,
    int n_alpha,
    int n_beta,
    const SOSCFOptions& opts = {});

// Stateful closed-shell SOSCF accelerator (Neese 2000): quasi-Newton
// (L-BFGS) orbital optimisation preconditioned by the diagonal orbital
// Hessian. Unlike the stateless ``soscf_step`` above — a single diagonal
// augmented-Hessian step with no curvature memory — this class accumulates
// (step, gradient-difference) pairs across SCF iterations and applies the
// L-BFGS two-loop recursion, so it captures the two-electron / XC coupling
// the diagonal model omits. With an empty history the first step reduces to
// the diagonal preconditioned step ``κ = −g / (ε_a − ε_i)`` (the same
// direction as ``soscf_step`` sans the tiny AH λ-shift), so it is a strict
// generalisation. Construct one per SCF run; call ``step`` each iteration
// once the driver has entered its SOSCF phase.
class SOSCF {
public:
    explicit SOSCF(SOSCFOptions opts = {});

    SOSCFStepResult step(const Eigen::MatrixXd& F,
                         const Eigen::MatrixXd& C_prev,
                         const Eigen::VectorXd& eps_prev,
                         int n_occ);

    // Drop the L-BFGS history (e.g. when the SCF leaves and re-enters the
    // SOSCF phase, so stale curvature from a different frame is not reused).
    void clear();

private:
    SOSCFOptions opts_;
    std::deque<Eigen::VectorXd> s_hist_;  // orbital-rotation steps κ
    std::deque<Eigen::VectorXd> y_hist_;  // gradient differences g_{k} − g_{k−1}
    Eigen::VectorXd g_prev_;
    Eigen::VectorXd kappa_prev_;
    bool have_prev_ = false;
};

// Stateful open-shell SOSCF accelerator. Stacks the α and β orbital-rotation
// gradients into a single vector and runs one L-BFGS history over both, so
// the accumulated curvature couples the spins (the diagonal Hessian itself
// stays per-spin block-diagonal, matching ``uhf_soscf_step``). The rotation
// is applied per spin.
class UHFSOSCF {
public:
    explicit UHFSOSCF(SOSCFOptions opts = {});

    UHFSOSCFStepResult step(const Eigen::MatrixXd& F_alpha,
                            const Eigen::MatrixXd& F_beta,
                            const Eigen::MatrixXd& C_alpha_prev,
                            const Eigen::MatrixXd& C_beta_prev,
                            const Eigen::VectorXd& eps_alpha_prev,
                            const Eigen::VectorXd& eps_beta_prev,
                            int n_alpha,
                            int n_beta);

    void clear();

private:
    SOSCFOptions opts_;
    std::deque<Eigen::VectorXd> s_hist_;
    std::deque<Eigen::VectorXd> y_hist_;
    Eigen::VectorXd g_prev_;
    Eigen::VectorXd kappa_prev_;
    bool have_prev_ = false;
};

}  // namespace vibeqc
