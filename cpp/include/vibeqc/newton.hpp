// Phase D2c — Newton SCF (full-Hessian Newton-step on the orbital-rotation
// manifold) for closed-shell RHF / RKS-style problems driven via JKBuilder.
//
// Naming note: the second-order-SCF family in vibe-qc is split per the
// roadmap (docs/roadmap.md § "SCF guess + convergence program"):
//   * D2c "Newton" — full orbital Hessian, exact step via CG. THIS FILE.
//     (Roadmap calls this "SOSCF strict"; we use "Newton" to keep the
//     surface honest about which Hessian is solved.)
//   * D2d "SOSCF"  — Neese 2000 approximate-second-order, diagonal-dominant
//     orbital-Hessian + AH λ-shift. Lives in vibeqc/soscf.hpp once landed
//     (separate from THIS file).
//   * D2e "TRAH"   — Helmich-Paris 2022, full AH + adaptive trust radius.
//     Lives in vibeqc/trah.hpp once landed.
// All three plug into the same orbital-rotation manifold + JKBuilder
// matvec abstractions; they differ in which Hessian approximation +
// step-control strategy they use.
//
// References:
//   * G. B. Bacskay, Chem. Phys. 61, 385 (1981) — original full-Hessian
//     orbital Newton method for HF SCF (the algorithm shipped here).
//   * H. J. A. Jensen, P. Jørgensen, J. Chem. Phys. 80, 1204 (1984) —
//     trust-region step.
//   * T. H. Fischer, J. Almlöf, J. Phys. Chem. 96, 9768 (1992) — the
//     ε_a − ε_i diagonal preconditioner + DIIS warm-up activation
//     threshold (≈ 1.0 default).
//   * F. Neese, Chem. Phys. Lett. 325, 93 (2000) — diagonal-dominant
//     approximation (ORCA's "SOSCF"; vibeqc/soscf.hpp future entry,
//     NOT what this file ships).
//   * B. Helmich-Paris, J. Chem. Phys. 156, 204104 (2022) — TRAH
//     (the full augmented-Hessian / λ-shift trust region; vibeqc/trah.hpp
//     future entry; we ship the simpler step-size cap here as the
//     baseline trust-region strategy).
//
// What we ship (this header).
//   * ``newton_step`` — one full Newton step on the closed-shell RHF
//     orbital-rotation manifold, using the *full* orbital Hessian
//     (4(ai|bj) − (ab|ij) − (aj|ib) + (ε_a − ε_i)δ_ab δ_ij) via a
//     preconditioned CG solver. The Hessian-vector product builds one
//     "perturbed-density Fock" (G[D^κ]) per CG iteration through the
//     supplied JKBuilder — same kernel used everywhere in the molecular
//     SCF stack.
//   * Trust-region: simple step-size cap on ‖κ‖_F (like quadratic_step;
//     the proper TRAH λ-shift is a follow-up).
//   * Activation policy lives in the SCF driver — once
//     ‖F D S − S D F‖_F drops below ``newton_threshold`` (default 1.0
//     per Fischer-Almlöf 1992), the driver swaps "diagonalize F" for
//     ``newton_step``.
//
// Distinction from ``quadratic_step`` (in ``quadratic_scf.hpp``).
//   * ``quadratic_step`` uses only the *diagonal* orbital Hessian
//     (κ_ai = -F_ai / (ε_a − ε_i + λ)) — ignores the two-electron
//     coupling. Cheap, robust, but only linearly convergent.
//   * ``newton_step`` uses the *full* orbital Hessian via CG —
//     quadratically convergent in ~3–5 iterations once activated.
//     Cost: one Fock build per CG iter (~5–20 per Newton step).
//
// UHF extension (``uhf_newton_step``).
//   * Per-spin orbital-rotation manifold: ``κ`` is a pair
//     (κ_α (n_vir_α × n_occ_α), κ_β (n_vir_β × n_occ_β)).
//   * Per-spin diagonal preconditioner: ``1/(ε_a^σ − ε_i^σ)``.
//   * Coupled Hessian-vector product through the shared J:
//        D^κ_σ      = build_perturbed_density(κ_σ, C_occ^σ, C_vir^σ)
//        F_α^pert   = J(D^κ_α + D^κ_β) − K(D^κ_α)
//        F_β^pert   = J(D^κ_α + D^κ_β) − K(D^κ_β)
//        [Aκ]^σ_ai  = (ε_a^σ − ε_i^σ) κ_ai^σ + F_σ^pert^MOσ_{ai}
//     The α↔β cross-spin coupling lives entirely in the J build on
//     the *total* perturbed density; K is spin-diagonal. No factor
//     of 2 on F^pert (unlike the closed-shell "2·G[D^κ]" — the UHF
//     normalisation is already "per-spin").
//   * RHS: -F_σ^MO occ-vir block per spin. Same JKBuilder used by
//     ``run_uhf_scf_with_jk`` so DF / COSX paths plug in
//     transparently.
//
// RKS / UKS extensions live in follow-up commits — adding the f_xc
// contribution needs the libxc second derivatives. The HF cases
// (closed-shell + open-shell) ship first.

#pragma once

#include <Eigen/Dense>
#include <cstddef>

namespace vibeqc {

class JKBuilder;
class XCKernelBuilder;
class UHFXCKernelBuilder;

struct NewtonOptions {
    // CG cap. Most Newton steps converge in 5-15 CG iters once the SCF
    // gradient is below the activation threshold. Bumping past 50 is
    // typically a sign of a near-zero-gap or non-PD Hessian — better
    // to back off the activation threshold than burn iterations.
    int cg_max_iter = 50;

    // Relative residual tolerance on the CG solver: ‖A κ + g‖ ≤
    // tol · ‖g‖. 1e-4 is a "good enough" Newton step (residual noise
    // doesn't dominate the SCF outer-loop energy).
    double cg_tol = 1e-4;

    // Trust-region cap on ‖κ‖_F. Larger ⇒ more aggressive Newton
    // steps; smaller ⇒ more conservative. 0.3 is a moderate default
    // — large enough to make full progress most steps, small enough
    // to keep the matrix exponential well-conditioned (expm_skew
    // scales-and-squares automatically beyond ‖K‖ ≈ 0.5).
    double trust_radius = 0.3;
};

struct NewtonStepResult {
    Eigen::MatrixXd C;     // (n_bf, n_kept) rotated MO coefficients
    Eigen::VectorXd eps;   // (n_kept,)     refreshed MO energies
    int cg_iter = 0;       // CG iterations used this step
    bool cg_converged = false;
    double kappa_norm = 0.0;  // ‖κ‖_F before trust-region capping
};

// One full Newton step on the closed-shell RHF orbital-rotation manifold.
// Drop-in replacement for ``diagonalize(F)`` when the SCF driver enters
// its second-order phase.
//
// Parameters
// ----------
// F           : (n_bf, n_bf) AO-basis Fock matrix (symmetric).
// C_prev      : (n_bf, n_kept) current MO coefficients (canonical-orth
//               compatible — rotation stays in the kept subspace).
// eps_prev    : (n_kept,) current MO energies. Used both as the gradient
//               source (occ-vir block of F^MO) and the diagonal CG
//               preconditioner (ε_a − ε_i)^{-1}.
// n_occ       : Number of doubly-occupied MOs.
// jk          : JKBuilder used to compute G[D^κ] = J[D^κ] − ½ K[D^κ]
//               for each CG matvec. Same builder used by the SCF driver,
//               so DF / COSX paths plug in transparently.
// opts        : NewtonOptions (see above).
//
// Returns
// -------
// NewtonStepResult with the rotated C, refreshed ε, and a small report
// (CG iters used, converged flag, pre-clip step norm). Drivers should
// log ``cg_iter`` to the SCF trace for diagnostics.
//
// Notes
// -----
// * No fallback if CG fails — the caller is responsible for backing off
//   to ``quadratic_step`` or DIIS extrapolation if ``cg_converged ==
//   false``. In practice the activation threshold is set so we're well
//   inside the trust radius when Newton kicks in; CG nearly always
//   converges in <30 iters.
// * For ``n_occ == n_kept`` (no virtual subspace) the rotation is the
//   identity — ``newton_step`` returns ``C_prev`` with refreshed ``eps``.
//
// **RKS extension (Phase D2c-KS).** ``xc_kernel`` defaults to nullptr —
// HF callers get the existing four-index two-electron-only Hessian
// matvec bit-for-bit. RKS callers pass a non-null
// ``XCKernelBuilder*`` (built via ``make_unpolarised_xc_kernel_builder``
// at the current SCF density) and the matvec adds
// ``xc_kernel->apply(D^κ)`` to the J − ½α_HF·K contribution before the
// 2·ov_block projection. ``alpha_hf`` scales the HF exchange piece
// for hybrid functionals (1.0 for pure HF; e.g. 0.20 for B3LYP);
// passed straight through to ``jk.build_g_rhf(D^κ, alpha_hf)``.
NewtonStepResult newton_step(const Eigen::MatrixXd& F,
                           const Eigen::MatrixXd& C_prev,
                           const Eigen::VectorXd& eps_prev,
                           int n_occ,
                           const JKBuilder& jk,
                           const NewtonOptions& opts = {},
                           const XCKernelBuilder* xc_kernel = nullptr,
                           double alpha_hf = 1.0);

// Open-shell extension: one full Newton step on the UHF orbital-
// rotation manifold (separate α / β rotations coupled through the
// shared J). Returns the rotated MO coefficients and refreshed MO
// energies for *both* spins.
//
// Parameters
// ----------
// F_alpha, F_beta     : (n_bf, n_bf) AO-basis Fock matrices, one
//                       per spin (symmetric).
// C_alpha_prev, C_beta_prev
//                     : (n_bf, n_kept) current α/β MO coefficients.
// eps_alpha_prev, eps_beta_prev
//                     : (n_kept,) current α/β MO energies.
// n_alpha, n_beta     : Occupations per spin (n_α + n_β = n_electrons,
//                       n_α − n_β = multiplicity − 1).
// jk                  : JKBuilder used to compute J / K for the
//                       perturbed-density Hessian-vector products.
// opts                : NewtonOptions; ``cg_max_iter`` / ``cg_tol`` /
//                       ``trust_radius`` shared across both spins
//                       (one Newton step on the combined α+β
//                       manifold).
//
// Returns
// -------
// UHFNewtonStepResult with per-spin rotated C and refreshed ε, plus a
// shared report (CG iters, converged flag, combined ‖κ‖_F pre-clip).
//
// Notes
// -----
// * For ``n_alpha == n_kept`` or ``n_beta == n_kept`` (one spin has
//   no virtual subspace) the corresponding spin rotation is the
//   identity but the other spin still gets a Newton step.
// * CG iterates on the stacked (κ_α, κ_β) trial vector. The matvec
//   builds J once on D^κ_α + D^κ_β and K once per spin — three
//   JKBuilder calls per CG iter.
struct UHFNewtonStepResult {
    Eigen::MatrixXd C_alpha;
    Eigen::MatrixXd C_beta;
    Eigen::VectorXd eps_alpha;
    Eigen::VectorXd eps_beta;
    int cg_iter = 0;
    bool cg_converged = false;
    double kappa_norm = 0.0;  // combined ‖(κ_α, κ_β)‖_F before clipping
};

// **UKS extension (Phase D2c-KS-UHF).** ``xc_kernel`` defaults to
// nullptr (HF behaviour bit-for-bit); UKS callers pass a non-null
// ``UHFXCKernelBuilder*`` pinned at the current per-spin densities
// and the per-spin matvec adds the W^XC_σ contribution. ``alpha_hf``
// scales the HF exchange piece for hybrids.
UHFNewtonStepResult uhf_newton_step(
    const Eigen::MatrixXd& F_alpha,
    const Eigen::MatrixXd& F_beta,
    const Eigen::MatrixXd& C_alpha_prev,
    const Eigen::MatrixXd& C_beta_prev,
    const Eigen::VectorXd& eps_alpha_prev,
    const Eigen::VectorXd& eps_beta_prev,
    int n_alpha,
    int n_beta,
    const JKBuilder& jk,
    const NewtonOptions& opts = {},
    const UHFXCKernelBuilder* xc_kernel = nullptr,
    double alpha_hf = 1.0);

// ---------------------------------------------------------------------------
// Internal SCF stability analysis (real UHF -> UHF block).
//
// A vanishing orbital gradient only certifies a stationary point of the
// SCF energy; the character of the extremum is decided by the lowest
// eigenvalue of the orbital-rotation Hessian: if it is negative,
// rotating the orbitals along the corresponding eigenvector lowers the
// energy further (stability criterion as stated in Lehtola, Molecules
// 25, 1218 (2020), doi:10.3390/molecules25051218, Section 10; original
// Hartree-Fock stability conditions: Seeger & Pople, J. Chem. Phys. 66,
// 3045 (1977), doi:10.1063/1.434318). The operator whose spectrum is
// probed here is exactly the coupled per-spin orbital Hessian already
// used by ``uhf_newton_step`` (see the UHF extension notes above); only
// REAL rotations within the UHF variational space are examined, i.e.
// the "internal" instability in Seeger-Pople's classification.
//
// The lowest eigenpair is extracted matrix-free with a Davidson
// iteration (Davidson, J. Comput. Phys. 17, 87 (1975)) preconditioned
// by the (eps_a - eps_i) diagonal; each iteration costs one
// Hessian-vector product = one J build + two K builds via the supplied
// JKBuilder.
// ---------------------------------------------------------------------------
struct UHFStabilityOptions {
    // Davidson matvec cap. Each matvec is one J + two K builds.
    int max_iter = 60;
    // Subspace cap before a thick restart onto the lowest Ritz block.
    int max_subspace = 24;
    // Residual convergence: ||H x - theta x|| <= residual_tol. The UHF
    // driver tightens this below its instability classification threshold;
    // direct callers retain the cheaper diagnostic default.
    double residual_tol = 1e-3;
};

struct UHFStabilityResult {
    bool converged = false;          // Davidson residual converged
    double lowest_eigenvalue = 0.0;  // valid when ``converged``
    int n_matvec = 0;
    // Unit-norm lowest-eigenvector direction (occ-vir blocks per spin,
    // [a, i] indexing as in uhf_newton_step). Empty when a spin has no
    // occupied x virtual rotation space.
    Eigen::MatrixXd kappa_alpha;
    Eigen::MatrixXd kappa_beta;
};

// Lowest eigenpair of the internal UHF orbital-rotation Hessian at the
// (converged) SCF point described by per-spin MO coefficients and
// energies. ``xc_kernel`` / ``alpha_hf`` extend the operator to UKS
// exactly as in ``uhf_newton_step`` (nullptr / 1.0 = pure HF).
UHFStabilityResult uhf_internal_stability_lowest(
    const Eigen::MatrixXd& C_alpha,
    const Eigen::MatrixXd& C_beta,
    const Eigen::VectorXd& eps_alpha,
    const Eigen::VectorXd& eps_beta,
    int n_alpha,
    int n_beta,
    const JKBuilder& jk,
    const UHFStabilityOptions& opts = {},
    const UHFXCKernelBuilder* xc_kernel = nullptr,
    double alpha_hf = 1.0);

}  // namespace vibeqc
