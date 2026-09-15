// Unrestricted Kohn-Sham SCF driver for open-shell molecules.
//
// Open-shell DFT. Mirrors UHF the way RKS mirrors RHF: separate α/β
// densities, libxc called in its spin-polarized mode, two coupled Fock
// matrices
//   F_α = Hcore + J(D_α + D_β) − α·K(D_α) + V_xc,α[ρ_α, ρ_β]
//   F_β = Hcore + J(D_α + D_β) − α·K(D_β) + V_xc,β[ρ_α, ρ_β]
// (α here is the HF-exchange fraction from the functional, not a spin
// index — apologies for the notation overlap.)
//
// Occupations follow the molecule's multiplicity, same formulas as UHF:
// n_α = (n_e + mult−1)/2,  n_β = (n_e − mult+1)/2.

#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

#include "basis.hpp"
#include "cosx.hpp"
#include "ecp.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "molecule.hpp"
#include "newton.hpp"   // NewtonOptions
#include "rhf.hpp"      // SCFIteration
#include "soscf.hpp"    // SOSCFOptions
#include "trah.hpp"     // TRAHOptions
#include "scf_restart.hpp"  // SCFRestartOptions
#include "davidson.hpp"

namespace vibeqc {

class Functional;

// Spin-resolved XC energy and first derivative matrices for one fixed pair of
// density matrices.  This is the shared first-order XC surface used by UKS
// and by constrained shared-orbital methods such as ROKS.  Keeping the
// projection in the native UKS implementation is especially important for
// full-grid external providers: their adjoints are derivatives of one
// integrated energy and cannot be reconstructed through the pointwise libxc
// API.
struct UKSXCPotential {
    Eigen::MatrixXd V_alpha;
    Eigen::MatrixXd V_beta;
    double energy = 0.0;
};

UKSXCPotential evaluate_uks_xc_potential(
    const Functional& functional,
    const BasisSet& basis,
    const Grid& grid,
    const Eigen::MatrixXd& density_alpha,
    const Eigen::MatrixXd& density_beta);

struct UKSOptions {
    std::string functional = "LDA";
    GridOptions grid;

    int max_iter = 100;
    double conv_tol_energy = 1e-8;
    // Max over spins of ||X^T(F D S - S D F)X||_F, X^T S X = I.
    double conv_tol_grad = 1e-6;
    double damping = 0.5;
    // Dynamic damping (Zerner-Hehenberger 1979). See RHFOptions::dynamic_damping.
    // Default true (ORCA-style adaptive damping).
    bool dynamic_damping = true;
    double dynamic_damping_min = 0.0;
    double dynamic_damping_max = 0.95;
    // Per-spin Kohn-Sham matrix mixing; see RHFOptions::fock_mixing.
    double fock_mixing = 0.0;
    bool use_diis = true;
    int diis_start_iter = 2;
    std::size_t diis_subspace_size = 8;

    // SCF Fock-extrapolation accelerator. See RHFOptions::scf_accelerator
    // (default ``EDIIS_DIIS``) and UHFOptions::scf_accelerator (single
    // coefficient set across both spins for EDIIS / EDIIS_DIIS).
    SCFAccelerator scf_accelerator = SCFAccelerator::EDIIS_DIIS;
    double ediis_diis_switch_threshold = 1e-1;

    // Adaptive-depth CDIIS parameters (Chupin et al. 2021). See
    // RHFOptions::diis_restart_tau / diis_adaptive_delta. Consulted only for
    // scf_accelerator R_CDIIS (τ) / AD_CDIIS (δ); both default to 1e-4.
    double diis_restart_tau = 1e-4;
    double diis_adaptive_delta = 1e-4;

    // PATOM (default) — runs in-field HF on atomic superposition.
    InitialGuess initial_guess = InitialGuess::AUTO;

    // READ guess: caller-supplied per-spin starting densities + optional
    // source file. Consumed only when ``initial_guess == READ``; the Python
    // run_uks wrapper fills these from ``read_path`` / ``read_from``.
    // See RHFOptions::read_density.
    Eigen::MatrixXd read_density_alpha;
    Eigen::MatrixXd read_density_beta;
    std::string read_path;

    // ATOMSPIN: per-atom spin seed for a broken-symmetry SAD start (+1/-1/0
    // in atom order). Empty (default) keeps the spin-symmetric SAD split.
    // Consulted only when the resolved guess is SAD. See UHFOptions.
    std::vector<int> atomic_spins;

    // SPINLOCK (broken-symmetry magnetic convergence; off by default). See
    // UHFOptions for the full description.
    //   SPIN_SCHEDULE: two-phase SCF, locked n_alpha-n_beta = spinlock_value
    //     for spinlock_iterations cycles, then release to the target.
    //   PATTERN_HOLD: MOM-hold the seeded pattern for spinlock_iterations.
    SpinlockMode spinlock_mode = SpinlockMode::OFF;
    int spinlock_iterations = 0;
    int spinlock_value = 0;

    // Canonical-orthogonalization threshold on the AO overlap matrix.
    // See RHFOptions::linear_dep_threshold. Default 1e-7.
    double linear_dep_threshold = 1e-7;

    // Phase 14c — effective core potentials. See RHFOptions for the
    // full description. Default: no ECPs (empty list).
    std::vector<ECPCenter> ecp_centers;
    std::string ecp_library;
    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_primitive_centers;
    std::vector<double> ecp_effective_charges;
    int ecp_total_ncore = 0;

    // Phase C1a-2 — Saunders-Hillier level shift (Hartree), applied
    // per-spin. See UHFOptions::level_shift for the formula and
    // RHFOptions::level_shift for the full description. Default 0.0
    // (no shift).
    double level_shift = 0.0;

    // Auto-reducing level shift. See RHFOptions::level_shift_warmup_cycles
    // and RHFOptions::level_shift_schedule for the full description;
    // applied per-spin to both α and β Fock builds.
    int level_shift_warmup_cycles = -1;
    std::vector<double> level_shift_schedule;

    // Auto level-shift on oscillation. When true (default), the SCF loop
    // monitors the energy history for oscillatory behaviour and automatically
    // engages a Saunders-Hillier level shift (``auto_level_shift_value``)
    // when oscillation is detected. The shift is applied per-spin; DIIS
    // history is cleared on engagement. See RHFOptions for the full
    // description.
    bool auto_level_shift_on_oscillation = true;
    double auto_level_shift_value = 0.3;
    int oscillation_detect_start_iter = 10;
    int oscillation_window = 10;
    int oscillation_min_sign_flips = 5;

    // Phase C1c-3 — second-order SCF fallback, applied per-spin. See
    // RHFOptions / UHFOptions for the full description. Default 0
    // (disabled).
    int quadratic_fallback_iter = 0;
    double quadratic_fallback_shift = 0.1;
    double quadratic_fallback_max_step = 0.1;

    // Phase D2c-KS-UHF — second-order ("Newton") convergence for UKS.
    // The Hessian matvec uses the polarised XC kernel
    // (make_polarised_xc_kernel_builder) pinned at the current per-spin
    // densities; the αβ coupling carries through v2rho2_αβ for LDA and,
    // since Phase 17e, additionally through v2rhosigma_* / v2sigma2_* /
    // the σ_αβ cross term for GGA / hybrid-GGA. Meta-GGA still raises
    // with a roadmap pointer (τ-dependent polarised fxc unplumbed).
    double newton_threshold = 0.0;
    NewtonOptions newton_opts = {};

    // Phase D2d-KS-UHF — Neese approximate-SOSCF. As with RKS, no XC
    // kernel needed: F_σ already carries V_xc,σ into the matvec via
    // the per-spin F^MO occ-vir block.
    // Default 0.0 (off, opt-in); see RHFOptions::soscf_threshold.
    double soscf_threshold = 0.0;
    SOSCFOptions soscf_opts = {};

    // Phase D2e-KS-UHF — TRAH. Same XC kernel plumbing as Newton +
    // Powell-ρ adaptive trust radius. Mutual-exclusion priority:
    // quadratic > Newton > TRAH > SOSCF.
    double trah_threshold = 0.0;
    TRAHOptions trah_opts = {};

    SCFRestartOptions restart_opts = {};

    // Density fitting (resolution of the identity). When ``density_fit
    // = true``, the per-iter J(D_total) and (for hybrid functionals
    // with α_HF > 0) K(D_α) / K(D_β) builds use the precomputed
    // B-tensor instead of the four-index ERI. The XC contribution is
    // unaffected — it lives on the molecular grid, independent of the
    // ERI tensor. See RHFOptions::density_fit and
    // cpp/include/vibeqc/df.hpp for math + references.
    bool density_fit = false;
    std::string aux_basis;

    // Direct SCF Fock-build mode. See RHFOptions::scf_mode for the
    // full description; UKS uses the same direct kernel for the
    // Hartree and exact-exchange builds on the α / β densities.
    // Orthogonal to density_fit + cosx.
    SCFMode scf_mode = SCFMode::AUTO;
    int scf_mode_auto_threshold = 140;
    double schwarz_threshold = 1e-10;

    // Incremental (Almlöf) ΔP Fock build for the direct path. See
    // RHFOptions::incremental_fock for the full description.
    // BUG 87: enabled by default for 3–10× direct-SCF speedup.
    bool incremental_fock = true;
    int incremental_fock_reset_freq = 8;

    // Two-phase Schwarz threshold for direct SCF. See
    // RHFOptions::schwarz_threshold_loose for the full description.
    double schwarz_threshold_loose = 1e-7;
    double schwarz_threshold_tighten_at = 1e-3;

    // Chain-of-spheres exchange (COSX). When ``cosx = true`` and the
    // functional has α_HF > 0, K(D_α) and K(D_β) are built via Neese's
    // 2009 seminumerical algorithm instead of the four-index / RI path.
    // See RKSOptions::cosx for the full description. Pure DFT (α_HF = 0)
    // makes the flag a no-op. Default ``cosx_grid`` is sparser than the
    // XC grid; see cosx.hpp::default_cosx_grid_options.
    bool cosx = false;
    GridOptions cosx_grid = default_cosx_grid_options();

    // COSX variant / grid-level / accuracy threshold (RIJCOSX upgrade,
    // M1). Consulted only when ``cosx == true``. See cosx.hpp.
    //   cosx_variant    — AUTO (default) resolves STANDARD vs FITTED from
    //                     ``thresh_cosx`` + basis cardinality.
    //   cosx_grid_level — -1 (default) AUTO: GridX tier + multi-stage for
    //                     normal-tolerance SCFs; legacy grid when
    //                     conv_tol_grad < 1e-6 (cosx_use_gridx). 0 forces
    //                     legacy; 1..4 force a GridX tier at any tolerance.
    //   thresh_cosx     — COSX accuracy target (< 1e-6 ⇒ FITTED).
    CosxVariant cosx_variant = CosxVariant::AUTO;
    int cosx_grid_level = -1;
    double thresh_cosx = 1e-6;

    // Davidson iterative diagonalization (subset eigensolver).
    // When true, replaces the dense SelfAdjointEigenSolver with a
    // block-Davidson algorithm that extracts only the n_occ lowest
    // eigenpairs.  Useful for large basis sets where O(N³)
    // diagonalization dominates the SCF wall-clock.
    // ``davidson_min_dim`` gates the switch: for small problems the
    // dense solver is faster; only activate Davidson above this many
    // kept orthogonal basis functions.
    bool use_davidson = false;
    DavidsonOptions davidson = {};
    int davidson_min_dim = 100;

    // DFT+U (Dudarev). Same per-spin convention as UHFOptions.
    std::vector<HubbardSiteCxx> dft_plus_u_sites;
    std::vector<std::vector<int>> dft_plus_u_ao_groups;

    // Internal SCF stability analysis (UHF-ABOVE-ROHF-VARIATIONAL-
    // INVERSION fix, 2026-08-05; extended to UKS 2026-08-08). After
    // the SCF converges, the lowest eigenvalue of the internal (real
    // UHF -> UHF) orbital-rotation Hessian is computed matrix-free with
    // the polarised XC kernel included in the matvec. The escape
    // protocol mirrors the UHF version: line-search along the unstable
    // mode and MOM-anchored SCF restart.
    //
    // Default ON where the complete polarised response is supported (LDA,
    // GGA, and global-hybrid LDA/GGA on a consistent JK energy surface). The
    // implicit automatic check is skipped for meta-GGA, range-separated,
    // VV10, DFT+U, and active hybrid RIJCOSX jobs until all corresponding
    // response terms are plumbed; an explicit request remains fail-closed.
    // Solvent wrappers likewise skip until coupled reaction-field response. Cost
    // when the solution is already stable: one Davidson solve + one XC kernel
    // build. Set ``stability_check = false`` to opt out.
    bool stability_check = true;
    // Set by the Python property setter. Keeps the automatic default
    // distinguishable from a user request without weakening the default for
    // supported UKS routes or silently accepting an incomplete response.
    bool stability_check_explicit = false;
    double stability_tol = 1e-4;
    int stability_max_retries = 3;
    int stability_davidson_max_iter = 60;

    // Multi-guess convergence (BUG 88). See UHFOptions::multi_guess_seeds.
    std::vector<std::uint64_t> multi_guess_seeds;
};

struct UKSResult {
    std::optional<BasisSet> restart_basis;
    std::optional<GuessSelection> guess_selection;
    double energy = 0.0;
    double e_electronic = 0.0;
    // True whenever the one-electron Hamiltonian included an ECP operator,
    // including zero-core model potentials for which ecp_total_ncore is zero.
    bool ecp_operator_applied = false;
    // See RHFResult for the verified/unknown provenance contract.
    bool ecp_provenance_verified = false;
    // Exact, operative XML-library ECP provenance. Empty for all-electron and
    // inline-primitive Hamiltonians.
    std::vector<ECPCenter> ecp_xml_centers;
    std::string ecp_xml_library;
    // Exact inline-primitive provenance; blocks and centers are paired by
    // index, while effective charges remain in molecule-atom order.
    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_primitive_centers;
    std::vector<double> ecp_effective_charges;
    // Physical core electrons replaced by the molecular ECP that produced
    // this reference. Meaningful only when ecp_provenance_verified is true.
    int ecp_total_ncore = 0;
    double e_coulomb = 0.0;
    double e_hf_exchange = 0.0;   // (−α/2) [tr(D_α K_α) + tr(D_β K_β)]
    double e_xc = 0.0;
    double e_nuclear = 0.0;
    // Dudarev DFT+U contribution (sum over per-spin contributions).
    double e_dft_plus_u = 0.0;
    int n_iter = 0;
    bool converged = false;
    // Maximum independent XC grid batches evaluated concurrently across
    // the SCF and post-convergence stability phases.
    int xc_batch_workers_used = 1;
    double s_squared = 0.0;
    double s_squared_ideal = 0.0;

    Eigen::VectorXd mo_energies_alpha;
    Eigen::MatrixXd mo_coeffs_alpha;
    Eigen::MatrixXd density_alpha;
    Eigen::MatrixXd fock_alpha;

    Eigen::VectorXd mo_energies_beta;
    Eigen::MatrixXd mo_coeffs_beta;
    Eigen::MatrixXd density_beta;
    Eigen::MatrixXd fock_beta;

    std::vector<SCFIteration> scf_trace;
    std::string functional;

    // Internal-stability report (see UKSOptions::stability_check).
    bool stability_checked = false;
    bool stability_analysis_converged = false;
    double stability_eigenvalue = 0.0;
    int n_stability_restarts = 0;
    bool internal_instability = false;
    // Wall / CPU seconds spent in the post-convergence internal-stability
    // phase (Davidson analysis plus any rotate-and-reconverge escape),
    // measured inside the driver. Reported as its own perf-log phase and
    // its own .out timing row so it is no longer averaged into "SCF avg.
    // per iteration" (issue #205); zero when the analysis did not run.
    // ``stability_cpu_s`` sums every OpenMP thread, matching the perf
    // log's other CPU columns.
    double stability_wall_s = 0.0;
    double stability_cpu_s = 0.0;
    // Iterations in the first-order SCF loop before the timed stability
    // phase began. A corrective follow replaces ``n_iter`` with the final
    // reconvergence attempt, while the runner's loop-only timing average
    // must retain the divisor belonging to the wall it reports.
    int n_iter_before_stability = 0;
};

UKSResult run_uks(const Molecule& mol,
                  const BasisSet& basis,
                  const UKSOptions& options = {});

// Lower-level open-shell KS-DFT SCF entry point. Mirrors
// run_rks_scf_with_jk (rks.hpp) for unrestricted hybrid / pure DFT.
// Takes per-spin electron counts, S, Hcore, E_nuc, a JKBuilder for
// the J / K piece, and a pre-built integration ``xc_grid``. The
// molecular ``run_uks`` builds Hcore + E_nuc + JKBuilder +
// ``build_grid(mol, opts.grid)`` and (optionally) SAD-fraction
// per-spin densities, then calls into this.
//
// ``init_alpha`` / ``init_beta``: optional per-spin initial
// density matrices. Both supplied or both empty — mixed is
// rejected as ambiguous.
UKSResult run_uks_scf_with_jk(const BasisSet& basis,
                              int n_alpha,
                              int n_beta,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const class JKBuilder& jk,
                              const Grid& xc_grid,
                              const UKSOptions& options = {},
                              const Eigen::MatrixXd& init_alpha = {},
                              const Eigen::MatrixXd& init_beta  = {},
                              const Grid* vv10_grid = nullptr,
                              const Molecule* guess_molecule = nullptr,
                              const GuessSelection* prepared_guess = nullptr);

}  // namespace vibeqc
