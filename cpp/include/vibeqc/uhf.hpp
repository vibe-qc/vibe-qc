// Unrestricted Hartree-Fock SCF driver for open-shell molecules.
//
// UHF relaxes the closed-shell constraint of RHF: alpha and beta electrons
// occupy independent sets of spatial orbitals. The coupled Fock equations
// are
//   F_α = Hcore + J(Dα + Dβ) - K(Dα)
//   F_β = Hcore + J(Dα + Dβ) - K(Dβ)
// with per-spin density matrices D_σ = C_σ^occ · (C_σ^occ)ᵀ (NO factor 2 —
// UHF densities are one-particle densities per spin, so
//   trace(D_α · S) = n_α,  trace(D_β · S) = n_β,
//   n_α + n_β = n_electrons,  n_α − n_β = multiplicity − 1 = 2S.
//
// Like RHF we use Hcore initial guess, symmetric orthogonalization, and
// DIIS (spin-coupled: one Pulay history over the stacked α+β error, one
// coefficient set extrapolating both Focks — the spins share J(Dα + Dβ),
// so per-spin histories can stall; see DIIS::extrapolate_spin_coupled).
// The <S^2> expectation value is reported as a spin-contamination
// diagnostic.

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
#include "rhf.hpp"   // SCFIteration
#include "newton.hpp" // NewtonOptions
#include "soscf.hpp"  // SOSCFOptions (Neese D2d)
#include "trah.hpp"   // TRAHOptions (Helmich-Paris D2e)
#include "scf_restart.hpp"  // SCFRestartOptions
#include "davidson.hpp"

namespace vibeqc {

struct UHFOptions {
    int max_iter = 100;
    double conv_tol_energy = 1e-8;  // |E[k] - E[k-1]|  (Hartree)
    double conv_tol_grad = 1e-6;    // max over spins of ||F D S - S D F||_F
    double damping = 0.5;
    // Dynamic damping (Zerner-Hehenberger 1979). See RHFOptions::dynamic_damping.
    // Default true (ORCA-style adaptive damping).
    bool dynamic_damping = true;
    double dynamic_damping_min = 0.0;
    double dynamic_damping_max = 0.95;
    // Per-spin Fock matrix mixing; see RHFOptions::fock_mixing.
    double fock_mixing = 0.0;
    bool use_diis = true;
    int diis_start_iter = 2;
    std::size_t diis_subspace_size = 8;

    // SCF Fock-extrapolation accelerator. See RHFOptions::scf_accelerator
    // (default ``EDIIS_DIIS``). For UHF the EDIIS extrapolation uses a
    // *single* set of coefficients applied to both spins (the energy
    // functional couples them through the system energy E_i and the
    // per-spin trace cross-term).
    SCFAccelerator scf_accelerator = SCFAccelerator::EDIIS_DIIS;
    double ediis_diis_switch_threshold = 1e-1;

    // Adaptive-depth CDIIS parameters (Chupin et al. 2021). See
    // RHFOptions::diis_restart_tau / diis_adaptive_delta. Consulted only for
    // scf_accelerator R_CDIIS (τ) / AD_CDIIS (δ); both default to 1e-4.
    double diis_restart_tau = 1e-4;
    double diis_adaptive_delta = 1e-4;

    // PATOM (default) — runs in-field HF on atomic superposition.
    InitialGuess initial_guess = InitialGuess::AUTO;

    // READ guess: caller-supplied per-spin starting densities (in the current
    // AO basis) + optional source file. Consumed only when
    // ``initial_guess == READ``; the Python run_uhf wrapper fills these from
    // ``read_path`` / ``read_from``. See RHFOptions::read_density.
    Eigen::MatrixXd read_density_alpha;
    Eigen::MatrixXd read_density_beta;
    std::string read_path;

    // ATOMSPIN: per-atom spin seed for a broken-symmetry SAD start, aligned
    // to atom order, +1 = majority alpha, -1 = majority beta, 0 = unpolarised.
    // Empty (default) keeps the spin-symmetric SAD split. Consulted only when
    // the resolved guess is SAD (incl. AUTO->SAD for open shells); pairing it
    // with any other guess raises. See GuessEngine::build_open_shell.
    std::vector<int> atomic_spins;

    // SPINLOCK (broken-symmetry magnetic convergence; off by default).
    //   SPIN_SCHEDULE: converge at n_alpha-n_beta = spinlock_value for the
    //     first spinlock_iterations cycles, then restart at the multiplicity
    //     target (CRYSTAL SPINLOCK n nstep; a two-phase SCF in run_uhf).
    //   PATTERN_HOLD: hold the seeded broken-symmetry occupied set by maximum
    //     overlap (MOM) for the first spinlock_iterations cycles, then release
    //     (protects an atomic_spins seed; see vibeqc/mom.hpp).
    SpinlockMode spinlock_mode = SpinlockMode::OFF;
    int spinlock_iterations = 0;   // N cycles to hold (both modes)
    int spinlock_value = 0;        // target n_alpha-n_beta for SPIN_SCHEDULE

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
    // per-spin. The UHF density convention is D_σ = C_occ · C_occ^T
    // (no factor of 2), so the shift formula is
    //     F_σ_shifted = F_σ + b · S − b · (S · D_σ · S)
    // which raises virtual α / β orbital eigenvalues by ``b`` and
    // leaves the occupied ones unchanged. Same semantics as
    // RHFOptions::level_shift otherwise. Default 0.0 (no shift).
    double level_shift = 0.0;

    // Auto-reducing level shift. See RHFOptions::level_shift_warmup_cycles
    // and RHFOptions::level_shift_schedule for the full description; the
    // per-spin shift magnitude is resolved by ``level_shift_at_iter`` and
    // applied to both α and β Fock builds each iteration.
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

    // Phase C1c-3 — second-order ("quadratic") SCF fallback, applied
    // per-spin. See RHFOptions::quadratic_fallback_* for the full
    // description; the only difference here is that the Newton step
    // runs independently on the α and β MO sets (each with its own
    // n_occ and ε spectrum). DIIS is skipped per-spin during the
    // quadratic phase, same as the RHF case.
    int quadratic_fallback_iter = 0;
    double quadratic_fallback_shift = 0.1;
    double quadratic_fallback_max_step = 0.1;

    // Phase D2c — second-order ("Newton") convergence, open-shell.
    // Mirrors RHFOptions::newton_threshold / newton_opts. The Newton
    // step rotates α and β orbitals jointly — a single preconditioned
    // CG iterates on the stacked (κ_α, κ_β) trial vector, with the
    // matvec building J once on (D^κ_α + D^κ_β) and K once per spin
    // (three JKBuilder calls per CG iter). See cpp/include/vibeqc/
    // newton.hpp for the math and the open-shell extension of D2c.
    //
    // Activation logic is identical to RHF: once
    // ``max(‖[F_α, D_α S]‖_F, ‖[F_β, D_β S]‖_F)`` drops below
    // ``newton_threshold``, the per-spin diagonalisations are replaced
    // by a coupled Newton step. ``newton_threshold = 0.0`` (default,
    // back-compat) disables Newton.
    double newton_threshold = 0.0;
    NewtonOptions newton_opts = {};

    // Phase D2d — approximate SOSCF (Neese 2000), open-shell.
    // Mirrors RHFOptions::soscf_threshold / soscf_opts. The diagonal-
    // dominant Hessian approximation has no cross-spin coupling, so
    // the per-spin AH eigsolves run independently (see uhf_soscf_step
    // in cpp/src/soscf.cpp). Mutually exclusive with Newton — when
    // both thresholds are set, Newton wins.
    // Default 0.0 (off, opt-in); see RHFOptions::soscf_threshold.
    double soscf_threshold = 0.0;
    SOSCFOptions soscf_opts = {};

    // Phase D2e — TRAH (Trust-Region Augmented Hessian; Helmich-Paris
    // 2022), open-shell. See RHFOptions::trah_threshold / trah_opts.
    // Same per-spin coupled CG matvec as uhf_newton_step but with the
    // adaptive trust radius driven by Powell's ρ test. Mutual
    // exclusion priority: quadratic > Newton > TRAH > SOSCF.
    double trah_threshold = 0.0;
    TRAHOptions trah_opts = {};

    SCFRestartOptions restart_opts = {};

    // Density fitting (resolution of the identity). When ``density_fit
    // = true``, the per-iter J(D_total) and K(D_α / D_β) builds use
    // the precomputed B-tensor instead of the four-index ERI. See
    // RHFOptions::density_fit and cpp/include/vibeqc/df.hpp for the
    // math + references. ``aux_basis`` is a libint-recognised auxiliary
    // basis name (e.g. "def2-svp-jk", "def2-universal-jkfit"); empty +
    // ``density_fit=true`` raises with a hint at the Python autodetect
    // helper ``vibeqc.default_aux_basis_for(name, kind="jk")``.
    bool density_fit = false;
    std::string aux_basis;

    // Direct SCF Fock-build mode. See RHFOptions::scf_mode for the
    // full description; UHF dispatch follows the same rules — DIRECT
    // (or AUTO + n_bf > scf_mode_auto_threshold) replaces the in-core
    // four-index path with on-the-fly Schwarz-screened libint quartet
    // evaluation. Orthogonal to density_fit + cosx.
    SCFMode scf_mode = SCFMode::AUTO;
    int scf_mode_auto_threshold = 140;
    double schwarz_threshold = 1e-10;

    // Incremental (Almlöf) ΔP Fock build for the direct path. See
    // RHFOptions::incremental_fock for the full description. UHF
    // calls ``build_g_rhf`` per spin block via the same DirectJK
    // builder; the cache holds the most-recently-built spin density
    // and is reset every ``incremental_fock_reset_freq`` calls
    // (counted across the α + β stream).
    // BUG 87: enabled by default for 3–10× direct-SCF speedup.
    bool incremental_fock = true;
    int incremental_fock_reset_freq = 8;

    // Two-phase Schwarz threshold for direct SCF. See
    // RHFOptions::schwarz_threshold_loose for the full description.
    double schwarz_threshold_loose = 1e-7;
    double schwarz_threshold_tighten_at = 1e-3;

    // Chain-of-spheres exchange (COSX). See RHFOptions::cosx for the
    // full description. The grid is built fresh per SCF call from
    // ``cosx_grid``.
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

    // DFT+U (Dudarev). Same shape + convention as RHFOptions, but
    // the per-spin formula natively applies — n_α = (S P_α S)_block,
    // V_U_α = U_eff (½ δ − n_α), same for β; energies sum directly
    // (no factor of 2 since each spin contributes its own term).
    std::vector<HubbardSiteCxx> dft_plus_u_sites;
    std::vector<std::vector<int>> dft_plus_u_ao_groups;

    // Internal SCF stability analysis (UHF-ABOVE-ROHF-VARIATIONAL-
    // INVERSION fix, 2026-08-05). After the SCF converges, the lowest
    // eigenvalue of the internal (real UHF -> UHF) orbital-rotation
    // Hessian is computed matrix-free (uhf_internal_stability_lowest in
    // newton.hpp; stability criterion per Lehtola, Molecules 25, 1218
    // (2020) Section 10 / Seeger & Pople, J. Chem. Phys. 66, 3045
    // (1977)). A converged SCF whose Hessian has a negative eigenvalue
    // is a saddle point — an excited SCF solution reported as
    // converged. On detection the driver line-searches the energy along
    // the unstable rotation, restarts the SCF from the rotated
    // densities, and re-checks, up to ``stability_max_retries`` times.
    //
    // Default ON: the silent failure mode this guards against (e.g. O2
    // m=5/cc-pVDZ converging +59 mHa ABOVE the ROHF energy on the same
    // system) invalidates open-shell results without any diagnostic.
    // Cost when the solution is already stable: one Davidson solve,
    // typically 10-40 J+2K builds. Set ``stability_check = false`` to
    // opt out (single-SCF legacy behaviour, bit-for-bit).
    bool stability_check = true;
    // Instability threshold on the lowest Hessian eigenvalue: unstable
    // when lambda_min < -stability_tol. Guards against declaring
    // instability on numerical noise around genuinely zero modes
    // (e.g. symmetry rotations of degenerate open shells).
    double stability_tol = 1e-4;
    // Maximum rotate-and-reconverge escapes before giving up (the
    // result then carries the negative eigenvalue for the caller /
    // output layer to report loudly).
    int stability_max_retries = 3;
    // Davidson matvec cap for each stability solve.  The sign-certifying
    // residual needs more than 64 products on FeCl3/cc-pVDZ (#398), so keep
    // bounded headroom above that measured case.
    int stability_davidson_max_iter = 80;

    // Multi-guess convergence (BUG 88). When non-empty, the Python
    // runner tries the SCF from each additional seed, fingerprints
    // each converged state, and returns the lowest-energy internally-
    // stable solution. Empty (default) = single SCF.
    std::vector<std::uint64_t> multi_guess_seeds;
};

struct UHFResult {
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
    // Dudarev DFT+U contribution. Open-shell: per-spin formula
    // applied separately to (P_α, P_β), energies summed directly.
    double e_dft_plus_u = 0.0;
    int n_iter = 0;
    bool converged = false;
    double s_squared = 0.0;          // <S^2> expectation; ideal value = S(S+1)
    double s_squared_ideal = 0.0;    // S(S+1) for the requested multiplicity

    // Alpha spin
    Eigen::VectorXd mo_energies_alpha;
    Eigen::MatrixXd mo_coeffs_alpha;
    Eigen::MatrixXd density_alpha;
    Eigen::MatrixXd fock_alpha;

    // Beta spin
    Eigen::VectorXd mo_energies_beta;
    Eigen::MatrixXd mo_coeffs_beta;
    Eigen::MatrixXd density_beta;
    Eigen::MatrixXd fock_beta;

    std::vector<SCFIteration> scf_trace;

    // Internal-stability report (see UHFOptions::stability_check).
    // ``stability_checked`` is true when the analysis ran on the final
    // converged solution; ``stability_analysis_converged`` reports the
    // Davidson solve itself. ``stability_eigenvalue`` is the lowest
    // internal-Hessian eigenvalue at the returned solution (>=
    // -stability_tol means internally stable). ``n_stability_restarts``
    // counts rotate-and-reconverge escapes taken to reach it.
    bool stability_checked = false;
    bool stability_analysis_converged = false;
    double stability_eigenvalue = 0.0;
    int n_stability_restarts = 0;
    // True when the returned solution is internally UNSTABLE (lowest
    // eigenvalue < -stability_tol) and the driver did not (or, for
    // deliberate state-targeting runs, must not) escape it: the energy
    // then corresponds to an excited SCF solution and the output layer
    // reports it loudly.
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
};

UHFResult run_uhf(const Molecule& mol,
                  const BasisSet& basis,
                  const UHFOptions& options = {});

// Lower-level UHF SCF entry point. Mirrors run_rhf_scf_with_jk
// (rhf.hpp): takes the AO overlap S, the one-electron Hamiltonian
// Hcore, the system's nuclear repulsion E_nuc, and a JKBuilder for
// the two-electron Fock piece. Drives the same per-spin SCF body
// (canonical orthogonalisation, per-spin DIIS, optional damping /
// level-shift / quadratic fallback). The wrapper run_uhf above
// builds Hcore + E_nuc + JKBuilder + (optionally) a SAD initial
// guess from the molecule and calls into this.
//
// ``initial_density_alpha`` / ``initial_density_beta`` are optional.
// When non-empty (rows == basis size each) they seed D_α / D_β
// directly; when empty they fall back to a Hcore-diagonalisation
// initial guess. Mixed (only one supplied) is rejected as ambiguous.
UHFResult run_uhf_scf_with_jk(const BasisSet& basis,
                              int n_alpha,
                              int n_beta,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const class JKBuilder& jk,
                              const UHFOptions& options = {},
                              const Eigen::MatrixXd& initial_density_alpha = {},
                              const Eigen::MatrixXd& initial_density_beta = {},
                              const Molecule* guess_molecule = nullptr,
                              const GuessSelection* prepared_guess = nullptr);

}  // namespace vibeqc
