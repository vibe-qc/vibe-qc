// Restricted Kohn-Sham SCF for closed-shell molecules.
//
// For a (possibly hybrid) density functional F with HF-exchange fraction α,
// the Fock / Kohn-Sham matrix is
//   F_KS = Hcore + J(D) − α · K(D) + V_xc[D]
// and the total energy
//   E = tr(D · Hcore) + (1/2) tr(D · J(D)) − (α/2) tr(D · K(D))
//     + E_xc[D] + E_nuc
// with D = 2 C_occ C_occᵀ (the RHF closed-shell total density).
//
// V_xc is computed by numerical integration over the molecular grid
// (Phase 9a) using AO values/gradients (Phase 9b) and libxc's
// (exc, v_ρ, v_σ) (Phase 9c). LDA, pure GGA, and hybrid-GGA are all
// supported through a single path.

#pragma once
#include "vibeqc/opentrustregion.hpp"

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
#include "newton.hpp"  // NewtonOptions
#include "rhf.hpp"     // SCFIteration
#include "soscf.hpp"   // SOSCFOptions
#include "trah.hpp"    // TRAHOptions
#include "scf_restart.hpp"  // SCFRestartOptions
#include "davidson.hpp"

namespace vibeqc {

struct RKSOptions {
    // Explicit whole-SCF orbital optimizer; existing native phase defaults stay unchanged.
    std::string orbital_optimizer = "native";
    OpenTrustRegionOptions opentrustregion;

    std::string functional = "LDA";   // name accepted by Functional(...)
    GridOptions grid;

    int max_iter = 100;
    double conv_tol_energy = 1e-8;
    double conv_tol_grad = 1e-6;
    double damping = 0.5;
    // Dynamic damping (Zerner-Hehenberger 1979). See RHFOptions::dynamic_damping.
    // Default true (ORCA-style adaptive damping).
    bool dynamic_damping = true;
    double dynamic_damping_min = 0.0;
    double dynamic_damping_max = 0.95;
    // Fock/Kohn-Sham matrix mixing; see RHFOptions::fock_mixing.
    double fock_mixing = 0.0;
    bool use_diis = true;
    int diis_start_iter = 2;
    std::size_t diis_subspace_size = 8;

    // SCF Fock-extrapolation accelerator. See RHFOptions::scf_accelerator
    // (default ``EDIIS_DIIS``).
    SCFAccelerator scf_accelerator = SCFAccelerator::EDIIS_DIIS;
    // EDIIS -> DIIS switch threshold on the RMS matrix-element commutator
    // norm. The raw Frobenius norm is size-extensive; see RHFOptions.
    double ediis_diis_switch_threshold = 1e-1;

    // Adaptive-depth CDIIS parameters (Chupin et al. 2021). See
    // RHFOptions::diis_restart_tau / diis_adaptive_delta. Consulted only for
    // scf_accelerator R_CDIIS (τ) / AD_CDIIS (δ); both default to 1e-4.
    double diis_restart_tau = 1e-4;
    double diis_adaptive_delta = 1e-4;

    // PATOM (default) — runs in-field HF on atomic superposition for a
    // qualitatively correct molecular density. Best default from benchmarking.
    InitialGuess initial_guess = InitialGuess::AUTO;

    // READ guess: caller-supplied starting density + optional source file.
    // Consumed only when ``initial_guess == READ``; the Python run_rks
    // wrapper fills ``read_density`` from ``read_path`` / ``read_from``.
    // See RHFOptions::read_density.
    Eigen::MatrixXd read_density;
    std::string read_path;

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

    // Phase C1a-2 — Saunders-Hillier level shift (Hartree). See
    // RHFOptions::level_shift for the full description. Default 0.0
    // (no shift).
    double level_shift = 0.0;

    // Auto-reducing level shift. See RHFOptions::level_shift_warmup_cycles
    // and RHFOptions::level_shift_schedule for the full description.
    int level_shift_warmup_cycles = -1;
    std::vector<double> level_shift_schedule;

    // Auto level-shift on oscillation. When true (default), the SCF loop
    // monitors the energy history for oscillatory behaviour and automatically
    // engages a Saunders-Hillier level shift (``auto_level_shift_value``)
    // when oscillation is detected. See RHFOptions for the full description.
    bool auto_level_shift_on_oscillation = true;
    double auto_level_shift_value = 0.3;
    int oscillation_detect_start_iter = 10;
    int oscillation_window = 10;
    int oscillation_min_sign_flips = 5;

    // Phase C1c-3 — second-order SCF fallback. See RHFOptions for the
    // full description. Default 0 (disabled).
    int quadratic_fallback_iter = 0;
    double quadratic_fallback_shift = 0.1;
    double quadratic_fallback_max_step = 0.1;

    // Phase D2c-KS — second-order ("Newton") convergence for KS.
    // Mirrors RHFOptions::newton_threshold / newton_opts. Once
    // ‖F D S − S D F‖_F < newton_threshold the per-iter
    // diagonalisation is replaced by a full-Hessian Newton-CG step;
    // the matvec now adds the XC contribution W^XC[D^κ] via
    // ``make_unpolarised_xc_kernel_builder`` (LDA / GGA closed-shell).
    // Default 0 (disabled). For meta-GGA functionals (when Phase 17e
    // adds them) the kernel builder will raise — fall back to
    // EDIIS_DIIS / quadratic_fallback in that regime.
    double newton_threshold = 0.0;
    NewtonOptions newton_opts = {};

    // Phase D2d-KS — Neese approximate-SOSCF (D2d). Same diagonal-
    // dominant AH eigsolve as RHF; **no XC kernel needed** because the
    // input F already contains V_xc (F = Hcore + J − ½αK + V_xc) and
    // SOSCF only consumes the OV block of F^MO + the diagonal Hessian.
    // Mutually exclusive with Newton (Newton wins when both set).
    // Default 0.0 (off, opt-in); see RHFOptions::soscf_threshold.
    double soscf_threshold = 0.0;
    SOSCFOptions soscf_opts = {};

    // Phase D2e-KS — TRAH (Helmich-Paris 2022). Same matvec as Newton
    // + Powell-ρ adaptive trust radius. Same XC kernel plumbing.
    // Mutual-exclusion priority: quadratic > Newton > TRAH > SOSCF.
    double trah_threshold = 0.0;
    TRAHOptions trah_opts = {};

    SCFRestartOptions restart_opts = {};

    // Density fitting (resolution of the identity). When ``density_fit
    // = true``, the per-iter J(D) build uses the precomputed B-tensor
    // instead of the four-index ERI. For hybrid functionals (α_HF > 0)
    // the K(D) build is also DF-ed. Pure DFT (α_HF = 0) only needs J,
    // which makes the J-only fit ``def2-universal-jfit`` sufficient
    // (smaller and ~30 % faster than the full JKfit) — though any
    // libint-recognised aux basis works. See RHFOptions::density_fit
    // and cpp/include/vibeqc/df.hpp for the math + references. Empty
    // ``aux_basis`` + ``density_fit=true`` raises with a hint at the
    // Python autodetect helper ``vibeqc.default_aux_basis_for(name,
    // kind="jk")``.
    bool density_fit = false;
    std::string aux_basis;

    // Direct SCF Fock-build mode. See RHFOptions::scf_mode for the
    // full description; RKS uses the same direct kernel for the
    // Hartree (J) and exact-exchange (K) builds. Orthogonal to
    // density_fit + cosx — when either is true, this is ignored.
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
    // functional has α_HF > 0, the K-matrix build is replaced by Neese's
    // 2009 seminumerical algorithm: K is reduced to a sum of rank-1
    // updates over the XC integration grid, with analytic 1/|r-r_g|
    // Coulomb-attraction integrals evaluated per grid point. Pair with
    // ``density_fit = true`` for the standard "RIJCOSX" hybrid-DFT
    // acceleration (ORCA's default for hybrid SCF). For pure DFT
    // (α_HF = 0) the flag is a no-op since K is not built.
    //
    // Default ``cosx_grid`` is sparser than the XC grid (~5700 vs
    // ~46000 angular points/atom on neutral organics). Energies match
    // direct K to ~0.2-0.5 mHa with the default — bumping cosx_grid
    // back to the full XC settings tightens to ~50 µHa. See
    // cpp/include/vibeqc/cosx.hpp::default_cosx_grid_options().
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

    // DFT+U (Dudarev). Same shape + spin convention as RHFOptions —
    // see cpp/include/vibeqc/rhf.hpp + cpp/include/vibeqc/dft_plus_u.hpp.
    // Empty by default. ``dft_plus_u_ao_groups`` is parallel to
    // ``dft_plus_u_sites``; precomputed by the Python wrapper.
    std::vector<HubbardSiteCxx> dft_plus_u_sites;
    std::vector<std::vector<int>> dft_plus_u_ao_groups;

    // Restricted-stability VERDICT (issue #144 prerequisite, decision-
    // neutral). After a converged RKS SCF, the lowest eigenvalue of the
    // orbital-rotation Hessian at the returned solution is computed
    // matrix-free: the coupled per-spin UHF machinery at equal spin
    // blocks with the polarised XC kernel pinned at (D/2, D/2), i.e. the
    // union of the closed-shell KS singlet and triplet sectors
    // (Bauernschmitt & Ahlrichs, Chem. Phys. Lett. 256, 454 (1996),
    // doi:10.1063/1.471637; HF limit: Seeger & Pople 1977,
    // doi:10.1063/1.434318). A negative eigenvalue below
    // -stability_tol means the converged solution is a saddle or an
    // externally unstable restricted root: the result carries
    // internal_instability=true and the output layer warns loudly.
    //
    // Verdict ONLY — no rotation, no escape, no promotion, no basin
    // search. What to DO with a negative verdict is the maintainer
    // decision in
    // agentic-loop/asks/ask-scf144-restricted-stability-contract-2026-08-27.md.
    //
    // Default ON (matching UHF/UKS). Cost when stable: one Davidson
    // solve plus the batched XC-kernel build. Set stability_check=False
    // for legacy single-SCF behaviour.
    bool stability_check = true;
    // Distinguishes the automatic default from an explicit request so
    // unsupported routes (+U, meta-GGA, RSH, VV10, COSX hybrids) skip
    // silently by default but fail closed when explicitly requested
    // (same contract as UKS).
    bool stability_check_explicit = false;
    double stability_tol = 1e-4;
    int stability_davidson_max_iter = 60;
};

struct RKSResult {
    OpenTrustRegionReport opentrustregion;
    std::optional<BasisSet> restart_basis;
    std::optional<GuessSelection> guess_selection;
    double energy = 0.0;              // total KS energy (Hartree)
    double e_electronic = 0.0;
    // True whenever the one-electron Hamiltonian included an ECP operator,
    // including zero-core model potentials for which ecp_total_ncore is zero.
    bool ecp_operator_applied = false;
    // See RHFResult for the verified/unknown provenance contract.
    bool ecp_provenance_verified = false;
    // Exact, operative XML-library ECP provenance. Empty for all-electron and
    // inline-primitive Hamiltonians; copied by rhf_result_from_rks.
    std::vector<ECPCenter> ecp_xml_centers;
    std::string ecp_xml_library;
    // Exact inline-primitive provenance; copied by rhf_result_from_rks.
    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_primitive_centers;
    std::vector<double> ecp_effective_charges;
    // Physical core electrons replaced by the molecular ECP that produced
    // this reference. Meaningful only when ecp_provenance_verified is true.
    int ecp_total_ncore = 0;
    double e_coulomb = 0.0;           // (1/2) tr(D J(D))
    double e_hf_exchange = 0.0;       // −(α/2) tr(D K(D))   (0 for pure DFT)
    double e_xc = 0.0;                // ∫ ρ ε_xc dr
    double e_nuclear = 0.0;
    // Dudarev DFT+U contribution. Zero unless RKSOptions.dft_plus_u_sites
    // is non-empty. Closed-shell sum over both spins from the per-spin
    // formula; double-counting handled inside the SCF loop.
    double e_dft_plus_u = 0.0;
    int n_iter = 0;
    bool converged = false;
    // Restricted-stability verdict (see RKSOptions::stability_check).
    // Same field contract as RHFResult: ``stability_checked`` means the
    // analysis ran; ``stability_analysis_converged`` reports the
    // Davidson solve; ``stability_eigenvalue`` is the lowest
    // orbital-rotation Hessian eigenvalue (>= -stability_tol means
    // stable); ``internal_instability`` is the negative verdict.
    bool stability_checked = false;
    bool stability_analysis_converged = false;
    double stability_eigenvalue = 0.0;
    bool internal_instability = false;
    double stability_wall_s = 0.0;
    double stability_cpu_s = 0.0;
    // Maximum independent XC grid batches evaluated concurrently during
    // this run. Exposed for performance regression and deployment evidence.
    int xc_batch_workers_used = 1;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    Eigen::MatrixXd fock;              // converged F (includes V_U if +U is on)
    std::vector<SCFIteration> scf_trace;
    std::string functional;
    // DIIS warm-start histories for CPCM macro-iteration reuse (BUG 100).
    // Carries the final (Fock, error) pairs from the DIIS subspace so the
    // next inner SCF can pre-populate its DIIS instead of starting fresh.
    std::vector<Eigen::MatrixXd> diis_fock_history;
    std::vector<Eigen::MatrixXd> diis_error_history;
};

RKSResult run_rks(const Molecule& mol,
                  const BasisSet& basis,
                  const RKSOptions& options = {});

// Adapter: build an RHFResult-shaped reference from a converged RKS
// result so that MP2 (which takes ``const RHFResult&``) can run on
// KS orbitals. This is the path the double-hybrid dispatch uses:
// vibe-qc's ``run_mp2`` only reads ``rhf.converged``,
// ``rhf.mo_coeffs``, ``rhf.mo_energies``, and ``rhf.energy`` from
// the reference, so a faithful field-for-field copy is all that's
// needed. ``e_hf`` on the returned struct is the KS total energy
// (the underlying-reference energy that MP2 reports in
// ``MP2Result::e_hf``). ``scf_trace`` and ``fock`` are copied across
// for downstream inspection; MP2 ignores them.
inline RHFResult rhf_result_from_rks(const RKSResult& rks) {
    RHFResult r;
    r.restart_basis = rks.restart_basis;
    r.guess_selection = rks.guess_selection;
    r.energy       = rks.energy;
    r.e_electronic = rks.e_electronic;
    r.ecp_operator_applied = rks.ecp_operator_applied;
    r.ecp_provenance_verified = rks.ecp_provenance_verified;
    r.ecp_xml_centers = rks.ecp_xml_centers;
    r.ecp_xml_library = rks.ecp_xml_library;
    r.ecp_primitive_blocks = rks.ecp_primitive_blocks;
    r.ecp_primitive_centers = rks.ecp_primitive_centers;
    r.ecp_effective_charges = rks.ecp_effective_charges;
    r.ecp_total_ncore = rks.ecp_total_ncore;
    r.n_iter       = rks.n_iter;
    r.converged    = rks.converged;
    r.mo_energies  = rks.mo_energies;
    r.mo_coeffs    = rks.mo_coeffs;
    r.density      = rks.density;
    r.fock         = rks.fock;
    r.scf_trace    = rks.scf_trace;
    r.opentrustregion = rks.opentrustregion;
    // Carry the restricted-stability verdict across the adapter so
    // double-hybrid / MP2-on-RKS consumers see the same diagnosis the
    // RKS result reported (issue #144).
    r.stability_checked = rks.stability_checked;
    r.stability_analysis_converged = rks.stability_analysis_converged;
    r.stability_eigenvalue = rks.stability_eigenvalue;
    r.internal_instability = rks.internal_instability;
    r.stability_wall_s = rks.stability_wall_s;
    r.stability_cpu_s = rks.stability_cpu_s;
    return r;
}

// Lower-level closed-shell KS-DFT SCF entry point. Mirrors
// run_rhf_scf_with_jk (rhf.hpp) for hybrid / pure DFT. Takes the
// AO overlap S, the one-electron Hamiltonian Hcore, the system's
// nuclear repulsion E_nuc, a JKBuilder for the J / K piece, and a
// pre-built integration ``xc_grid`` for the XC numerical
// quadrature. The molecular ``run_rks`` builds Hcore + E_nuc +
// JKBuilder + ``build_grid(mol, opts.grid)`` and calls into this.
//
// Splitting the XC grid out as a parameter is what makes this
// usable for periodic-Γ KS-DFT — the caller substitutes a periodic
// Becke-partitioned grid (``build_grid_periodic``) and the
// remaining SCF + XC machinery is unchanged.
//
// ``initial_density`` (closed-shell density matrix; rows == basis
// size) seeds D directly when non-empty; otherwise the SCF starts
// from a Hcore-diagonalisation guess. The functional / hybrid
// fraction comes from ``options.functional``.
RKSResult run_rks_scf_with_jk(const BasisSet& basis,
                              int n_electrons,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const class JKBuilder& jk,
                              const Grid& xc_grid,
                              const RKSOptions& options = {},
                              const Eigen::MatrixXd& initial_density = {},
                              const Grid* vv10_grid = nullptr,
                              const std::vector<Eigen::MatrixXd>* warm_fock_history = nullptr,
                              const std::vector<Eigen::MatrixXd>* warm_error_history = nullptr,
                              const Molecule* guess_molecule = nullptr,
                              const GuessSelection* prepared_guess = nullptr);

}  // namespace vibeqc
