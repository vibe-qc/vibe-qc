// Restricted Hartree-Fock SCF driver for a single closed-shell molecule.

#pragma once
#include "vibeqc/opentrustregion.hpp"

#include <Eigen/Dense>
#include <optional>
#include <cstddef>
#include <string>

#include "basis.hpp"
#include "cosx.hpp"
#include "davidson.hpp"      // Davidson iterative diagonalizer
#include "dft_plus_u.hpp"   // HubbardSiteCxx for the optional +U Fock term
#include "ecp.hpp"
#include "ediis.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "jk_builder.hpp"  // SCFMode (shared with UHF / RKS / UKS via transitive include)
#include "molecule.hpp"
#include "newton.hpp"
#include "soscf.hpp"   // SOSCFOptions (Neese D2d, distinct from D2c Newton)
#include "trah.hpp"    // TRAHOptions (Helmich-Paris D2e, full Hessian + adaptive trust)
#include "scf_restart.hpp"  // SCFRestartOptions (deterministic orbital-rotation restart)

namespace vibeqc {

struct RHFOptions {
    // Explicit whole-SCF orbital optimizer; existing native phase defaults stay unchanged.
    std::string orbital_optimizer = "native";
    OpenTrustRegionOptions opentrustregion;

    int max_iter = 100;
    double conv_tol_energy = 1e-8;  // |E[k] - E[k-1]| < tol  (Hartree)
    double conv_tol_grad = 1e-6;    // ||F D S - S D F||_Frobenius

    // Density damping: D_next_input = damping * D_prev + (1-damping) * D_new.
    // 0.0 = no damping; 0.7 = heavy damping (mostly keep the previous iterate).
    // Ignored once DIIS takes over (it's a better accelerator on its own).
    double damping = 0.5;

    // Dynamic (adaptive) damping (Zerner-Hehenberger 1979). When true, the
    // ``damping`` α is adjusted iteration-by-iteration based on the energy
    // decrease: increased toward ``dynamic_damping_max`` when the SCF
    // energy oscillates upward, decreased toward ``dynamic_damping_min``
    // when the energy is monotonically decreasing. The ``damping`` field
    // is used as the initial α; final α stays in [min, max].
    // Default true (ORCA-style adaptive damping): ``damping`` starts at
    // 0.5 and the Zerner-Hehenberger heuristic adapts it within
    // [dynamic_damping_min, dynamic_damping_max]. For monotonically
    // decreasing energies α relaxes toward 0.0, so well-behaved systems
    // pay no penalty; oscillating systems are stabilised.
    bool dynamic_damping = true;
    double dynamic_damping_min = 0.0;
    double dynamic_damping_max = 0.95;

    // Fock/Kohn-Sham matrix mixing. ``fock_mixing`` is the weight of the
    // previous matrix in the matrix diagonalised to produce the next
    // density:
    //     F_diag = (1 - fock_mixing) F_current + fock_mixing F_previous.
    // This mirrors CRYSTAL's FMIXING keyword (FMIXING 30 -> 0.30) and is
    // independent of density damping. Default 0.0 (off).
    double fock_mixing = 0.0;

    // DIIS extrapolation (Pulay). Normally a large net speedup over damping:
    // ~10 iterations to converge H2O / 6-31G* instead of ~20-30.
    bool use_diis = true;
    int diis_start_iter = 2;        // Must be >= 2 (need ≥ 2 Fs to extrapolate)
    std::size_t diis_subspace_size = 8;

    // SCF Fock-extrapolation accelerator. ``DIIS`` is Pulay's commutator
    // DIIS; ``EDIIS`` is energy-DIIS (Kudin/Scuseria/Cancès 2002);
    // ``EDIIS_DIIS`` (default) is the production hybrid (Garza/Scuseria
    // 2012) that uses EDIIS while ‖e‖_F > ediis_diis_switch_threshold
    // and DIIS below it — matches PySCF / ORCA defaults and is the
    // most-robust choice for far-from-convergence starts. ``KDIIS``
    // (Kollmar 1997) uses an orbital-rotation-gradient error vector.
    // Ignored when use_diis == false. The same diis_subspace_size caps
    // the EDIIS history.
    SCFAccelerator scf_accelerator = SCFAccelerator::EDIIS_DIIS;

    // EDIIS -> DIIS switch threshold for SCFAccelerator::EDIIS_DIIS. The
    // hybrid uses EDIIS while the RMS matrix-element commutator norm is above
    // this and switches to DIIS once below. Default 1e-1 matches PySCF's
    // intensive threshold convention; the raw Frobenius norm is size-extensive.
    double ediis_diis_switch_threshold = 1e-1;

    // Adaptive-depth commutator-DIIS parameters (Chupin, Dupuy, Legendre
    // & Séré, ESAIM: M2AN 55, 2785 (2021)). Consulted only when
    // scf_accelerator is R_CDIIS (restart aggressiveness τ ∈ (0,1)) or
    // AD_CDIIS (depth-window δ > 0); see make_diis in ediis.hpp.
    double diis_restart_tau = 1e-4;
    double diis_adaptive_delta = 1e-4;

    // Initial density guess. PATOM (default) runs a few in-field HF steps
    // on top of the atomic superposition to build a qualitatively correct
    // molecular density. Benchmarked as the best default across the molecular
    // test suite (H2, H2O, CH4) — reduces iteration counts by 1-3 vs SAD.
    // ORCA combines PATOM-like guess with SOSCF for 2-iteration convergence.
    InitialGuess initial_guess = InitialGuess::AUTO;

    // READ guess: caller-supplied starting density (closed-shell, in the
    // current AO basis). Consumed only when ``initial_guess == READ``; the
    // Python ``run_rhf`` wrapper builds it from ``read_path`` / ``read_from``
    // (a prior result, .qvf, or .molden), projecting onto this basis if the
    // geometry/basis differ. Empty + READ is an error.
    Eigen::MatrixXd read_density;

    // READ source file (``.qvf`` / ``.molden``) for the Python run_rhf
    // wrapper. Ignored by the C++ driver itself (which only consumes
    // ``read_density``); the wrapper reads this, builds the density, and
    // fills ``read_density``. Empty = restart from an in-memory ``read_from``.
    std::string read_path;

    // Canonical-orthogonalization threshold on the AO overlap matrix S.
    // Any eigenvalue of S below this is projected out before the SCF
    // starts, shrinking the MO dimension from n_basis to n_basis - n_dropped.
    // For well-conditioned bases (all eigenvalues above threshold) this
    // is equivalent to plain symmetric orthogonalization — same SCF
    // result to machine precision. For near-linearly-dependent bases
    // (large diffuse sets on tight-contact geometries) the projection is
    // what makes the SCF converge at all.
    // Default 1e-7 matches PySCF / ORCA convention. Set 0.0 to disable
    // the projection (useful only for back-compat regression tests).
    double linear_dep_threshold = 1e-7;

    // Phase 14c — effective core potentials. When ``ecp_centers`` is
    // non-empty, the SCF adds
    //     V_ECP = compute_ecp_matrix(basis, ecp_centers, ecp_library)
    // to ``Hcore`` before the iteration starts. The Molecule retains
    // its physical ionic charge and full electron count; the SCF
    // subtracts the ECP-replaced cores itself. Empty
    // ``ecp_library`` means "ecp10mdf" (a sensible Stuttgart-Köln
    // small-core default for K–Kr); for other element ranges or
    // ECP families set this explicitly.
    std::vector<ECPCenter> ecp_centers;
    std::string ecp_library;

    // Phase 14g — inline primitive ECPs for basis sidecars whose
    // per-element cores are not available as a libecpint XML library
    // (vDZP / composite-3c carriers). XML-library and inline-primitive
    // inputs are mutually exclusive; the high-level driver rejects a mixed
    // request rather than silently discarding either operator.
    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_primitive_centers;
    std::vector<double> ecp_effective_charges;
    int ecp_total_ncore = 0;

    // Phase C1a-2 — Saunders-Hillier level shift (Hartree). Adds
    //     F_shifted = F + b · S − (b/2) · S · D · S
    // before diagonalisation, where ``b = level_shift``. This raises
    // virtual orbital eigenvalues by ``b`` while leaving the occupied
    // ones unchanged (the shift is "inert at the converged density":
    // S · D_∞ · S = 2 · S · C_occ · C_occ^T · S, and the projector
    // ½ S D S projects onto the occupied subspace). The SCF fixed
    // point is unchanged — only the iteration dynamics are damped.
    // Useful when DIIS oscillates between near-degenerate occupied /
    // virtual swaps on small-HOMO–LUMO-gap molecules. The shift is
    // skipped during the C1c quadratic phase (Newton step in MO
    // space already includes the (ε_a − ε_i + λ) preconditioner).
    // Default 0.0 (no shift). Typical values: 0.1 – 0.5 Hartree.
    double level_shift = 0.0;

    // Auto-reducing level shift. ``level_shift_warmup_cycles`` controls
    // how the base ``level_shift`` decays across iterations (shared with
    // the periodic drivers): ``-1`` = auto (up to five shifted startup
    // cycles, then released with ≥1 unshifted tail cycle); ``0`` =
    // persistent (held every iteration, the legacy behaviour); ``N > 0``
    // = explicit warm-up length. ``level_shift_schedule`` is an explicit
    // per-iteration curve (CRYSTAL ``LEVSHIFT B IRESET`` style): empty ⇒
    // derive from the warm-up logic; non-empty ⇒ use it directly, with
    // the last entry reused for iterations past its length (0.0 releases).
    // Both are resolved by ``level_shift_at_iter`` in level_shift.hpp.
    int level_shift_warmup_cycles = -1;
    std::vector<double> level_shift_schedule;

    // Auto level-shift on oscillation. When true (default), the SCF loop
    // monitors the energy history for oscillatory behaviour and automatically
    // engages a Saunders-Hillier level shift (``auto_level_shift_value``)
    // when oscillation is detected. This protects against the most common
    // default-convergence failure: DIIS oscillating between near-degenerate
    // occupied/virtual swaps on transition-metal systems, heterocycles, and
    // small-gap organics. The shift is held persistently once engaged; DIIS
    // history is cleared on engagement. Set false to rely on the explicit
    // ``level_shift`` alone.
    //
    // Mirrors the ROHF Python driver's ``auto_level_shift_on_oscillation``
    // (``python/vibeqc/rohf.py``), which has converged FeCl3 ROHF and over
    // a dozen previously-failing open-shell cases since v0.15.x.
    bool auto_level_shift_on_oscillation = true;
    double auto_level_shift_value = 0.3;
    // How many iterations before oscillation detection activates. The
    // detector needs a history of ``oscillation_window`` energy deltas with
    // ``oscillation_min_sign_flips`` sign changes to trigger.
    int oscillation_detect_start_iter = 10;
    int oscillation_window = 10;
    int oscillation_min_sign_flips = 5;

    // Phase C1c-3 — second-order ("quadratic") SCF fallback. When
    // standard SCF (damping + DIIS) fails to converge — typically on
    // small-gap insulators where DIIS oscillates between near-
    // degenerate occ/vir swaps — switch from "diagonalize F" to a
    // Newton step in MO space (κ_{ai} = -F_{ai}^MO / (ε_a − ε_i + λ);
    // C_new = C_prev · exp(κ); D_new = 2·C_occ·C_occ^T). The step is
    // preconditioned by orbital-energy differences (the diagonal of
    // the orbital-rotation Hessian) and trust-region capped at
    // ``quadratic_fallback_max_step`` to keep the matrix exponential
    // well-conditioned.
    //
    // Activation. ``quadratic_fallback_iter > 0`` enables the
    // fallback after iteration ``quadratic_fallback_iter``. ``= 0``
    // (default) disables — SCF behaves exactly as before. DIIS is
    // skipped during the quadratic phase (the Newton step is its
    // own update mechanism, and mixing with extrapolation undoes
    // the trust-region cap).
    //
    // See vibeqc/quadratic_scf.hpp for the kernel. Same option
    // semantics as PeriodicRHFOptions::quadratic_fallback_*; the
    // periodic side shipped first (Phase C1c-1 / C1c-2) and the
    // molecular side mirrors it here.
    int quadratic_fallback_iter = 0;
    double quadratic_fallback_shift = 0.1;
    double quadratic_fallback_max_step = 0.1;

    // Phase D2c — second-order ("Newton") convergence. Once the SCF
    // commutator norm ‖F D S − S D F‖_F drops below
    // ``newton_threshold``, switch from "diagonalize F" to a full
    // Newton step on the orbital-rotation manifold (see
    // cpp/include/vibeqc/newton.hpp). The Newton step solves the
    // *full* orbital Hessian via preconditioned CG (one Fock build
    // per CG iter, ~5–15 iters typical), giving quadratic convergence
    // in the asymptotic regime.
    //
    // ``newton_threshold = 0.0`` (default) disables Newton — SCF
    // behaves exactly as before. A typical "production" setting is
    // ``scf_accelerator = EDIIS_DIIS`` (warm-up) +
    // ``newton_threshold = 1.0`` (Fischer-Almlöf 1992 convention) —
    // EDIIS+DIIS handles iterations 1–N, then Newton closes out the
    // last 3–5 iters with quadratic convergence.
    //
    // Newton is skipped during the C1c quadratic-fallback phase
    // (``quadratic_fallback_iter > 0`` && ``iter > ...``); the two
    // are alternative second-order schemes and shouldn't run together.
    double newton_threshold = 0.0;
    NewtonOptions newton_opts = {};

    // Phase D2d — approximate SOSCF (Neese 2000). Once the SCF
    // commutator norm ‖F D S − S D F‖_F drops below
    // ``soscf_threshold``, switch from "diagonalize F" to a step
    // computed by the augmented-Hessian eigsolve with diagonal-
    // dominant Hessian (see cpp/include/vibeqc/soscf.hpp). Distinct
    // from D2c Newton (full Hessian, expensive) and from D2c's older
    // single-step quadratic_step (no AH eigsolve). SOSCF is cheaper
    // per step than Newton (no Fock build per CG iter) but linearly
    // convergent rather than quadratically; useful when the Newton
    // matvec is too expensive or unstable.
    //
    // ``soscf_threshold = 0.0`` (default) disables SOSCF. Set it > 0 to
    // switch from first-order DIIS to the second-order SOSCF finalizer
    // (now a quasi-Newton L-BFGS accelerator; see cpp/src/soscf.cpp) once
    // the commutator-error norm drops below it. Left off by default: SOSCF
    // does not robustly accelerate every system (e.g. it stalls on porphine
    // RHF/STO-3G), so it is opt-in rather than a default. Newton + SOSCF are
    // mutually exclusive; if both thresholds are nonzero Newton wins.
    double soscf_threshold = 0.0;
    SOSCFOptions soscf_opts = {};

    // Phase D2e — TRAH (trust-region augmented Hessian; Helmich-Paris
    // 2022). Like Newton (D2c) — full orbital Hessian via preconditioned
    // CG — but with an adaptive trust radius driven by Powell's ρ test
    // (actual / model-predicted ΔE). Useful when Newton's fixed step-
    // cap is too rigid; the trust radius shrinks when steps overshoot
    // and expands when they're consistently undersized.
    //
    // Mutual exclusion with Newton + SOSCF (when more than one
    // threshold is set, the priority is quadratic > Newton > TRAH >
    // SOSCF). ``trah_threshold = 0.0`` (default) disables TRAH.
    double trah_threshold = 0.0;
    TRAHOptions trah_opts = {};

    // Deterministic SCF restart via randomized orbital rotations.
    // See cpp/include/vibeqc/scf_restart.hpp.
    SCFRestartOptions restart_opts = {};

    // Density fitting / resolution-of-the-identity. When ``density_fit
    // = true``, the RHF Fock build replaces the four-index ERI tensor
    // with the J / K factorisation
    //   (μν|λσ) ≈ Σ_P B^P_{μν} B^P_{λσ}
    // (Whitten 1973, Dunlap-Connolly-Sabin 1979, Eichkorn-Treutler-Öhm-
    // Häser-Ahlrichs 1995). See cpp/include/vibeqc/df.hpp for the math
    // and the bundled JKfit families.
    //
    // ``aux_basis`` is a libint-recognised auxiliary basis name (e.g.
    // "def2-svp-jk", "cc-pvtz-jkfit"). Empty + ``density_fit=true``
    // raises with a clear error directing the user at
    // ``vibeqc.default_aux_basis_for(orbital_basis_name, kind="jk")``
    // for autodetection on the Python side.
    bool density_fit = false;
    std::string aux_basis;

    // Direct SCF Fock-build mode. When ``scf_mode = SCFMode::DIRECT``
    // (or ``AUTO`` and the basis is larger than
    // ``scf_mode_auto_threshold`` functions), the four-index in-core
    // ERI tensor is replaced by on-the-fly libint quartet evaluation
    // screened by the strict Cauchy-Schwarz bound. Memory drops from
    // O(n_bf⁴) to O(n_shells² + n_bf²); CPU time on the ~50-atom /
    // def2-SVP regime drops by 2–5× because the in-core path eats
    // cache. See cpp/include/vibeqc/jk_direct.hpp for the kernel.
    //
    // Orthogonal to ``density_fit`` + ``cosx`` — when either of those
    // is true, DF / RIJCOSX kernels supersede the four-index path and
    // this field is ignored.
    //
    // ``schwarz_threshold`` (default 1e-10, ORCA convention) is the
    // per-quartet skip bound: |⟨μν|λσ⟩| ≤ Q_μν Q_λσ. Tighter is
    // stricter (fewer skips); 1e-8 is OK for energies, gradients want
    // 1e-12. Ignored when ``scf_mode = CONVENTIONAL``.
    //
    // ``scf_mode_auto_threshold`` (default 140 basis functions) is
    // the AUTO cutoff: below it the in-core path is faster (no per-
    // iter integral re-evaluation); above it the in-core tensor is
    // multi-GB and the direct path wins. Calibrated against
    // benchmarks/orca_vs_vibeqc_speed.md (BUG 87 — threshold lowered
    // from 200 to 140; STO-3G on ~38-atom systems is ~134 BF and the
    // in-core path is 17× faster there; cc-pVQZ on 3 atoms is 165 BF
    // and the in-core path uses 3.7 GB — 140 splits the difference).
    SCFMode scf_mode = SCFMode::AUTO;
    int scf_mode_auto_threshold = 140;
    double schwarz_threshold = 1e-10;

    // Incremental (Almlöf) ΔP Fock build. When ``incremental_fock =
    // true`` and the active path is DirectJKBuilder, the builder
    // caches D_prev + G_2e_prev across SCF iterations; each call
    // computes ΔD = D − D_prev and returns G_prev + G_2e[ΔD].
    // Because the per-shell density envelope inside the Schwarz
    // screen sees |ΔD| rather than |D|, converged blocks vanish
    // from the work as the SCF progresses — typical 3–10× total-
    // SCF speedup at the n-hexadecane / def2-SVP scale.
    //
    // Full rebuild every ``incremental_fock_reset_freq`` iterations
    // to bound floating-point drift (the Almlöf incremental-build
    // reset frequency; 8 is the default). Set to 1 to effectively
    // disable the incremental path while keeping the option set.
    //
    // No-op when scf_mode resolves to CONVENTIONAL or when
    // density_fit / cosx is enabled (those paths have their own
    // amortisation).
    //
    // BUG 87: enabled by default — the incremental ΔD cache gives
    // 3–10× SCF speedup on the direct path with no accuracy penalty.
    bool incremental_fock = true;
    int incremental_fock_reset_freq = 8;

    // Two-phase Schwarz threshold for direct SCF. Start with the
    // (looser) ``schwarz_threshold_loose`` so the per-shell density
    // envelope catches mid-SCF quartets aggressively; once the SCF
    // gradient norm |F D S − S D F|_F drops below
    // ``schwarz_threshold_tighten_at``, switch to the tight
    // ``schwarz_threshold`` and discard the incremental cache so
    // the converged result hits the user's accuracy budget.
    //
    // Two-phase loose→tight refinement of Schwarz screening (Häser &
    // Ahlrichs, J. Comput. Chem. 10, 104 (1989)) — the loose phase makes
    // the incremental ΔP screening actually trigger in the early
    // iterations where most of the time is spent.
    //
    // Defaults: loose = 1e-7 (ORCA convention), tighten_at = 1e-3
    // (gradient-norm cutoff for the switch). Set
    // ``schwarz_threshold_loose <= schwarz_threshold`` to disable
    // the two-phase coarsening (uniform tight threshold throughout).
    //
    // No-op when scf_mode resolves to CONVENTIONAL.
    double schwarz_threshold_loose = 1e-7;
    double schwarz_threshold_tighten_at = 1e-3;

    // Chain-of-spheres exchange (COSX). When ``cosx = true`` and
    // ``density_fit = true``, the K(D) build is replaced by Neese's
    // 2009 seminumerical algorithm: K is reduced to a sum of rank-1
    // updates over a numerical integration grid, with analytic
    // 1/|r-r_g| Coulomb-attraction integrals evaluated per grid point.
    // This is the "RIJCOSX" path (RI-J + COSX-K). See
    // cpp/include/vibeqc/cosx.hpp for the math + Neese 2009 reference.
    //
    // The grid is built fresh per SCF call from ``cosx_grid``; defaults
    // to the same Treutler-Ahlrichs M4 + Lebedev settings used for the
    // DFT XC grid. A typical COSX setup uses a sparser grid; tune
    // ``cosx_grid`` for the speed-vs-accuracy trade-off.
    bool cosx = false;
    GridOptions cosx_grid = default_cosx_grid_options();

    // COSX variant / grid-level / accuracy threshold (RIJCOSX upgrade,
    // M1). Consulted only when ``cosx == true``. See cosx.hpp for the
    // variant + tier definitions.
    //   cosx_variant    — AUTO (default) resolves STANDARD vs FITTED from
    //                     ``thresh_cosx`` + the orbital-basis cardinality
    //                     (resolve_cosx_variant). STANDARD = Neese 2009
    //                     no-fit K; FITTED = overlap-fitted K.
    //   cosx_grid_level — -1 (default) AUTO: use a 2021 GridX tier with the
    //                     coarse→fine multi-stage progression for normal-
    //                     tolerance SCFs (DZ/TZ→GridX2, QZ→GridX3), and fall
    //                     back to the legacy grid when ``conv_tol_grad`` is
    //                     tighter than the GridX commutator floor (1e-6, see
    //                     ``cosx_use_gridx``). 0 forces the legacy grid;
    //                     1..4 force a GridX tier at any tolerance.
    //   thresh_cosx     — COSX accuracy target driving AUTO variant
    //                     selection (< 1e-6 ⇒ FITTED).
    CosxVariant cosx_variant = CosxVariant::AUTO;
    int cosx_grid_level = -1;
    double thresh_cosx = 1e-6;

    // DFT+U (Dudarev rotationally-invariant). Empty by default — when
    // ``dft_plus_u_sites`` is non-empty, the SCF Fock build adds the
    // per-spin Hubbard +U potential
    //
    //   V_U^A_{mm'} = U_eff (δ_{mm'} / 2 − n^A_l_{mm'}),
    //
    // and the total energy gains
    //   E_U = Σ_A (U_eff / 2) (tr n^A_l − tr (n^A_l)²)
    // (summed over both spins for closed-shell, where n_α = n_β = n_total / 2).
    //
    // ``dft_plus_u_ao_groups`` is the parallel array of AO indices for
    // each site; precomputed once at SCF setup (typically by the Python
    // wrapper from ``ao_group_indices(basis)``) so the SCF loop pays
    // the shell walk zero times. See cpp/include/vibeqc/dft_plus_u.hpp
    // for the full math + spin-convention contract.
    std::vector<HubbardSiteCxx> dft_plus_u_sites;
    std::vector<std::vector<int>> dft_plus_u_ao_groups;

    // Davidson iterative diagonalization for large basis sets.
    bool use_davidson = false;
    DavidsonOptions davidson = {};
    int davidson_min_dim = 100;

    // Restricted-stability VERDICT (issue #144 prerequisite, decision-
    // neutral). After a converged RHF SCF, the lowest eigenvalue of the
    // orbital-rotation Hessian at the returned solution is computed
    // matrix-free (the coupled per-spin UHF machinery evaluated at
    // equal spin blocks, i.e. the union of the Seeger-Pople singlet and
    // triplet sectors; Seeger & Pople, J. Chem. Phys. 66, 3045 (1977),
    // doi:10.1063/1.434318). A negative eigenvalue below -stability_tol
    // means the converged solution is a saddle or an externally
    // unstable restricted root: the result carries
    // internal_instability=true and the output layer warns loudly that
    // the energy is an excited SCF solution.
    //
    // Verdict ONLY — this surface does not rotate, escape, promote to
    // UHF, or search other basins. What to DO with a negative verdict
    // is the maintainer decision recorded in
    // agentic-loop/asks/ask-scf144-restricted-stability-contract-2026-08-27.md;
    // this option group only makes the verdict visible.
    //
    // Default ON (matching UHF/UKS): a silent wrong root is the defect
    // class #144 tracks. Cost when stable: one Davidson solve, typically
    // 10-40 J + K builds. Set stability_check=False for legacy
    // single-SCF behaviour.
    bool stability_check = true;
    // Distinguishes the automatic default from an explicit user request
    // so unsupported routes (e.g. +U) can skip silently by default but
    // fail closed when explicitly requested (same contract as UKS).
    bool stability_check_explicit = false;
    // Instability threshold: unstable when lambda_min < -stability_tol.
    double stability_tol = 1e-4;
    // Davidson matvec cap for the stability solve.
    int stability_davidson_max_iter = 60;
};

// Per-iteration record populated during the SCF loop. Captured for every
// iteration (even after convergence) so Python can format a log or a trace
// plot without the C++ side owning any output format.
struct SCFIteration {
    int iter = 0;
    double energy = 0.0;         // Total energy at this iteration (Hartree)
    double delta_e = 0.0;        // E[k] - E[k-1]; 0 on iter 1
    double grad_norm = 0.0;      // ||F D S - S D F||_Frobenius
    int diis_subspace = 0;       // DIIS history size used this iter (0 if off)
    int newton_cg_iter = 0;       // Newton CG iters this step (0 unless Newton)
    double trah_level_shift = 0.0;  // TRAH level shift λ this step (0 unless
                                    // TRAH took a trust-region boundary step;
                                    // > 0 ⇒ on the ‖κ‖ = Δ boundary)
    double wall_s = 0.0;  // Measured wall-clock seconds charged to this
                          // iteration: time since the previous iteration's
                          // trace record (Fock build + energy + DIIS +
                          // diagonalization all covered). Exactly 0.0 means
                          // UNMEASURED (a driver that did not fill it) —
                          // renderers must show "--" for 0.0, never render
                          // it as a "0.000 s" measurement (BUG87-A: the
                          // shipped .perf printed 0.000 on every iteration
                          // of a 2258 s SCF).
    // Which Fock extrapolation actually reached the SCF this iteration
    // (GitLab #682): 0 = none (below diis_start_iter, accelerator off, or a
    // second-order / quadratic / hold phase), 1 = Pulay DIIS (including the
    // adaptive-depth R_CDIIS / AD_CDIIS policies), 2 = KDIIS, 3 = EDIIS,
    // 4 = ADIIS. The EDIIS+DIIS and ADIIS+DIIS hybrids record the branch
    // their switch metric selected, so the citation router can credit the
    // accelerators the run executed rather than the one it was configured
    // with. Last member so positional aggregate initialisation stays valid.
    int accelerator_step = 0;
};

struct RHFResult {
    OpenTrustRegionReport opentrustregion;
    std::optional<BasisSet> restart_basis;
    std::optional<GuessSelection> guess_selection;
    double energy = 0.0;            // total HF energy (Hartree)
    double e_electronic = 0.0;      // = energy - nuclear_repulsion
    // True whenever the one-electron Hamiltonian included an ECP operator,
    // including zero-core model potentials for which ecp_total_ncore is zero.
    bool ecp_operator_applied = false;
    // True only when the authoritative high-level molecular driver built the
    // one-electron Hamiltonian and stamped the exact ECP route below. False
    // means the provenance is unknown (as for run_rhf_scf_with_jk), not that
    // the caller-supplied Hamiltonian is known to be all-electron.
    bool ecp_provenance_verified = false;
    // Exact XML-library input that produced this reference. Empty for the
    // all-electron and inline-primitive routes. The library is normalized to
    // the operative name ("ecp10mdf" when the options field was empty).
    std::vector<ECPCenter> ecp_xml_centers;
    std::string ecp_xml_library;
    // Exact inline-primitive input that produced this reference. All three
    // fields are empty for the all-electron and XML-library routes. Blocks and
    // centers are paired by index; effective charges remain in molecule-atom
    // order so derivative callers can reproduce the same Hamiltonian.
    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_primitive_centers;
    std::vector<double> ecp_effective_charges;
    // Number of physical core electrons replaced by the ECP in the
    // molecular SCF that produced this reference. Meaningful only when
    // ecp_provenance_verified is true; zero also covers all-electron and
    // zero-core model-potential references.
    int ecp_total_ncore = 0;
    // Dudarev (1998) DFT+U contribution to the total energy. Zero when
    // the RHFOptions ``dft_plus_u_sites`` vector is empty (the common
    // case). Summed over both spins for closed-shell — i.e.
    // ``energy = e_electronic + e_nuclear + e_dft_plus_u`` with the
    // electronic energy computed from the HF-only part of the Fock
    // matrix (Hcore + G, *without* V_U) to avoid the standard +U
    // double-counting trap.
    double e_dft_plus_u = 0.0;
    // Restricted-stability verdict (see RHFOptions::stability_check).
    // ``stability_checked`` is true when the analysis ran on the final
    // converged solution; ``stability_analysis_converged`` reports the
    // Davidson solve itself; ``stability_eigenvalue`` is the lowest
    // orbital-rotation Hessian eigenvalue at the returned solution
    // (>= -stability_tol means stable); ``internal_instability`` is the
    // negative verdict passed to the output layer.
    bool stability_checked = false;
    bool stability_analysis_converged = false;
    double stability_eigenvalue = 0.0;
    bool internal_instability = false;
    // Wall / CPU seconds spent in the post-convergence stability phase.
    double stability_wall_s = 0.0;
    double stability_cpu_s = 0.0;
    int n_iter = 0;
    bool converged = false;
    Eigen::VectorXd mo_energies;    // ascending (Hartree)
    Eigen::MatrixXd mo_coeffs;      // columns = MOs, rows = AOs
    Eigen::MatrixXd density;        // D = 2 C_occ C_occ^T
    Eigen::MatrixXd fock;           // converged F (includes V_U if +U is on)
    std::vector<SCFIteration> scf_trace;  // one entry per SCF iteration
};

RHFResult run_rhf(const Molecule& mol,
                  const BasisSet& basis,
                  const RHFOptions& options = {});

// Lower-level closed-shell SCF entry point. Takes the AO overlap S,
// the one-electron Hamiltonian Hcore, the (system-dependent) nuclear
// repulsion E_nuc, and a JKBuilder for the two-electron Fock piece —
// then runs the standard SCF iteration (canonical orthogonalisation,
// DIIS, optional damping / level-shift / quadratic fallback,
// post-convergence Fock rebuild). The molecular ``run_rhf`` above is
// a thin wrapper around this function: it builds Hcore + E_nuc from
// the molecular integrals and JKBuilder via the
// density_fit / cosx options.
//
// This is the entry point that lets callers drive *any* closed-shell
// RHF / Γ-only periodic-RHF / model-Hamiltonian SCF through the same
// code path. The periodic-Γ caller builds Hcore from
// ``compute_*_lattice`` + Γ-fold, E_nuc from
// ``nuclear_repulsion_per_cell``, and the JKBuilder from
// ``make_periodic_gamma_jk_builder``; everything else reuses the
// molecular SCF infrastructure (DIIS, level shift, quadratic
// fallback, …) without changes.
//
// ``initial_density`` is optional. When non-empty (rows == basis size)
// it seeds D directly (used for SAD / external-density restart);
// when empty it falls back to diagonalising Hcore.
//
// Notes that don't apply here vs. ``run_rhf``:
//   * No SAD path — the SAD construction in ``run_rhf`` reads atomic
//     densities from the Molecule, which doesn't translate to an
//     arbitrary Hcore caller. Pass a SAD density via
//     ``initial_density`` if needed.
//   * No ECP integration — Hcore is the caller's responsibility. ECP fields
//     on ``options`` cannot certify an arbitrary matrix, so the returned
//     result keeps ``ecp_provenance_verified == false`` and carries no ECP
//     payload. Only the high-level molecular driver stamps exact provenance.
//     Build it via ``compute_ecp_one_electron`` for molecular
//     systems, or ``compute_*_lattice`` + Γ-fold for periodic-Γ.
RHFResult run_rhf_scf_with_jk(const BasisSet& basis,
                              int n_electrons,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const class JKBuilder& jk,
                              const RHFOptions& options = {},
                              const Eigen::MatrixXd& initial_density = {},
                              const Molecule* guess_molecule = nullptr,
                              const GuessSelection* prepared_guess = nullptr);

}  // namespace vibeqc
