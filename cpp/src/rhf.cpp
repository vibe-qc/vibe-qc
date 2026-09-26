#include "vibeqc/rhf.hpp"
#include "vibeqc/orbital_scf.hpp"

#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "vibeqc/diis.hpp"
#include "vibeqc/dynamic_damping.hpp"
#include "vibeqc/ediis.hpp"
#include "vibeqc/diagnostics.hpp"
#include "vibeqc/fock.hpp"
#include "vibeqc/kdiis.hpp"
#include "vibeqc/level_shift.hpp"
#include "vibeqc/newton.hpp"
#include "vibeqc/soscf.hpp"
#include "vibeqc/trah.hpp"
#include "vibeqc/cosx.hpp"
#include "vibeqc/cosx_staged.hpp"
#include "vibeqc/grid.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/jk_builder.hpp"
#include "vibeqc/linear_dependence.hpp"
#include "vibeqc/quadratic_scf.hpp"
#include "vibeqc/scf_convergence.hpp"
#include "vibeqc/scf_restart.hpp"
#include "vibeqc/scf_mixing.hpp"
#include "vibeqc/degenerate_frontier.hpp"

namespace vibeqc {

namespace {

// D = 2 * C_occ * C_occ^T  (total closed-shell density, trace D*S = n_elec)
Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int nocc) {
    const auto Cocc = C.leftCols(nocc);
    return 2.0 * Cocc * Cocc.transpose();
}

}  // namespace

RHFResult run_rhf(const Molecule& mol,
                  const BasisSet& basis,
                  const RHFOptions& opts) {
    validate_orbital_optimizer(opts);
    validate_initial_guess(opts.initial_guess);
    validate_guess_ecp(opts.initial_guess, molecular_guess_ecp_context(opts), &mol);
    validate_scf_max_iter(opts.max_iter, "run_rhf");
    if (mol.n_electrons() % 2 != 0) {
        throw std::invalid_argument(
            "run_rhf: closed-shell RHF requires an even electron count, got "
            + std::to_string(mol.n_electrons()));
    }
    if (mol.multiplicity() != 1) {
        throw std::invalid_argument(
            "run_rhf: closed-shell RHF requires multiplicity = 1, got "
            + std::to_string(mol.multiplicity()));
    }
    validate_fraction_01("run_rhf: damping", opts.damping);
    validate_fraction_01("run_rhf: fock_mixing", opts.fock_mixing);

    const auto ecp_input = validate_molecular_ecp_dispatch(
        opts.ecp_centers, opts.ecp_library, opts.ecp_primitive_blocks,
        opts.ecp_primitive_centers, opts.ecp_effective_charges,
        opts.ecp_total_ncore, "run_rhf");

    // ---- One-electron integrals --------------------------------------------
    const Eigen::MatrixXd S = compute_overlap(basis);
    const Eigen::MatrixXd T = compute_kinetic(basis);
    const auto ecp_h = ecp_input == MolecularECPInput::INLINE_PRIMITIVES
        ? compute_ecp_one_electron_from_primitives(
              basis, mol, opts.ecp_primitive_centers,
              opts.ecp_primitive_blocks, opts.ecp_effective_charges,
              opts.ecp_total_ncore)
        : compute_ecp_one_electron(
              basis, mol, opts.ecp_centers, opts.ecp_library);
    const Eigen::MatrixXd Hcore = T + ecp_h.V + ecp_h.V_ecp;
    const double E_nuc = ecp_h.E_nuc;

    // Effective electron count: subtract ECP-replaced core electrons.
    const int n_elec = mol.n_electrons() - ecp_h.total_ncore;
    if (n_elec < 0) {
        throw std::invalid_argument(
            "run_rhf: ECP cores remove more electrons ("
            + std::to_string(ecp_h.total_ncore)
            + ") than the molecule has ("
            + std::to_string(mol.n_electrons())
            + "). Molecule.n_electrons() must be the full physical electron "
              "count; the SCF subtracts the ECP core itself.");
    }
    const auto with_ecp_provenance =
        [ncore = ecp_h.total_ncore,
         operator_applied = ecp_input != MolecularECPInput::NONE,
         xml_centers = ecp_input == MolecularECPInput::XML_LIBRARY
             ? opts.ecp_centers : std::vector<ECPCenter>{},
         xml_library = ecp_input == MolecularECPInput::XML_LIBRARY
             ? (opts.ecp_library.empty() ? std::string("ecp10mdf")
                                         : opts.ecp_library)
             : std::string{},
         primitive_blocks = ecp_input == MolecularECPInput::INLINE_PRIMITIVES
             ? opts.ecp_primitive_blocks : std::vector<ECPPrimitiveBlock>{},
         primitive_centers = ecp_input == MolecularECPInput::INLINE_PRIMITIVES
             ? opts.ecp_primitive_centers
             : std::vector<std::array<double, 3>>{},
         effective_charges = ecp_input == MolecularECPInput::INLINE_PRIMITIVES
             ? opts.ecp_effective_charges : std::vector<double>{}](
            RHFResult result) {
            result.ecp_operator_applied = operator_applied;
            result.ecp_provenance_verified = true;
            result.ecp_xml_centers = xml_centers;
            result.ecp_xml_library = xml_library;
            result.ecp_primitive_blocks = primitive_blocks;
            result.ecp_primitive_centers = primitive_centers;
            result.ecp_effective_charges = effective_charges;
            result.ecp_total_ncore = ncore;
            return result;
        };

    // ---- Two-electron infrastructure ---------------------------------------
    // The SCF loop talks to a single ``JKBuilder`` interface; concrete
    // implementations cover the three paths (direct 4-index, RIJK, and
    // RIJCOSX). See cpp/include/vibeqc/jk_builder.hpp for the abstraction.
    std::unique_ptr<BasisSet> aux;     // owned for the lifetime of jk
    Grid cosx_grid_built;              // populated only on the RIJCOSX path
    std::vector<GridOptions> cosx_stages;  // COSX grid progression (GridX)
    CosxVariant cosx_var = CosxVariant::FITTED;
    bool cosx_multistage = false;      // true on the multi-stage GridX path
    std::unique_ptr<JKBuilder> jk;
    if (opts.density_fit) {
        if (opts.aux_basis.empty()) {
            throw std::invalid_argument(
                "run_rhf: density_fit=true requires aux_basis to be set "
                "(e.g. \"def2-svp-jk\"). Use "
                "vibeqc.default_aux_basis_for(orbital_basis_name, kind=\"jk\") "
                "for autodetection.");
        }
        aux = std::make_unique<BasisSet>(mol, opts.aux_basis);
        if (opts.cosx) {
            const int cosx_card =
                cosx_basis_cardinality_from_name(basis.name());
            cosx_var = resolve_cosx_variant(
                opts.cosx_variant, opts.thresh_cosx, cosx_card,
                opts.cosx_grid_level);
            if (!cosx_use_gridx(opts.cosx_grid_level, opts.conv_tol_grad)) {
                // Legacy single COSX grid: explicit opt-out
                // (cosx_grid_level==0) or the auto fallback for a
                // conv_tol_grad tighter than the GridX commutator floor.
                cosx_grid_built = build_grid(
                    mol, resolve_cosx_grid_options(opts.cosx_grid, 0,
                                                   cosx_card));
            } else {
                // GridX multi-stage progression (P2). The stage-0 (coarse)
                // grid also seeds the initial guess; finer stages are built
                // on the fly in run_cosx_staged_scf.
                cosx_stages = cosx_grid_stages_for_level(
                    resolve_cosx_grid_level(opts.cosx_grid_level, cosx_card));
                cosx_grid_built = build_grid(mol, cosx_stages.front());
                cosx_multistage = true;
            }
            jk = make_cosx_jk_builder(basis, *aux, cosx_grid_built, cosx_var);
        } else {
            jk = make_df_jk_builder(basis, *aux);
        }
    } else {
        const SCFMode mode = resolve_scf_mode(
            opts.scf_mode, static_cast<int>(basis.nbasis()),
            opts.scf_mode_auto_threshold);
        if (mode == SCFMode::DIRECT) {
            // Two-phase Schwarz: open at the loose threshold, the
            // SCF inner loop tightens once grad-norm crosses the
            // ``schwarz_threshold_tighten_at`` cutoff. Falls back
            // to the tight threshold immediately when the user has
            // set loose ≤ tight (disables coarsening).
            const double initial_thr =
                (opts.schwarz_threshold_loose > opts.schwarz_threshold)
                    ? opts.schwarz_threshold_loose
                    : opts.schwarz_threshold;
            jk = make_direct_jk_builder(
                basis, initial_thr,
                opts.incremental_fock, opts.incremental_fock_reset_freq);
        } else {
            jk = make_four_index_jk_builder(basis);
        }
    }

    auto prepared = prepare_closed_guess(
        &mol, basis, n_elec / 2, opts.initial_guess, S, Hcore, *jk,
        {}, opts.read_density, opts.linear_dep_threshold, nullptr, molecular_guess_ecp_context(opts));
    const auto& guess_D = prepared.density;

    if (cosx_multistage) {
        // Step through the coarse → fine COSX grid stages, carrying the
        // density forward; the final stage converges to opts' tolerance.
        Eigen::MatrixXd D = guess_D;
        auto seg = [&](const JKBuilder& seg_jk,
                       const RHFOptions& so) -> RHFResult {
            RHFResult r = run_rhf_scf_with_jk(basis, n_elec, S, Hcore, E_nuc,
                                              seg_jk, so, D, &mol, &prepared.selection);
            prepared.selection.transport = InitialGuess::READ;
            D = r.density;
            return r;
        };
        return with_ecp_provenance(run_cosx_staged_scf<RHFResult>(
            mol, cosx_stages, opts, *jk, seg));
    }
    return with_ecp_provenance(run_rhf_scf_with_jk(
        basis, n_elec, S, Hcore, E_nuc, *jk, opts, guess_D, &mol, &prepared.selection));
}

RHFResult run_rhf_scf_with_jk(const BasisSet& basis,
                              int n_electrons,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const JKBuilder& jk,
                              const RHFOptions& opts,
                              const Eigen::MatrixXd& initial_density,
                              const Molecule* guess_molecule,
                              const GuessSelection* prepared_guess) {
    validate_initial_guess(opts.initial_guess);
    validate_guess_ecp(opts.initial_guess, molecular_guess_ecp_context(opts), guess_molecule);
    validate_orbital_optimizer(opts);
    validate_scf_max_iter(opts.max_iter, "run_rhf_scf_with_jk");
    if (n_electrons < 0 || n_electrons % 2 != 0) {
        throw std::invalid_argument("closed-shell SCF requires a nonnegative even electron count");
    }
    auto constructed = prepare_closed_guess(
        guess_molecule, basis, n_electrons / 2, opts.initial_guess, S, Hcore,
        jk, initial_density, opts.read_density, opts.linear_dep_threshold,
        prepared_guess, molecular_guess_ecp_context(opts));


    validate_fraction_01("run_rhf_scf_with_jk: damping", opts.damping);
    validate_fraction_01(
        "run_rhf_scf_with_jk: fock_mixing", opts.fock_mixing);
    if (S.rows() != Hcore.rows() || S.cols() != Hcore.cols()
        || S.rows() != static_cast<Eigen::Index>(basis.nbasis())) {
        throw std::invalid_argument(
            "run_rhf_scf_with_jk: S / Hcore shapes do not match "
            "basis.nbasis()");
    }
    const int nocc = n_electrons / 2;

    // Canonical orthogonalization: X is (n_basis, n_kept) where
    // n_kept <= n_basis; X^T S X = I. For well-conditioned S this is
    // equivalent to plain symmetric orthogonalization. For near-
    // linearly-dependent S, the near-null subspace is projected out
    // before the SCF starts so the Fock diagonalization stays stable.
    // Loewdin metric and AO->atom map for the degenerate-frontier tie-break
    // (#210). Built once per SCF; the tie-break runs only on an iteration
    // whose frontier gap is below tolerance, which for an ordinary molecule
    // is never. RHF needs this at least as much as UHF: it has no stability
    // check, so nothing downstream masks a basin picked at random.
    const Eigen::MatrixXd s_sqrt_ao = symmetric_sqrt(S);
    std::vector<int> ao_atom_index;
    {
        ao_atom_index.reserve(basis.nbasis());
        const auto shell_list = basis.shells();
        for (std::size_t shell = 0; shell < shell_list.size(); ++shell) {
            const int atom = basis.shell_atom_index(shell);
            const int l = shell_list[shell].l;
            const std::size_t n_functions = shell_list[shell].pure
                ? static_cast<std::size_t>(2 * l + 1)
                : static_cast<std::size_t>((l + 1) * (l + 2) / 2);
            for (std::size_t f = 0; f < n_functions; ++f) {
                ao_atom_index.push_back(atom);
            }
        }
    }

    const auto orth = canonical_orthogonalizer(S, opts.linear_dep_threshold);
    if (orth.n_kept == 0) {
        throw std::runtime_error(
            "RHF: AO basis has no non-null directions above the "
            "linear-dependence threshold (threshold = "
            + std::to_string(opts.linear_dep_threshold) + ")");
    }
    if (nocc > orth.n_kept) {
        throw std::runtime_error(
            "RHF: canonical orthogonalization dropped too many basis "
            "directions for this electron count (n_occ = "
            + std::to_string(nocc) + ", n_kept = "
            + std::to_string(orth.n_kept) + "); loosen "
            "linear_dep_threshold or pick a less redundant basis");
    }
    const Eigen::MatrixXd& X = orth.X;

    // ---- Hcore initial guess -----------------------------------------------
    const bool use_davidson_here =
        opts.use_davidson && orth.n_kept >= opts.davidson_min_dim;
    DavidsonOptions dav_opts = opts.davidson;
    if (use_davidson_here) {
        if (dav_opts.n_eig == 0) dav_opts.n_eig = orth.n_kept;
    }
    Eigen::MatrixXd dav_guess_ortho;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fock_solver;
    auto diagonalize_fock = [&](const Eigen::MatrixXd& F)
        -> std::pair<Eigen::MatrixXd, Eigen::VectorXd> {
        const Eigen::MatrixXd Fp = X.transpose() * F * X;
        if (use_davidson_here) {
            if (dav_guess_ortho.size() != 0) {
                dav_opts.guess_vectors = dav_guess_ortho;
            }
            DavidsonResult dres = davidson_solve(Fp, dav_opts);
            if (!dres.converged) {
                throw std::runtime_error(
                    "RHF: Davidson diagonalization did not converge after "
                    + std::to_string(dres.n_iter) + " iterations");
            }
            dav_guess_ortho = dres.eigenvectors;
            return {X * dres.eigenvectors, dres.eigenvalues};
        }
        fock_solver.compute(Fp);
        if (fock_solver.info() != Eigen::Success) {
            throw std::runtime_error("RHF: Fock diagonalization failed");
        }
        return {X * fock_solver.eigenvectors(), fock_solver.eigenvalues()};
    };

    auto [C0, eps0] = diagonalize_fock(Hcore);
    Eigen::MatrixXd D;
    if (constructed.density.size() != 0) {
        if (constructed.density.rows() != S.rows()
            || constructed.density.cols() != S.cols()) {
            throw std::invalid_argument(
                "run_rhf_scf_with_jk: initial_density shape does not "
                "match S");
        }
        D = constructed.density;
    } else {
        D = build_density(C0, nocc);
    }
    Eigen::MatrixXd D_prev = D;

    // Track MO basis between iterations so the C1c Newton step has a
    // current MO frame to operate in. Updated each iteration after
    // either diagonalisation or the quadratic step.
    Eigen::MatrixXd C_prev_mo = C0;
    Eigen::VectorXd eps_prev_mo = eps0;

    // ---- SCF iterations ---------------------------------------------------
    RHFResult result;
    result.guess_selection = constructed.selection;
    result.restart_basis = basis;
    result.mo_energies = eps0;
    result.mo_coeffs = C0;

    if (opts.orbital_optimizer == "opentrustregion") {
        const auto run = run_orbital_scf(opts, X, S, Hcore, E_nuc, jk,
            D, {}, nocc, nocc, true, 1.0);
        assign_orbital_restricted(result, run, E_nuc);
        return result;
    }

    // ORCA NOITER compatibility: evaluate the initial density once, but do
    // not take an SCF update or append an iteration trace entry.
    if (opts.max_iter == 0) {
        const Eigen::MatrixXd G = jk.build_g_rhf(D);
        Eigen::MatrixXd F = Hcore + G;
        const double E_elec =
            0.5 * (D.array() * (Hcore + F).array()).sum();
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                0.5 * D, S);
            F += vu.V;
            e_dft_plus_u = 2.0 * vu.energy;
        }
        auto [C, eps] = diagonalize_fock(F);
        result.mo_energies = eps;
        result.mo_coeffs = C;
        result.density = D;
        result.fock = F;
        result.energy = E_elec + E_nuc + e_dft_plus_u;
        result.e_electronic = E_elec;
        result.e_dft_plus_u = e_dft_plus_u;
        return result;
    }

    DIIS diis = make_diis(opts);
    EDIIS ediis(opts.diis_subspace_size);
    ADIIS adiis(opts.diis_subspace_size);
    KDIIS kdiis(opts.diis_subspace_size);
    SOSCF soscf(opts.soscf_opts);  // stateful L-BFGS SOSCF accelerator

    // TRAH (D2e) state: trust radius adapted across iterations from
    // Powell's ρ test. ``trah_predicted_decrease`` is the model
    // prediction from the PREVIOUS TRAH step; we test ρ against it
    // when the current iteration's actual ΔE is known.
    double trah_trust_radius = opts.trah_opts.initial_trust_radius;
    double trah_predicted_decrease = 0.0;
    double trah_kappa_norm = 0.0;   // ‖κ‖ of the previous TRAH step
    bool trah_active_prev = false;

    // Dynamic-damping state (Zerner-Hehenberger 1979). When
    // opts.dynamic_damping is false, ``current_damping`` stays equal to
    // opts.damping for the entire run (back-compat: identical to the
    // pre-dynamic-damping behaviour). When true, it adapts based on the
    // sign + magnitude of the energy decrease per iteration.
    double current_damping = opts.damping;
    bool have_prev_E = false;

    // An incremental difference-Fock energy is not comparable with the
    // nonincremental tight-screened full-density map.  When its gradient
    // first passes, force one non-comparable full-build row, then require the
    // following full-build row to satisfy both ordinary tolerances (IID 418).
    bool full_fock_validation_pending = false;
    int fock_map_epoch_start_iter = 1;

    double E_prev = 0.0;
    Eigen::MatrixXd F_prev_mixed;
    bool have_prev_fock = false;

    // Auto-level-shift-on-oscillation state shared by the molecular drivers.
    // ``oscillation_engaged`` flips true once the SCF energy history shows
    // oscillatory behaviour (frequent dE sign flips). Once engaged, a
    // persistent Saunders-Hillier shift is applied for the rest of the run,
    // and the DIIS history is cleared so extrapolation restarts from the
    // shifted iterates. The shift value and detector parameters are read
    // from ``opts.auto_level_shift_*``. Inactive when
    // ``opts.auto_level_shift_on_oscillation`` is false.
    bool oscillation_engaged = false;
    double effective_level_shift = opts.level_shift;
    int effective_warmup = opts.level_shift_warmup_cycles;

    // Deterministic orbital-rotation restart state (BUG 64).
    int num_restarts = 0;
    int stall_iters = 0;
    double best_grad_norm = 1e300;
    double best_energy = 0.0;
    Eigen::MatrixXd best_C = C0;
    Eigen::VectorXd best_eps = eps0;
    Eigen::MatrixXd best_D = D;

    // Two-phase direct-SCF Schwarz tightening (loose→tight refinement
    // of Schwarz screening; Häser & Ahlrichs, J. Comput. Chem. 10,
    // 104 (1989)). The builder is constructed at the loose
    // threshold; this flag flips once the gradient norm drops below
    // ``schwarz_threshold_tighten_at`` and we ask the builder to
    // tighten + discard its incremental cache. No-op when scf_mode
    // resolved to CONVENTIONAL (the JKBuilder there is a fixed-cost
    // 4-index ERI tensor; ``set_schwarz_threshold`` is a no-op for
    // those concretes).
    const bool schwarz_two_phase =
        opts.schwarz_threshold_loose > opts.schwarz_threshold;
    bool schwarz_tightened = !schwarz_two_phase;
    // Per-iteration wall clock for the SCFIteration trace
    // (BUG87-A truthful-timers fix): each trace row is charged
    // with the wall time since the previous row was recorded.
    auto t_iter_prev = std::chrono::steady_clock::now();
    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        const bool validating_first_full_build =
            full_fock_validation_pending;
        full_fock_validation_pending = false;

        // Quadratic-fallback phase: once iter > quadratic_fallback_iter we
        // replace the diagonalize-F update with a Newton step in MO space
        // (κ_{ai} = -F_{ai}^MO / (ε_a − ε_i + λ); C_new = C_prev · exp(κ);
        // D_new = 2 · C_occ · C_occ^T). DIIS extrapolation is skipped in
        // that phase (the Newton step has its own update mechanism, and
        // mixing with DIIS undoes the trust-region cap). Damping is also
        // skipped — the trust-region cap on ‖κ‖_F is the analogous safety.
        const bool in_quadratic_phase =
            opts.quadratic_fallback_iter > 0
            && iter > opts.quadratic_fallback_iter;

        // Damp the density used to build F (after the first iteration). We
        // apply damping only until DIIS takes over — once DIIS is driving
        // the extrapolation it does a better job of handling oscillations
        // and damping just slows it down.
        const bool diis_active =
            opts.use_diis && iter >= opts.diis_start_iter
            && !in_quadratic_phase;
        const Eigen::MatrixXd D_used =
            (iter == 1 || current_damping == 0.0 || diis_active
             || in_quadratic_phase)
                ? D
                : (current_damping * D_prev + (1.0 - current_damping) * D);

        const Eigen::MatrixXd G = jk.build_g_rhf(D_used);
        Eigen::MatrixXd F = Hcore + G;

        // Electronic energy:  E_elec = (1/2) tr( D_used * (Hcore + F) )
        // (Evaluated at the non-extrapolated F so the reported energy is
        // consistent with the density that produced it.)
        //
        // Computed from the *HF-only* F (Hcore + G); the optional
        // Dudarev +U contribution is added separately to E_total below
        // — see the note in [`cpp/include/vibeqc/dft_plus_u.hpp`] on
        // why ``(1/2) tr(D V_U) ≠ E_U`` and we must avoid letting V_U
        // leak into the (Hcore + F) trace.
        const double E_elec =
            0.5 * (D_used.array() * (Hcore + F).array()).sum();

        // Dudarev +U Fock contribution. Per-spin convention: pass
        // P_σ = D_used / 2 (closed-shell ⇒ each spin sees half the
        // total density). V_U is identical for both spins by
        // closed-shell symmetry, so we add it to the single RHF Fock
        // once; the energy doubles to sum over both spins.
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                0.5 * D_used, S);
            F += vu.V;
            e_dft_plus_u = 2.0 * vu.energy;
        }

        const double E_total = E_elec + E_nuc + e_dft_plus_u;

        // Orbital gradient: [F, D*S] = FDS - SDF (zero at convergence).
        const Eigen::MatrixXd error = F * D_used * S - S * D_used * F;
        const double grad_norm = error.norm();
        const double dE = E_total - E_prev;

        VIBEQC_DIAG("rhf", vibeqc::DiagLevel::VERBOSE,
            "iter %3d  E=% .10f  dE=%+.3e  |grad|=%.3e",
            iter, E_total, (iter == 1 ? 0.0 : dE), grad_norm);

        // Two-phase Schwarz tightening: switch the direct JKBuilder
        // to its tight threshold once the SCF gradient norm crosses
        // ``schwarz_threshold_tighten_at``. The builder also drops
        // its incremental ΔP cache (if any) — the bounds change so
        // ``G_prev`` is no longer consistent with the new screen.
        bool schwarz_tightened_this_iter = false;
        if (!schwarz_tightened
            && grad_norm < opts.schwarz_threshold_tighten_at) {
            VIBEQC_DIAG("rhf", vibeqc::DiagLevel::DEBUG,
                "Schwarz tighten at iter %d: grad %.3e < %.3e",
                iter, grad_norm, opts.schwarz_threshold_tighten_at);
            jk.set_schwarz_threshold(opts.schwarz_threshold);
            jk.reset_state();
            schwarz_tightened = true;
            schwarz_tightened_this_iter =
                jk.uses_runtime_schwarz_threshold();
            if (schwarz_tightened_this_iter) {
                full_fock_validation_pending = true;
                fock_map_epoch_start_iter = iter + 1;
            }
        }

        const bool coarse_fock_active =
            jk.incremental_active()
            || schwarz_tightened_this_iter
            || (!schwarz_tightened
                && jk.uses_runtime_schwarz_threshold());
        const bool begin_full_fock_refinement =
            needs_full_fock_refinement(
                iter, grad_norm, opts.conv_tol_grad,
                coarse_fock_active);
        const bool converged =
            !validating_first_full_build
            && !begin_full_fock_refinement
            && is_scf_converged(
                iter, dE, grad_norm,
                opts.conv_tol_energy, opts.conv_tol_grad);
        if (converged) {
            VIBEQC_DIAG("rhf", vibeqc::DiagLevel::DEBUG,
                "converged at iter %d: |dE|=%.3e < %.0e, |grad|=%.3e < %.0e",
                iter, std::abs(dE), opts.conv_tol_energy,
                grad_norm, opts.conv_tol_grad);
        }

        // Newton activation (Phase D2c). Once the orbital-gradient norm
        // drops below ``newton_threshold``, swap the diagonalize-F update
        // for a full Newton step on the orbital-rotation manifold — see
        // cpp/include/vibeqc/newton.hpp. DIIS extrapolation, damping and
        // level shift are all skipped in this phase (the Newton step is
        // its own update mechanism, and mixing them undoes the trust
        // region). Disabled when ``newton_threshold == 0`` (default,
        // back-compat) or while the C1c quadratic fallback owns the
        // step (the two are alternative second-order schemes).
        const bool in_newton_phase =
            opts.newton_threshold > 0.0
            && grad_norm < opts.newton_threshold
            && !in_quadratic_phase;

        // SOSCF activation (Phase D2d, Neese 2000). Augmented-Hessian
        // step with diagonal-dominant Hessian; cheaper per step than
        // Newton (no Fock build), linearly convergent. Mutually
        // exclusive with Newton — if both thresholds are set, Newton
        // wins (full-Hessian quadratic convergence is preferred).
        // D2e TRAH (Helmich-Paris 2022) activation. Full Hessian like
        // Newton, but with an adaptive trust radius driven by Powell's
        // ρ test. Mutual exclusion priority: quadratic > Newton > TRAH
        // > SOSCF (Newton's fixed cap is more predictable on PD
        // Hessians; TRAH wins over SOSCF because the full Hessian is
        // more accurate than the diagonal-dominant approximation).
        const bool in_trah_phase =
            opts.trah_threshold > 0.0
            && grad_norm < opts.trah_threshold
            && !in_quadratic_phase
            && !in_newton_phase;

        const bool in_soscf_phase =
            opts.soscf_threshold > 0.0
            && grad_norm < opts.soscf_threshold
            && !in_quadratic_phase
            && !in_newton_phase
            && !in_trah_phase;

        // SCF accelerator (DIIS / KDIIS / EDIIS / EDIIS+DIIS hybrid).
        // Feed the selected accelerator(s) every iteration so history
        // builds up before we need it; the extrapolated F replaces the
        // raw F once ``diis_active`` (start_iter crossed). Skipped
        // entirely during the quadratic, Newton or SOSCF phases —
        // those are second-order updates that handle their own
        // convergence.
        int diis_subspace_this_iter = 0;
        // Extrapolation that actually reached F this iteration (#682); see
        // SCFIteration::accelerator_step for the codes.
        int accelerator_step_this_iter = 0;
        if (opts.use_diis && !in_quadratic_phase
            && !in_newton_phase && !in_soscf_phase && !in_trah_phase) {
            Eigen::MatrixXd F_extrap = F;
            switch (opts.scf_accelerator) {
                // R_CDIIS / AD_CDIIS share the DIIS extrapolation; the
                // adaptive depth policy is baked into ``diis`` at
                // construction (see make_diis).
                case SCFAccelerator::DIIS:
                case SCFAccelerator::R_CDIIS:
                case SCFAccelerator::AD_CDIIS: {
                    F_extrap = diis.extrapolate(F, error);
                    diis_subspace_this_iter =
                        static_cast<int>(diis.subspace_size());
                    accelerator_step_this_iter = 1;
                    break;
                }
                case SCFAccelerator::KDIIS: {
                    F_extrap = kdiis.extrapolate(F, C_prev_mo,
                                                  eps_prev_mo, nocc);
                    diis_subspace_this_iter =
                        static_cast<int>(kdiis.subspace_size());
                    accelerator_step_this_iter = 2;
                    break;
                }
                case SCFAccelerator::EDIIS: {
                    F_extrap = ediis.extrapolate(F, D_used, E_total);
                    diis_subspace_this_iter =
                        static_cast<int>(ediis.subspace_size());
                    accelerator_step_this_iter = 3;
                    break;
                }
                case SCFAccelerator::EDIIS_DIIS: {
                    Eigen::MatrixXd F_diis = diis.extrapolate(F, error);
                    Eigen::MatrixXd F_ediis =
                        ediis.extrapolate(F, D_used, E_total);
                    diis_subspace_this_iter =
                        static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm, static_cast<Eigen::Index>(F.rows()));
                    const bool take_ediis =
                        switch_metric > opts.ediis_diis_switch_threshold;
                    // A DIIS-branch cycle discards F_ediis; retract it so
                    // the anti-replay guard keys on consumed returns only.
                    if (!take_ediis) ediis.discard_last_extrapolation();
                    F_extrap = take_ediis ? F_ediis : F_diis;
                    accelerator_step_this_iter = take_ediis ? 3 : 1;
                    break;
                }
                case SCFAccelerator::ADIIS: {
                    F_extrap = adiis.extrapolate(F, D_used);
                    diis_subspace_this_iter =
                        static_cast<int>(adiis.subspace_size());
                    accelerator_step_this_iter = 4;
                    break;
                }
                case SCFAccelerator::ADIIS_DIIS: {
                    Eigen::MatrixXd F_diis = diis.extrapolate(F, error);
                    Eigen::MatrixXd F_adiis = adiis.extrapolate(F, D_used);
                    diis_subspace_this_iter =
                        static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm, static_cast<Eigen::Index>(F.rows()));
                    const bool take_adiis =
                        switch_metric > opts.ediis_diis_switch_threshold;
                    if (!take_adiis) adiis.discard_last_extrapolation();
                    F_extrap = take_adiis ? F_adiis : F_diis;
                    accelerator_step_this_iter = take_adiis ? 4 : 1;
                    break;
                }
            }
            if (diis_active) {
                F = F_extrap;
            } else {
                // Below diis_start_iter nothing extrapolated reaches the
                // SCF, so every branch's return is discarded. Same
                // produced-vs-consumed retraction as the hybrid above.
                ediis.discard_last_extrapolation();
                adiis.discard_last_extrapolation();
                accelerator_step_this_iter = 0;
            }
        }
        if (!in_quadratic_phase && opts.fock_mixing != 0.0) {
            if (have_prev_fock) {
                F = mix_fock_matrices(F, F_prev_mixed, opts.fock_mixing);
            }
            F_prev_mixed = F;
            have_prev_fock = true;
        }

        // ---- per-iteration diagnostics (DEBUG) ---------------------------
        {
            const char* phase = "standard";
            if (in_quadratic_phase)      phase = "quadratic";
            else if (in_newton_phase)    phase = "newton";
            else if (in_trah_phase)      phase = "trah";
            else if (in_soscf_phase)     phase = "soscf";
            if (diis_active && diis_subspace_this_iter > 0) {
                VIBEQC_DIAG("rhf", vibeqc::DiagLevel::DEBUG,
                    "iter %d  phase=%s  diis=%d  damp=%.2f",
                    iter, phase, diis_subspace_this_iter, current_damping);
            } else {
                VIBEQC_DIAG("rhf", vibeqc::DiagLevel::DEBUG,
                    "iter %d  phase=%s  damp=%.2f",
                    iter, phase, current_damping);
            }
        }

        // Update C / eps. Three branches:
        //   * Newton phase: full Newton step on orbital-rotation manifold
        //     (preconditioned CG against the orbital Hessian). One Fock
        //     build per CG iter via the same JKBuilder.
        //   * Quadratic-fallback phase (C1c): diagonal-Hessian Newton
        //     step (κ_ai = -F_ai / (ε_a − ε_i + λ)).
        //   * Standard: diagonalize F with optional Saunders-Hillier
        //     level shift (F_shift = F + b·S − (b/2)·S·D·S).
        Eigen::MatrixXd C_new;
        Eigen::VectorXd eps_new;
        int newton_cg_iter_this_step = 0;
        double trah_level_shift_this_step = 0.0;
        if (in_newton_phase) {
            auto step = newton_step(F, C_prev_mo, eps_prev_mo, nocc,
                                   jk, opts.newton_opts);
            C_new = std::move(step.C);
            eps_new = std::move(step.eps);
            newton_cg_iter_this_step = step.cg_iter;
        } else if (in_trah_phase) {
            // D2e TRAH: same full-Hessian CG as Newton but with an
            // adaptive trust radius. The ρ test that drives the
            // trust-radius update happens later in the loop, once
            // this iteration's actual ΔE is known.
            auto step = trah_step(F, C_prev_mo, eps_prev_mo, nocc, jk,
                                  opts.trah_opts, trah_trust_radius);
            C_new = std::move(step.C);
            eps_new = std::move(step.eps);
            newton_cg_iter_this_step = step.cg_iter;
            trah_predicted_decrease = step.predicted_decrease;
            trah_kappa_norm = step.kappa_norm;
            trah_level_shift_this_step = step.level_shift;
        } else if (in_soscf_phase) {
            // D2d Neese SOSCF: AH eigsolve on diagonal-dominant
            // Hessian, no Fock build per step. The (n_pairs+1) eigsolve
            // is cheap; trust-region cap on ‖κ‖_F same as Newton.
            auto step = soscf.step(F, C_prev_mo, eps_prev_mo, nocc);
            C_new = std::move(step.C);
            eps_new = std::move(step.eps);
        } else if (in_quadratic_phase) {
            auto step = quadratic_step(F, C_prev_mo, eps_prev_mo, nocc,
                                       opts.quadratic_fallback_shift,
                                       opts.quadratic_fallback_max_step);
            C_new = std::move(step.C);
            eps_new = std::move(step.eps);
        } else {
            // Standard diagonalize with the unified auto-reducing
            // Saunders-Hillier shift resolved for this iteration.
            // When oscillation has been auto-detected, the shift is
            // resolved from the engaged effective value rather than
            // ``opts.level_shift`` (which is zero for most runs).
            const double b = level_shift_at_iter(
                oscillation_engaged ? effective_level_shift
                                    : opts.level_shift,
                oscillation_engaged ? effective_warmup
                                    : opts.level_shift_warmup_cycles,
                opts.level_shift_schedule,
                opts.max_iter,
                iter);
            if (b > 0.0) {
                VIBEQC_DIAG("rhf", vibeqc::DiagLevel::DEBUG,
                    "level shift b=%.3e at iter %d", b, iter);
            }
            std::tie(C_new, eps_new) = diagonalize_fock(apply_level_shift(
                F, S, D_used, b, LevelShiftDensity::TOTAL));
        }

        // Record this iteration in the trace (after the C/eps update so
        // ``newton_cg_iter`` reflects the step actually taken this iter).
        result.scf_trace.push_back(SCFIteration{
            iter,
            E_total,
            (iter == 1) ? 0.0 : dE,
            grad_norm,
            diis_subspace_this_iter,
            newton_cg_iter_this_step,
            trah_level_shift_this_step,
        });
        result.scf_trace.back().accelerator_step = accelerator_step_this_iter;
        {
            // Live progress for terminal, NDJSON, and vq manifest consumers;
            // the Python diagnostics bridge owns all user-facing formatting.
            VIBEQC_DIAG("progress", vibeqc::DiagLevel::QUIET,
                "iter=%d energy=%.17g dE=%.17g grad=%.17g diis=%d",
                iter, E_total, result.scf_trace.back().delta_e, grad_norm,
                diis_subspace_this_iter);
        }
        {
            // Truthful per-iteration wall (BUG87-A): everything since the
            // previous trace row, so Fock build + energy + DIIS +
            // diagonalization are all covered.
            const auto t_iter_now = std::chrono::steady_clock::now();
            result.scf_trace.back().wall_s =
                std::chrono::duration<double>(t_iter_now - t_iter_prev)
                    .count();
            t_iter_prev = t_iter_now;
        }

        // Auto-level-shift-on-oscillation detection. If the recent energy
        // changes form a scale-significant limit cycle while the commutator
        // norm is stalled, engage a persistent Saunders-Hillier shift,
        // clear the DIIS history, and continue. The shift damps the virtual
        // block and steers the SCF
        // into the correct basin. Fires once; oscillation is a mid-SCF
        // phenomenon, not a startup transient.
        const bool refresh_restart_baseline =
            iter == 1 || validating_first_full_build
            || E_total < best_energy;
        const int projected_stall_iters = stall_iters
            + ((!refresh_restart_baseline
                && grad_norm > best_grad_norm * 1.01) ? 1 : 0);
        const bool coarse_stall_transition_imminent =
            opts.restart_opts.enabled
            && !in_newton_phase && !in_trah_phase
            && projected_stall_iters >= opts.restart_opts.max_stall_iters
            && !converged
            && coarse_fock_active;
        if (opts.auto_level_shift_on_oscillation && !oscillation_engaged
            && iter >= opts.oscillation_detect_start_iter
            && !in_quadratic_phase && !in_newton_phase
            && !in_trah_phase && !in_soscf_phase
            && !coarse_stall_transition_imminent
            && !validating_first_full_build
            && !begin_full_fock_refinement
            && iter - fock_map_epoch_start_iter + 1
                   >= opts.oscillation_window
            && detect_scf_oscillation(result.scf_trace,
                                      opts.oscillation_window,
                                      opts.oscillation_min_sign_flips,
                                      opts.conv_tol_energy)) {
            oscillation_engaged = true;
            effective_level_shift = opts.auto_level_shift_value;
            effective_warmup = 0;  // persistent — held every iteration
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            VIBEQC_DIAG("rhf", vibeqc::DiagLevel::STANDARD,
                "auto level-shift engaged at iter %d (b=%.2f Ha)",
                iter, effective_level_shift);
        }

        // ---- deterministic orbital-rotation restart (BUG 64) --------
        if (iter == 1) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
        } else if (validating_first_full_build) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_C = C_new;
            best_eps = eps_new;
            best_D = build_density(C_new, nocc);
        } else if (E_total < best_energy) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_C = C_new;
            best_eps = eps_new;
            best_D = build_density(C_new, nocc);
        }
        if (grad_norm <= best_grad_norm * 1.01) {
            if (grad_norm < best_grad_norm) {
                best_grad_norm = grad_norm;
            }
        } else {
            ++stall_iters;
        }
        // IID 129 follow-up: the shared stall gate tests ``!converged``,
        // not the gradient.  Screened difference-Fock bias can leave only
        // the energy criterion outstanding after the gradient has passed.
        // The orbital-rotation branch retains its gradient guard below.
        const bool restart_stall_detected =
            opts.restart_opts.enabled
            && !in_newton_phase && !in_trah_phase
            && stall_iters >= opts.restart_opts.max_stall_iters
            && !converged;
        const bool full_fock_refinement_from_stall =
            restart_stall_detected && coarse_fock_active;
        const bool transition_to_full_fock =
            begin_full_fock_refinement || full_fock_refinement_from_stall;
        const bool fock_map_changed_this_iter =
            transition_to_full_fock || schwarz_tightened_this_iter;
        if (transition_to_full_fock) {
            // IIDs 129/418: a screened difference-Fock recurrence may
            // accumulate errors from prior increments.  Whether detected by
            // a high-gradient stall or by a passed gradient with dithering
            // energy, enter one common tight-screened, nonincremental phase.
            if (!schwarz_tightened) {
                jk.set_schwarz_threshold(opts.schwarz_threshold);
                schwarz_tightened = true;
            }
            jk.set_incremental(false);
            jk.reset_state();
            stall_iters = 0;
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_C = C_new;
            best_eps = eps_new;
            best_D = build_density(C_new, nocc);
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            soscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
            full_fock_validation_pending = true;
            fock_map_epoch_start_iter = iter + 1;
            if (begin_full_fock_refinement) {
                VIBEQC_DIAG("rhf", vibeqc::DiagLevel::STANDARD,
                    "direct-SCF coarse Fock phase ended at iter %d after "
                    "gradient convergence; certifying energy on consecutive "
                    "tight-screened full-density builds "
                    "(|dE|=%.3e, grad=%.3e)",
                    iter, std::abs(dE), grad_norm);
            } else {
                VIBEQC_DIAG("rhf", vibeqc::DiagLevel::STANDARD,
                    "direct-SCF coarse Fock phase ended at iter %d after "
                    "drift stall; certifying energy on consecutive "
                    "tight-screened "
                    "full-density builds (grad=%.3e)",
                    iter, grad_norm);
            }
        } else if (restart_stall_detected
                   && grad_norm > opts.conv_tol_grad
                   && num_restarts < opts.restart_opts.max_restarts) {
            ++num_restarts;
            stall_iters = 0;
            VIBEQC_DIAG("rhf", vibeqc::DiagLevel::STANDARD,
                "restart %d/%d at iter %d: grad=%.3e, best_grad=%.3e",
                num_restarts, opts.restart_opts.max_restarts,
                iter, grad_norm, best_grad_norm);
            auto [C_rot, eps_rot] = rotate_orbitals_in_subspaces(
                best_C, best_eps, nocc, &F,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 2,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 2 + 1);
            D = build_density(C_rot, nocc);
            C_new = C_rot;
            eps_new = eps_rot;
            diis.clear(); ediis.clear(); adiis.clear(); kdiis.clear();
            soscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_C = C_rot;
            best_eps = eps_rot;
            best_D = D;
        }
        if (schwarz_tightened_this_iter && !transition_to_full_fock) {
            stall_iters = 0;
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            soscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
        }
        if (fock_map_changed_this_iter) {
            trah_trust_radius = opts.trah_opts.initial_trust_radius;
            trah_predicted_decrease = 0.0;
            trah_kappa_norm = 0.0;
        }

        C_prev_mo = C_new;
        eps_prev_mo = eps_new;
        D_prev = D_used;
        // #210: an exactly degenerate frontier makes `leftCols` a coin flip on
        // the eigensolver's basis. Resolve it deterministically; a no-op
        // whenever the frontier is gapped, which is virtually always.
        {
            const Eigen::MatrixXd Cocc = occupied_block_deterministic(
                C_new, eps_new, nocc, s_sqrt_ao, ao_atom_index);
            D = nocc > 0
                ? Eigen::MatrixXd(2.0 * Cocc * Cocc.transpose())
                : Eigen::MatrixXd(Eigen::MatrixXd::Zero(C_new.rows(), C_new.rows()));
        }

        result.mo_energies = eps_new;
        result.mo_coeffs = C_new;
        result.density = D_used;
        result.fock = F;
        result.energy = E_total;
        result.e_electronic = E_elec;
        result.e_dft_plus_u = e_dft_plus_u;
        result.n_iter = iter;

        if (converged) {
            VIBEQC_DIAG("rhf", vibeqc::DiagLevel::DEBUG,
                "final self-consistency pass at iter %d", iter);
            // Final self-consistency pass: rebuild F with the fresh density
            // D = make_density(C_new) so the returned MO energies correspond
            // to F(D_final) rather than F(D_{final-1}). This matches the
            // convention used by PySCF / Psi4 and makes orbital energies
            // compare at machine precision instead of ~1e-7.
            //
            // result.energy / e_electronic are likewise recomputed on the
            // rebuilt (J, K, D) so the returned energy reproduces from
            // the returned matrices (2026-05-18 audit P3; see
            // tests/test_scf_final_consistency.py). The energy therefore
            // differs from scf_trace[-1].energy by at most ~conv_tol;
            // test_scf_log.py asserts that band.
            //
            // The energy uses the SCF's iterated K convention, i.e. it
            // is computed BEFORE the COSX one-center correction below:
            // the correction upgrades only the returned Fock / MOs.
            // Folding it into the energy would shift COSX totals off the
            // iterated SCF surface (~1e-4 Ha) and break both the trace
            // band and the parity decomposition self-consistency
            // (test_parity_hf_dft.py); a no-op for non-COSX builders.
            Eigen::MatrixXd J_final = jk.build_J(D);
            Eigen::MatrixXd K_final = jk.build_K(D);
            // +U-free trace: V_U must not leak into (Hcore + F); the
            // Dudarev energy is added separately below (see the
            // per-iteration block + dft_plus_u.hpp for the math note).
            const double E_elec_final =
                (D.array() * Hcore.array()).sum()
                + 0.5  * (D.array() * J_final.array()).sum()
                - 0.25 * (D.array() * K_final.array()).sum();
            jk.apply_one_center_correction(K_final, D);
            const Eigen::MatrixXd G_final = J_final - 0.5 * K_final;
            Eigen::MatrixXd F_final = Hcore + G_final;
            if (!opts.dft_plus_u_sites.empty()) {
                const auto vu_final = compute_dft_plus_u(
                    opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                    0.5 * D, S);
                F_final += vu_final.V;
                // Re-report e_dft_plus_u against the fresh density so
                // the result is consistent with the returned F_final.
                result.e_dft_plus_u = 2.0 * vu_final.energy;
            }
            auto [C_final, eps_final] = diagonalize_fock(F_final);
            result.mo_energies = eps_final;
            result.mo_coeffs  = C_final;
            result.fock       = F_final;
            result.density    = D;
            result.energy        = E_elec_final + E_nuc + result.e_dft_plus_u;
            result.e_electronic  = E_elec_final;
            result.converged  = true;

            // ---- Restricted-stability VERDICT (issue #144) ------------
            // Verdict only: report the lowest orbital-rotation Hessian
            // eigenvalue of the converged restricted solution, never
            // rotate / escape / promote. What to do with a negative
            // verdict is the maintainer decision in
            // agentic-loop/asks/ask-scf144-restricted-stability-contract-
            // 2026-08-27.md; this phase only makes the verdict visible
            // (mirrors UHF/UKS: skip silently on unsupported routes by
            // default, fail closed on an explicit request).
            if (opts.stability_check) {
                const auto t_stab_wall0 = std::chrono::steady_clock::now();
                const std::clock_t t_stab_cpu0 = std::clock();
                const auto charge_stability_phase = [&]() {
                    result.stability_wall_s = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_stab_wall0).count();
                    result.stability_cpu_s =
                        static_cast<double>(std::clock() - t_stab_cpu0)
                        / static_cast<double>(CLOCKS_PER_SEC);
                };
                const auto unsupported_stability = [&](bool unsupported,
                                                       const char* missing_response,
                                                       const char* opt_out) {
                    if (!unsupported) return false;
                    if (opts.stability_check_explicit) {
                        throw std::runtime_error(
                            std::string("RHF internal stability analysis was "
                                        "explicitly requested, but ")
                            + missing_response
                            + " is not yet plumbed. Set stability_check=False "
                              "to run "
                            + opt_out + " without a stability verdict.");
                    }
                    // Default-on check must not manufacture a verdict from
                    // an incomplete orbital Hessian. Preserve the converged
                    // first-order result, leave stability_checked=false.
                    return true;
                };
                if (!unsupported_stability(
                        !opts.dft_plus_u_sites.empty(),
                        "the DFT+U response",
                        "the supported first-order DFT+U SCF")
                    && !unsupported_stability(
                        jk.has_post_scf_exchange_correction(),
                        "the COSX one-centre exchange correction response and "
                        "a consistent corrected-orbital energy surface",
                        "the supported first-order hybrid COSX SCF")) {
                    UHFStabilityOptions sopts;
                    sopts.max_iter = opts.stability_davidson_max_iter;
                    // A residual looser than the instability threshold
                    // cannot certify the sign of the reported eigenvalue.
                    sopts.residual_tol =
                        std::min(1e-5, 0.1 * opts.stability_tol);
                    // Equal spin blocks on the restricted solution: the
                    // coupled per-spin orbital Hessian over (κ_α, κ_β)
                    // decomposes into the Seeger-Pople singlet (A+B) and
                    // triplet (A−B) sectors, so the lowest eigenvalue over
                    // the combined space is the restricted stability
                    // verdict (Seeger & Pople 1977, doi:10.1063/1.434318;
                    // KS analogue: Bauernschmitt & Ahlrichs 1996,
                    // doi:10.1063/1.471637).
                    const auto stab = uhf_internal_stability_lowest(
                        C_final, C_final, eps_final, eps_final,
                        nocc, nocc, jk, sopts, nullptr, 1.0);
                    result.stability_checked = true;
                    result.stability_analysis_converged = stab.converged;
                    result.stability_eigenvalue = stab.lowest_eigenvalue;
                    result.internal_instability =
                        stab.converged
                        && stab.lowest_eigenvalue < -opts.stability_tol;
                }
                charge_stability_phase();
            }
            return result;
        }

        // Dynamic-damping update for the next iteration's α (Zerner-
        // Hehenberger 1979). Skipped (and current_damping stays at
        // opts.damping) when opts.dynamic_damping is false.
        if (opts.dynamic_damping && !fock_map_changed_this_iter) {
            current_damping = update_dynamic_damping(
                current_damping, E_total, E_prev, have_prev_E,
                opts.dynamic_damping_min, opts.dynamic_damping_max);
        }
        have_prev_E = !fock_map_changed_this_iter;

        // TRAH trust-radius update from Powell's ρ test. The
        // ``trah_predicted_decrease`` was set by the PREVIOUS
        // iteration's trah_step (if any); apply ρ now that this
        // iteration's actual ΔE is in hand. Skip on iteration 1
        // (no previous step to test) and on iterations where the
        // previous step wasn't TRAH (predicted_decrease stays at 0).
        if (trah_active_prev && !fock_map_changed_this_iter) {
            const double actual_decrease = E_prev - E_total;
            trah_trust_radius = update_trust_radius(
                trah_trust_radius, actual_decrease,
                trah_predicted_decrease,
                trah_kappa_norm,   // ‖κ‖ of the previous TRAH step —
                                   // lets the radius expand when the
                                   // step used its budget (ρ > rho_expand)
                opts.trah_opts);
        }
        trah_active_prev = fock_map_changed_this_iter
            ? false : in_trah_phase;

        E_prev = E_total;
    }

    result.converged = false;
    return result;
}

}  // namespace vibeqc
