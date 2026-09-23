#include "vibeqc/uhf.hpp"
#include "vibeqc/orbital_scf.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <memory>
#include <stdexcept>

#include "vibeqc/diis.hpp"
#include "vibeqc/diagnostics.hpp"
#include "vibeqc/dynamic_damping.hpp"
#include "vibeqc/ediis.hpp"
#include "vibeqc/cosx.hpp"
#include "vibeqc/cosx_staged.hpp"
#include "vibeqc/grid.hpp"
#include "vibeqc/kdiis.hpp"
#include "vibeqc/level_shift.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/jk_builder.hpp"
#include "vibeqc/linear_dependence.hpp"
#include "vibeqc/mom.hpp"
#include "vibeqc/quadratic_scf.hpp"
#include "vibeqc/scf_mixing.hpp"
#include "vibeqc/scf_convergence.hpp"
#include "vibeqc/scf_restart.hpp"
#include "vibeqc/newton.hpp"
#include "vibeqc/soscf.hpp"
#include "vibeqc/trah.hpp"
#include "vibeqc/davidson.hpp"
#include "vibeqc/degenerate_frontier.hpp"

namespace vibeqc {

namespace {

Eigen::MatrixXd build_density_from_occ(const Eigen::MatrixXd& C, int nocc) {
    if (nocc <= 0) {
        return Eigen::MatrixXd::Zero(C.rows(), C.rows());
    }
    const auto Cocc = C.leftCols(nocc);
    return Cocc * Cocc.transpose();
}

// <S^2> for UHF (Szabo-Ostlund exercise 3.40, equivalent Pople formula):
//   <S^2> = S(S+1) + n_β - Σ_{i∈occ_α, j∈occ_β} |<α_i | S | β_j>|^2
double compute_s_squared(
    const Eigen::MatrixXd& C_alpha, int n_alpha,
    const Eigen::MatrixXd& C_beta, int n_beta,
    const Eigen::MatrixXd& S) {
    const double dS = 0.5 * (n_alpha - n_beta);
    const double ideal = dS * (dS + 1);
    if (n_alpha == 0 || n_beta == 0) return ideal;
    // Mulliken-style orbital overlap matrix between occupied alpha and beta.
    const Eigen::MatrixXd Cao = C_alpha.leftCols(n_alpha);
    const Eigen::MatrixXd Cbo = C_beta.leftCols(n_beta);
    const Eigen::MatrixXd M = Cao.transpose() * S * Cbo;       // (n_α, n_β)
    const double sum_sq = (M.array() * M.array()).sum();
    return ideal + static_cast<double>(n_beta) - sum_sq;
}

}  // namespace

UHFResult run_uhf(const Molecule& mol,
                  const BasisSet& basis,
                  const UHFOptions& opts) {
    validate_orbital_optimizer(opts);
    if (opts.orbital_optimizer == "opentrustregion"
        && (opts.spinlock_mode != SpinlockMode::OFF || !opts.atomic_spins.empty()))
        throw std::invalid_argument("OpenTrustRegion does not support spin schedules, MOM holds or targeted atomic spin states; select native");
    validate_initial_guess(opts.initial_guess);
    validate_guess_ecp(opts.initial_guess, molecular_guess_ecp_context(opts), &mol);
    validate_scf_max_iter(opts.max_iter, "run_uhf");
    validate_atomic_spin_selection(
        &mol, GuessEngine::resolve_auto_for_molecule(mol, opts.initial_guess, false, true),
        opts.atomic_spins);
    const auto ecp_input = validate_molecular_ecp_dispatch(
        opts.ecp_centers, opts.ecp_library, opts.ecp_primitive_blocks,
        opts.ecp_primitive_centers, opts.ecp_effective_charges,
        opts.ecp_total_ncore, "run_uhf");
    // ---- Integrals ----
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
            "run_uhf: ECP cores remove more electrons ("
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
            UHFResult result) {
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
    const int mult = mol.multiplicity();
    const int n_alpha = (n_elec + mult - 1) / 2;
    const int n_beta  = (n_elec - mult + 1) / 2;

    if (n_alpha + n_beta != n_elec || n_alpha - n_beta != mult - 1) {
        throw std::invalid_argument(
            "UHF: multiplicity and electron count are inconsistent");
    }
    if (n_beta < 0) {
        throw std::invalid_argument("UHF: negative beta electron count");
    }
    validate_fraction_01("UHF: damping", opts.damping);
    validate_fraction_01("UHF: fock_mixing", opts.fock_mixing);

    // ---- Two-electron infrastructure ---------------------------------------
    // Same JKBuilder dispatch as run_rhf — see jk_builder.hpp.
    std::unique_ptr<BasisSet> aux;
    Grid cosx_grid_built;
    std::vector<GridOptions> cosx_stages;  // COSX grid progression (GridX)
    CosxVariant cosx_var = CosxVariant::FITTED;
    bool cosx_multistage = false;
    const bool spinlock_active =
        (opts.spinlock_mode == SpinlockMode::SPIN_SCHEDULE
         && opts.spinlock_iterations > 0 && opts.max_iter > 0);
    std::unique_ptr<JKBuilder> jk;
    if (opts.density_fit) {
        if (opts.aux_basis.empty()) {
            throw std::invalid_argument(
                "run_uhf: density_fit=true requires aux_basis to be set "
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
                // Legacy single COSX grid: explicit opt-out or the auto
                // fallback for a conv_tol_grad tighter than the GridX floor.
                cosx_grid_built = build_grid(
                    mol, resolve_cosx_grid_options(opts.cosx_grid, 0,
                                                   cosx_card));
            } else if (spinlock_active) {
                // SPINLOCK runs two sequential SCFs; don't combine it with
                // the multi-stage progression — use the single fine grid
                // (still the 2021 5-region AngularGrid for this level).
                cosx_grid_built = build_grid(
                    mol, cosx_grid_options_for_level(
                             resolve_cosx_grid_level(opts.cosx_grid_level,
                                                     cosx_card)));
            } else {
                // GridX multi-stage progression (P2). Stage-0 (coarse) grid
                // seeds the guess; finer stages are built on the fly.
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
            // Two-phase Schwarz: open at the loose threshold. See
            // run_rhf for the full comment.
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

    auto prepared = prepare_open_guess(
        &mol, basis, n_alpha, n_beta, opts.initial_guess, S, Hcore, *jk,
        {}, {}, opts.read_density_alpha, opts.read_density_beta,
        opts.atomic_spins, opts.linear_dep_threshold, false, nullptr, molecular_guess_ecp_context(opts));
    const auto& gDa = prepared.alpha;
    const auto& gDb = prepared.beta;

    // SPINLOCK spin-schedule (mode A): converge at a locked n_alpha-n_beta for
    // the first spinlock_iterations cycles, then restart at the multiplicity
    // target from that density (CRYSTAL SPINLOCK n nstep). Two sequential SCFs
    // so the core loop stays untouched; the locked phase polarises (e.g.
    // high-spin) before relaxing to the target.
    if (spinlock_active) {
        const int n_elec = n_alpha + n_beta;
        if (opts.spinlock_value > n_elec || -opts.spinlock_value > n_elec
            || ((n_elec + opts.spinlock_value) % 2) != 0) {
            throw std::runtime_error(
                "UHF SPINLOCK: spinlock_value=" + std::to_string(opts.spinlock_value)
                + " is incompatible with " + std::to_string(n_elec)
                + " electrons (need |value| <= n_elec and matching parity)");
        }
        const int na_lock = (n_elec + opts.spinlock_value) / 2;
        const int nb_lock = (n_elec - opts.spinlock_value) / 2;
        UHFOptions opts_lock = opts;
        opts_lock.spinlock_mode = SpinlockMode::OFF;
        opts_lock.max_iter = opts.spinlock_iterations;
        // The locked phase deliberately converges an off-target spin
        // state; never stability-rotate it away (the release phase
        // below keeps the configured stability handling).
        opts_lock.stability_check = false;
        const auto locked = run_uhf_scf_with_jk(
            basis, na_lock, nb_lock, S, Hcore, E_nuc, *jk, opts_lock,
            gDa, gDb, &mol, &prepared.selection);
        prepared.selection.transport = InitialGuess::READ;
        UHFOptions opts_release = opts;
        opts_release.spinlock_mode = SpinlockMode::OFF;
        return with_ecp_provenance(run_uhf_scf_with_jk(
            basis, n_alpha, n_beta, S, Hcore, E_nuc, *jk, opts_release,
            locked.density_alpha, locked.density_beta, &mol, &prepared.selection));
    }
    if (cosx_multistage) {
        Eigen::MatrixXd Da = gDa, Db = gDb;
        auto seg = [&](const JKBuilder& seg_jk,
                       const UHFOptions& so) -> UHFResult {
            UHFResult r = run_uhf_scf_with_jk(basis, n_alpha, n_beta, S, Hcore,
                                              E_nuc, seg_jk, so, Da, Db, &mol, &prepared.selection);
            prepared.selection.transport = InitialGuess::READ;
            Da = r.density_alpha;
            Db = r.density_beta;
            return r;
        };
        return with_ecp_provenance(run_cosx_staged_scf<UHFResult>(
            mol, cosx_stages, opts, *jk, seg));
    }
    return with_ecp_provenance(run_uhf_scf_with_jk(
        basis, n_alpha, n_beta, S, Hcore, E_nuc, *jk, opts, gDa, gDb, &mol, &prepared.selection));
}

namespace {

// One full SCF drive to convergence (or max_iter). The public
// run_uhf_scf_with_jk wraps this with the post-convergence internal
// stability analysis; the SCF body itself is unchanged.
// ``mom_anchor_alpha`` / ``mom_anchor_beta`` (optional): FIXED occupied
// reference orbitals ((n_bf, n_occ_sigma) each). When supplied, the first
// ``mom_anchor_iters`` cycles select occupations by maximum overlap with
// this anchor instead of aufbau (initial-reference MOM; the drifting-
// reference variant is the SPINLOCK PATTERN_HOLD above, both on Gilbert,
// Besley & Gill 2008 via mom.hpp). Used by the stability escape: a
// rotated saddle-escape start needs its occupation pattern held against
// aufbau reversion while the mean field self-consists around it —
// measured on ferrocene m=3/def2-SVP, a plain density-only restart is
// recaptured by the original basin from every line-search point.
UHFResult run_uhf_scf_once(const BasisSet& basis,
                           int n_alpha,
                           int n_beta,
                           const Eigen::MatrixXd& S,
                           const Eigen::MatrixXd& Hcore,
                           double E_nuc,
                           const JKBuilder& jk,
                           const UHFOptions& opts,
                           const Eigen::MatrixXd& init_alpha,
                           const Eigen::MatrixXd& init_beta,
                           const Eigen::MatrixXd* mom_anchor_alpha = nullptr,
                           const Eigen::MatrixXd* mom_anchor_beta = nullptr,
                           int mom_anchor_iters = 0) {
    validate_orbital_optimizer(opts);
    validate_scf_max_iter(opts.max_iter, "run_uhf_scf_with_jk");
    if (n_alpha < 0 || n_beta < 0) {
        throw std::invalid_argument(
            "run_uhf_scf_with_jk: per-spin electron counts must be "
            "non-negative");
    }
    validate_fraction_01("run_uhf_scf_with_jk: damping", opts.damping);
    validate_fraction_01(
        "run_uhf_scf_with_jk: fock_mixing", opts.fock_mixing);
    if (S.rows() != Hcore.rows() || S.cols() != Hcore.cols()
        || S.rows() != static_cast<Eigen::Index>(basis.nbasis())) {
        throw std::invalid_argument(
            "run_uhf_scf_with_jk: S / Hcore shapes do not match "
            "basis.nbasis()");
    }
    if ((init_alpha.size() == 0) != (init_beta.size() == 0)) {
        throw std::invalid_argument(
            "run_uhf_scf_with_jk: init_alpha and init_beta must both be "
            "supplied or both be empty");
    }

    // Loewdin metric and AO->atom map for the degenerate-frontier tie-break
    // (#210). Both are built once per SCF; the tie-break itself runs only on
    // an iteration whose frontier gap is below tolerance, which for an
    // ordinary molecule is never.
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

    // Canonical orthogonalization; see rhf.cpp for the shared notes.
    const auto orth = canonical_orthogonalizer(S, opts.linear_dep_threshold);
    if (orth.n_kept == 0) {
        throw std::runtime_error(
            "UHF: AO basis has no non-null directions above the "
            "linear-dependence threshold (threshold = "
            + std::to_string(opts.linear_dep_threshold) + ")");
    }
    if (std::max(n_alpha, n_beta) > orth.n_kept) {
        throw std::runtime_error(
            "UHF: canonical orthogonalization dropped too many basis "
            "directions for this electron count (max(n_α, n_β) = "
            + std::to_string(std::max(n_alpha, n_beta)) + ", n_kept = "
            + std::to_string(orth.n_kept) + ")");
    }
    const Eigen::MatrixXd& X = orth.X;

    // ---- Davidson iterative diagonalization setup -------------------------
    const bool use_davidson_here =
        opts.use_davidson && orth.n_kept >= opts.davidson_min_dim;
    DavidsonOptions dav_opts = opts.davidson;
    if (use_davidson_here) {
        if (dav_opts.n_eig == 0) dav_opts.n_eig = orth.n_kept;
    }
    Eigen::MatrixXd dav_guess_ortho;  // for eigenvector recycling

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fock_solver;
    auto diagonalize = [&](const Eigen::MatrixXd& F)
        -> std::pair<Eigen::MatrixXd, Eigen::VectorXd> {
        const Eigen::MatrixXd Fp = X.transpose() * F * X;
        if (use_davidson_here) {
            if (dav_guess_ortho.size() != 0) {
                dav_opts.guess_vectors = dav_guess_ortho;
            }
            DavidsonResult dres = davidson_solve(Fp, dav_opts);
            if (!dres.converged) {
                throw std::runtime_error(
                    "UHF: Davidson diagonalization did not converge after "
                    + std::to_string(dres.n_iter) + " iterations");
            }
            dav_guess_ortho = dres.eigenvectors;
            return {X * dres.eigenvectors, dres.eigenvalues};
        }
        fock_solver.compute(Fp);
        if (fock_solver.info() != Eigen::Success) {
            throw std::runtime_error("UHF: Fock diagonalization failed");
        }
        return {X * fock_solver.eigenvectors(), fock_solver.eigenvalues()};
    };

    // ---- Initial densities. Hcore-diag (no init given) or external.
    auto [C0, eps0] = diagonalize(Hcore);
    Eigen::MatrixXd C_alpha = C0;
    Eigen::MatrixXd C_beta  = C0;
    Eigen::MatrixXd D_alpha, D_beta;
    if (init_alpha.size() != 0) {
        if (init_alpha.rows() != S.rows() || init_alpha.cols() != S.cols()
            || init_beta.rows()  != S.rows() || init_beta.cols()  != S.cols()) {
            throw std::invalid_argument(
                "run_uhf_scf_with_jk: init_alpha / init_beta shapes do "
                "not match S");
        }
        D_alpha = init_alpha;
        D_beta  = init_beta;
    } else {
        D_alpha = build_density_from_occ(C_alpha, n_alpha);
        D_beta  = build_density_from_occ(C_beta,  n_beta);
    }
    Eigen::MatrixXd D_alpha_prev = D_alpha;
    Eigen::MatrixXd D_beta_prev  = D_beta;

    // Track per-spin MO frame between iterations so the C1c Newton step
    // has a current MO basis to operate in. Updated each iteration after
    // either diagonalisation or the quadratic step.
    Eigen::MatrixXd C_alpha_prev_mo = C_alpha;
    Eigen::MatrixXd C_beta_prev_mo  = C_beta;
    Eigen::VectorXd eps_alpha_prev_mo = eps0;
    Eigen::VectorXd eps_beta_prev_mo  = eps0;

    UHFResult result;
    result.restart_basis = basis;
    result.mo_energies_alpha = eps0;
    result.mo_energies_beta  = eps0;
    result.mo_coeffs_alpha = C_alpha;
    result.mo_coeffs_beta  = C_beta;

    if (opts.orbital_optimizer == "opentrustregion") {
        if (opts.spinlock_mode != SpinlockMode::OFF || !opts.atomic_spins.empty())
            throw std::invalid_argument("OpenTrustRegion does not support targeted spin states or spinlock; select native");
        const auto run = run_orbital_scf(opts, X, S, Hcore, E_nuc, jk,
            D_alpha, D_beta, n_alpha, n_beta, false, 1.0);
        assign_orbital_unrestricted(result, run, E_nuc, S, n_alpha, n_beta);
        return result;
    }

    // ORCA NOITER compatibility: evaluate the initial spin densities once,
    // without taking an SCF update or recording an iteration.
    if (opts.max_iter == 0) {
        const Eigen::MatrixXd D_total = D_alpha + D_beta;
        const Eigen::MatrixXd J = jk.build_J_slot(D_total, 0);
        const Eigen::MatrixXd K_alpha = jk.build_K_slot(D_alpha, 0);
        const Eigen::MatrixXd K_beta = jk.build_K_slot(D_beta, 1);
        Eigen::MatrixXd F_alpha = Hcore + J - K_alpha;
        Eigen::MatrixXd F_beta = Hcore + J - K_beta;
        const double E_elec = 0.5 * (
              (D_alpha.array() * (Hcore + F_alpha).array()).sum()
            + (D_beta.array() * (Hcore + F_beta).array()).sum());
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu_a = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                D_alpha, S);
            const auto vu_b = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                D_beta, S);
            F_alpha += vu_a.V;
            F_beta += vu_b.V;
            e_dft_plus_u = vu_a.energy + vu_b.energy;
        }
        auto [Ca, eps_a] = diagonalize(F_alpha);
        auto [Cb, eps_b] = diagonalize(F_beta);
        result.mo_energies_alpha = eps_a;
        result.mo_energies_beta = eps_b;
        result.mo_coeffs_alpha = Ca;
        result.mo_coeffs_beta = Cb;
        result.density_alpha = D_alpha;
        result.density_beta = D_beta;
        result.fock_alpha = F_alpha;
        result.fock_beta = F_beta;
        result.energy = E_elec + E_nuc + e_dft_plus_u;
        result.e_electronic = E_elec;
        result.e_dft_plus_u = e_dft_plus_u;
        const double dS = 0.5 * (n_alpha - n_beta);
        result.s_squared_ideal = dS * (dS + 1.0);
        result.s_squared = compute_s_squared(Ca, n_alpha, Cb, n_beta, S);
        return result;
    }

    // One spin-coupled Pulay history (see DIIS::extrapolate_spin_coupled):
    // F_α and F_β share J(D_α + D_β), so a single coefficient set must
    // extrapolate both. Per-spin histories stall the SCF tail (OH/def2-TZVP
    // UHF regression, tests/test_uhf_open_shell_convergence.py). The
    // adaptive depth policy (R_CDIIS / AD_CDIIS) is baked in via make_diis.
    DIIS diis = make_diis(opts);
    EDIIS ediis(opts.diis_subspace_size);
    ADIIS adiis(opts.diis_subspace_size);
    KDIIS kdiis(opts.diis_subspace_size);
    UHFSOSCF usoscf(opts.soscf_opts);  // stateful L-BFGS SOSCF accelerator

    // Dynamic-damping state (see RHF). Inactive by default.
    double current_damping = opts.damping;
    bool have_prev_E = false;
    bool full_fock_validation_pending = false;
    int fock_map_epoch_start_iter = 1;

    // TRAH (D2e) trust-radius state across iterations.
    double trah_trust_radius = opts.trah_opts.initial_trust_radius;
    double trah_predicted_decrease = 0.0;
    double trah_kappa_norm = 0.0;   // ‖κ‖ of the previous TRAH step
    bool trah_active_prev = false;

    double E_prev = 0.0;
    Eigen::MatrixXd F_alpha_prev_mixed;
    Eigen::MatrixXd F_beta_prev_mixed;
    bool have_prev_fock = false;

    // Auto-level-shift-on-oscillation state — see rhf.cpp for the full
    // description. The engaged shift is applied per-spin through the
    // same ``level_shift_at_iter`` resolution.
    bool oscillation_engaged = false;
    double effective_level_shift = opts.level_shift;
    int effective_warmup = opts.level_shift_warmup_cycles;

    // Deterministic orbital-rotation restart state (BUG 64).
    int num_restarts = 0;
    int stall_iters = 0;
    double best_grad_norm = 1e300;
    double best_energy = 0.0;
    Eigen::MatrixXd best_Ca = C_alpha;
    Eigen::MatrixXd best_Cb = C_beta;
    Eigen::VectorXd best_eps_a = eps0;
    Eigen::VectorXd best_eps_b = eps0;
    Eigen::MatrixXd best_Da = D_alpha;
    Eigen::MatrixXd best_Db = D_beta;

    // Two-phase direct-SCF Schwarz tightening — see run_rhf for the
    // full comment. Loose threshold while grad-norm > tighten_at,
    // tight threshold (and incremental cache reset) thereafter.
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

        // Quadratic-fallback phase: Newton step in MO space replaces the
        // diagonalize-F update once iter > quadratic_fallback_iter. DIIS
        // and damping are skipped during the quadratic phase, same as the
        // RHF case; the trust-region cap on ‖κ‖_F is the analogous safety.
        // The Newton step runs independently per spin (each spin has its
        // own n_occ and ε spectrum).
        const bool in_quadratic_phase =
            opts.quadratic_fallback_iter > 0
            && iter > opts.quadratic_fallback_iter;

        // SPINLOCK PATTERN_HOLD: the accelerator is suspended (no history
        // recorded, no extrapolation, damping stays live) while the hold
        // is active. Fock extrapolation across held-window iterates steers
        // the SCF toward the symmetric attractor by continuous orbital
        // rotation, a collapse the occupation-selecting MOM hold cannot
        // see, and poisons the post-release history with out-of-basin
        // iterates. The history starts fresh at release. Mirrors the
        // periodic drivers (periodic_uks_multi_k_ewald.py & siblings).
        const bool anchor_hold_active =
            mom_anchor_alpha != nullptr && mom_anchor_beta != nullptr
            && mom_anchor_iters > 0 && iter <= mom_anchor_iters;
        const bool hold_active =
            (opts.spinlock_mode == SpinlockMode::PATTERN_HOLD
             && opts.spinlock_iterations > 0
             && iter <= opts.spinlock_iterations)
            || anchor_hold_active;

        const bool diis_active =
            opts.use_diis && iter >= opts.diis_start_iter
            && !in_quadratic_phase && !hold_active;

        // Damp densities (until DIIS takes over). Gate on the iteration-
        // local current_damping, not opts.damping: with damping=0.0 +
        // dynamic_damping=true the dynamic update is the only source of
        // a non-zero mixing factor (matches rhf.cpp).
        auto damp_or_pass = [&](const Eigen::MatrixXd& D_new,
                                const Eigen::MatrixXd& D_prev)
            -> Eigen::MatrixXd {
            if (iter == 1 || current_damping == 0.0 || diis_active
                || in_quadratic_phase) return D_new;
            return current_damping * D_prev + (1.0 - current_damping) * D_new;
        };
        const Eigen::MatrixXd Da_used = damp_or_pass(D_alpha, D_alpha_prev);
        const Eigen::MatrixXd Db_used = damp_or_pass(D_beta,  D_beta_prev);

        // Common Coulomb from the total density; per-spin exchange.
        // Slot 0 = J(D_total), 0 = K(D_α), 1 = K(D_β) so the
        // incremental-Fock ΔD cache (DirectJKBuilder) doesn't collide
        // across spins. Falls back to the stateless path on non-
        // Direct builders + when ``incremental_fock = false``.
        const Eigen::MatrixXd D_total = Da_used + Db_used;
        const Eigen::MatrixXd J = jk.build_J_slot(D_total, 0);
        const Eigen::MatrixXd K_alpha = jk.build_K_slot(Da_used, 0);
        const Eigen::MatrixXd K_beta  = jk.build_K_slot(Db_used, 1);

        Eigen::MatrixXd F_alpha = Hcore + J - K_alpha;
        Eigen::MatrixXd F_beta  = Hcore + J - K_beta;

        // UHF electronic energy from the +U-free Fock (Hcore + J - K_σ).
        // The Dudarev +U term is added separately below — same double-
        // counting discipline as RHF/RKS.
        const double E_elec = 0.5 * (
              (Da_used.array() * (Hcore + F_alpha).array()).sum()
            + (Db_used.array() * (Hcore + F_beta ).array()).sum());

        // Dudarev +U Fock contribution. Per-spin convention applies
        // natively to UHF: call the kernel once per spin with that
        // spin's density and add the returned V_U_σ to F_σ. Energies
        // sum directly (each call returns the per-spin contribution).
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu_a = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                Da_used, S);
            const auto vu_b = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                Db_used, S);
            F_alpha += vu_a.V;
            F_beta  += vu_b.V;
            e_dft_plus_u = vu_a.energy + vu_b.energy;
        }

        const double E_total = E_elec + E_nuc + e_dft_plus_u;

        // Per-spin orbital gradients [F_σ, D_σ · S] = F D S - S D F.
        const Eigen::MatrixXd err_alpha = F_alpha * Da_used * S - S * Da_used * F_alpha;
        const Eigen::MatrixXd err_beta  = F_beta  * Db_used * S - S * Db_used * F_beta;
        const double grad_norm = std::max(err_alpha.norm(), err_beta.norm());
        const double dE = E_total - E_prev;

        VIBEQC_DIAG("uhf", vibeqc::DiagLevel::VERBOSE,
            "iter %3d  E=% .10f  dE=%+.3e  |grad|=%.3e",
            iter, E_total, (iter == 1 ? 0.0 : dE), grad_norm);

        // Two-phase Schwarz tightening; see run_rhf for full comment.
        bool schwarz_tightened_this_iter = false;
        if (!schwarz_tightened
            && grad_norm < opts.schwarz_threshold_tighten_at) {
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
            VIBEQC_DIAG("uhf", vibeqc::DiagLevel::DEBUG,
                "converged at iter %d: |dE|=%.3e < %.0e, |grad|=%.3e < %.0e",
                iter, std::abs(dE), opts.conv_tol_energy,
                grad_norm, opts.conv_tol_grad);
        }

        // Newton activation (Phase D2c, open-shell). Once the
        // max-over-spins commutator norm drops below
        // ``newton_threshold``, swap the per-spin diagonalisations for
        // a *coupled* Newton step (one CG iteration solves both α and
        // β rotations jointly; see uhf_newton_step in newton.cpp). DIIS
        // extrapolation, damping, and per-spin level shift are all
        // skipped in this phase — the Newton step is its own update
        // mechanism, and mixing them undoes the trust region.
        // Disabled when ``newton_threshold == 0`` (default, back-compat)
        // or while the C1c quadratic fallback owns the step (the two
        // are alternative second-order schemes).
        const bool in_newton_phase =
            opts.newton_threshold > 0.0
            && grad_norm < opts.newton_threshold
            && !in_quadratic_phase;

        // D2e TRAH (open-shell). Same per-spin coupled CG as Newton
        // but with the adaptive trust radius driven by Powell's ρ
        // test. Priority: quadratic > Newton > TRAH > SOSCF.
        const bool in_trah_phase =
            opts.trah_threshold > 0.0
            && grad_norm < opts.trah_threshold
            && !in_quadratic_phase
            && !in_newton_phase;

        // D2d Neese SOSCF activation, open-shell. Mutually exclusive
        // with Newton + TRAH.
        const bool in_soscf_phase =
            opts.soscf_threshold > 0.0
            && grad_norm < opts.soscf_threshold
            && !in_quadratic_phase
            && !in_newton_phase
            && !in_trah_phase;

        // SCF accelerator (per-spin DIIS / coupled-spin EDIIS / hybrid).
        // Skipped during quadratic, Newton, TRAH, or SOSCF phases, and
        // skipped entirely (not even recorded) while the PATTERN_HOLD
        // window is active; see the hold_active note above.
        int diis_sub = 0;
        // Extrapolation that actually reached F this iteration (#682); see
        // SCFIteration::accelerator_step for the codes.
        int accelerator_step_this_iter = 0;
        if (opts.use_diis && !in_quadratic_phase && !hold_active
            && !in_newton_phase && !in_trah_phase && !in_soscf_phase) {
            Eigen::MatrixXd Fa_ext = F_alpha;
            Eigen::MatrixXd Fb_ext = F_beta;
            switch (opts.scf_accelerator) {
                // R_CDIIS / AD_CDIIS share the spin-coupled DIIS
                // extrapolation; the adaptive depth policy is baked into
                // ``diis`` at construction (see make_diis).
                case SCFAccelerator::DIIS:
                case SCFAccelerator::R_CDIIS:
                case SCFAccelerator::AD_CDIIS: {
                    auto d = diis.extrapolate_spin_coupled(
                        F_alpha, F_beta, err_alpha, err_beta);
                    Fa_ext = std::move(d.first);
                    Fb_ext = std::move(d.second);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    accelerator_step_this_iter = 1;
                    break;
                }
                case SCFAccelerator::KDIIS: {
                    auto k = kdiis.extrapolate(
                        F_alpha, F_beta,
                        C_alpha_prev_mo, C_beta_prev_mo,
                        eps_alpha_prev_mo, eps_beta_prev_mo,
                        n_alpha, n_beta);
                    Fa_ext = std::move(k.first);
                    Fb_ext = std::move(k.second);
                    diis_sub = static_cast<int>(kdiis.subspace_size());
                    accelerator_step_this_iter = 2;
                    break;
                }
                case SCFAccelerator::EDIIS: {
                    auto e = ediis.extrapolate(F_alpha, F_beta,
                                               Da_used, Db_used, E_total);
                    Fa_ext = std::move(e.first);
                    Fb_ext = std::move(e.second);
                    diis_sub = static_cast<int>(ediis.subspace_size());
                    accelerator_step_this_iter = 3;
                    break;
                }
                case SCFAccelerator::EDIIS_DIIS: {
                    auto d = diis.extrapolate_spin_coupled(
                        F_alpha, F_beta, err_alpha, err_beta);
                    Eigen::MatrixXd Fa_d = std::move(d.first);
                    Eigen::MatrixXd Fb_d = std::move(d.second);
                    auto e = ediis.extrapolate(F_alpha, F_beta,
                                               Da_used, Db_used, E_total);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm,
                            static_cast<Eigen::Index>(F_alpha.rows()),
                            2);
                    if (switch_metric > opts.ediis_diis_switch_threshold) {
                        Fa_ext = std::move(e.first);
                        Fb_ext = std::move(e.second);
                        accelerator_step_this_iter = 3;
                    } else {
                        // A DIIS-branch cycle discards the EDIIS pair;
                        // retract it so the anti-replay guard keys on
                        // consumed returns only.
                        ediis.discard_last_extrapolation();
                        Fa_ext = std::move(Fa_d);
                        Fb_ext = std::move(Fb_d);
                        accelerator_step_this_iter = 1;
                    }
                    break;
                }
                case SCFAccelerator::ADIIS: {
                    auto a = adiis.extrapolate(F_alpha, F_beta,
                                               Da_used, Db_used);
                    Fa_ext = std::move(a.first);
                    Fb_ext = std::move(a.second);
                    diis_sub = static_cast<int>(adiis.subspace_size());
                    accelerator_step_this_iter = 4;
                    break;
                }
                case SCFAccelerator::ADIIS_DIIS: {
                    auto d = diis.extrapolate_spin_coupled(
                        F_alpha, F_beta, err_alpha, err_beta);
                    Eigen::MatrixXd Fa_d = std::move(d.first);
                    Eigen::MatrixXd Fb_d = std::move(d.second);
                    auto a = adiis.extrapolate(F_alpha, F_beta,
                                               Da_used, Db_used);
                    diis_sub = static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm,
                            static_cast<Eigen::Index>(F_alpha.rows()),
                            2);
                    if (switch_metric > opts.ediis_diis_switch_threshold) {
                        Fa_ext = std::move(a.first);
                        Fb_ext = std::move(a.second);
                        accelerator_step_this_iter = 4;
                    } else {
                        adiis.discard_last_extrapolation();
                        Fa_ext = std::move(Fa_d);
                        Fb_ext = std::move(Fb_d);
                        accelerator_step_this_iter = 1;
                    }
                    break;
                }
            }
            if (diis_active) {
                F_alpha = Fa_ext;
                F_beta  = Fb_ext;
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
                F_alpha = mix_fock_matrices(
                    F_alpha, F_alpha_prev_mixed, opts.fock_mixing);
                F_beta = mix_fock_matrices(
                    F_beta, F_beta_prev_mixed, opts.fock_mixing);
            }
            F_alpha_prev_mixed = F_alpha;
            F_beta_prev_mixed = F_beta;
            have_prev_fock = true;
        }

        // Per-spin update. Three branches:
        //   * Newton phase: coupled Newton step on the α+β orbital-
        //     rotation manifold (preconditioned CG against the full
        //     UHF orbital Hessian — shared J, per-spin K). One Fock
        //     build (one J + two K) per CG iter via the JKBuilder.
        //   * Quadratic-fallback phase (C1c): diagonal-Hessian Newton
        //     step per spin (κ_ai^σ = -F_σ^MO_ai / (ε_a^σ − ε_i^σ + λ)).
        //     Independent α / β rotations — no cross-spin coupling.
        //   * Standard: per-spin diagonalisation with optional
        //     Saunders-Hillier level shift
        //     (F_σ_shift = F_σ + b·S − b·(S·D_σ·S)).
        Eigen::MatrixXd Ca_new, Cb_new;
        Eigen::VectorXd eps_a_new, eps_b_new;
        int newton_cg_iter_this_step = 0;
        double trah_level_shift_this_step = 0.0;
        if (in_newton_phase) {
            auto step = uhf_newton_step(F_alpha, F_beta,
                                        C_alpha_prev_mo, C_beta_prev_mo,
                                        eps_alpha_prev_mo, eps_beta_prev_mo,
                                        n_alpha, n_beta,
                                        jk, opts.newton_opts);
            Ca_new = std::move(step.C_alpha);
            Cb_new = std::move(step.C_beta);
            eps_a_new = std::move(step.eps_alpha);
            eps_b_new = std::move(step.eps_beta);
            newton_cg_iter_this_step = step.cg_iter;
        } else if (in_trah_phase) {
            // D2e TRAH, open-shell: per-spin coupled CG with adaptive
            // trust radius. Driver updates trah_trust_radius post-iter
            // via Powell's ρ.
            auto step = uhf_trah_step(F_alpha, F_beta,
                                      C_alpha_prev_mo, C_beta_prev_mo,
                                      eps_alpha_prev_mo, eps_beta_prev_mo,
                                      n_alpha, n_beta, jk,
                                      opts.trah_opts, trah_trust_radius);
            Ca_new = std::move(step.C_alpha);
            Cb_new = std::move(step.C_beta);
            eps_a_new = std::move(step.eps_alpha);
            eps_b_new = std::move(step.eps_beta);
            newton_cg_iter_this_step = step.cg_iter;
            trah_predicted_decrease = step.predicted_decrease;
            trah_kappa_norm = step.kappa_norm;
            trah_level_shift_this_step = step.level_shift;
        } else if (in_soscf_phase) {
            // D2d Neese SOSCF, open-shell: per-spin AH eigsolves
            // (no cross-spin coupling in the diagonal-dominant
            // Hessian approximation).
            auto step = usoscf.step(F_alpha, F_beta,
                                    C_alpha_prev_mo, C_beta_prev_mo,
                                    eps_alpha_prev_mo, eps_beta_prev_mo,
                                    n_alpha, n_beta);
            Ca_new = std::move(step.C_alpha);
            Cb_new = std::move(step.C_beta);
            eps_a_new = std::move(step.eps_alpha);
            eps_b_new = std::move(step.eps_beta);
        } else if (in_quadratic_phase) {
            auto step_a = quadratic_step(F_alpha, C_alpha_prev_mo,
                                         eps_alpha_prev_mo, n_alpha,
                                         opts.quadratic_fallback_shift,
                                         opts.quadratic_fallback_max_step);
            auto step_b = quadratic_step(F_beta, C_beta_prev_mo,
                                         eps_beta_prev_mo, n_beta,
                                         opts.quadratic_fallback_shift,
                                         opts.quadratic_fallback_max_step);
            Ca_new = std::move(step_a.C);
            eps_a_new = std::move(step_a.eps);
            Cb_new = std::move(step_b.C);
            eps_b_new = std::move(step_b.eps);
        } else {
            // Standard per-spin diagonalize with the unified auto-reducing
            // Saunders-Hillier shift resolved for this iteration.
            // ``apply_level_shift`` returns F untouched when b is 0, so the
            // S·Dσ·S matmuls are skipped once the warm-up releases.
            // Da_used / Db_used are *spin* densities (occupations in {0, 1}),
            // hence SPIN: weight 1, not the closed-shell ½.
            // When oscillation has been auto-detected, the shift is resolved
            // from the engaged effective value rather than opts.level_shift.
            const double b = level_shift_at_iter(
                oscillation_engaged ? effective_level_shift
                                    : opts.level_shift,
                oscillation_engaged ? effective_warmup
                                    : opts.level_shift_warmup_cycles,
                opts.level_shift_schedule,
                opts.max_iter,
                iter);
            std::tie(Ca_new, eps_a_new) = diagonalize(apply_level_shift(
                F_alpha, S, Da_used, b, LevelShiftDensity::SPIN));
            std::tie(Cb_new, eps_b_new) = diagonalize(apply_level_shift(
                F_beta, S, Db_used, b, LevelShiftDensity::SPIN));
            // SPINLOCK pattern-hold: for the first spinlock_iterations SCF
            // cycles, hold the seeded broken-symmetry occupation by selecting
            // occupied orbitals via maximum overlap with the previous cycle
            // (MOM) rather than pure aufbau, then release. Keeps an ATOMSPIN /
            // atomic_spins seed from collapsing to the symmetric solution
            // during early iterations. iter 1 establishes the pattern by
            // aufbau (no previous orbitals yet); MOM holds it for iters
            // 2..spinlock_iterations.
            if (opts.spinlock_mode == SpinlockMode::PATTERN_HOLD
                && opts.spinlock_iterations > 0
                && iter <= opts.spinlock_iterations && iter > 1
                && C_alpha_prev_mo.cols() >= n_alpha
                && C_beta_prev_mo.cols() >= n_beta) {
                mom_reorder_occupied(Ca_new, eps_a_new, S,
                                     C_alpha_prev_mo.leftCols(n_alpha), n_alpha);
                mom_reorder_occupied(Cb_new, eps_b_new, S,
                                     C_beta_prev_mo.leftCols(n_beta), n_beta);
            }
            // Stability-escape anchor: hold the occupation pattern of the
            // FIXED rotated reference (initial-reference MOM) from iter 1,
            // so the saddle-escape start survives until the mean field
            // self-consists in the new basin.
            if (anchor_hold_active) {
                mom_reorder_occupied(Ca_new, eps_a_new, S,
                                     *mom_anchor_alpha, n_alpha);
                mom_reorder_occupied(Cb_new, eps_b_new, S,
                                     *mom_anchor_beta, n_beta);
            }
        }

        // Record this iteration in the trace (after the C/eps update so
        // ``newton_cg_iter`` reflects the step actually taken this iter).
        result.scf_trace.push_back(SCFIteration{
            iter,
            E_total,
            (iter == 1) ? 0.0 : dE,
            grad_norm,
            diis_sub,
            newton_cg_iter_this_step,
            trah_level_shift_this_step,
        });
        result.scf_trace.back().accelerator_step = accelerator_step_this_iter;
        {
            // Python fans this precision-preserving diagnostic out to the
            // live terminal, NDJSON, and vq manifest sinks.
            VIBEQC_DIAG("progress", vibeqc::DiagLevel::QUIET,
                "iter=%d energy=%.17g dE=%.17g grad=%.17g diis=%d",
                iter, E_total, result.scf_trace.back().delta_e, grad_norm,
                diis_sub);
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

        // Auto-level-shift-on-oscillation detection — see rhf.cpp for the
        // full description.
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
            VIBEQC_DIAG("uhf", vibeqc::DiagLevel::STANDARD,
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
            best_Ca = Ca_new;
            best_Cb = Cb_new;
            best_eps_a = eps_a_new;
            best_eps_b = eps_b_new;
            best_Da = build_density_from_occ(Ca_new, n_alpha);
            best_Db = build_density_from_occ(Cb_new, n_beta);
        } else if (E_total < best_energy) {
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_Ca = Ca_new;
            best_Cb = Cb_new;
            best_eps_a = eps_a_new;
            best_eps_b = eps_b_new;
            best_Da = build_density_from_occ(Ca_new, n_alpha);
            best_Db = build_density_from_occ(Cb_new, n_beta);
        }
        if (grad_norm <= best_grad_norm * 1.01) {
            if (grad_norm < best_grad_norm) {
                best_grad_norm = grad_norm;
            }
        } else {
            ++stall_iters;
        }
        // IID 129 follow-up: an energy-only stall must still end the coarse
        // Fock phase; orbital rotation keeps its gradient guard below.
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
            if (!schwarz_tightened) {
                jk.set_schwarz_threshold(opts.schwarz_threshold);
                schwarz_tightened = true;
            }
            jk.set_incremental(false);
            jk.reset_state();
            stall_iters = 0;
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_Ca = Ca_new;
            best_Cb = Cb_new;
            best_eps_a = eps_a_new;
            best_eps_b = eps_b_new;
            best_Da = build_density_from_occ(Ca_new, n_alpha);
            best_Db = build_density_from_occ(Cb_new, n_beta);
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            usoscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
            full_fock_validation_pending = true;
            fock_map_epoch_start_iter = iter + 1;
            if (begin_full_fock_refinement) {
                VIBEQC_DIAG("uhf", vibeqc::DiagLevel::STANDARD,
                    "direct-SCF coarse Fock phase ended at iter %d after "
                    "gradient convergence; certifying energy on consecutive "
                    "tight-screened full-density builds "
                    "(|dE|=%.3e, grad=%.3e)",
                    iter, std::abs(dE), grad_norm);
            } else {
                VIBEQC_DIAG("uhf", vibeqc::DiagLevel::STANDARD,
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
            VIBEQC_DIAG("uhf", vibeqc::DiagLevel::STANDARD,
                "restart %d/%d at iter %d: grad=%.3e, best_grad=%.3e",
                num_restarts, opts.restart_opts.max_restarts,
                iter, grad_norm, best_grad_norm);
            auto [Ca_rot, epsa_rot] = rotate_orbitals_in_subspaces(
                best_Ca, best_eps_a, n_alpha, &F_alpha,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 4,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 4 + 1);
            auto [Cb_rot, epsb_rot] = rotate_orbitals_in_subspaces(
                best_Cb, best_eps_b, n_beta, &F_beta,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 4 + 2,
                opts.restart_opts.seed + static_cast<std::uint64_t>(num_restarts) * 4 + 3);
            D_alpha = build_density_from_occ(Ca_rot, n_alpha);
            D_beta = build_density_from_occ(Cb_rot, n_beta);
            Ca_new = Ca_rot;
            Cb_new = Cb_rot;
            eps_a_new = epsa_rot;
            eps_b_new = epsb_rot;
            diis.clear(); ediis.clear(); adiis.clear(); kdiis.clear();
            usoscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
            best_energy = E_total;
            best_grad_norm = grad_norm;
            best_Ca = Ca_rot;
            best_Cb = Cb_rot;
            best_eps_a = epsa_rot;
            best_eps_b = epsb_rot;
            best_Da = D_alpha;
            best_Db = D_beta;
        }
        if (schwarz_tightened_this_iter && !transition_to_full_fock) {
            stall_iters = 0;
            diis.clear();
            ediis.clear();
            adiis.clear();
            kdiis.clear();
            usoscf.clear();
            current_damping = opts.damping;
            have_prev_fock = false;
        }
        if (fock_map_changed_this_iter) {
            trah_trust_radius = opts.trah_opts.initial_trust_radius;
            trah_predicted_decrease = 0.0;
            trah_kappa_norm = 0.0;
        }

        // SPINLOCK pattern-hold: for the first spinlock_iterations SCF
        C_alpha_prev_mo   = Ca_new;
        C_beta_prev_mo    = Cb_new;
        eps_alpha_prev_mo = eps_a_new;
        eps_beta_prev_mo  = eps_b_new;

        D_alpha_prev = Da_used;
        D_beta_prev  = Db_used;
        C_alpha = Ca_new;
        C_beta  = Cb_new;
        // #210: an exactly degenerate frontier makes `leftCols` a coin flip
        // on the eigensolver's basis. Resolve it deterministically; a no-op
        // whenever the frontier is gapped, which is virtually always.
        {
            const Eigen::MatrixXd Ca_occ = occupied_block_deterministic(
                C_alpha, eps_a_new, n_alpha, s_sqrt_ao, ao_atom_index);
            const Eigen::MatrixXd Cb_occ = occupied_block_deterministic(
                C_beta, eps_b_new, n_beta, s_sqrt_ao, ao_atom_index);
            D_alpha = n_alpha > 0 ? Eigen::MatrixXd(Ca_occ * Ca_occ.transpose())
                                  : Eigen::MatrixXd(Eigen::MatrixXd::Zero(C_alpha.rows(), C_alpha.rows()));
            D_beta = n_beta > 0 ? Eigen::MatrixXd(Cb_occ * Cb_occ.transpose())
                                : Eigen::MatrixXd(Eigen::MatrixXd::Zero(C_beta.rows(), C_beta.rows()));
        }

        result.mo_energies_alpha = eps_a_new;
        result.mo_energies_beta  = eps_b_new;
        result.mo_coeffs_alpha = C_alpha;
        result.mo_coeffs_beta  = C_beta;
        result.density_alpha = Da_used;
        result.density_beta  = Db_used;
        result.fock_alpha = F_alpha;
        result.fock_beta  = F_beta;
        result.energy = E_total;
        result.e_electronic = E_elec;
        result.e_dft_plus_u = e_dft_plus_u;
        result.n_iter = iter;

        if (converged) {
            // Final self-consistency pass (same convention as RHF):
            // rebuild F_α / F_β on the fresh (D_α, D_β) so the returned
            // MOs and Fock correspond to F(D_final), not F(D_{final-1}).
            // result.energy / e_electronic are likewise recomputed on the
            // rebuilt pair so the returned energy reproduces from the
            // returned matrices (2026-05-18 audit P3; see
            // tests/test_scf_final_consistency.py). The energy therefore
            // differs from scf_trace[-1].energy by at most ~conv_tol;
            // test_scf_log.py asserts that band.
            //
            // The energy uses the SCF's iterated K convention, i.e. it
            // is computed BEFORE the COSX one-center correction: the
            // correction upgrades only the returned Fock / MOs (see the
            // matching note in rhf.cpp; no-op for non-COSX builders).
            const Eigen::MatrixXd D_total_f = D_alpha + D_beta;
            const Eigen::MatrixXd J_f = jk.build_J_slot(D_total_f, 0);
            Eigen::MatrixXd Ka_f = jk.build_K_slot(D_alpha, 0);
            Eigen::MatrixXd Kb_f = jk.build_K_slot(D_beta, 1);
            // +U-free trace: V_U must not leak into the energy trace;
            // the Dudarev energy is added separately below.
            //   E_elec = tr(D_tot Hcore) + (1/2) tr(D_tot J)
            //          - (1/2) [tr(D_α K_α) + tr(D_β K_β)]
            const double E_elec_f =
                  (D_total_f.array() * Hcore.array()).sum()
                + 0.5 * (D_total_f.array() * J_f.array()).sum()
                - 0.5 * ((D_alpha.array() * Ka_f.array()).sum()
                       + (D_beta .array() * Kb_f.array()).sum());
            jk.apply_one_center_correction(Ka_f, D_alpha);
            jk.apply_one_center_correction(Kb_f, D_beta);
            Eigen::MatrixXd Fa_f = Hcore + J_f - Ka_f;
            Eigen::MatrixXd Fb_f = Hcore + J_f - Kb_f;
            if (!opts.dft_plus_u_sites.empty()) {
                const auto vu_a_f = compute_dft_plus_u(
                    opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                    D_alpha, S);
                const auto vu_b_f = compute_dft_plus_u(
                    opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                    D_beta, S);
                Fa_f += vu_a_f.V;
                Fb_f += vu_b_f.V;
                result.e_dft_plus_u = vu_a_f.energy + vu_b_f.energy;
            }
            auto [Ca_f, eps_a_f] = diagonalize(Fa_f);
            auto [Cb_f, eps_b_f] = diagonalize(Fb_f);
            // If the SCF converged while the PATTERN_HOLD window was still
            // active, the converged state is the MOM-held occupied set,
            // which need not be aufbau in its own Fock. Re-select by max
            // overlap with the held pattern so the reported energy / <S²>
            // / MOs describe the state the SCF actually converged to (an
            // aufbau slice here would silently swap to a different state).
            if (opts.spinlock_mode == SpinlockMode::PATTERN_HOLD
                && opts.spinlock_iterations > 0
                && iter <= opts.spinlock_iterations
                && C_alpha_prev_mo.cols() >= n_alpha
                && C_beta_prev_mo.cols() >= n_beta) {
                mom_reorder_occupied(Ca_f, eps_a_f, S,
                                     C_alpha_prev_mo.leftCols(n_alpha),
                                     n_alpha);
                mom_reorder_occupied(Cb_f, eps_b_f, S,
                                     C_beta_prev_mo.leftCols(n_beta),
                                     n_beta);
            }
            // Same for a stability-escape anchor hold that was still
            // active at convergence.
            if (mom_anchor_alpha != nullptr && mom_anchor_beta != nullptr
                && mom_anchor_iters > 0 && iter <= mom_anchor_iters) {
                mom_reorder_occupied(Ca_f, eps_a_f, S,
                                     *mom_anchor_alpha, n_alpha);
                mom_reorder_occupied(Cb_f, eps_b_f, S,
                                     *mom_anchor_beta, n_beta);
            }
            result.mo_energies_alpha = eps_a_f;
            result.mo_energies_beta  = eps_b_f;
            result.mo_coeffs_alpha = Ca_f;
            result.mo_coeffs_beta  = Cb_f;
            result.fock_alpha = Fa_f;
            result.fock_beta  = Fb_f;
            result.density_alpha = D_alpha;
            result.density_beta  = D_beta;
            result.energy = E_elec_f + E_nuc + result.e_dft_plus_u;
            result.e_electronic = E_elec_f;
            result.converged = true;

            const double dS = 0.5 * (n_alpha - n_beta);
            result.s_squared_ideal = dS * (dS + 1.0);
            result.s_squared = compute_s_squared(
                result.mo_coeffs_alpha, n_alpha,
                result.mo_coeffs_beta,  n_beta,
                S);
            return result;
        }

        // Dynamic-damping update (Zerner-Hehenberger 1979). Inactive
        // unless opts.dynamic_damping is true.
        if (opts.dynamic_damping && !fock_map_changed_this_iter) {
            current_damping = update_dynamic_damping(
                current_damping, E_total, E_prev, have_prev_E,
                opts.dynamic_damping_min, opts.dynamic_damping_max);
        }
        have_prev_E = !fock_map_changed_this_iter;

        if (trah_active_prev && !fock_map_changed_this_iter) {
            const double actual_decrease = E_prev - E_total;
            trah_trust_radius = update_trust_radius(
                trah_trust_radius, actual_decrease,
                trah_predicted_decrease, trah_kappa_norm,
                opts.trah_opts);
        }
        trah_active_prev = fock_map_changed_this_iter
            ? false : in_trah_phase;

        E_prev = E_total;
    }

    const double dS = 0.5 * (n_alpha - n_beta);
    result.s_squared_ideal = dS * (dS + 1.0);
    result.converged = false;
    return result;
}

// UHF total energy of a given (D_alpha, D_beta) pair — one J + two K
// builds. Used by the stability line search below to evaluate rotated
// determinants without touching SCF state:
//   E = tr(D_tot Hcore) + 1/2 tr(D_tot J)
//     - 1/2 [tr(D_a K_a) + tr(D_b K_b)] + E_nuc
double uhf_energy_of_densities(const Eigen::MatrixXd& Hcore,
                               double E_nuc,
                               const JKBuilder& jk,
                               const Eigen::MatrixXd& Da,
                               const Eigen::MatrixXd& Db) {
    const Eigen::MatrixXd D_tot = Da + Db;
    const Eigen::MatrixXd J = jk.build_J(D_tot);
    const Eigen::MatrixXd Ka = jk.build_K(Da);
    const Eigen::MatrixXd Kb = jk.build_K(Db);
    return (D_tot.array() * Hcore.array()).sum()
        + 0.5 * (D_tot.array() * J.array()).sum()
        - 0.5 * ((Da.array() * Ka.array()).sum()
               + (Db.array() * Kb.array()).sum())
        + E_nuc;
}

// Rotate one spin's occupied space along the occ-vir generator
// ``theta * kappa`` and return the resulting occupied coefficients
// C_occ' = (C exp(K)).leftCols(n_occ), K the padded antisymmetric
// generator (same convention as uhf_newton_step). The spin density is
// C_occ' C_occ'^T; the occupied block itself doubles as the fixed MOM
// anchor for the escape restart.
Eigen::MatrixXd rotated_occupied(const Eigen::MatrixXd& C,
                                 const Eigen::MatrixXd& kappa_ov,
                                 int n_occ,
                                 double theta) {
    const Eigen::Index n_kept = C.cols();
    if (n_occ <= 0) {
        return Eigen::MatrixXd(C.rows(), 0);
    }
    if (kappa_ov.size() == 0) {
        return C.leftCols(n_occ);
    }
    const Eigen::Index n_vir = n_kept - n_occ;
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_kept, n_kept);
    K.bottomLeftCorner(n_vir, n_occ) = theta * kappa_ov;
    K.topRightCorner(n_occ, n_vir) = -theta * kappa_ov.transpose();
    return (C * expm_skew(K)).leftCols(n_occ);
}

}  // namespace

UHFResult run_uhf_scf_with_jk(const BasisSet& basis,
                              int n_alpha,
                              int n_beta,
                              const Eigen::MatrixXd& S,
                              const Eigen::MatrixXd& Hcore,
                              double E_nuc,
                              const JKBuilder& jk,
                              const UHFOptions& opts,
                              const Eigen::MatrixXd& init_alpha,
                              const Eigen::MatrixXd& init_beta,
                              const Molecule* guess_molecule,
                              const GuessSelection* prepared_guess) {
    validate_scf_max_iter(opts.max_iter, "run_uhf_scf_with_jk");
    if (opts.spinlock_mode == SpinlockMode::SPIN_SCHEDULE
        && opts.spinlock_iterations > 0 && opts.max_iter > 0) {
        throw std::invalid_argument(
            "run_uhf_scf_with_jk does not implement SPIN_SCHEDULE phases; "
            "use run_uhf for a locked/released population schedule");
    }
    auto constructed = prepare_open_guess(
        guess_molecule, basis, n_alpha, n_beta, opts.initial_guess, S, Hcore,
        jk, init_alpha, init_beta, opts.read_density_alpha, opts.read_density_beta,
        opts.atomic_spins, opts.linear_dep_threshold, false, prepared_guess, molecular_guess_ecp_context(opts));

    UHFResult result = run_uhf_scf_once(
        basis, n_alpha, n_beta, S, Hcore, E_nuc, jk, opts,
        constructed.alpha, constructed.beta);
    result.guess_selection = constructed.selection;
    if (opts.orbital_optimizer == "opentrustregion") return result;

    // ---- Internal stability analysis + corrective restart ----------------
    //
    // A converged SCF only certifies a stationary point; whether it is a
    // local minimum is decided by the lowest eigenvalue of the internal
    // orbital-rotation Hessian: negative means rotating along the
    // eigenvector lowers the energy further (Lehtola, Molecules 25, 1218
    // (2020), Section 10; original conditions Seeger & Pople, J. Chem.
    // Phys. 66, 3045 (1977)). Measured trigger for this guard: O2 m=5/
    // cc-pVDZ, where the default guess converges cleanly onto a saddle at
    // -149.1039798772 Ha, +59.15 mHa ABOVE the ROHF energy on the same
    // system — violating the variational identity E(UHF) <= E(ROHF) — with
    // lambda_min = -0.0869; the true minimum -149.1799299198 Ha lies one
    // line-searched rotation away (bug UHF-ABOVE-ROHF-VARIATIONAL-
    // INVERSION, agentic-loop/bug-claims.md).
    //
    // Escape protocol: a small fixed-step rotation is NOT sufficient —
    // DIIS re-captures the excited basin (measured: steps 0.1-0.5 all
    // fall back; the E(theta) valley sits near theta ~= 1 for the
    // unit-norm eigenvector). So the energy is line-searched on a theta
    // grid along the unstable mode (one J + two K builds per point) and
    // the SCF restarts from the argmin densities.
    //
    // The +U Fock contribution is not part of the Hessian action, so the
    // analysis is skipped for +U jobs rather than run with the wrong
    // operator.
    if (!opts.stability_check || !result.converged
        || !opts.dft_plus_u_sites.empty()) {
        return result;
    }

    // Post-convergence stability-phase timing (issue #205). Start the clock
    // only after the skip gate: an opt-out, non-converged SCF, or unsupported
    // +U response performs no stability work and therefore retains exact-zero
    // wall/CPU fields instead of manufacturing a microsecond phase. On large
    // open-shell jobs the real analysis measured 0.8-3.0x the wall of the
    // entire SCF loop it certifies, so that work remains separately visible.
    // Wall comes from steady_clock; CPU from std::clock(), which on POSIX
    // accumulates every OpenMP thread -- the same convention as the Python
    // PerfScope's time.process_time().
    const auto t_stability_wall0 = std::chrono::steady_clock::now();
    const std::clock_t t_stability_cpu0 = std::clock();
    const auto charge_stability_phase = [&](UHFResult& charged) {
        charged.stability_wall_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_stability_wall0).count();
        charged.stability_cpu_s =
            static_cast<double>(std::clock() - t_stability_cpu0)
            / static_cast<double>(CLOCKS_PER_SEC);
    };
    UHFStabilityOptions sopts;
    sopts.max_iter = opts.stability_davidson_max_iter;
    // A residual looser than the instability threshold cannot certify the
    // sign of the reported eigenvalue. Keep one decimal guard digit while
    // avoiding needless over-solving for unusually loose user thresholds.
    sopts.residual_tol = std::min(1e-5, 0.1 * opts.stability_tol);

    // Deliberate state-targeting workflows (ATOMSPIN broken-symmetry
    // seeds, SPINLOCK pattern holds) converge on purpose to a chosen
    // basin that need not be the global minimum. For those the analysis
    // still runs — the eigenvalue lands on the result so the output
    // layer can report an instability — but the driver must NOT rotate
    // the state away from what the user explicitly asked for.
    const bool deliberate_state_targeting =
        (opts.spinlock_mode != SpinlockMode::OFF
         && opts.spinlock_iterations > 0)
        || !opts.atomic_spins.empty();
    if (deliberate_state_targeting) {
        const auto stab = uhf_internal_stability_lowest(
            result.mo_coeffs_alpha, result.mo_coeffs_beta,
            result.mo_energies_alpha, result.mo_energies_beta,
            n_alpha, n_beta, jk, sopts);
        result.stability_checked = true;
        result.stability_analysis_converged = stab.converged;
        result.stability_eigenvalue = stab.lowest_eigenvalue;
        result.internal_instability =
            stab.converged
            && stab.lowest_eigenvalue < -opts.stability_tol;
        charge_stability_phase(result);
        return result;
    }

    for (int restart = 0; ; ++restart) {
        const auto stab = uhf_internal_stability_lowest(
            result.mo_coeffs_alpha, result.mo_coeffs_beta,
            result.mo_energies_alpha, result.mo_energies_beta,
            n_alpha, n_beta, jk, sopts);
        result.stability_checked = true;
        result.stability_analysis_converged = stab.converged;
        result.stability_eigenvalue = stab.lowest_eigenvalue;
        result.internal_instability =
            stab.converged
            && stab.lowest_eigenvalue < -opts.stability_tol;
        if (!stab.converged) {
            // Davidson did not settle: report honestly, act on nothing.
            break;
        }
        if (!result.internal_instability) {
            break;  // internally stable — done
        }
        if (restart >= opts.stability_max_retries) {
            // Give up escaping; the negative eigenvalue + verdict stay
            // on the result for the output layer to report loudly.
            break;
        }

        // Line-search and reconverge BOTH signs of the internal-instability
        // eigenvector. Seeger and Pople, J. Chem. Phys. 66, 3045 (1977),
        // Section IV (Eqs. 37-41), require both searches because the two
        // descents can reach different minima. Choosing one sign from the
        // unrelaxed grid energy is insufficient: SCF can recapture one
        // branch while the other descends.
        static constexpr double kThetaMagnitudes[] = {
            0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0,
        };
        struct EscapeSeed {
            bool found = false;
            double energy = 0.0;
            Eigen::MatrixXd C_alpha_occ;
            Eigen::MatrixXd C_beta_occ;
        };
        auto line_search = [&](double sign) {
            EscapeSeed seed;
            seed.energy = result.energy;
            for (const double magnitude : kThetaMagnitudes) {
                const double theta = sign * magnitude;
                const Eigen::MatrixXd Ca_occ = rotated_occupied(
                    result.mo_coeffs_alpha, stab.kappa_alpha,
                    n_alpha, theta);
                const Eigen::MatrixXd Cb_occ = rotated_occupied(
                    result.mo_coeffs_beta, stab.kappa_beta,
                    n_beta, theta);
                const Eigen::MatrixXd Da = Ca_occ * Ca_occ.transpose();
                const Eigen::MatrixXd Db = Cb_occ * Cb_occ.transpose();
                const double E = uhf_energy_of_densities(
                    Hcore, E_nuc, jk, Da, Db);
                if (E < seed.energy) {
                    seed.found = true;
                    seed.energy = E;
                    seed.C_alpha_occ = Ca_occ;
                    seed.C_beta_occ = Cb_occ;
                }
            }
            return seed;
        };

        // Restart from one signed line-search minimum, holding the rotated
        // occupation pattern by fixed-reference maximum overlap for the
        // first cycles (mom.hpp; Gilbert-Besley-Gill 2008). A density-only
        // restart is measurably recaptured by the original basin on larger
        // systems (ferrocene m=3/def2-SVP).
        constexpr int kAnchorHoldIters = 15;
        constexpr int kMaxRestartChains = 5;
        auto reconverge = [&](const EscapeSeed& seed, int anchor_iters) {
            const Eigen::MatrixXd Da =
                seed.C_alpha_occ * seed.C_alpha_occ.transpose();
            const Eigen::MatrixXd Db =
                seed.C_beta_occ * seed.C_beta_occ.transpose();
            const Eigen::MatrixXd* anchor_alpha =
                anchor_iters > 0 ? &seed.C_alpha_occ : nullptr;
            const Eigen::MatrixXd* anchor_beta =
                anchor_iters > 0 ? &seed.C_beta_occ : nullptr;
            UHFResult candidate = run_uhf_scf_once(
                basis, n_alpha, n_beta, S, Hcore, E_nuc, jk, opts,
                Da, Db, anchor_alpha, anchor_beta, anchor_iters);

            // If the first drive runs out of iterations while remaining in
            // the downhill basin, chain fresh-DIIS restarts from its final
            // densities. Do not require every finite-iteration history to
            // lower the energy: near a stationary point the commutator can
            // shrink while the energy approaches its limit by a few nHa from
            // below (Cr2), and the old monotonic gate stopped exactly there.
            for (int chain = 0;
                 !candidate.converged && chain < kMaxRestartChains; ++chain) {
                if (candidate.energy >= result.energy - 1e-9) {
                    break;
                }
                candidate = run_uhf_scf_once(
                    basis, n_alpha, n_beta, S, Hcore, E_nuc, jk, opts,
                    candidate.density_alpha, candidate.density_beta);
            }
            return candidate;
        };

        bool found_reconverged_descent = false;
        UHFResult retried;
        for (const double sign : {-1.0, 1.0}) {
            const EscapeSeed seed = line_search(sign);
            if (!seed.found) {
                continue;
            }
            // Fixed-reference MOM protects sparse hole patterns (measured on
            // ferrocene), but it can over-constrain a collective instability
            // such as Cr2 and prevent its downhill density from settling.
            // If that guarded attempt does not produce a converged descent,
            // release the occupations immediately and retry from the same
            // variationally downhill density with a fresh accelerator.
            for (const int anchor_iters : {kAnchorHoldIters, 0}) {
                UHFResult candidate = reconverge(seed, anchor_iters);
                if (candidate.converged
                    && candidate.energy < result.energy - 1e-10) {
                    if (!found_reconverged_descent
                        || candidate.energy < retried.energy) {
                        found_reconverged_descent = true;
                        retried = std::move(candidate);
                    }
                    break;
                }
            }
        }
        if (!found_reconverged_descent) {
            // Neither sign produced a lower converged solution. Keep the
            // original result with the instability flagged.
            break;
        }
        retried.stability_checked = true;
        retried.stability_analysis_converged = true;
        retried.stability_eigenvalue = stab.lowest_eigenvalue;
        retried.n_stability_restarts = result.n_stability_restarts + 1;
        result = retried;
        result.guess_selection = constructed.selection;
        result.guess_selection->transport = InitialGuess::READ;
    }
    charge_stability_phase(result);
    return result;
}

}  // namespace vibeqc
