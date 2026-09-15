#include "vibeqc/periodic_rhf.hpp"

#include "vibeqc/bloch.hpp"
#include "vibeqc/diis.hpp"
#include "vibeqc/dynamic_damping.hpp"
#include "vibeqc/ediis.hpp"
#include "vibeqc/kdiis.hpp"
#include "vibeqc/ewald.hpp"
#include "vibeqc/grid.hpp"
#include "vibeqc/guess.hpp"
#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/level_shift.hpp"
#include "vibeqc/periodic_fock.hpp"
#include "vibeqc/periodic_jk_builder.hpp"
#include "vibeqc/scf_mixing.hpp"
#include "vibeqc/scf_convergence.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace vibeqc {

namespace {

// Fold a LatticeMatrixSet to the Γ block as a real matrix. At k = 0 the
// Bloch phases all reduce to +1, and sum-of-real-blocks is real.
Eigen::MatrixXd fold_gamma_real(const LatticeMatrixSet& set) {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(set.nbf, set.nbf);
    for (const auto& block : set.blocks) M += block;
    return M;
}

// Symmetric orthogonalisation X = S^{-½}. Throws if S is near-singular.
Eigen::MatrixXd s_inverse_sqrt(const Eigen::MatrixXd& S,
                               double lindep_threshold = 1.0e-8) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
    const Eigen::VectorXd& eigs = solver.eigenvalues();
    const double min_eig = eigs.minCoeff();
    if (min_eig < lindep_threshold) {
        throw std::runtime_error(
            "run_rhf_periodic_gamma: S(Γ) near-singular (min eigenvalue = "
            + std::to_string(min_eig)
            + "); unit cell too small for the AO basis");
    }
    const Eigen::MatrixXd& U = solver.eigenvectors();
    return U * eigs.unaryExpr([](double v) { return 1.0 / std::sqrt(v); })
                 .asDiagonal() * U.transpose();
}

Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int nocc) {
    const auto Cocc = C.leftCols(nocc);
    return 2.0 * Cocc * Cocc.transpose();
}

}  // namespace

namespace {

// Direct, half-summed, distance-truncated Madelung energy. Used by
// default; in 3D this is only conditionally convergent, so the caller
// should move to EWALD_3D via ``CoulombMethod`` for quantitative work.
double nuclear_repulsion_direct_truncated(const PeriodicSystem& system,
                                          const LatticeSumOptions& opts) {
    const auto cells = direct_lattice_cells(system, opts.nuclear_cutoff_bohr);
    const auto& atoms = system.unit_cell;
    const int n = static_cast<int>(atoms.size());
    double e_nuc = 0.0;
    for (const auto& c : cells) {
        const bool is_zero = c.index.isZero();
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                if (is_zero && b <= a) continue;  // intra-cell: upper triangle only
                const double dx = atoms[a].xyz[0] - atoms[b].xyz[0] - c.r_cart[0];
                const double dy = atoms[a].xyz[1] - atoms[b].xyz[1] - c.r_cart[1];
                const double dz = atoms[a].xyz[2] - atoms[b].xyz[2] - c.r_cart[2];
                const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (r < 1.0e-14) continue;  // coincident → skip
                const double contrib = static_cast<double>(atoms[a].Z) *
                                       static_cast<double>(atoms[b].Z) / r;
                // Intra-cell pairs counted once; inter-cell pairs halved
                // because the ±g pair both appear in `cells`.
                e_nuc += is_zero ? contrib : 0.5 * contrib;
            }
        }
    }
    return e_nuc;
}

double nuclear_repulsion_slab_ewald_2d(const PeriodicSystem& system,
                                       const LatticeSumOptions& opts) {
    if (system.dim != 2) {
        throw std::invalid_argument(
            "nuclear_repulsion_per_cell: SLAB_EWALD_2D requires dim == 2");
    }

    const auto& atoms = system.unit_cell;
    const int n = static_cast<int>(atoms.size());
    if (n == 0) return 0.0;

    Eigen::Matrix3Xd positions(3, n);
    Eigen::VectorXd charges(n);
    for (int i = 0; i < n; ++i) {
        positions.col(i) = Eigen::Vector3d(
            atoms[i].xyz[0], atoms[i].xyz[1], atoms[i].xyz[2]);
        charges[i] = static_cast<double>(atoms[i].Z);
    }

    EwaldOptions eopts;
    eopts.alpha = opts.slab_ewald_alpha > 0.0 ? opts.slab_ewald_alpha : 0.4;
    eopts.real_cutoff_bohr = opts.nuclear_cutoff_bohr;

    const Eigen::Vector3d a1 = system.lattice.col(0);
    const Eigen::Vector3d a2 = system.lattice.col(1);
    const Eigen::Vector3d cross = a1.cross(a2);
    const double area = cross.norm();
    if (area < 1.0e-14) {
        throw std::invalid_argument(
            "nuclear_repulsion_per_cell: SLAB_EWALD_2D has degenerate "
            "in-plane lattice vectors");
    }
    const Eigen::Vector3d nhat = cross / area;

    // Use the background-neutralized primitive as the implementation core,
    // then remove the exact uniform-sheet contribution to recover the bare
    // point-charge four-term block. This is the E_nn partner of the slab
    // V_ne/J decomposition used by the Gamma-only SCF.
    const double z_background = 0.0;
    const double e_with_background =
        ewald_2d_point_charge_energy_with_background(
            system.lattice, positions, charges, z_background, eopts);

    const double q_background = -charges.sum();
    double sheet_dot = 0.0;
    for (int i = 0; i < n; ++i) {
        const double z_i = positions.col(i).dot(nhat);
        sheet_dot += charges[i] * std::abs(z_i - z_background);
    }
    const double e_sheet =
        -(2.0 * M_PI / area) * q_background * sheet_dot;

    return e_with_background - e_sheet;
}

}  // namespace

double nuclear_repulsion_per_cell(const PeriodicSystem& system,
                                  const LatticeSumOptions& opts) {
    switch (opts.coulomb_method) {
        case CoulombMethod::EWALD_3D: {
            EwaldOptions eopts;
            eopts.real_cutoff_bohr = opts.nuclear_cutoff_bohr;
            return ewald_nuclear_repulsion(system, eopts);
        }
        case CoulombMethod::SLAB_EWALD_2D:
            return nuclear_repulsion_slab_ewald_2d(system, opts);
        case CoulombMethod::NEUTRALIZED_1D:
            throw std::runtime_error(
                "nuclear_repulsion_per_cell: NEUTRALIZED_1D wire Ewald is "
                "not yet implemented. Use DIRECT_TRUNCATED for now.");
        case CoulombMethod::DIRECT_TRUNCATED:
        default:
            return nuclear_repulsion_direct_truncated(system, opts);
    }
}

PeriodicRHFResult run_rhf_periodic_gamma(const PeriodicSystem& system,
                                         const BasisSet& basis,
                                         const PeriodicRHFOptions& opts) {
    std::vector<InitialGuess> guess_capabilities{
        InitialGuess::HCORE, InitialGuess::SAD, InitialGuess::PATOM,
        InitialGuess::HUECKEL, InitialGuess::MINAO, InitialGuess::READ};
    if (system.dim == 3) guess_capabilities.push_back(InitialGuess::SAP);
    const InitialGuess effective_guess = GuessEngine::resolve_auto_for_molecule(
        system.unit_cell_molecule(), opts.initial_guess, true, false, guess_capabilities);
    const auto guess_mol = system.unit_cell_molecule();
    const auto guess_ecp = periodic_guess_ecp_context(opts, &guess_mol);
    validate_guess_ecp(effective_guess, guess_ecp, &guess_mol);
    if (guess_ecp.active() && system.dim != 3) throw GuessCapabilityError(
        "native periodic ECP operators require a 3D cell");
    if (!opts.atomic_spins.empty()) {
        throw GuessCapabilityError("native periodic RHF cannot represent an atomic spin seed");
    }
    if (opts.spinlock_mode != SpinlockMode::OFF && opts.spinlock_iterations > 0) {
        throw GuessCapabilityError("native periodic RHF cannot execute a spin schedule");
    }
    const int n_elec = system.n_electrons() - guess_ecp.total_ncore;
    if (n_elec % 2 != 0) {
        throw std::invalid_argument(
            "run_rhf_periodic_gamma: closed-shell RHF requires even electron "
            "count per unit cell, got " + std::to_string(n_elec));
    }
    if (system.multiplicity != 1) {
        throw std::invalid_argument(
            "run_rhf_periodic_gamma: closed-shell RHF requires multiplicity = 1");
    }
    validate_fraction_01("run_rhf_periodic_gamma: damping", opts.damping);
    validate_fraction_01("run_rhf_periodic_gamma: fock_mixing",
                         opts.fock_mixing);

    const int nocc = n_elec / 2;

    // ---- One-electron integrals (real-space, folded to Γ) ------------------
    const auto S_set = compute_overlap_lattice(basis, system, opts.lattice_opts);
    const auto T_set = compute_kinetic_lattice(basis, system, opts.lattice_opts);
    const auto V_set = guess_ecp.active()
        ? compute_nuclear_lattice_with_charges(
            basis, system, opts.lattice_opts, guess_ecp.effective_charges)
        : compute_nuclear_lattice(basis, system, opts.lattice_opts);

    const Eigen::MatrixXd S = fold_gamma_real(S_set);
    Eigen::MatrixXd Hcore = fold_gamma_real(T_set) + fold_gamma_real(V_set);
    if (guess_ecp.active()) Hcore += fold_gamma_real(compute_ecp_lattice_from_primitives(
        basis, system, opts.lattice_opts, guess_ecp.primitive_centers,
        guess_ecp.primitive_blocks));
    const Eigen::MatrixXd X = s_inverse_sqrt(S);

    const bool use_davidson_here =
        opts.use_davidson && static_cast<int>(S.rows()) >= opts.davidson_min_dim;
    DavidsonOptions dav_opts = opts.davidson;
    if (use_davidson_here) {
        if (dav_opts.n_eig == 0) dav_opts.n_eig = static_cast<int>(S.rows());
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
    auto diag = [&](const Eigen::MatrixXd& F) {
        const Eigen::MatrixXd Fp = X.transpose() * F * X;
        if (use_davidson_here) {
            DavidsonResult dres = davidson_solve(Fp, dav_opts);
            if (!dres.converged) {
                throw std::runtime_error(
                    "run_rhf_periodic_gamma: Davidson did not converge");
            }
            return std::make_pair(X * dres.eigenvectors, dres.eigenvalues);
        }
        solver.compute(Fp);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "run_rhf_periodic_gamma: Fock(Γ) diagonalisation failed");
        }
        return std::make_pair(X * solver.eigenvectors(), solver.eigenvalues());
    };

    // ---- Initial guess (via the unified GuessEngine) ----------------------
    // For periodic, SAD lives at g=0 only — D depends on the unit-cell
    // atoms, not the lattice. HCORE / AUTO fall through to the Hcore-diag
    // path below.
    //
    // The Hcore diagonalisation is run unconditionally because KDIIS's
    // MO-basis error vector needs the initial C/eps, regardless of which
    // guess populates D.
    auto [C0, eps0] = diag(Hcore);
    Eigen::MatrixXd D;
    if (effective_guess == InitialGuess::READ) {
        const auto& source = opts.read_density;
        if (source.rows() != S.rows() || source.cols() != S.cols() || !source.allFinite()
            || (source - source.transpose()).norm() > 1e-8)
            throw std::invalid_argument("READ: density must be finite, Hermitian and match the AO basis");
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(source, Eigen::EigenvaluesOnly);
        if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() < -1e-8)
            throw std::invalid_argument("READ: density must be positive semidefinite");
        D = normalize_guess_density(source, S, 2 * nocc);
    } else if (effective_guess == InitialGuess::SAP) {
        // Periodic SAP: Fock-mode guess F_SAP(Γ) = T(Γ) + V_SAP(Γ), where
        // V_SAP is the lattice-summed SAP potential (Ewald split,
        // compute_vsap_lattice). Diagonalise it for the initial density — the
        // periodic analogue of the molecular SAP guess (V_SAP replaces V_ne).
        // A molecular Becke grid on the unit cell is adequate for the smooth
        // long-range part of V_SAP; the analytical erfc short-range carries
        // the sharp on-site structure.
        const Grid grid = build_grid(system.unit_cell_molecule());
        const auto Vsap_set = guess_ecp.active()
            ? compute_vsap_ecp_lattice(basis, system, opts.lattice_opts, guess_ecp)
            : compute_vsap_lattice(basis, system, grid, "sap_helfem_large", opts.lattice_opts);
        const Eigen::MatrixXd F_sap =
            fold_gamma_real(T_set) + fold_gamma_real(Vsap_set);
        D = build_density(diag(F_sap).first, nocc);
    } else if (effective_guess == InitialGuess::HUECKEL) {
        // Periodic HUECKEL: Fock-mode GWH guess built directly in the lattice
        // AO basis from S(g) and per-AO atomic energies, then diagonalised at
        // Γ just like SAP.
        const auto F_huckel_set = compute_huckel_fock_lattice(
            system.unit_cell_molecule(), basis, S_set, guess_ecp);
        D = build_density(diag(fold_gamma_real(F_huckel_set)).first, nocc);
    } else if (effective_guess == InitialGuess::MINAO) {
        // Periodic MINAO: project the ANO-RCC minimal-basis reference density
        // onto the working basis using the lattice-summed Γ overlaps
        // (S(Γ) and the lattice-summed cross overlap S_tm(Γ)). Density-mode
        // guess, like SAD — D is the on-site g=0 block.
        D = compute_minao_density_periodic(
            system.unit_cell_molecule(), basis, system, S, 2 * nocc,
            opts.lattice_opts, guess_ecp);
    } else if (effective_guess == InitialGuess::PATOM) {
        // Periodic PATOM: start from the SAD density and run the one-step
        // in-field re-polarisation with the Gamma periodic J/K builder.
        auto jk_guess = make_periodic_gamma_jk_builder(
            basis, system, opts.lattice_opts);
        const auto g = GuessEngine::build_closed_shell(
            system.unit_cell_molecule(), basis, nocc,
            effective_guess, &S, &Hcore, jk_guess.get(),
            SystemHints{/*is_periodic=*/true}, 1e-7, guess_ecp);
        D = g.D;
    } else {
        const auto g = GuessEngine::build_closed_shell(
            system.unit_cell_molecule(), basis, nocc,
            effective_guess, /*S=*/&S, /*Hcore=*/nullptr, /*jk=*/nullptr,
            SystemHints{/*is_periodic=*/true}, 1e-7, guess_ecp);
        if (g.D.size() != 0) {
            D = g.D;
        } else {
            D = build_density(C0, nocc);
        }
    }

    // ---- Nuclear repulsion per cell ---------------------------------------
    // Only the ionic repulsion uses effective charges. Atomic guesses and
    // every basis/chemical-identity consumer keep the physical system.
    auto ionic_system = system;
    if (guess_ecp.active()) {
        for (std::size_t a = 0; a < ionic_system.unit_cell.size(); ++a)
            ionic_system.unit_cell[a].Z = static_cast<int>(std::round(guess_ecp.effective_charges[a]));
    }
    const double e_nuc = nuclear_repulsion_per_cell(ionic_system, opts.lattice_opts);

    // ---- SCF loop ---------------------------------------------------------
    PeriodicRHFResult result;
    result.restart_basis = basis;
    result.restart_lattice = system.lattice;
    result.guess_selection = {opts.initial_guess, effective_guess, effective_guess};
    result.restart_kpoints = Eigen::MatrixXd::Zero(1, 3);
    result.restart_weights = Eigen::VectorXd::Ones(1);

    result.overlap = S;
    result.e_nuclear = e_nuc;

    DIIS diis = make_diis(opts);
    EDIIS ediis(opts.diis_subspace_size);
    ADIIS adiis(opts.diis_subspace_size);
    KDIIS kdiis(opts.diis_subspace_size);
    Eigen::MatrixXd D_prev = D;
    Eigen::MatrixXd F_prev_mixed;
    bool have_prev_fock = false;
    double E_prev = 0.0;
    Eigen::VectorXd eps_last;

    // Dynamic-damping state (Zerner-Hehenberger 1979). When
    // opts.dynamic_damping is false, current_damping == opts.damping
    // for the entire run (back-compat).
    double current_damping = opts.damping;
    bool have_prev_E = false;

    // Previous-iter MO basis (for KDIIS error vector). Initialized
    // from the initial-guess diagonalization.
    Eigen::MatrixXd C_prev_mo = C0;
    Eigen::VectorXd eps_prev_mo = eps0;

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        const bool diis_active =
            opts.use_diis && iter >= opts.diis_start_iter;
        const Eigen::MatrixXd D_used =
            (iter == 1 || current_damping == 0.0 || diis_active)
                ? D
                : (current_damping * D_prev + (1.0 - current_damping) * D);

        const JKMatrices jk =
            build_jk_gamma_molecular_limit(basis, system, opts.lattice_opts,
                                           D_used);
        Eigen::MatrixXd F = Hcore + jk.J - 0.5 * jk.K;

        // E_elec from the +U-free F first (same double-counting
        // discipline as cpp/src/rhf.cpp:278+ — see the note there).
        const double E_elec = 0.5 * (D_used.array() * (Hcore + F).array()).sum();

        // Dudarev +U at Γ. Closed-shell convention identical to the
        // molecular RHF path: pass P_σ = D_used / 2, multiply the
        // returned per-spin energy by 2.
        double e_dft_plus_u = 0.0;
        if (!opts.dft_plus_u_sites.empty()) {
            const auto vu = compute_dft_plus_u(
                opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                0.5 * D_used, S);
            F += vu.V;
            e_dft_plus_u = 2.0 * vu.energy;
        }

        const double E_total = E_elec + e_nuc + e_dft_plus_u;

        const Eigen::MatrixXd error = F * D_used * S - S * D_used * F;
        const double grad_norm = error.norm();
        const double dE = E_total - E_prev;

        const bool converged = is_scf_converged(
            iter, dE, grad_norm,
            opts.conv_tol_energy, opts.conv_tol_grad);

        int diis_subspace = 0;
        if (opts.use_diis) {
            Eigen::MatrixXd F_ext = F;
            switch (opts.scf_accelerator) {
                // R_CDIIS / AD_CDIIS share the DIIS extrapolation; the
                // adaptive depth policy is baked into ``diis`` (make_diis).
                case SCFAccelerator::DIIS:
                case SCFAccelerator::R_CDIIS:
                case SCFAccelerator::AD_CDIIS: {
                    F_ext = diis.extrapolate(F, error);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    break;
                }
                case SCFAccelerator::KDIIS: {
                    F_ext = kdiis.extrapolate(F, C_prev_mo,
                                              eps_prev_mo, nocc);
                    diis_subspace = static_cast<int>(kdiis.subspace_size());
                    break;
                }
                case SCFAccelerator::EDIIS: {
                    F_ext = ediis.extrapolate(F, D_used, E_total);
                    diis_subspace = static_cast<int>(ediis.subspace_size());
                    break;
                }
                case SCFAccelerator::EDIIS_DIIS: {
                    Eigen::MatrixXd F_d = diis.extrapolate(F, error);
                    Eigen::MatrixXd F_e =
                        ediis.extrapolate(F, D_used, E_total);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    // Intensive RMS metric (ediis.hpp) -- the raw
                    // Frobenius norm is size-extensive and over-holds
                    // the EDIIS regime (ddd8d325 defect class).
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm, static_cast<Eigen::Index>(F.rows()));
                    const bool take_ediis =
                        switch_metric > opts.ediis_diis_switch_threshold;
                    // A DIIS-branch cycle discards F_e; retract it so the
                    // anti-replay guard keys on consumed returns only.
                    if (!take_ediis) ediis.discard_last_extrapolation();
                    F_ext = take_ediis ? F_e : F_d;
                    break;
                }
                case SCFAccelerator::ADIIS: {
                    F_ext = adiis.extrapolate(F, D_used);
                    diis_subspace = static_cast<int>(adiis.subspace_size());
                    break;
                }
                case SCFAccelerator::ADIIS_DIIS: {
                    Eigen::MatrixXd F_d = diis.extrapolate(F, error);
                    Eigen::MatrixXd F_a = adiis.extrapolate(F, D_used);
                    diis_subspace = static_cast<int>(diis.subspace_size());
                    const double switch_metric =
                        ediis_diis_switch_metric(
                            grad_norm, static_cast<Eigen::Index>(F.rows()));
                    const bool take_adiis =
                        switch_metric > opts.ediis_diis_switch_threshold;
                    if (!take_adiis) adiis.discard_last_extrapolation();
                    F_ext = take_adiis ? F_a : F_d;
                    break;
                }
            }
            if (diis_active) {
                F = F_ext;
            } else {
                // Below diis_start_iter nothing extrapolated reaches the
                // SCF, so every branch's return is discarded. Same
                // produced-vs-consumed retraction as the hybrid above.
                ediis.discard_last_extrapolation();
                adiis.discard_last_extrapolation();
            }
        }
        if (opts.fock_mixing != 0.0) {
            if (have_prev_fock) {
                F = mix_fock_matrices(F, F_prev_mixed, opts.fock_mixing);
            }
            F_prev_mixed = F;
            have_prev_fock = true;
        }

        result.scf_trace.push_back(SCFIteration{
            iter, E_total, (iter == 1) ? 0.0 : dE, grad_norm, diis_subspace});

        // Saunders-Hillier level shift for this iteration, resolved and then
        // applied by the same two helpers the molecular drivers use (base
        // shift + warm-up / explicit schedule, then the shift operator). The
        // shift raises the virtual eigenvalues by b and leaves the occupied
        // block fixed, so it is inert at the converged density: it changes
        // the path, never the fixed point. ``apply_level_shift`` returns F
        // untouched when b is 0, so the S·D·S matmuls are skipped once the
        // warm-up releases. ``D_used`` is the closed-shell total density.
        const double b_shift = level_shift_at_iter(opts, iter);
        auto [C_new, eps_new] = diag(apply_level_shift(
            F, S, D_used, b_shift, LevelShiftDensity::TOTAL));
        D_prev = D_used;
        D = build_density(C_new, nocc);

        result.mo_energies = eps_new;
        result.mo_coeffs = C_new;
        result.density = D_used;
        result.fock = F;
        result.energy = E_total;
        result.e_electronic = E_elec;
        result.e_dft_plus_u = e_dft_plus_u;
        result.n_iter = iter;
        eps_last = eps_new;

        if (converged) {
            // Self-consistency pass on the fresh density (matches molecular
            // RHF convention for clean MO-energy output).
            const auto jk_f =
                build_jk_gamma_molecular_limit(basis, system,
                                               opts.lattice_opts, D);
            Eigen::MatrixXd F_final =
                Hcore + jk_f.J - 0.5 * jk_f.K;
            const double e_elec_final =
                0.5 * (D.array() * (Hcore + F_final).array()).sum();
            double e_dft_plus_u_final = 0.0;
            if (!opts.dft_plus_u_sites.empty()) {
                const auto vu_f = compute_dft_plus_u(
                    opts.dft_plus_u_sites, opts.dft_plus_u_ao_groups,
                    0.5 * D, S);
                F_final += vu_f.V;
                e_dft_plus_u_final = 2.0 * vu_f.energy;
            }
            auto [C_f, eps_f] = diag(F_final);
            result.mo_energies = eps_f;
            result.mo_coeffs = C_f;
            result.density = D;
            result.fock = F_final;
            result.e_electronic = e_elec_final;
            result.e_dft_plus_u = e_dft_plus_u_final;
            result.energy = e_elec_final + e_nuc + e_dft_plus_u_final;
            result.converged = true;
            break;
        }

        // Track previous-iter MO basis for KDIIS error vector.
        C_prev_mo = C_new;
        eps_prev_mo = eps_new;

        // Dynamic-damping update for next iter (Zerner-Hehenberger 1979).
        if (opts.dynamic_damping) {
            current_damping = update_dynamic_damping(
                current_damping, E_total, E_prev, have_prev_E,
                opts.dynamic_damping_min, opts.dynamic_damping_max);
        }
        have_prev_E = true;

        E_prev = E_total;
    }

    return result;
}

}  // namespace vibeqc
