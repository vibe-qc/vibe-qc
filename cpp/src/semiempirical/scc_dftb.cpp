#include "vibeqc/semiempirical/scc_dftb.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "vibeqc/diis.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"

namespace vibeqc {
namespace semiempirical {

namespace {

double repulsive_energy(const Molecule& mol,
                        const SemiempiricalParameters& params) {
    double E_rep = 0.0;
    const auto& atoms = mol.atoms();
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        for (std::size_t b = a + 1; b < atoms.size(); ++b) {
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R > 1e-12) {
                E_rep += params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
            }
        }
    }
    return E_rep;
}

}  // namespace

SCCDFTBResult run_scc_dftb(const Molecule& mol,
                           const SemiempiricalParameters& params,
                           const SCCOptions& opts) {
    // ---- Validate ----
    if (mol.multiplicity() != 1) {
        throw std::invalid_argument(
            "run_scc_dftb: only closed-shell supported");
    }
    for (const auto& atom : mol.atoms()) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error(
                "run_scc_dftb: element Z=" + std::to_string(atom.Z)
                + " not in parameter set");
        }
    }
    // Valence electrons only (see run_dftb0).
    const int n_val_e =
        SemiempiricalHamiltonianBuilder::valence_electron_count(mol, params);
    if (n_val_e % 2 != 0) {
        throw std::invalid_argument(
            "run_scc_dftb: only even valence electron counts supported");
    }

    // ---- Build basis, overlap, H⁰, γ matrix ----
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    // Cap at n_basis — a valence-minimal basis holds at most n_basis
    // doubly-occupied MOs; uncapped counts read past the eigensolution
    // (UB; see run_dftb0).
    int n_occ = n_val_e / 2;
    if (n_occ > n_basis) n_occ = n_basis;
    const auto n_atoms = static_cast<int>(mol.atoms().size());

    Eigen::MatrixXd S = SemiempiricalHamiltonianBuilder::build_overlap(basis);
    Eigen::MatrixXd H0 = SemiempiricalHamiltonianBuilder::build_hamiltonian_zero(
        basis, S, mol, params);
    Eigen::MatrixXd gamma = SemiempiricalHamiltonianBuilder::gamma_matrix(
        mol, params, opts.gamma_form);

    // ---- Initialize charges ----
    Eigen::VectorXd dq;
    if (opts.initial_charges.size() == 0) {
        dq = Eigen::VectorXd::Zero(n_atoms);
    } else {
        if (opts.initial_charges.size() != n_atoms ||
            !opts.initial_charges.allFinite()) {
            throw std::invalid_argument(
                "SCC-DFTB: initial_charges must be finite and match n_atoms");
        }
        dq = opts.initial_charges;
    }
    double E_rep = repulsive_energy(mol, params);

    // ---- SCF loop ----
    SCCDFTBResult result;
    const double mix = std::max(0.0, std::min(1.0, opts.charge_mixing));
    // Adaptive damping state. Vector Aitken relaxation estimates the scalar
    // mixing fraction from consecutive Mulliken residuals. Unlike the old
    // one-way "halve on anticorrelation" rule, it can recover after an early
    // oscillation instead of spending the rest of an otherwise contractive
    // SCC cycle at the 0.01 floor. The caller's requested fraction remains a
    // hard upper bound, and the convergence test itself is unchanged.
    double mix_cur = mix;
    Eigen::VectorXd resid_prev;
    KPointOccupationOptions occupation_options;

    // ---- DIIS charge accelerator (Pulay 1980/1982) ----
    //
    // The SCC fixed-point problem r(q) = q_out(q) − q = 0 is solved
    // analogously to the SCF Fock problem. Without DIIS the loop relies
    // on 2-point vector Aitken relaxation (see else-branch below),
    // which can stall or oscillate on heteronuclear systems whose charge
    // response couples strongly through the γ matrix (e.g. purine rings).
    // DIIS builds a subspace from the last N (q_out, r) pairs and finds
    // the linear combination Σ c_i q_out^i that minimises ‖Σ c_i r_i‖;
    // this is numerically equivalent to Anderson mixing and is the
    // standard SCC-DFTB convergence strategy (DFTB+, tblite, xtb).
    std::unique_ptr<DIIS> charge_diis;
    if (opts.use_diis) {
        int diis_depth = std::max(2, std::min(12, opts.diis_subspace));
        charge_diis = std::make_unique<DIIS>(diis_depth);
    }

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        // Build SCC Hamiltonian
        Eigen::MatrixXd H = SemiempiricalHamiltonianBuilder::build_scc_hamiltonian(
            H0, S, basis, mol, params, gamma, dq);

        // Diagonalize
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H, S);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("SCC-DFTB: eigendecomposition failed at iter "
                                     + std::to_string(iter));
        }
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();

        // Build an electron-conserving closed-shell density. A nonzero
        // electronic temperature smooths frontier-level crossings without
        // changing the SCC Hamiltonian or charge functional.
        occupation_options.smearing_temperature =
            opts.electronic_temperature;
        auto occupation = compute_closed_shell_kpoint_occupations(
            {eps}, {1.0}, n_val_e, n_occ, occupation_options);
        Eigen::MatrixXd D =
            C * occupation.occupations_per_k.front().asDiagonal() *
            C.transpose();

        // Compute new charges
        Eigen::VectorXd dq_new = SemiempiricalHamiltonianBuilder::mulliken_charges(
            basis, D, S, mol, params);

        // Check convergence
        Eigen::VectorXd resid = dq_new - dq;
        double max_change = resid.cwiseAbs().maxCoeff();
        bool converged = (max_change < opts.conv_tol_charge);

        // ---- Charge mixing / acceleration ----
        if (charge_diis) {
            // DIIS extrapolation on the SCC potential V = γ·Δq, which
            // is a smoother mapping than the raw charge vector. This
            // follows the standard SCC-DFTB acceleration strategy used
            // by DFTB+ and tblite (Anderson mixing on the potential).
            // Pulay, Chem. Phys. Lett. 73, 393 (1980);
            //        J. Comput. Chem. 3, 556 (1982).
            Eigen::VectorXd V_new = gamma * dq_new;
            Eigen::VectorXd V_old = gamma * dq;
            Eigen::VectorXd resid_V = V_new - V_old;

            Eigen::MatrixXd V_mat = V_new;
            Eigen::MatrixXd err_mat = resid_V;
            Eigen::MatrixXd V_diis =
                charge_diis->extrapolate(V_mat, err_mat);
            Eigen::VectorXd V_diis_vec = V_diis.col(0);

            // The raw DIIS/Anderson extrapolation can overstep when the
            // SCC response is nonlinear (heteronuclear / π-conjugated
            // systems).  Production SCC-DFTB codes damp the Anderson
            // step by the same mixing fraction used in simple mixing:
            //   q_new = (1 − α) q_old + α q_anderson.
            // We recover q_anderson = γ⁻¹ V_diis, then damp it.
            //
            // A shallow DIIS history (< 2 iterates) cannot extrapolate
            // at all; extrapolate() returns the raw V_new, equivalent
            // to mix = 1.0.  We require at least 3 iterates before
            // trusting the extrapolation.
            //
            // DIIS is excellent at the initial charge sloshing but can
            // oscillate in the final convergence tail (max |Δq| ≲ 1e-3).
            // When the residual is already small, we hand off to Aitken
            // relaxation, which is more contractive in that regime.
            const bool subspace_ready =
                charge_diis->subspace_size() >= 3
                && max_change > 1.0e-5;

            bool diis_used = false;
            if (subspace_ready) {
                // Recover charges from the DIIS-optimised potential.
                // γ is SPD for any finite geometry, so the linear solve
                // is well-conditioned. Fall back to mixing when the
                // solve fails or the recovered charges are unphysical.
                bool recovered = false;
                Eigen::VectorXd dq_candidate;
                if (gamma.rows() > 0) {
                    Eigen::LLT<Eigen::MatrixXd> llt(gamma);
                    if (llt.info() == Eigen::Success) {
                        dq_candidate = llt.solve(V_diis_vec);
                        recovered = dq_candidate.allFinite();
                    }
                }
                if (!recovered) {
                    // Diagonal fallback: divide by on-site Hubbard U.
                    dq_candidate.resize(n_atoms);
                    for (int a = 0; a < n_atoms; ++a) {
                        dq_candidate(a) =
                            V_diis_vec(a) / gamma(a, a);
                    }
                    recovered = dq_candidate.allFinite();
                }

                // Sanity-check the recovered charges: no atom should
                // carry a charge fluctuation exceeding its valence
                // count by more than a generous margin. DIIS can
                // over-extrapolate and produce unphysical ionicity
                // when the subspace is nearly linearly dependent.
                if (recovered) {
                    bool physical = true;
                    for (int a = 0; a < n_atoms; ++a) {
                        int Z_a = mol.atoms()[a].Z;
                        int n_val =
                            params.valence_electrons(Z_a);
                        if (std::abs(dq_candidate(a))
                            > static_cast<double>(n_val) + 2.0) {
                            physical = false;
                            break;
                        }
                    }
                    if (physical) {
                        // Damp the Anderson step:
                        //   dq_new_input = dq + mix * (dq_anderson − dq).
                        // This is the standard production recipe that
                        // keeps the SCC loop contractive even when the
                        // raw Anderson extrapolation is poor.
                        dq += mix_cur
                              * (dq_candidate - dq);
                        diis_used = true;
                    }
                }
            }

            if (diis_used) {
                // Enforce total charge conservation.
                double drift =
                    (dq.sum() - static_cast<double>(mol.charge()))
                    / static_cast<double>(n_atoms);
                dq.array() -= drift;
            } else {
                // DIIS extrapolation rejected (shallow history,
                // singular solve, or unphysical charges).  Fall back
                // to damped Aitken mixing for this step.
                if (resid_prev.size() > 0 && mix > 0.0) {
                    const Eigen::VectorXd delta_resid =
                        resid - resid_prev;
                    const double denom = delta_resid.squaredNorm();
                    if (denom > 1e-24) {
                        const double estimate =
                            -mix_cur * resid_prev.dot(delta_resid)
                            / denom;
                        if (std::isfinite(estimate)
                            && estimate > 0.0) {
                            const double mix_floor =
                                std::min(0.01, mix);
                            mix_cur = std::max(
                                mix_floor,
                                std::min(mix, estimate));
                        } else if (resid.dot(resid_prev) < 0.0) {
                            mix_cur = std::max(
                                0.5 * mix_cur,
                                std::min(0.01, mix));
                        }
                    }
                }
                dq += mix_cur * resid;
            }

            resid_prev = resid;
        } else {
            // Fallback: vector Aitken relaxation (Irons & Tuck,
            // Int. J. Numer. Methods Eng. 1, 275–277, 1969):
            //   α_k = −α_{k−1} r_{k−1}·(r_k−r_{k−1}) / ||r_k−r_{k−1}||².
            // Finite-temperature retries use a separately validated
            // one-way damping path.
            if (resid_prev.size() > 0 && opts.electronic_temperature > 0.0) {
                if (resid.dot(resid_prev) < 0.0) {
                    mix_cur = std::max(0.5 * mix_cur, std::min(0.01, mix));
                }
            } else if (resid_prev.size() > 0 && mix > 0.0) {
                const Eigen::VectorXd delta_resid = resid - resid_prev;
                const double denom = delta_resid.squaredNorm();
                if (denom > 1e-24) {
                    const double estimate =
                        -mix_cur * resid_prev.dot(delta_resid) / denom;
                    if (std::isfinite(estimate) && estimate > 0.0) {
                        const double mix_floor = std::min(0.01, mix);
                        mix_cur = std::max(mix_floor, std::min(mix, estimate));
                    } else if (resid.dot(resid_prev) < 0.0) {
                        mix_cur = std::max(0.5 * mix_cur, std::min(0.01, mix));
                    }
                }
            }
            resid_prev = resid;
            dq += mix_cur * resid;
        }

        if (converged) {
            // Variational free energy (Elstner et al., PRB 58, 7260
            // (1998), Eq. 4):  E = tr(D H⁰) + ½ ΣΔqγΔq + E_rep.
            // At finite electronic temperature the Mermin -T*S term makes
            // this the stationary potential whose derivative is the force.
            // tr(D H⁰) is computed directly rather than as Σnε − tr(D H¹).
            //
            // Use dq_new (the Mulliken charges consistent with D), not the
            // mixed/extrapolated dq, so that E_scc is variational in D.
            // The mixed dq steers the next iteration; dq_new is the
            // self-consistent charge at this density.
            double E_elec = D.cwiseProduct(H0).sum();
            double E_entropy =
                opts.electronic_temperature * occupation.entropy;

            // Second-order charge-fluctuation energy: ½ Σ_{AB} γ_{AB} Δq_A Δq_B
            double E_scc = 0.5 * dq_new.dot(gamma * dq_new);

            result.energy = E_elec - E_entropy + E_scc + E_rep;
            result.e_electronic = E_elec - E_entropy;
            result.e_repulsive = E_rep;
            result.e_scc = E_scc;
            result.mo_energies = std::move(eps);
            result.mo_coeffs = std::move(C);
            result.density = std::move(D);
            result.overlap = S;
            result.hamiltonian = H;
            result.charges = dq_new;
            result.n_basis = n_basis;
            result.n_occ = n_occ;
            result.n_iter = iter;
            result.converged = true;
            return result;
        }
    }

    // Not converged — return last iteration. Evaluate the functional
    // consistently at the returned density: E₂ uses the Mulliken charges
    // OF that density, not the mixed iterate (an inconsistent (D, Δq)
    // pair reports an energy that is not the functional value at all).
    Eigen::MatrixXd H = SemiempiricalHamiltonianBuilder::build_scc_hamiltonian(
        H0, S, basis, mol, params, gamma, dq);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H, S);
    Eigen::VectorXd eps = solver.eigenvalues();
    Eigen::MatrixXd C = solver.eigenvectors();
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    Eigen::MatrixXd D = 2.0 * C_occ * C_occ.transpose();
    dq = SemiempiricalHamiltonianBuilder::mulliken_charges(basis, D, S, mol, params);
    double E_elec = D.cwiseProduct(H0).sum();
    double E_scc = 0.5 * dq.dot(gamma * dq);

    result.energy = E_elec + E_scc + E_rep;
    result.e_electronic = E_elec;
    result.e_repulsive = E_rep;
    result.e_scc = E_scc;
    result.mo_energies = eps;
    result.mo_coeffs = C;
    result.density = std::move(D);
    result.overlap = S;
    result.hamiltonian = H;
    result.charges = dq;
    result.n_basis = n_basis;
    result.n_occ = n_occ;
    result.n_iter = opts.max_iter;
    result.converged = false;
    return result;
}


USCCDFTBResult run_uscc_dftb(const Molecule& mol,
                              const SemiempiricalParameters& params,
                              const SCCOptions& opts) {
    for (const auto& atom : mol.atoms()) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error("run_uscc_dftb: unsupported element Z="
                                     + std::to_string(atom.Z));
        }
    }
    int mult = mol.multiplicity();
    // Valence electrons only (see run_dftb0).
    int n_el =
        SemiempiricalHamiltonianBuilder::valence_electron_count(mol, params);
    int n_alpha = (n_el + mult - 1) / 2;
    int n_beta = (n_el - mult + 1) / 2;

    Molecule mol_unit = mol; // for unit_cell_molecule equivalent
    BasisSet basis = SemiempiricalBasis::build(mol_unit, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    if (n_alpha > n_basis) n_alpha = n_basis;
    if (n_beta > n_basis) n_beta = n_basis;
    const int n_atoms = static_cast<int>(mol.atoms().size());

    Eigen::MatrixXd S = SemiempiricalHamiltonianBuilder::build_overlap(basis);
    Eigen::MatrixXd H0 = SemiempiricalHamiltonianBuilder::build_hamiltonian_zero(
        basis, S, mol, params);
    Eigen::MatrixXd gamma = SemiempiricalHamiltonianBuilder::gamma_matrix(
        mol, params, opts.gamma_form);

    Eigen::VectorXd dq;
    if (opts.initial_charges.size() == 0) {
        dq = Eigen::VectorXd::Zero(n_atoms);
    } else {
        if (opts.initial_charges.size() != n_atoms ||
            !opts.initial_charges.allFinite()) {
            throw std::invalid_argument(
                "USCC-DFTB: initial_charges must be finite and match n_atoms");
        }
        dq = opts.initial_charges;
    }
    const double mix = std::max(0.0, std::min(1.0, opts.charge_mixing));
    // Bounded vector Aitken state (see run_scc_dftb).
    double mix_cur = mix;
    Eigen::VectorXd resid_prev;

    // Repulsive energy
    double E_rep = 0.0;
    const auto& atoms = mol.atoms();
    for (std::size_t a = 0; a < atoms.size(); ++a)
        for (std::size_t b = a + 1; b < atoms.size(); ++b) {
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R > 1e-12) E_rep += params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
        }

    USCCDFTBResult result;

    // ---- DIIS charge accelerator ----
    std::unique_ptr<DIIS> charge_diis;
    if (opts.use_diis) {
        int diis_depth = std::max(2, std::min(12, opts.diis_subspace));
        charge_diis = std::make_unique<DIIS>(diis_depth);
    }

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        Eigen::VectorXd V = gamma * dq;
        Eigen::MatrixXd H_scc = H0;
        // Add SCC correction using the same pattern as closed-shell
        const auto& shells = basis.shells();
        const auto& lib_shells = basis.libint();
        const auto shell2bf = lib_shells.shell2bf();
        for (int mu = 0; mu < n_basis; ++mu) {
            int a_mu = -1;
            for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                int bf0 = shell2bf[s]; int nf = lib_shells[s].size();
                if (mu >= bf0 && mu < bf0 + nf) { a_mu = shells[s].atom_index; break; }
            }
            double V_mu = V(a_mu);
            for (int nu = 0; nu < n_basis; ++nu) {
                int a_nu = -1;
                for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                    int bf0 = shell2bf[s]; int nf = lib_shells[s].size();
                    if (nu >= bf0 && nu < bf0 + nf) { a_nu = shells[s].atom_index; break; }
                }
                // Elstner Eq. 14 with flipped sign for the Δq > 0 = cation
                // convention (see build_scc_hamiltonian).
                H_scc(mu, nu) -= 0.5 * S(mu, nu) * (V_mu + V(a_nu));
            }
        }

        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, S);
        if (solver.info() != Eigen::Success) throw std::runtime_error("USCC: diag failed");
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();

        Eigen::MatrixXd C_occ_a = C.leftCols(n_alpha);
        Eigen::MatrixXd C_occ_b = C.leftCols(n_beta);
        Eigen::MatrixXd D_a = C_occ_a * C_occ_a.transpose();
        Eigen::MatrixXd D_b = C_occ_b * C_occ_b.transpose();
        Eigen::MatrixXd D_total = D_a + D_b;

        Eigen::VectorXd dq_new = SemiempiricalHamiltonianBuilder::mulliken_charges(
            basis, D_total, S, mol, params);

        Eigen::VectorXd resid = dq_new - dq;
        double max_change = resid.cwiseAbs().maxCoeff();
        bool converged = (max_change < opts.conv_tol_charge);

        // ---- Charge mixing / acceleration ----
        if (charge_diis) {
            Eigen::VectorXd V_new = gamma * dq_new;
            Eigen::VectorXd V_old = gamma * dq;
            Eigen::VectorXd resid_V = V_new - V_old;

            Eigen::MatrixXd V_mat = V_new;
            Eigen::MatrixXd err_mat = resid_V;
            Eigen::MatrixXd V_diis =
                charge_diis->extrapolate(V_mat, err_mat);
            Eigen::VectorXd V_diis_vec = V_diis.col(0);

            // See run_scc_dftb for rationale — shallow DIIS history
            // returns the raw V_new; taking the full charge step
            // destabilises heteronuclear / π-conjugated systems.
            // DIIS hands off to Aitken when the residual falls below
            // 1e-5 to avoid oscillation in the convergence tail.
            const bool subspace_ready =
                charge_diis->subspace_size() >= 3
                && max_change > 1.0e-5;

            bool diis_used = false;
            if (subspace_ready) {
                bool recovered = false;
                Eigen::VectorXd dq_candidate;
                if (gamma.rows() > 0) {
                    Eigen::LLT<Eigen::MatrixXd> llt(gamma);
                    if (llt.info() == Eigen::Success) {
                        dq_candidate = llt.solve(V_diis_vec);
                        recovered = dq_candidate.allFinite();
                    }
                }
                if (!recovered) {
                    dq_candidate.resize(n_atoms);
                    for (int a = 0; a < n_atoms; ++a) {
                        dq_candidate(a) =
                            V_diis_vec(a) / gamma(a, a);
                    }
                    recovered = dq_candidate.allFinite();
                }

                if (recovered) {
                    bool physical = true;
                    for (int a = 0; a < n_atoms; ++a) {
                        int Z_a = mol.atoms()[a].Z;
                        int n_val =
                            params.valence_electrons(Z_a);
                        if (std::abs(dq_candidate(a))
                            > static_cast<double>(n_val) + 2.0) {
                            physical = false;
                            break;
                        }
                    }
                    if (physical) {
                        dq += mix_cur
                              * (dq_candidate - dq);
                        diis_used = true;
                    }
                }
            }

            if (diis_used) {
                double drift =
                    (dq.sum() - static_cast<double>(mol.charge()))
                    / static_cast<double>(n_atoms);
                dq.array() -= drift;
            } else {
                if (resid_prev.size() > 0 && mix > 0.0) {
                    const Eigen::VectorXd delta_resid =
                        resid - resid_prev;
                    const double denom = delta_resid.squaredNorm();
                    if (denom > 1e-24) {
                        const double estimate =
                            -mix_cur * resid_prev.dot(delta_resid)
                            / denom;
                        if (std::isfinite(estimate)
                            && estimate > 0.0) {
                            const double mix_floor =
                                std::min(0.01, mix);
                            mix_cur = std::max(
                                mix_floor,
                                std::min(mix, estimate));
                        } else if (resid.dot(resid_prev) < 0.0) {
                            mix_cur = std::max(
                                0.5 * mix_cur,
                                std::min(0.01, mix));
                        }
                    }
                }
                dq += mix_cur * resid;
            }

            resid_prev = resid;
        } else {
            if (resid_prev.size() > 0 && mix > 0.0) {
                const Eigen::VectorXd delta_resid = resid - resid_prev;
                const double denom = delta_resid.squaredNorm();
                if (denom > 1e-24) {
                    const double estimate =
                        -mix_cur * resid_prev.dot(delta_resid) / denom;
                    if (std::isfinite(estimate) && estimate > 0.0) {
                        const double mix_floor = std::min(0.01, mix);
                        mix_cur = std::max(mix_floor, std::min(mix, estimate));
                    } else if (resid.dot(resid_prev) < 0.0) {
                        mix_cur = std::max(0.5 * mix_cur, std::min(0.01, mix));
                    }
                }
            }
            resid_prev = resid;
            dq += mix_cur * resid;
        }

        if (converged) {
            // Variational assembly, as in run_scc_dftb.
            // Use dq_new (consistent with D_total) for the energy.
            double E_elec = D_total.cwiseProduct(H0).sum();
            double E_scc = 0.5 * dq_new.dot(gamma * dq_new);
            result.energy = E_elec + E_scc + E_rep;
            result.e_electronic = E_elec; result.e_repulsive = E_rep; result.e_scc = E_scc;
            result.mo_energies = std::move(eps); result.mo_coeffs = std::move(C);
            result.density_alpha = std::move(D_a); result.density_beta = std::move(D_b);
            result.overlap = S; result.hamiltonian = H_scc; result.charges = dq_new;
            result.n_basis = n_basis; result.n_alpha = n_alpha; result.n_beta = n_beta;
            result.n_iter = iter; result.converged = true;
            return result;
        }
    }

    // Not converged — still populate the full result from the last mixed
    // charges (consistent functional at the returned density, as in
    // run_scc_dftb). Pre-fix this path returned default-initialized
    // matrices, and compute_uscc_dftb_gradient on such a result read past
    // the empty eigenvector matrix (segfault).
    {
        Eigen::VectorXd V = gamma * dq;
        Eigen::MatrixXd H_scc = H0;
        const auto& shells = basis.shells();
        const auto& lib_shells = basis.libint();
        const auto shell2bf = lib_shells.shell2bf();
        std::vector<int> ao_atom(n_basis);
        for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
            int bf0 = shell2bf[s]; int nf = lib_shells[s].size();
            for (int i = 0; i < nf; ++i) ao_atom[bf0 + i] = shells[s].atom_index;
        }
        for (int mu = 0; mu < n_basis; ++mu) {
            double V_mu = V(ao_atom[mu]);
            for (int nu = 0; nu < n_basis; ++nu) {
                H_scc(mu, nu) -= 0.5 * S(mu, nu) * (V_mu + V(ao_atom[nu]));
            }
        }
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, S);
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();
        Eigen::MatrixXd C_occ_a = C.leftCols(n_alpha);
        Eigen::MatrixXd C_occ_b = C.leftCols(n_beta);
        Eigen::MatrixXd D_a = C_occ_a * C_occ_a.transpose();
        Eigen::MatrixXd D_b = C_occ_b * C_occ_b.transpose();
        Eigen::MatrixXd D_total = D_a + D_b;
        dq = SemiempiricalHamiltonianBuilder::mulliken_charges(
            basis, D_total, S, mol, params);
        double E_elec = D_total.cwiseProduct(H0).sum();
        double E_scc = 0.5 * dq.dot(gamma * dq);
        result.energy = E_elec + E_scc + E_rep;
        result.e_electronic = E_elec; result.e_repulsive = E_rep; result.e_scc = E_scc;
        result.mo_energies = std::move(eps); result.mo_coeffs = std::move(C);
        result.density_alpha = std::move(D_a); result.density_beta = std::move(D_b);
        result.overlap = S; result.hamiltonian = std::move(H_scc); result.charges = dq;
        result.n_basis = n_basis; result.n_alpha = n_alpha; result.n_beta = n_beta;
    }
    result.n_iter = opts.max_iter; result.converged = false;
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
