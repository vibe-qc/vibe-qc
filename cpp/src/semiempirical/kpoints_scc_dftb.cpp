#include "vibeqc/semiempirical/kpoints_scc_dftb.hpp"

#include "vibeqc/semiempirical/core/pair_lattice.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "vibeqc/diis.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/periodic_scc_dftb.hpp"

namespace vibeqc {
namespace semiempirical {

void validate_scc_dftb_kpoint_options(
    const SCCOptions& options,
    const char* caller) {
    const std::string prefix = std::string(caller) + ": ";
    if (options.max_iter < 1) {
        throw std::invalid_argument(prefix + "max_iter must be >= 1");
    }
    if (!std::isfinite(options.conv_tol_charge) ||
        options.conv_tol_charge <= 0.0) {
        throw std::invalid_argument(
            prefix + "conv_tol_charge must be finite and > 0");
    }
    if (!std::isfinite(options.charge_mixing) ||
        options.charge_mixing <= 0.0 || options.charge_mixing > 1.0) {
        throw std::invalid_argument(
            prefix + "charge_mixing must be finite and in (0, 1]");
    }
}

KPointSCCDFTBResult run_scc_dftb_kpoints(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    const SCCOptions& scc_opts,
    double cutoff_bohr,
    const KPointOccupationOptions& occupation_options) {

    validate_kpoint_occupation_options(
        occupation_options, "run_scc_dftb_kpoints");
    const int n_val_e = validate_dftb_kpoint_request(
        system, params, kmesh, cutoff_bohr, "run_scc_dftb_kpoints");
    validate_scc_dftb_kpoint_options(scc_opts, "run_scc_dftb_kpoints");
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    const auto cells_for_guard =
        atom_pair_interaction_cells(system, cutoff_bohr);
    require_nonzero_lattice_image(
        cells_for_guard, cutoff_bohr, "run_scc_dftb_kpoints");

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    // Cap at n_basis — a valence-minimal basis holds at most n_basis
    // doubly-occupied MOs per k-point; uncapped counts read past the
    // eigensolution (UB; see run_dftb0).
    int n_occ = n_val_e / 2;
    if (n_occ > n_basis) n_occ = n_basis;
    const int n_atoms = static_cast<int>(system.unit_cell.size());

    // Build aligned H0(g), S(g), and cell blocks once.
    const auto lattice =
        build_h0_s_per_cell(basis, system, params, cutoff_bohr);
    const auto& cells = lattice.cells;

    // Periodic γ matrix
    const auto gamma =
        build_periodic_dftb_gamma_matrix(system, params, cutoff_bohr);

    // Repulsive energy (reuse gamma-point DFTB0)
    const auto& atoms = system.unit_cell;
    double E_rep = 0.0;
    for (const auto& cell : cells) {
        bool is_zero = (cell.index.array() == 0).all();
        for (int a = 0; a < n_atoms; ++a) {
            int start_b = is_zero ? a + 1 : 0;
            for (int b = start_b; b < n_atoms; ++b) {
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + cell.r_cart[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + cell.r_cart[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + cell.r_cart[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > cutoff_bohr) continue;
                double factor = is_zero ? 1.0 : 0.5;
                E_rep += factor * params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
            }
        }
    }

    // SCC SCF loop
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(n_atoms);
    const double mix = scc_opts.charge_mixing;
    // Adaptive damping state (see run_scc_dftb).
    double mix_cur = mix;
    Eigen::VectorXd resid_prev;
    const auto& shells = basis.shells();
    const auto& lib_shells = basis.libint();
    const auto shell2bf = lib_shells.shell2bf();
    std::vector<int> ao_atom(n_basis);
    for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
        const int bf0 = shell2bf[s];
        const int nf = lib_shells[s].size();
        for (int i = 0; i < nf; ++i) {
            ao_atom[bf0 + i] = shells[s].atom_index;
        }
    }

    KPointSCCDFTBResult result;
    result.smearing_temperature = occupation_options.smearing_temperature;

    // ---- DIIS charge accelerator ----
    std::unique_ptr<DIIS> charge_diis;
    if (scc_opts.use_diis) {
        int diis_depth = std::max(2, std::min(12, scc_opts.diis_subspace));
        charge_diis = std::make_unique<DIIS>(diis_depth);
    }

    std::vector<Eigen::MatrixXd> H_scc_blocks = lattice.h0;
    std::vector<ComplexMatrix> overlap_per_k;
    overlap_per_k.reserve(kmesh.size());
    for (const auto& kpoint : kmesh.kpoints) {
        ComplexMatrix overlap = ComplexMatrix::Zero(n_basis, n_basis);
        for (std::size_t ci = 0; ci < cells.size(); ++ci) {
            const double phase = kpoint.dot(cells[ci].r_cart);
            const std::complex<double> eikr(
                std::cos(phase), std::sin(phase));
            overlap +=
                eikr * lattice.overlap[ci].cast<std::complex<double>>();
        }
        overlap = 0.5 * (overlap + overlap.adjoint().eval());
        overlap_per_k.push_back(std::move(overlap));
    }
    for (int iter = 1; iter <= scc_opts.max_iter; ++iter) {
        // SCC potential
        Eigen::VectorXd V = gamma * dq;

        // Build SCC correction to H⁰(g) for each cell
        // H1_{mu,nu}(g) = -0.5 S_{mu,nu}(g) (V_A + V_B).
        for (std::size_t ci = 0; ci < cells.size(); ++ci) {
            auto& H_g = H_scc_blocks[ci];
            H_g = lattice.h0[ci];
            const auto& S_g = lattice.overlap[ci];
            for (int mu = 0; mu < n_basis; ++mu) {
                const double Vmu = V(ao_atom[mu]);
                for (int nu = 0; nu < n_basis; ++nu) {
                    // Elstner Eq. 14 with flipped sign (Δq > 0 = cation,
                    // see build_scc_hamiltonian).
                    H_g(mu, nu) -=
                        0.5 * S_g(mu, nu) * (Vmu + V(ao_atom[nu]));
                }
            }
        }

        // Diagonalize at each k-point and accumulate Mulliken populations.
        Eigen::VectorXd populations = Eigen::VectorXd::Zero(n_atoms);
        double E_band = 0.0;
        std::vector<Eigen::VectorXd> eps_per_k;
        eps_per_k.reserve(kmesh.size());
        std::vector<BandDiag> bands_per_k;
        bands_per_k.reserve(kmesh.size());

        for (std::size_t ik = 0; ik < kmesh.size(); ++ik) {
            const auto& k = kmesh.kpoints[ik];

            Eigen::MatrixXcd Hk = Eigen::MatrixXcd::Zero(n_basis, n_basis);
            for (std::size_t ci = 0; ci < cells.size(); ++ci) {
                double phase = k.dot(cells[ci].r_cart);
                std::complex<double> eikr(std::cos(phase), std::sin(phase));
                Hk += eikr * H_scc_blocks[ci].cast<std::complex<double>>();
            }

            Hk = 0.5 * (Hk + Hk.adjoint().eval());
            auto bands = diagonalize_bloch(Hk, overlap_per_k[ik]);
            eps_per_k.push_back(bands.energies);
            bands_per_k.push_back(std::move(bands));
        }

        auto occupation = compute_closed_shell_kpoint_occupations(
            eps_per_k, kmesh.weights, n_val_e, n_occ, occupation_options);
        for (std::size_t ik = 0; ik < kmesh.size(); ++ik) {
            const double wk = kmesh.weights.empty()
                                  ? 1.0 / static_cast<double>(kmesh.size())
                                  : kmesh.weights[ik];
            const auto& bands = bands_per_k[ik];
            Eigen::MatrixXcd Dk =
                bands.coefficients *
                occupation.occupations_per_k[ik].asDiagonal() *
                bands.coefficients.adjoint();
            for (int band = 0; band < n_basis; ++band) {
                E_band += wk * occupation.occupations_per_k[ik](band) *
                          bands.energies(band);
            }
            const Eigen::MatrixXcd DS = Dk * overlap_per_k[ik];

            for (int mu = 0; mu < n_basis; ++mu) {
                populations(ao_atom[mu]) += wk * DS(mu, mu).real();
            }
        }

        Eigen::VectorXd dq_new(n_atoms);
        for (int a = 0; a < n_atoms; ++a) {
            dq_new(a) = params.valence_electrons(atoms[a].Z) - populations(a);
        }
        const double charge_excess =
            (dq_new.sum() - static_cast<double>(system.charge)) / n_atoms;
        dq_new.array() -= charge_excess;

        Eigen::VectorXd resid = dq_new - dq;
        double max_change = resid.cwiseAbs().maxCoeff();
        bool converged = (max_change < scc_opts.conv_tol_charge);

        // ---- Charge mixing / acceleration ----
        if (charge_diis) {
            // DIIS on the charge vector directly (same recipe as run_scc_dftb).
            Eigen::MatrixXd dq_mat = dq_new;
            Eigen::MatrixXd err_mat = resid;
            Eigen::MatrixXd dq_diis_mat =
                charge_diis->extrapolate(dq_mat, err_mat);

            const bool subspace_ready =
                charge_diis->subspace_size() >= 3
                && max_change > 1.0e-5;

            bool diis_used = false;
            if (subspace_ready) {
                Eigen::VectorXd dq_candidate = dq_diis_mat.col(0);
                bool physical = dq_candidate.allFinite();
                if (physical) {
                    for (int a = 0; a < n_atoms; ++a) {
                        if (std::abs(dq_candidate(a))
                            > params.valence_electrons(atoms[a].Z) + 2.0) {
                            physical = false;
                            break;
                        }
                    }
                }
                // DIIS trust-region guard (same rationale as run_scc_dftb).
                if (physical) {
                    const double E_scc_cur = 0.5 * dq.dot(gamma * dq);
                    const double E_scc_cand =
                        0.5 * dq_candidate.dot(gamma * dq_candidate);
                    const double delta_E_scc = E_scc_cand - E_scc_cur;
                    const double abs_floor =
                        0.1 * static_cast<double>(n_atoms);
                    const double trust_radius = std::max(
                        abs_floor, std::abs(E_scc_cur));
                    if (std::abs(delta_E_scc) > trust_radius) {
                        physical = false;
                    }
                }
                if (physical) {
                    dq += mix_cur * (dq_candidate - dq);
                    diis_used = true;
                }
            }

            if (diis_used) {
                const double drift =
                    (dq.sum() - static_cast<double>(system.charge)) / n_atoms;
                dq.array() -= drift;
            } else {
                if (resid_prev.size() > 0 && mix > 0.0) {
                    const Eigen::VectorXd delta_resid =
                        resid - resid_prev;
                    const double denom = delta_resid.squaredNorm();
                    if (denom > 1e-24) {
                        const double estimate =
                            -mix_cur * resid_prev.dot(delta_resid) / denom;
                        if (std::isfinite(estimate) && estimate > 0.0) {
                            const double mix_floor =
                                std::min(0.01, mix);
                            mix_cur = std::max(
                                mix_floor, std::min(mix, estimate));
                        } else if (resid.dot(resid_prev) < 0.0) {
                            mix_cur = std::max(
                                0.5 * mix_cur, std::min(0.01, mix));
                        }
                    }
                }
                dq += mix_cur * resid;
            }
            resid_prev = resid;
        } else {
            // Fallback: vector Aitken relaxation
            if (resid_prev.size() > 0 && mix > 0.0) {
                const Eigen::VectorXd delta_resid =
                    resid - resid_prev;
                const double denom = delta_resid.squaredNorm();
                if (denom > 1e-24) {
                    const double estimate =
                        -mix_cur * resid_prev.dot(delta_resid) / denom;
                    if (std::isfinite(estimate) && estimate > 0.0) {
                        const double mix_floor =
                            std::min(0.01, mix);
                        mix_cur = std::max(
                            mix_floor, std::min(mix, estimate));
                    } else if (resid.dot(resid_prev) < 0.0) {
                        mix_cur = std::max(
                            0.5 * mix_cur, std::min(0.01, mix));
                    }
                }
            }
            resid_prev = resid;
            dq += mix_cur * resid;
        }

        if (converged) {
            // Issue #434: refuse BEFORE assembling a success record, and
            // only here -- on the accepted spectrum. An intermediate iterate
            // that transits a near-degenerate frontier is not an error.
            reject_unresolved_frontier_cut(
                occupation, occupation_options, "run_scc_dftb_kpoints");
            // Variational assembly, k-space form. The occupied eigenvalue
            // sum contains H1, whose expectation is -V dot population.
            const double E_h1 = -V.dot(populations);
            const double E_h0 = E_band - E_h1;
            const double E_scc = 0.5 * dq.dot(gamma * dq);
            result.energy = E_h0 + E_scc + E_rep;
            result.free_energy =
                result.energy - occupation_options.smearing_temperature *
                                    occupation.entropy;
            result.e_electronic = E_h0;
            result.e_repulsive = E_rep;
            result.e_scc = E_scc;
            result.fermi_level = occupation.fermi_level;
            result.entropy = occupation.entropy;
            result.occupations_per_k =
                std::move(occupation.occupations_per_k);
            result.eps_per_k = std::move(eps_per_k);
            // Measured band edges + gaps from the occupations actually
            // used (issue #426; convention documented on KPointBandEdges).
            // The unconverged tail below deliberately leaves the edge
            // fields NaN: a diagnostics record measured nothing.
            store_kpoint_band_edges(
                result,
                compute_kpoint_band_edges(
                    result.eps_per_k, result.occupations_per_k));
            for (const auto& energies : result.eps_per_k) {
                for (int i = 0; i < energies.size(); ++i) {
                    result.band_energies.push_back(energies(i));
                }
            }
            result.charges = dq;
            result.n_basis = n_basis;
            result.n_occ = n_occ;
            result.n_kpoints = static_cast<int>(kmesh.size());
            result.n_iter = iter;
            result.converged = true;
            return result;
        }
    }

    result.charges = dq;
    result.n_basis = n_basis;
    result.n_occ = n_occ;
    result.n_kpoints = static_cast<int>(kmesh.size());
    result.n_iter = scc_opts.max_iter;
    result.converged = false;
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
