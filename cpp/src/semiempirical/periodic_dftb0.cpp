#include "vibeqc/semiempirical/periodic_dftb0.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

#include "vibeqc/integrals.hpp"
#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/core/pair_lattice.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"

namespace vibeqc {
namespace semiempirical {

PeriodicDFTB0Result run_dftb0_gamma(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const PeriodicDFTB0Options& opts) {

    // ---- Validate ----
    if (system.multiplicity != 1) {
        throw std::invalid_argument(
            "run_dftb0_gamma: only closed-shell supported");
    }
    for (const auto& atom : system.unit_cell) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error(
                "run_dftb0_gamma: element Z=" + std::to_string(atom.Z)
                + " not in parameter set");
        }
    }

    // ---- Build minimal basis for unit cell ----
    Molecule mol = system.unit_cell_molecule();
    // Valence electrons only per cell (see run_dftb0).
    int n_val_e = -system.charge;
    for (const auto& atom : system.unit_cell) {
        n_val_e += params.valence_electrons(atom.Z);
    }
    if (n_val_e % 2 != 0) {
        throw std::invalid_argument(
            "run_dftb0_gamma: only even valence electron counts supported");
    }
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    // Cap at n_basis — a valence-minimal basis holds at most n_basis
    // doubly-occupied MOs; uncapped counts read past the eigensolution
    // (UB; see run_dftb0).
    int n_occ = n_val_e / 2;
    if (n_occ > n_basis) n_occ = n_basis;

    // ---- Enumerate lattice cells ----
    // Pair-distance image selection (issue #316): translations carrying
    // any atom pair within the cutoff, per pair_lattice.hpp rule (*).
    LatticeSumOptions lopt;
    lopt.cutoff_bohr = opts.cutoff_bohr;
    auto cells = atom_pair_interaction_cells(system, opts.cutoff_bohr);

    if (opts.gamma_only_0) {
        cells = {cells[0]};  // only the g=0 cell
    } else {
        require_nonzero_lattice_image(
            cells, opts.cutoff_bohr, "run_dftb0_gamma");
    }

    const auto& atoms = system.unit_cell;
    const int n_atoms = static_cast<int>(atoms.size());
    const double kappa = params.kappa();

    // ---- Build AO-to-atom mapping ----
    const auto& lib_shells = basis.libint();
    const auto shell2bf = lib_shells.shell2bf();
    const auto& shells = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar(n_atoms);
    for (int a = 0; a < n_atoms; ++a) {
        hbar[a] = params.average_on_site(atoms[a].Z);
    }
    for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
        int n_funcs = lib_shells[s].size();
        int bf_start = shell2bf[s];
        for (int i = 0; i < n_funcs; ++i) {
            ao_atom[bf_start + i] = shells[s].atom_index;
        }
    }

    // ---- Compute overlap lattice sum S(g) ----
    // Explicit-cell overlap on the pair-complete list, then zero every
    // AO pair beyond the pair cutoff so the summed S/H carry exactly the
    // interaction set (*) of pair_lattice.hpp.
    auto S_lattice = compute_overlap_lattice_explicit(basis, system, cells);
    mask_blocks_beyond_pair_cutoff(
        S_lattice.blocks, cells, ao_atom, atoms, opts.cutoff_bohr);

    // ---- Lattice-sum H_Γ and S_Γ ----
    Eigen::MatrixXd H_gamma = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd S_gamma = Eigen::MatrixXd::Zero(n_basis, n_basis);

    for (std::size_t ci = 0; ci < cells.size(); ++ci) {
        const auto& S_g = S_lattice.blocks[ci];  // S(g) for this cell
        Eigen::MatrixXd H_g = Eigen::MatrixXd::Zero(n_basis, n_basis);

        const bool is_zero_cell =
            (cells[ci].index.array() == 0).all();

        // Build H⁰(g) from S(g) using Wolfsberg-Helmholtz
        for (int mu = 0; mu < n_basis; ++mu) {
            int a_mu = ao_atom[mu];
            for (int nu = 0; nu < n_basis; ++nu) {
                int a_nu = ao_atom[nu];

                if (is_zero_cell && a_mu == a_nu && mu == nu) {
                    // On-site: use ε_l (only in g=0 cell)
                    H_g(mu, nu) = params.on_site_energy(atoms[a_mu].Z,
                        shells[mu < static_cast<int>(shells.size()) ? mu : 0].l);
                    // Fix: need proper l lookup. Use a simpler fallback.
                    // For now use the on-site energy directly
                } else if (S_g(mu, nu) != 0.0) {
                    double h_avg = 0.5 * (hbar[a_mu] + hbar[a_nu]);
                    H_g(mu, nu) = 0.5 * kappa * S_g(mu, nu) * h_avg;
                }
            }
        }

        // For zero cell: overwrite on-site diagonal properly
        if (is_zero_cell) {
            // Map from AO index to (atom, l) using shells info
            int ao_idx = 0;
            for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                int a = shells[s].atom_index;
                int l = shells[s].l;
                int nf = lib_shells[s].size();
                double eps = params.on_site_energy(atoms[a].Z, l);
                for (int i = 0; i < nf; ++i, ++ao_idx) {
                    H_g(ao_idx, ao_idx) = eps;
                }
            }
        }

        H_gamma += H_g;
        S_gamma += S_g;
    }

    // ---- Solve generalized eigenvalue problem ----
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        H_gamma, S_gamma);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "run_dftb0_gamma: generalized eigenvalue solver failed");
    }

    Eigen::VectorXd eps = solver.eigenvalues();
    Eigen::MatrixXd C = solver.eigenvectors();

    // Issue #339: a degenerate Gamma frontier receives equal fractional
    // occupations (Weinert & Davenport T -> 0 ensemble) so the density and
    // every analytic derivative are unique. Open-gap spectra get the exact
    // hard-Aufbau vector, and the assembly below stays bit-identical to the
    // historical 2 * C_occ C_occ^T path for that case.
    Eigen::VectorXd occupations =
        gamma_degenerate_frontier_occupations(eps, n_occ);
    bool hard_aufbau = true;
    for (int i = 0; i < n_basis; ++i) {
        const double expected = (i < n_occ) ? 2.0 : 0.0;
        if (occupations(i) != expected) {
            hard_aufbau = false;
            break;
        }
    }

    // ---- Electronic energy ----
    double E_elec = 0.0;
    if (hard_aufbau) {
        for (int i = 0; i < n_occ; ++i) {
            E_elec += 2.0 * eps(i);
        }
    } else {
        for (int i = 0; i < n_basis; ++i) {
            E_elec += occupations(i) * eps(i);
        }
    }

    // ---- Repulsive energy (lattice sum) ----
    double E_rep = 0.0;
    for (const auto& cell : cells) {
        const auto& g = cell.r_cart;
        bool is_zero = (cell.index.array() == 0).all();

        for (int a = 0; a < n_atoms; ++a) {
            int start_b = is_zero ? a + 1 : 0;  // avoid double-count in g=0
            for (int b = start_b; b < n_atoms; ++b) {
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + g[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + g[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + g[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > opts.cutoff_bohr) continue;

                double factor = is_zero ? 1.0 : 0.5;
                E_rep += factor * params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
            }
        }
    }

    // ---- Density ----
    Eigen::MatrixXd D;
    if (hard_aufbau) {
        Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        D = 2.0 * C_occ * C_occ.transpose();
    } else {
        D = C * occupations.asDiagonal() * C.transpose();
    }

    // ---- Result ----
    PeriodicDFTB0Result result;
    result.energy = E_elec + E_rep;
    result.e_electronic = E_elec;
    result.e_repulsive = E_rep;
    result.mo_energies = std::move(eps);
    result.mo_coeffs = std::move(C);
    result.density = std::move(D);
    result.occupations = std::move(occupations);
    result.overlap_gamma = std::move(S_gamma);
    result.hamiltonian_gamma = std::move(H_gamma);
    result.n_basis = n_basis;
    result.n_occ = n_occ;
    result.n_cells = static_cast<int>(cells.size());
    result.cutoff_bohr = opts.cutoff_bohr;
    result.gamma_only_0 = opts.gamma_only_0;

    return result;
}


// ---------------------------------------------------------------------------
// Unrestricted periodic DFTB0 (Gamma-point)
// ---------------------------------------------------------------------------

PeriodicUDFTB0Result run_udftb0_gamma(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const PeriodicDFTB0Options& opts) {

    for (const auto& atom : system.unit_cell) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error(
                "run_udftb0_gamma: element Z=" + std::to_string(atom.Z)
                + " not in parameter set");
        }
    }
    int mult = system.multiplicity;
    // Valence electrons only per cell (see run_dftb0).
    int n_el = -system.charge;
    for (const auto& atom : system.unit_cell) {
        n_el += params.valence_electrons(atom.Z);
    }
    int n_alpha = (n_el + mult - 1) / 2;
    int n_beta  = (n_el - mult + 1) / 2;

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    if (n_alpha > n_basis) n_alpha = n_basis;
    if (n_beta  > n_basis) n_beta  = n_basis;

    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    LatticeSumOptions lopt;
    lopt.cutoff_bohr = opts.cutoff_bohr;
    auto cells = atom_pair_interaction_cells(system, opts.cutoff_bohr);
    if (opts.gamma_only_0) {
        cells = {cells[0]};
    } else {
        require_nonzero_lattice_image(
            cells, opts.cutoff_bohr, "run_udftb0_gamma");
    }

    const auto& atoms = system.unit_cell;
    const int n_atoms = static_cast<int>(atoms.size());
    const double kappa = params.kappa();

    const auto& lib_shells = basis.libint();
    const auto shell2bf = lib_shells.shell2bf();
    const auto& shells = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar(n_atoms);
    for (int a = 0; a < n_atoms; ++a)
        hbar[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
        int n_funcs = lib_shells[s].size();
        int bf_start = shell2bf[s];
        for (int i = 0; i < n_funcs; ++i)
            ao_atom[bf_start + i] = shells[s].atom_index;
    }

    auto S_lattice = compute_overlap_lattice_explicit(basis, system, cells);
    mask_blocks_beyond_pair_cutoff(
        S_lattice.blocks, cells, ao_atom, atoms, opts.cutoff_bohr);

    Eigen::MatrixXd H_gamma = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd S_gamma = Eigen::MatrixXd::Zero(n_basis, n_basis);

    for (std::size_t ci = 0; ci < cells.size(); ++ci) {
        const auto& S_g = S_lattice.blocks[ci];
        Eigen::MatrixXd H_g = Eigen::MatrixXd::Zero(n_basis, n_basis);
        const bool is_zero_cell = (cells[ci].index.array() == 0).all();

        for (int mu = 0; mu < n_basis; ++mu) {
            int a_mu = ao_atom[mu];
            for (int nu = 0; nu < n_basis; ++nu) {
                int a_nu = ao_atom[nu];
                if (is_zero_cell && a_mu == a_nu && mu == nu) {
                } else if (S_g(mu, nu) != 0.0) {
                    double h_avg = 0.5 * (hbar[a_mu] + hbar[a_nu]);
                    H_g(mu, nu) = 0.5 * kappa * S_g(mu, nu) * h_avg;
                }
            }
        }

        if (is_zero_cell) {
            int ao_idx = 0;
            for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                int a = shells[s].atom_index;
                int l = shells[s].l;
                int nf = lib_shells[s].size();
                double eps = params.on_site_energy(atoms[a].Z, l);
                for (int i = 0; i < nf; ++i, ++ao_idx)
                    H_g(ao_idx, ao_idx) = eps;
            }
        }

        H_gamma += H_g;
        S_gamma += S_g;
    }

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_gamma, S_gamma);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("run_udftb0_gamma: diag failed");

    Eigen::VectorXd eps = solver.eigenvalues();
    Eigen::MatrixXd C = solver.eigenvectors();

    // Issue #439: each spin channel of a degenerate Gamma frontier gets the
    // same equal-fractional treatment the restricted driver received in
    // #339, applied per spin (full occupancy 1.0 per channel).  Alpha and
    // beta share one spectrum here -- DFTB0 is non-self-consistent -- but
    // they cut it at different counts, so the manifolds are resolved
    // independently.  A channel with an open gap returns the exact
    // hard-Aufbau vector, keeping gapped systems bit-identical.
    Eigen::VectorXd occ_alpha =
        gamma_degenerate_frontier_occupations(eps, n_alpha, -1.0, 1.0);
    Eigen::VectorXd occ_beta =
        gamma_degenerate_frontier_occupations(eps, n_beta, -1.0, 1.0);
    const auto is_hard_aufbau = [n_basis](
        const Eigen::VectorXd& occupations, int n_channel) {
        for (int i = 0; i < n_basis; ++i) {
            const double expected = (i < n_channel) ? 1.0 : 0.0;
            if (occupations(i) != expected) return false;
        }
        return true;
    };
    const bool hard_alpha = is_hard_aufbau(occ_alpha, n_alpha);
    const bool hard_beta = is_hard_aufbau(occ_beta, n_beta);

    double E_elec = 0.0;
    if (hard_alpha) {
        for (int i = 0; i < n_alpha; ++i) E_elec += eps(i);
    } else {
        for (int i = 0; i < n_basis; ++i) E_elec += occ_alpha(i) * eps(i);
    }
    if (hard_beta) {
        for (int i = 0; i < n_beta; ++i) E_elec += eps(i);
    } else {
        for (int i = 0; i < n_basis; ++i) E_elec += occ_beta(i) * eps(i);
    }

    double E_rep = 0.0;
    for (const auto& cell : cells) {
        const auto& g = cell.r_cart;
        bool is_zero = (cell.index.array() == 0).all();
        for (int a = 0; a < n_atoms; ++a) {
            int start_b = is_zero ? a + 1 : 0;
            for (int b = start_b; b < n_atoms; ++b) {
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + g[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + g[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + g[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > opts.cutoff_bohr) continue;
                double factor = is_zero ? 1.0 : 0.5;
                E_rep += factor * params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
            }
        }
    }

    Eigen::MatrixXd Da;
    if (hard_alpha) {
        Eigen::MatrixXd C_occ_a = C.leftCols(n_alpha);
        Da = C_occ_a * C_occ_a.transpose();
    } else {
        Da = C * occ_alpha.asDiagonal() * C.transpose();
    }
    Eigen::MatrixXd Db;
    if (hard_beta) {
        Eigen::MatrixXd C_occ_b = C.leftCols(n_beta);
        Db = C_occ_b * C_occ_b.transpose();
    } else {
        Db = C * occ_beta.asDiagonal() * C.transpose();
    }

    PeriodicUDFTB0Result result;
    result.energy = E_elec + E_rep;
    result.e_electronic = E_elec;
    result.e_repulsive = E_rep;
    result.mo_energies = std::move(eps);
    result.mo_coeffs = std::move(C);
    result.density_alpha = std::move(Da);
    result.density_beta = std::move(Db);
    result.occupations_alpha = std::move(occ_alpha);
    result.occupations_beta = std::move(occ_beta);
    result.overlap_gamma = std::move(S_gamma);
    result.hamiltonian_gamma = std::move(H_gamma);
    result.n_basis = n_basis;
    result.n_alpha = n_alpha;
    result.n_beta = n_beta;
    result.n_cells = static_cast<int>(cells.size());
    result.cutoff_bohr = opts.cutoff_bohr;
    result.gamma_only_0 = opts.gamma_only_0;
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
