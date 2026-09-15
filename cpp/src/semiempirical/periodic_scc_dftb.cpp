#include "vibeqc/semiempirical/periodic_scc_dftb.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "vibeqc/diis.hpp"
#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/core/pair_lattice.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"

namespace vibeqc {
namespace semiempirical {

namespace {

// One spin channel's Gamma density from its occupations (issue #439).  A
// hard-Aufbau channel (open frontier gap) reproduces the historical
// C_occ * C_occ^T assembly bit-for-bit; a channel cutting a degenerate
// manifold builds C * diag(occ) * C^T, which is an invariant of that
// eigenspace and therefore unique.
Eigen::MatrixXd uscc_gamma_spin_density(
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& occupations,
    int n_channel,
    int n_basis) {
    bool hard_aufbau = true;
    for (int i = 0; i < n_basis; ++i) {
        const double expected = (i < n_channel) ? 1.0 : 0.0;
        if (occupations(i) != expected) {
            hard_aufbau = false;
            break;
        }
    }
    if (hard_aufbau) {
        Eigen::MatrixXd C_occ = C.leftCols(n_channel);
        return C_occ * C_occ.transpose();
    }
    return C * occupations.asDiagonal() * C.transpose();
}

// Build Gamma-point H⁰ and S from lattice-summed S(g) blocks.
// Returns (H_gamma, S_gamma).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_h0_s0_gamma(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const LatticeSumOptions& lopt) {

    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    const auto cells =
        atom_pair_interaction_cells(system, lopt.cutoff_bohr);
    auto S_lattice = compute_overlap_lattice_explicit(basis, system, cells);
    const int n_basis = static_cast<int>(basis.nbasis());
    const auto& atoms = system.unit_cell;
    const double kappa = params.kappa();

    // AO-to-atom map
    const auto& lib_shells = basis.libint();
    const auto shell2bf = lib_shells.shell2bf();
    const auto& shells = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        hbar[a] = params.average_on_site(atoms[a].Z);
    }
    for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
        int nf = lib_shells[s].size();
        int bf0 = shell2bf[s];
        for (int i = 0; i < nf; ++i) {
            ao_atom[bf0 + i] = shells[s].atom_index;
        }
    }
    mask_blocks_beyond_pair_cutoff(
        S_lattice.blocks, cells, ao_atom, atoms, lopt.cutoff_bohr);

    Eigen::MatrixXd H_gamma = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd S_gamma = Eigen::MatrixXd::Zero(n_basis, n_basis);

    for (std::size_t ci = 0; ci < cells.size(); ++ci) {
        const auto& S_g = S_lattice.blocks[ci];
        Eigen::MatrixXd H_g = Eigen::MatrixXd::Zero(n_basis, n_basis);
        bool is_zero = (cells[ci].index.array() == 0).all();

        for (int mu = 0; mu < n_basis; ++mu) {
            int a_mu = ao_atom[mu];
            for (int nu = 0; nu < n_basis; ++nu) {
                int a_nu = ao_atom[nu];
                if (S_g(mu, nu) == 0.0) continue;

                if (!is_zero || a_mu != a_nu) {
                    double h_avg = 0.5 * (hbar[a_mu] + hbar[a_nu]);
                    H_g(mu, nu) = 0.5 * kappa * S_g(mu, nu) * h_avg;
                }
            }
        }

        if (is_zero) {
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

    return {H_gamma, S_gamma};
}

}  // namespace

// Periodic γ matrix: γ_{AB}^{per} = Σ_g 1/√(|R_{AB}+g|² + η_{AB}²),
// summed over the pair-distance interaction set |R_{AB}+g| <= cutoff
// (issue #316; pair_lattice.hpp rule (*)).  Selecting terms by the pair
// distance instead of the translation length keeps every image within
// interaction range of each atom — the CCM interaction region of
// Bredow, Geudtner & Jug, J. Comput. Chem. 22, 89 (2001), pp. 90-91 —
// and makes the primitive-cell sum fold term-for-term onto the matched
// Gamma-supercell sum.
// The lattice sum runs through the shared Ewald-split kernel
// (semiempirical/core/periodic_gamma.hpp):
//
//   Gamma_AB = Phi_Ewald(R_B - R_A) + sum_images [gamma_AB(r) - 1/r] + on-site
//
// Before 2026-09-07 this function summed the bare Klopman-Ohno kernel by
// real-space truncation alone.  That is a sum of 1/R, divergent in three
// dimensions: charge neutrality should cancel it, but a *pair-distance*
// cutoff gives different pairs different image counts, so each pair keeps a
// different divergent constant and the cancellation is incomplete.  Issue
// #425 measured the result oscillating over ~10 Ha between cutoffs.
//
// Both halves of Eq. 17/18 are needed to fix it, and the maintainer's D1
// note is explicit that the functional form alone is not enough: swapping
// the kernel without Ewald still swings the neutrality-projected
// combination -4.33..+3.20 Ha over cutoffs 12-200 bohr, against
// -4.51..+3.12 for Klopman-Ohno.  So the Coulomb tail goes through Ewald
// here for either form, and `form` decides whether the remainder that is
// summed directly over images decays exponentially (Elstner, absolutely
// convergent) or as R^-3 (Klopman-Ohno, conditionally convergent).
Eigen::MatrixXd build_periodic_dftb_gamma_matrix(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    double cutoff_bohr,
    ShellGammaForm form) {

    const auto& atoms = system.unit_cell;
    const int n = static_cast<int>(atoms.size());
    const auto cells = atom_pair_interaction_cells(system, cutoff_bohr);

    std::vector<GammaSite> sites;
    std::vector<Eigen::Vector3d> positions;
    sites.reserve(atoms.size());
    positions.reserve(atoms.size());
    for (int a = 0; a < n; ++a) {
        double u = params.hubbard_u(atoms[a].Z);
        if (u <= 0.0) u = 0.4;
        GammaSite site;
        site.atom = a;
        site.hardness = u;
        sites.push_back(site);
        positions.emplace_back(
            atoms[static_cast<std::size_t>(a)].xyz[0],
            atoms[static_cast<std::size_t>(a)].xyz[1],
            atoms[static_cast<std::size_t>(a)].xyz[2]);
    }

    std::vector<Eigen::Vector3d> translations;
    for (int axis = 0; axis < system.dim; ++axis) {
        translations.push_back(system.lattice.col(axis));
    }

    // Unit-weight records: every atom pair and lattice image inside the pair
    // cutoff, the same rule (*) selection the overlap and H0 use (#316).
    const auto records = [&positions, &cells, cutoff_bohr, n](
        const ImageRecordSink& sink) {
        const double cutoff_sq = cutoff_bohr * cutoff_bohr;
        for (const auto& cell : cells) {
            const bool zero_cell = (cell.index.array() == 0).all();
            for (int a = 0; a < n; ++a) {
                for (int b = 0; b < n; ++b) {
                    if (zero_cell && a == b) continue;
                    const Eigen::Vector3d d =
                        positions[static_cast<std::size_t>(b)] + cell.r_cart
                        - positions[static_cast<std::size_t>(a)];
                    if (d.squaredNorm() > cutoff_sq) continue;
                    ImageRecord record;
                    record.a = a;
                    record.b = b;
                    record.shift = cell.r_cart;
                    record.weight = 1.0;
                    sink(record);
                }
            }
        }
    };

    ShellGammaSpec spec;
    spec.form = form;
    spec.ko_average = KlopmanOhnoAverage::InverseHardnessMean;
    const EwaldCoulombKernel ewald(translations);
    return build_periodic_shell_gamma(
        sites, positions, spec, ewald, records);
}

PeriodicSCCDFTBResult run_scc_dftb_gamma(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const PeriodicSCCOptions& opts) {

    if (system.multiplicity != 1) {
        throw std::invalid_argument("run_scc_dftb_gamma: closed-shell only");
    }
    for (const auto& atom : system.unit_cell) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error("run_scc_dftb_gamma: unsupported element Z="
                                     + std::to_string(atom.Z));
        }
    }
    // Valence electrons only per cell (see run_dftb0).
    int n_val_e = -system.charge;
    for (const auto& atom : system.unit_cell) {
        n_val_e += params.valence_electrons(atom.Z);
    }
    if (n_val_e % 2 != 0) {
        throw std::invalid_argument("run_scc_dftb_gamma: closed-shell only");
    }

    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, opts.cutoff_bohr);
    require_nonzero_lattice_image(
        cells, opts.cutoff_bohr, "run_scc_dftb_gamma");

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    // Cap at n_basis — a valence-minimal basis holds at most n_basis
    // doubly-occupied MOs; uncapped counts read past the eigensolution
    // (UB; see run_dftb0).
    int n_occ = n_val_e / 2;
    if (n_occ > n_basis) n_occ = n_basis;
    const int n_atoms = static_cast<int>(system.unit_cell.size());

    LatticeSumOptions lopt;
    lopt.cutoff_bohr = opts.cutoff_bohr;

    // Build H⁰_Γ and S_Γ once
    auto [H0_gamma, S_gamma] = build_h0_s0_gamma(basis, system, params, lopt);

    // Periodic γ matrix
    Eigen::MatrixXd gamma = build_periodic_dftb_gamma_matrix(
        system, params, opts.cutoff_bohr, opts.gamma_form);

    // Repulsive energy (reuse periodic DFTB0)
    PeriodicDFTB0Options popt;
    popt.cutoff_bohr = opts.cutoff_bohr;
    auto dftb0_result = run_dftb0_gamma(system, params, popt);
    double E_rep = dftb0_result.e_repulsive;

    // ---- SCC SCF loop ----
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(n_atoms);
    const double mix = std::max(0.0, std::min(1.0, opts.charge_mixing));
    // Adaptive damping state (see run_scc_dftb).
    double mix_cur = mix;
    Eigen::VectorXd resid_prev;
    // ---- DIIS charge accelerator ----
    std::unique_ptr<DIIS> charge_diis;
    if (opts.use_diis) {
        int diis_depth = std::max(2, std::min(12, opts.diis_subspace));
        charge_diis = std::make_unique<DIIS>(diis_depth);
    }

    PeriodicSCCDFTBResult result;
    result.cutoff_bohr = opts.cutoff_bohr;
    result.gamma_form = opts.gamma_form;

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        // Electrostatic potential: V = γ · Δq
        Eigen::VectorXd V = gamma * dq;

        // Build SCC Hamiltonian: H_Γ^{SCC} = H⁰_Γ + ½ S_Γ · (V_A + V_B)
        Eigen::MatrixXd H_scc = H0_gamma;
        const auto& shells = basis.shells();
        const auto& lib_shells = basis.libint();
        const auto shell2bf = lib_shells.shell2bf();
        for (int mu = 0; mu < n_basis; ++mu) {
            // Find atom for mu
            int a_mu = -1;
            for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                int bf0 = shell2bf[s];
                int nf = lib_shells[s].size();
                if (mu >= bf0 && mu < bf0 + nf) {
                    a_mu = shells[s].atom_index;
                    break;
                }
            }
            double V_mu = V(a_mu);
            for (int nu = 0; nu < n_basis; ++nu) {
                int a_nu = -1;
                for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                    int bf0 = shell2bf[s];
                    int nf = lib_shells[s].size();
                    if (nu >= bf0 && nu < bf0 + nf) {
                        a_nu = shells[s].atom_index;
                        break;
                    }
                }
                double V_nu = V(a_nu);
                // Elstner Eq. 14 with flipped sign for the Δq > 0 = cation
                // convention (see build_scc_hamiltonian).
                H_scc(mu, nu) -= 0.5 * S_gamma(mu, nu) * (V_mu + V_nu);
            }
        }

        // Diagonalize
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
            H_scc, S_gamma);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Periodic SCC: eigendecomposition failed");
        }
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();
        // Issue #339: degenerate frontier -> equal fractional occupations,
        // unique density/gradient/stress. Open-gap spectra keep the exact
        // hard-Aufbau assembly below (bit-identical SCC trajectory).
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
        Eigen::MatrixXd D;
        if (hard_aufbau) {
            Eigen::MatrixXd C_occ = C.leftCols(n_occ);
            D = 2.0 * C_occ * C_occ.transpose();
        } else {
            D = C * occupations.asDiagonal() * C.transpose();
        }

        // Mulliken charges
        Eigen::VectorXd dq_new = SemiempiricalHamiltonianBuilder::mulliken_charges(
            basis, D, S_gamma, mol, params);

        Eigen::VectorXd resid = dq_new - dq;
        double max_change = resid.cwiseAbs().maxCoeff();
        bool converged = (max_change < opts.conv_tol_charge);

        // ---- Charge mixing / acceleration ----
        if (charge_diis) {
            Eigen::MatrixXd dq_mat = dq_new;
            Eigen::MatrixXd err_mat = resid;
            Eigen::MatrixXd dq_diis_mat =
                charge_diis->extrapolate(dq_mat, err_mat);

            // Shallow DIIS history returns raw dq_new (mix=1.0);
            // damp instead to avoid destabilising the SCC loop
            // (see run_scc_dftb).
            bool diis_used = false;
            if (charge_diis->subspace_size() >= 2) {
                Eigen::VectorXd dq_candidate = dq_diis_mat.col(0);
                if (dq_candidate.allFinite()) {
                    bool physical = true;
                    for (int a = 0; a < n_atoms; ++a) {
                        int Z_a = system.unit_cell[a].Z;
                        int n_val =
                            params.valence_electrons(Z_a);
                        if (std::abs(dq_candidate(a))
                            > static_cast<double>(n_val) + 2.0) {
                            physical = false;
                            break;
                        }
                    }
                    if (physical) {
                        dq = dq_candidate;
                        diis_used = true;
                    }
                }
            }

            if (diis_used) {
                double drift =
                    (dq.sum()
                     - static_cast<double>(system.charge))
                    / static_cast<double>(n_atoms);
                dq.array() -= drift;
            } else {
                if (resid_prev.size() > 0
                    && resid.dot(resid_prev) < 0.0) {
                    mix_cur =
                        std::max(0.5 * mix_cur, 0.01);
                }
                dq += mix_cur * resid;
            }
            resid_prev = resid;
        } else {
            if (resid_prev.size() > 0 && resid.dot(resid_prev) < 0.0) {
                mix_cur = std::max(0.5 * mix_cur, 0.01);
            }
            resid_prev = resid;
            dq += mix_cur * resid;
        }

        if (converged) {
            // Variational assembly (see run_scc_dftb):
            // E = tr(D H⁰_Γ) + ½ ΣΔqγΔq + E_rep.
            double E_elec = D.cwiseProduct(H0_gamma).sum();
            double E_scc = 0.5 * dq.dot(gamma * dq);

            result.energy = E_elec + E_scc + E_rep;
            result.e_electronic = E_elec;
            result.e_repulsive = E_rep;
            result.e_scc = E_scc;
            result.mo_energies = std::move(eps);
            result.mo_coeffs = std::move(C);
            result.density = std::move(D);
            result.occupations = std::move(occupations);
            result.overlap_gamma = S_gamma;
            result.hamiltonian_gamma = std::move(H_scc);
            result.charges = dq;
            result.n_basis = n_basis;
            result.n_occ = n_occ;
            result.n_cells = static_cast<int>(cells.size());
            result.n_iter = iter;
            result.converged = true;
            return result;
        }
    }

    // Not converged
    Eigen::VectorXd V = gamma * dq;
    Eigen::MatrixXd H_scc = H0_gamma;
    for (int mu = 0; mu < n_basis; ++mu) {
        // simplified fallback using on-site values
        double Vmu = 0.0;
        for (int s = 0; s < static_cast<int>(basis.shells().size()); ++s) {
            int bf0 = basis.libint().shell2bf()[s];
            int nf = basis.libint().shells()[s].size();
            if (mu >= bf0 && mu < bf0 + nf) {
                Vmu = V(basis.shells()[s].atom_index);
                break;
            }
        }
        for (int nu = 0; nu < n_basis; ++nu) {
            double Vnu = 0.0;
            for (int s = 0; s < static_cast<int>(basis.shells().size()); ++s) {
                int bf0 = basis.libint().shell2bf()[s];
                int nf = basis.libint().shells()[s].size();
                if (nu >= bf0 && nu < bf0 + nf) {
                    Vnu = V(basis.shells()[s].atom_index);
                    break;
                }
            }
            H_scc(mu, nu) -= 0.5 * S_gamma(mu, nu) * (Vmu + Vnu);
        }
    }
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, S_gamma);
    Eigen::VectorXd eps = solver.eigenvalues();
    Eigen::MatrixXd C = solver.eigenvectors();
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
    Eigen::MatrixXd D;
    if (hard_aufbau) {
        Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        D = 2.0 * C_occ * C_occ.transpose();
    } else {
        D = C * occupations.asDiagonal() * C.transpose();
    }
    // Consistent functional at the returned density (see run_scc_dftb).
    dq = SemiempiricalHamiltonianBuilder::mulliken_charges(
        basis, D, S_gamma, mol, params);
    double E_elec = D.cwiseProduct(H0_gamma).sum();
    double E_scc = 0.5 * dq.dot(gamma * dq);

    result.energy = E_elec + E_scc + E_rep;
    result.e_electronic = E_elec;
    result.e_repulsive = E_rep;
    result.e_scc = E_scc;
    result.mo_energies = eps;
    result.mo_coeffs = C;
    result.density = std::move(D);
    result.occupations = std::move(occupations);
    result.overlap_gamma = S_gamma;
    result.hamiltonian_gamma = H_scc;
    result.charges = dq;
    result.n_basis = n_basis;
    result.n_occ = n_occ;
    result.n_cells = static_cast<int>(cells.size());
    result.n_iter = opts.max_iter;
    result.converged = false;
    return result;
}


// ---------------------------------------------------------------------------
// Unrestricted periodic SCC-DFTB (Gamma-point)
// ---------------------------------------------------------------------------

PeriodicUSCCDFTBResult run_uscc_dftb_gamma(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const PeriodicSCCOptions& opts) {

    for (const auto& atom : system.unit_cell) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error("run_uscc_dftb_gamma: unsupported element Z="
                                     + std::to_string(atom.Z));
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

    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, opts.cutoff_bohr);
    require_nonzero_lattice_image(
        cells, opts.cutoff_bohr, "run_uscc_dftb_gamma");

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    if (n_alpha > n_basis) n_alpha = n_basis;
    if (n_beta  > n_basis) n_beta  = n_basis;
    const int n_atoms = static_cast<int>(system.unit_cell.size());

    LatticeSumOptions lopt;
    lopt.cutoff_bohr = opts.cutoff_bohr;

    auto [H0_gamma, S_gamma] = build_h0_s0_gamma(basis, system, params, lopt);
    Eigen::MatrixXd gamma_mat = build_periodic_dftb_gamma_matrix(
        system, params, opts.cutoff_bohr, opts.gamma_form);

    PeriodicDFTB0Options popt;
    popt.cutoff_bohr = opts.cutoff_bohr;
    auto udftb0_result = run_udftb0_gamma(system, params, popt);
    double E_rep = udftb0_result.e_repulsive;

    Eigen::VectorXd dq = Eigen::VectorXd::Zero(n_atoms);
    const double mix = std::max(0.0, std::min(1.0, opts.charge_mixing));
    // Adaptive damping state (see run_scc_dftb).
    double mix_cur = mix;
    Eigen::VectorXd resid_prev;
    // ---- DIIS charge accelerator ----
    std::unique_ptr<DIIS> charge_diis;
    if (opts.use_diis) {
        int diis_depth = std::max(2, std::min(12, opts.diis_subspace));
        charge_diis = std::make_unique<DIIS>(diis_depth);
    }

    const auto& shells = basis.shells();
    const auto& lib_shells = basis.libint();
    const auto shell2bf = lib_shells.shell2bf();

    PeriodicUSCCDFTBResult result;
    result.cutoff_bohr = opts.cutoff_bohr;
    result.gamma_form = opts.gamma_form;

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        Eigen::VectorXd V = gamma_mat * dq;

        Eigen::MatrixXd H_scc = H0_gamma;
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
                // Elstner Eq. 14 with flipped sign (Δq > 0 = cation).
                H_scc(mu, nu) -= 0.5 * S_gamma(mu, nu) * (V_mu + V(a_nu));
            }
        }

        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, S_gamma);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Periodic USCC: diag failed");
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();

        // Issue #439: per-spin degenerate-frontier occupations, the #339
        // convention applied to each channel (full occupancy 1.0).  A
        // channel with an open gap keeps the exact hard-Aufbau density.
        Eigen::VectorXd occ_a =
            gamma_degenerate_frontier_occupations(eps, n_alpha, -1.0, 1.0);
        Eigen::VectorXd occ_b =
            gamma_degenerate_frontier_occupations(eps, n_beta, -1.0, 1.0);
        Eigen::MatrixXd D_a =
            uscc_gamma_spin_density(C, occ_a, n_alpha, n_basis);
        Eigen::MatrixXd D_b =
            uscc_gamma_spin_density(C, occ_b, n_beta, n_basis);
        Eigen::MatrixXd D_total = D_a + D_b;

        Eigen::VectorXd dq_new = SemiempiricalHamiltonianBuilder::mulliken_charges(
            basis, D_total, S_gamma, mol, params);

        Eigen::VectorXd resid = dq_new - dq;
        double max_change = resid.cwiseAbs().maxCoeff();
        bool converged = (max_change < opts.conv_tol_charge);

        // ---- Charge mixing / acceleration ----
        if (charge_diis) {
            Eigen::MatrixXd dq_mat = dq_new;
            Eigen::MatrixXd err_mat = resid;
            Eigen::MatrixXd dq_diis_mat =
                charge_diis->extrapolate(dq_mat, err_mat);

            // Shallow DIIS history returns raw dq_new (mix=1.0);
            // damp instead to avoid destabilising the SCC loop
            // (see run_scc_dftb).
            bool diis_used = false;
            if (charge_diis->subspace_size() >= 2) {
                Eigen::VectorXd dq_candidate = dq_diis_mat.col(0);
                if (dq_candidate.allFinite()) {
                    bool physical = true;
                    for (int a = 0; a < n_atoms; ++a) {
                        int Z_a = system.unit_cell[a].Z;
                        int n_val =
                            params.valence_electrons(Z_a);
                        if (std::abs(dq_candidate(a))
                            > static_cast<double>(n_val) + 2.0) {
                            physical = false;
                            break;
                        }
                    }
                    if (physical) {
                        dq = dq_candidate;
                        diis_used = true;
                    }
                }
            }

            if (diis_used) {
                double drift =
                    (dq.sum()
                     - static_cast<double>(system.charge))
                    / static_cast<double>(n_atoms);
                dq.array() -= drift;
            } else {
                if (resid_prev.size() > 0
                    && resid.dot(resid_prev) < 0.0) {
                    mix_cur =
                        std::max(0.5 * mix_cur, 0.01);
                }
                dq += mix_cur * resid;
            }
            resid_prev = resid;
        } else {
            if (resid_prev.size() > 0 && resid.dot(resid_prev) < 0.0) {
                mix_cur = std::max(0.5 * mix_cur, 0.01);
            }
            resid_prev = resid;
            dq += mix_cur * resid;
        }

        if (converged) {
            // Variational assembly, as in run_scc_dftb_gamma.
            double E_elec = D_total.cwiseProduct(H0_gamma).sum();
            double E_scc = 0.5 * dq.dot(gamma_mat * dq);

            result.energy = E_elec + E_scc + E_rep;
            result.e_electronic = E_elec; result.e_repulsive = E_rep; result.e_scc = E_scc;
            result.mo_energies = std::move(eps); result.mo_coeffs = std::move(C);
            result.density_alpha = std::move(D_a); result.density_beta = std::move(D_b);
            result.occupations_alpha = std::move(occ_a);
            result.occupations_beta = std::move(occ_b);
            result.overlap_gamma = S_gamma; result.hamiltonian_gamma = std::move(H_scc);
            result.charges = dq;
            result.n_basis = n_basis; result.n_alpha = n_alpha; result.n_beta = n_beta;
            result.n_cells = static_cast<int>(cells.size()); result.n_iter = iter;
            result.converged = true;
            return result;
        }
    }

    // Not converged — still populate the full result from the last mixed
    // charges (see run_uscc_dftb: an empty result segfaults the gradient).
    {
        Eigen::VectorXd V = gamma_mat * dq;
        Eigen::MatrixXd H_scc = H0_gamma;
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
                H_scc(mu, nu) -= 0.5 * S_gamma(mu, nu) * (V_mu + V(a_nu));
            }
        }
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, S_gamma);
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();
        Eigen::VectorXd occ_a =
            gamma_degenerate_frontier_occupations(eps, n_alpha, -1.0, 1.0);
        Eigen::VectorXd occ_b =
            gamma_degenerate_frontier_occupations(eps, n_beta, -1.0, 1.0);
        Eigen::MatrixXd D_a =
            uscc_gamma_spin_density(C, occ_a, n_alpha, n_basis);
        Eigen::MatrixXd D_b =
            uscc_gamma_spin_density(C, occ_b, n_beta, n_basis);
        Eigen::MatrixXd D_total = D_a + D_b;
        dq = SemiempiricalHamiltonianBuilder::mulliken_charges(
            basis, D_total, S_gamma, mol, params);
        double E_elec = D_total.cwiseProduct(H0_gamma).sum();
        double E_scc = 0.5 * dq.dot(gamma_mat * dq);
        result.energy = E_elec + E_scc + E_rep;
        result.e_electronic = E_elec; result.e_repulsive = E_rep; result.e_scc = E_scc;
        result.mo_energies = std::move(eps); result.mo_coeffs = std::move(C);
        result.density_alpha = std::move(D_a); result.density_beta = std::move(D_b);
        result.occupations_alpha = std::move(occ_a);
        result.occupations_beta = std::move(occ_b);
        result.overlap_gamma = S_gamma; result.hamiltonian_gamma = std::move(H_scc);
        result.charges = dq;
        result.n_basis = n_basis; result.n_alpha = n_alpha; result.n_beta = n_beta;
        result.n_cells = static_cast<int>(cells.size());
    }
    result.n_iter = opts.max_iter;
    result.converged = false;
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
