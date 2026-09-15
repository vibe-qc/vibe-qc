#include "vibeqc/semiempirical/dftb0.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>
#include <string>

#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"

namespace vibeqc {
namespace semiempirical {

DFTB0Result run_dftb0(const Molecule& mol,
                      const SemiempiricalParameters& params) {
    // ---- Validate ----
    if (mol.multiplicity() != 1) {
        throw std::invalid_argument(
            "run_dftb0: only closed-shell (multiplicity=1) supported in Stage 1");
    }

    // Check all elements are parameterized
    for (const auto& atom : mol.atoms()) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error(
                "run_dftb0: element Z=" + std::to_string(atom.Z)
                + " not found in parameter set. Stage 1 supports H, C, N, O.");
        }
    }

    // DFTB fills valence electrons only — the minimal basis has no core
    // orbitals. Counting all Z electrons (pre-fix) over-filled the band
    // by the core count (H₂O: 5 pairs instead of 4), broke the Mulliken
    // reference (Σ Δq = −n_core), and pushed occupancy into levels the
    // model has no room for.
    const int n_val_e =
        SemiempiricalHamiltonianBuilder::valence_electron_count(mol, params);
    if (n_val_e % 2 != 0) {
        throw std::invalid_argument(
            "run_dftb0: only even valence electron counts supported in Stage 1");
    }

    // ---- Build minimal basis ----
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const auto n_basis = static_cast<int>(basis.nbasis());
    // Cap at n_basis (as run_udftb0 does): a valence-minimal basis holds
    // at most n_basis doubly-occupied MOs; uncapped counts read past the
    // eigensolution — undefined behaviour returning heap-dependent
    // energies and gradients.
    int n_occ = n_val_e / 2;
    if (n_occ > n_basis) n_occ = n_basis;

    // ---- Build overlap S ----
    Eigen::MatrixXd S = SemiempiricalHamiltonianBuilder::build_overlap(basis);

    // ---- Build H⁰ ----
    Eigen::MatrixXd H0 = SemiempiricalHamiltonianBuilder::build_hamiltonian_zero(
        basis, S, mol, params);

    // ---- Solve generalized eigenvalue problem H⁰ C = S C ε ----
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H0, S);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "run_dftb0: generalized eigenvalue solver failed to converge");
    }

    Eigen::VectorXd eps = solver.eigenvalues();         // ascending
    Eigen::MatrixXd C = solver.eigenvectors();           // S-orthonormal

    // ---- Electronic energy ----
    double E_elec = 0.0;
    for (int i = 0; i < n_occ; ++i) {
        E_elec += 2.0 * eps(i);
    }

    // ---- Repulsive energy ----
    double E_rep = 0.0;
    const auto& atoms = mol.atoms();
    const std::size_t n_atoms = atoms.size();
    for (std::size_t a = 0; a < n_atoms; ++a) {
        for (std::size_t b = a + 1; b < n_atoms; ++b) {
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (R > 1e-12) {
                E_rep += params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
            }
        }
    }

    // ---- Density matrix D = 2 C_occ C_occ^T ----
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    Eigen::MatrixXd D = 2.0 * C_occ * C_occ.transpose();
    Eigen::VectorXd charges =
        SemiempiricalHamiltonianBuilder::mulliken_charges(
            basis, D, S, mol, params);

    // ---- Assemble result ----
    DFTB0Result result;
    result.energy = E_elec + E_rep;
    result.e_electronic = E_elec;
    result.e_repulsive = E_rep;
    result.mo_energies = std::move(eps);
    result.mo_coeffs = std::move(C);
    result.density = std::move(D);
    result.overlap = std::move(S);
    result.hamiltonian = std::move(H0);
    result.charges = std::move(charges);
    result.n_basis = n_basis;
    result.n_occ = n_occ;

    return result;
}


UDFTB0Result run_udftb0(const Molecule& mol,
                        const SemiempiricalParameters& params) {
    // Validate
    for (const auto& atom : mol.atoms()) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error("run_udftb0: element Z="
                + std::to_string(atom.Z) + " not in parameter set");
        }
    }
    int mult = mol.multiplicity();
    // Valence electrons only (see run_dftb0).
    int n_el =
        SemiempiricalHamiltonianBuilder::valence_electron_count(mol, params);
    int n_alpha = (n_el + mult - 1) / 2;
    int n_beta = (n_el - mult + 1) / 2;
    // Cap at n_basis (to be set after basis construction)
    if (n_alpha < n_beta || n_beta < 0) {
        throw std::invalid_argument("run_udftb0: invalid multiplicity");
    }

    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    if (n_alpha > n_basis) n_alpha = n_basis;
    if (n_beta > n_basis) n_beta = n_basis;
    Eigen::MatrixXd S = SemiempiricalHamiltonianBuilder::build_overlap(basis);
    Eigen::MatrixXd H0 = SemiempiricalHamiltonianBuilder::build_hamiltonian_zero(
        basis, S, mol, params);

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H0, S);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("run_udftb0: eigendecomposition failed");
    }
    Eigen::VectorXd eps = solver.eigenvalues();
    Eigen::MatrixXd C = solver.eigenvectors();

    // Band energy: sum of occupied α + β eigenvalues
    double E_elec = 0.0;
    for (int i = 0; i < n_alpha; ++i) E_elec += eps(i);
    for (int i = 0; i < n_beta; ++i) E_elec += eps(i);

    // Density matrices
    Eigen::MatrixXd C_occ_a = C.leftCols(n_alpha);
    Eigen::MatrixXd C_occ_b = C.leftCols(n_beta);
    Eigen::MatrixXd D_alpha = C_occ_a * C_occ_a.transpose();
    Eigen::MatrixXd D_beta = C_occ_b * C_occ_b.transpose();
    Eigen::VectorXd charges =
        SemiempiricalHamiltonianBuilder::mulliken_charges(
            basis, D_alpha + D_beta, S, mol, params);

    // Repulsive energy (same as closed-shell)
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

    UDFTB0Result result;
    result.energy = E_elec + E_rep;
    result.e_electronic = E_elec;
    result.e_repulsive = E_rep;
    result.mo_energies = std::move(eps);
    result.mo_coeffs = std::move(C);
    result.density_alpha = std::move(D_alpha);
    result.density_beta = std::move(D_beta);
    result.overlap = std::move(S);
    result.hamiltonian = std::move(H0);
    result.charges = std::move(charges);
    result.n_basis = n_basis;
    result.n_alpha = n_alpha;
    result.n_beta = n_beta;
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
