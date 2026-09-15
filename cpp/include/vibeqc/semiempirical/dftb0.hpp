// Non-self-consistent DFTB0 energy driver.
//
// Performs a single diagonalization of H⁰ in the minimal valence basis,
// computes the electronic energy, adds the pairwise repulsive energy,
// and returns the full result (orbitals, density, matrices).

#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>

#include "vibeqc/basis.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/parameters.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// DFTB0 result — returned by run_dftb0.
// ---------------------------------------------------------------------------

struct DFTB0Result : ParameterIdentifiedResult {
    double energy = 0.0;            // total DFTB0 energy (Hartree)
    double e_electronic = 0.0;      // Σ_i n_i ε_i
    double e_repulsive = 0.0;       // pairwise repulsive energy
    Eigen::VectorXd mo_energies;    // MO energies (Hartree), ascending
    Eigen::MatrixXd mo_coeffs;      // MO coefficients, columns = MOs
    Eigen::MatrixXd density;        // D = 2 C_occ C_occ^T (closed-shell)
    Eigen::MatrixXd overlap;        // overlap matrix S
    Eigen::MatrixXd hamiltonian;    // H⁰ matrix
    Eigen::VectorXd charges;        // conventional Mulliken net atomic charges
    int n_basis = 0;
    int n_occ = 0;
};

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

// Run a non-self-consistent DFTB0 energy calculation on a closed-shell
// molecule. Uses the default parameter set for H, C, N, O. Throws
// std::runtime_error if the molecule contains an unsupported element.
DFTB0Result run_dftb0(const Molecule& mol,
                      const SemiempiricalParameters& params);


// ---------------------------------------------------------------------------
// Unrestricted DFTB0 result
// ---------------------------------------------------------------------------

struct UDFTB0Result : ParameterIdentifiedResult {
    double energy = 0.0;
    double e_electronic = 0.0;
    double e_repulsive = 0.0;
    Eigen::VectorXd mo_energies;       // MO energies (same for both spins)
    Eigen::MatrixXd mo_coeffs;         // MO coefficients
    Eigen::MatrixXd density_alpha;     // D_alpha = C_occ^α C_occ^α^T
    Eigen::MatrixXd density_beta;      // D_beta = C_occ^β C_occ^β^T
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd hamiltonian;
    Eigen::VectorXd charges;           // total-density Mulliken net charges
    int n_basis = 0;
    int n_alpha = 0;                   // number of α electrons
    int n_beta = 0;                    // number of β electrons
};

// Run an unrestricted DFTB0 calculation for open-shell molecules.
UDFTB0Result run_udftb0(const Molecule& mol,
                        const SemiempiricalParameters& params);

}  // namespace semiempirical
}  // namespace vibeqc
