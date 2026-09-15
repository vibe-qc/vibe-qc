// Self-consistent-charge DFTB (SCC-DFTB) driver.
//
// Extends DFTB0 with charge self-consistency via Mulliken analysis.
// The SCF loop iterates:
//   1. Build H^{SCC} = H⁰ + ½ S (V_A + V_B)
//   2. Diagonalize H^{SCC} C = S C ε
//   3. Build density D = 2 C_occ C_occ^T
//   4. Compute Mulliken charges Δq_A
//   5. Mix charges, check convergence
//   6. Repeat
//
// Total energy:
//   E_SCC = Σ_i n_i ε_i + E_rep + ½ Σ_{AB} γ_{AB} Δq_A Δq_B

#pragma once

#include <Eigen/Dense>
#include <vector>

#include "vibeqc/semiempirical/dftb0.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/semiempirical/parameters.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// SCC-DFTB options
// ---------------------------------------------------------------------------

struct SCCOptions {
    int max_iter = 100;
    double conv_tol_charge = 1e-6;   // max |Δq_new − Δq_old| (e)
    double charge_mixing = 0.2;      // fraction of new charges to mix in (0–1)
    double electronic_temperature = 0.0;  // Fermi smearing (Ha), 0 = Aufbau
    Eigen::VectorXd initial_charges;  // optional Δq guess (n_atoms)
    bool use_diis = false;           // Pulay DIIS accelerator for charge vector
    int diis_subspace = 6;           // DIIS history depth (2–12)
    // Functional form of the SCC-DFTB gamma (maintainer decision D1,
    // 2026-08-28).  Staged: the flag exists first and the molecular default
    // stays Klopman-Ohno until the measured repin table lands, at which
    // point molecular and periodic cut over together to Elstner so that one
    // method name means one gamma.  Periodic routes already default to
    // Elstner, which is what gives their lattice sum a thermodynamic limit
    // (#425).
    ShellGammaForm gamma_form = ShellGammaForm::KlopmanOhno;
};

// ---------------------------------------------------------------------------
// SCC-DFTB result
// ---------------------------------------------------------------------------

struct SCCDFTBResult : ParameterIdentifiedResult {
    double energy = 0.0;            // free energy = electronic + SCC + repulsive
    double e_electronic = 0.0;      // tr(D H⁰) - T*S (T=0 by default)
    double e_repulsive = 0.0;       // pairwise repulsive energy
    double e_scc = 0.0;             // ½ Σ_{AB} γ_{AB} Δq_A Δq_B (≥ 0)
    Eigen::VectorXd mo_energies;    // MO energies (Hartree), ascending
    Eigen::MatrixXd mo_coeffs;      // MO coefficients, columns = MOs
    Eigen::MatrixXd density;        // D = 2 C_occ C_occ^T
    Eigen::MatrixXd overlap;        // overlap matrix S
    Eigen::MatrixXd hamiltonian;    // converged H^{SCC} matrix
    Eigen::VectorXd charges;        // Mulliken charge fluctuations Δq_A (n_atoms)
    int n_basis = 0;
    int n_occ = 0;
    int n_iter = 0;
    bool converged = false;
};

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

SCCDFTBResult run_scc_dftb(const Molecule& mol,
                           const SemiempiricalParameters& params,
                           const SCCOptions& opts = {});


// ---------------------------------------------------------------------------
// Unrestricted SCC-DFTB result
// ---------------------------------------------------------------------------

struct USCCDFTBResult : ParameterIdentifiedResult {
    double energy = 0.0;
    double e_electronic = 0.0;
    double e_repulsive = 0.0;
    double e_scc = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density_alpha;
    Eigen::MatrixXd density_beta;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd hamiltonian;
    Eigen::VectorXd charges;
    int n_basis = 0;
    int n_alpha = 0;
    int n_beta = 0;
    int n_iter = 0;
    bool converged = false;
};

USCCDFTBResult run_uscc_dftb(const Molecule& mol,
                              const SemiempiricalParameters& params,
                              const SCCOptions& opts = {});

}  // namespace semiempirical
}  // namespace vibeqc
