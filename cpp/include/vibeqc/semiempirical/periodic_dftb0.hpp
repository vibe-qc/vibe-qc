// Periodic Gamma-point DFTB0 energy driver.
//
// Extends the molecular DFTB0 to periodic systems (1D/2D/3D) at the
// Gamma point. The Hamiltonian and overlap are lattice-summed over
// direct-lattice cells within a cutoff:
//
//   H_Γ = Σ_g H⁰(g)      S_Γ = Σ_g S(g)
//
// where H⁰(g) is the two-center Wolfsberg-Helmholtz Hamiltonian
// between the reference cell and image cell g. On-site energies ε_l
// are applied only in the g=0 (same-cell) block.
//
// The repulsive energy is summed over lattice cells:
//   E_rep = ½ Σ_g Σ_{A,B}' V_rep(R_{A,B_g})
//
// Gamma-point driver. General k-point support lives in kpoints_dftb0.hpp.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/parameters.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Periodic DFTB0 options
// ---------------------------------------------------------------------------

struct PeriodicDFTB0Options {
    // Cutoff for the real-space lattice sum (bohr). Cells with |g| > cutoff
    // are excluded. Default 15.0 bohr is sufficient for tightly-bound systems.
    double cutoff_bohr = 15.0;

    // If true, only include the g=0 cell (molecular limit, useful for testing).
    bool gamma_only_0 = false;
};

// ---------------------------------------------------------------------------
// Periodic DFTB0 result
// ---------------------------------------------------------------------------

struct PeriodicDFTB0Result : ParameterIdentifiedResult {
    double energy = 0.0;             // total energy per unit cell (Hartree)
    double e_electronic = 0.0;       // band-structure energy Σ_i n_i ε_i
    double e_repulsive = 0.0;        // pairwise repulsive energy per cell
    Eigen::VectorXd mo_energies;     // MO energies (Hartree), ascending
    Eigen::MatrixXd mo_coeffs;       // MO coefficients at Gamma
    Eigen::MatrixXd density;         // density matrix D = 2 C_occ C_occ^T
    // Per-orbital occupations of the Gamma spectrum (issue #339). Exactly
    // hard Aufbau (2.0 / 0.0) for an open gap; a degenerate frontier
    // manifold carries equal fractional occupations.
    Eigen::VectorXd occupations;
    Eigen::MatrixXd overlap_gamma;   // lattice-summed overlap S_Γ
    Eigen::MatrixXd hamiltonian_gamma; // lattice-summed H_Γ
    int n_basis = 0;
    int n_occ = 0;
    int n_cells = 0;                 // number of lattice cells included
    // Effective lattice-sum provenance of the SCF that produced this result
    // (issue #340). 0.0 marks a legacy/default-initialized result without
    // provenance; the analytic derivative routes fail closed on it.
    double cutoff_bohr = 0.0;
    bool gamma_only_0 = false;
};

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

PeriodicDFTB0Result run_dftb0_gamma(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const PeriodicDFTB0Options& opts = {});


// ---------------------------------------------------------------------------
// Unrestricted periodic DFTB0 (Gamma-point)
// ---------------------------------------------------------------------------

struct PeriodicUDFTB0Result : ParameterIdentifiedResult {
    double energy = 0.0;
    double e_electronic = 0.0;
    double e_repulsive = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density_alpha;
    Eigen::MatrixXd density_beta;
    // Per-spin Gamma occupations actually used to build the densities
    // (issue #439).  Full occupancy is 1.0 per channel.  A channel whose
    // frontier has an open gap carries the exact hard-Aufbau vector; a
    // channel that cuts a roundoff-degenerate manifold carries the equal
    // fractional occupations of the Weinert & Davenport T -> 0 ensemble, so
    // its density and every analytic derivative are unique.  Recorded so a
    // consumer can tell the two apart -- previously the unrestricted result
    // exposed no occupations at all.
    Eigen::VectorXd occupations_alpha;
    Eigen::VectorXd occupations_beta;
    Eigen::MatrixXd overlap_gamma;
    Eigen::MatrixXd hamiltonian_gamma;
    int n_basis = 0;
    int n_alpha = 0;
    int n_beta = 0;
    int n_cells = 0;
    // Effective lattice-sum provenance of the SCF that produced this result
    // (issue #340); see PeriodicDFTB0Result.
    double cutoff_bohr = 0.0;
    bool gamma_only_0 = false;
};

PeriodicUDFTB0Result run_udftb0_gamma(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const PeriodicDFTB0Options& opts = {});

}  // namespace semiempirical
}  // namespace vibeqc
