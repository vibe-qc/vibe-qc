// Periodic PM6 NDDO driver at Gamma-point.
//
// Lattice-summed extension of the molecular PM6 Fock builder.
// The NDDO two-center Fock terms and core-core repulsion are summed
// over direct-lattice cells within a cutoff:
//
//   F_Gamma = Σ_g F(g)     where F(g) contains two-center terms
//                            between unit cell and image cell g.
//
// One-center terms are applied only in the g=0 (same-cell) block.
// The basis remains orthogonal (S=I) for PM6.
//
// The sibling periodic_omx.hpp declarations are a fail-closed compatibility
// surface; Bloch-periodic OM1/OM2/OM3 are not implemented.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// ---------------------------------------------------------------------------
// Periodic PM6 options
// ---------------------------------------------------------------------------

struct PeriodicPM6Options {
    // Cutoff for the real-space lattice sum (bohr).
    double cutoff_bohr = 15.0;

    // SCF convergence tolerance (max |Delta D| and max |[F, D]|)
    double conv_tol = 1e-7;

    // Maximum SCF iterations
    int max_iter = 100;

    // If true, only include the g=0 cell (molecular limit, useful for testing).
    bool gamma_only_0 = false;
};

// ---------------------------------------------------------------------------
// Periodic PM6 result
// ---------------------------------------------------------------------------

struct PeriodicPM6Result : ParameterIdentifiedResult {
    double energy = 0.0;             // total energy per unit cell (Ha)
    double e_electronic = 0.0;       // electronic energy ½Tr[P·(H+F)]
    double e_core = 0.0;             // core-core repulsion per unit cell
    Eigen::VectorXd mo_energies;     // MO energies (Ha), ascending
    Eigen::MatrixXd mo_coeffs;       // MO coefficients at Gamma
    Eigen::MatrixXd density;         // density matrix D = 2 C_occ C_occ^T
    Eigen::MatrixXd fock_gamma;      // lattice-summed Fock matrix F_Γ
    int n_basis = 0;
    int n_occ = 0;
    int n_cells = 0;                 // number of lattice cells included
    int n_iter = 0;
    bool converged = false;
};

// ---------------------------------------------------------------------------
// Run periodic PM6 SCF calculation at Gamma-point.
// ---------------------------------------------------------------------------

PeriodicPM6Result run_pm6_gamma(
    const PeriodicSystem& system,
    const PM6ParameterSet& params,
    const PeriodicPM6Options& opts = {});

// Internal batch entry point that reuses a validated lattice-cell list.
PeriodicPM6Result run_pm6_gamma_with_cells(
    const PeriodicSystem& system,
    const PM6ParameterSet& params,
    const PeriodicPM6Options& opts,
    const std::vector<LatticeCell>& cells);

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
