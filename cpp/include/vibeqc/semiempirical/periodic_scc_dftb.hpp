// Periodic SCC-DFTB at Gamma-point.
//
// Extends periodic DFTB0 with charge self-consistency. The SCC correction
// to the Gamma-point Hamiltonian is:
//
//   H_Γ^{SCC} = H_Γ - ½ S_Γ · diag(V_A + V_B)
//
// where V_A = Σ_C γ_{AC}^{periodic} Δq_C is the lattice-summed potential.
// The periodic γ matrix sums over lattice images:
//   γ_{AB}^{per} = Σ_g 1/√(|R_{AB} + g|² + η_{AB}²)
//
// Gamma-point driver. General k-point support lives in kpoints_scc_dftb.hpp.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/periodic_dftb0.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Periodic SCC-DFTB options
// ---------------------------------------------------------------------------

struct PeriodicSCCOptions {
    double cutoff_bohr = 15.0;       // lattice sum cutoff
    // Functional form of the SCC-DFTB gamma (maintainer decision D1,
    // 2026-08-28: "land the Elstner gamma behind a flag, periodic routes
    // defaulting to it").  Elstner's remainder decays exponentially, so the
    // Ewald-split lattice sum converges absolutely and the route has a
    // thermodynamic limit; the in-house Klopman-Ohno remainder falls off as
    // R^-3 and does not (#425).  The molecular default stays Klopman-Ohno
    // until the measured repin table lands, per the same staging.
    ShellGammaForm gamma_form = ShellGammaForm::Elstner;
    int max_iter = 100;
    double conv_tol_charge = 1e-6;   // max |Δq_new − Δq_old|
    double charge_mixing = 0.2;
    bool use_diis = false;           // Pulay DIIS accelerator for charge vector
    int diis_subspace = 6;           // DIIS history depth (2–12)
};

// ---------------------------------------------------------------------------
// Periodic SCC-DFTB result
// ---------------------------------------------------------------------------

struct PeriodicSCCDFTBResult : ParameterIdentifiedResult {
    double energy = 0.0;
    double e_electronic = 0.0;
    double e_repulsive = 0.0;
    double e_scc = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    // Per-orbital occupations of the Gamma spectrum (issue #339); see
    // PeriodicDFTB0Result::occupations.
    Eigen::VectorXd occupations;
    Eigen::MatrixXd overlap_gamma;
    Eigen::MatrixXd hamiltonian_gamma;
    Eigen::VectorXd charges;         // Δq_A per atom in unit cell
    int n_basis = 0;
    int n_occ = 0;
    int n_cells = 0;
    int n_iter = 0;
    bool converged = false;
    // Effective lattice-sum provenance of the SCF that produced this result
    // (issue #340); see PeriodicDFTB0Result.
    double cutoff_bohr = 0.0;
    // Gamma functional form the SCF used.  Provenance for the same reason
    // the cutoff is: the analytic gradient and stress must differentiate
    // the energy that was actually converged, and the two forms are
    // different functions of the geometry.
    ShellGammaForm gamma_form = ShellGammaForm::Elstner;
};

// Ewald-split lattice-summed SCC-DFTB gamma; see the definition for why
// both halves of Elstner Eq. 17/18 are required (#425).
Eigen::MatrixXd build_periodic_dftb_gamma_matrix(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    double cutoff_bohr,
    ShellGammaForm form = ShellGammaForm::Elstner);

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

PeriodicSCCDFTBResult run_scc_dftb_gamma(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const PeriodicSCCOptions& opts = {});


// ---------------------------------------------------------------------------
// Unrestricted periodic SCC-DFTB (Gamma-point)
// ---------------------------------------------------------------------------

struct PeriodicUSCCDFTBResult : ParameterIdentifiedResult {
    double energy = 0.0;
    double e_electronic = 0.0;
    double e_repulsive = 0.0;
    double e_scc = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density_alpha;
    Eigen::MatrixXd density_beta;
    // Per-spin Gamma occupations actually used to build the densities
    // (issue #439); see PeriodicUDFTB0Result for the convention.
    Eigen::VectorXd occupations_alpha;
    Eigen::VectorXd occupations_beta;
    Eigen::MatrixXd overlap_gamma;
    Eigen::MatrixXd hamiltonian_gamma;
    Eigen::VectorXd charges;
    int n_basis = 0;
    int n_alpha = 0;
    int n_beta = 0;
    int n_cells = 0;
    int n_iter = 0;
    bool converged = false;
    // Effective lattice-sum provenance of the SCF that produced this result
    // (issue #340); see PeriodicDFTB0Result.
    double cutoff_bohr = 0.0;
    // Gamma functional form the SCF used.  Provenance for the same reason
    // the cutoff is: the analytic gradient and stress must differentiate
    // the energy that was actually converged, and the two forms are
    // different functions of the geometry.
    ShellGammaForm gamma_form = ShellGammaForm::Elstner;
};

PeriodicUSCCDFTBResult run_uscc_dftb_gamma(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const PeriodicSCCOptions& opts = {});

}  // namespace semiempirical
}  // namespace vibeqc
