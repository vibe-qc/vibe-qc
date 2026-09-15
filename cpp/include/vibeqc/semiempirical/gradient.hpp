// Analytic nuclear gradient for DFTB0 and SCC-DFTB.
//
// Molecular DFTB0 gradient:
//   dE/dR_A = Σ_{μν} M_{μν} · dS_{μν}/dR_A + dE_rep/dR_A
//   M_{μν} = D_{μν} · ½κ · h̄_AB − W_{μν}   (two-center, A≠B)
//   M_{μν} = −W_{μν}                         (on-site, A=B)
//
// SCC-DFTB analytic gradient:
//   dE/dR_A = DFTB0-style gradient using converged SCC {D, W, H^{SCC}}
//           + ½ Σ_{BC} Δq_B Δq_C · ∂γ_{BC}/∂R_A
//
// dS/dR is computed via libint's overlap derivative engine (deriv_order=1).
// dE_rep/dR is computed analytically from the repulsive pair potential.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/semiempirical/dftb0.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/scc_dftb.hpp"
#include "vibeqc/semiempirical/periodic_dftb0.hpp"
#include "vibeqc/semiempirical/periodic_scc_dftb.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/methods/xtb/periodic_gfn2.hpp"

namespace vibeqc {
namespace semiempirical {

// Returns an (n_atoms, 3) matrix of gradient components in Hartree/bohr.
Eigen::MatrixXd compute_dftb0_gradient(
    const Molecule& mol,
    const DFTB0Result& result,
    const SemiempiricalParameters& params);

// Unrestricted DFTB0 analytic gradient.
Eigen::MatrixXd compute_udftb0_gradient(
    const Molecule& mol,
    const UDFTB0Result& result,
    const SemiempiricalParameters& params);

// SCC-DFTB analytic gradient.
Eigen::MatrixXd compute_scc_dftb_gradient(
    const Molecule& mol,
    const SCCDFTBResult& result,
    const SemiempiricalParameters& params);

// Unrestricted SCC-DFTB analytic gradient.
Eigen::MatrixXd compute_uscc_dftb_gradient(
    const Molecule& mol,
    const USCCDFTBResult& result,
    const SemiempiricalParameters& params);

// Historical CP-SCC entry point (SE5). The variational SCC energy
// assembly makes the charge-response correction identically zero at
// the converged solution, so this now simply delegates to
// compute_scc_dftb_gradient. Kept for API stability.
Eigen::MatrixXd compute_scc_dftb_gradient_response(
    const Molecule& mol,
    const SCCDFTBResult& result,
    const SemiempiricalParameters& params);

// Repulsive energy gradient only (for testing).
Eigen::MatrixXd dftb0_repulsive_gradient(
    const Molecule& mol,
    const SemiempiricalParameters& params);


// ---------------------------------------------------------------------------
// Periodic gradients (Gamma-point, SE3)
// ---------------------------------------------------------------------------

// Periodic DFTB0 atomic gradient (Gamma-point).
Eigen::MatrixXd compute_periodic_dftb0_gradient(
    const PeriodicSystem& system,
    const PeriodicDFTB0Result& result,
    const SemiempiricalParameters& params);

// Periodic SCC-DFTB atomic gradient (Gamma-point). The overlap contraction
// includes every periodic Wolfsberg-Helmholtz image block, and the charge
// term differentiates the same image-summed gamma matrix used by the SCF.
// A converged SCC result is required.
Eigen::MatrixXd compute_periodic_scc_dftb_gradient(
    const PeriodicSystem& system,
    const PeriodicSCCDFTBResult& result,
    const SemiempiricalParameters& params);

// Periodic UDFTB0 atomic gradient (Gamma-point).
Eigen::MatrixXd compute_periodic_udftb0_gradient(
    const PeriodicSystem& system,
    const PeriodicUDFTB0Result& result,
    const SemiempiricalParameters& params);

// Periodic USCC-DFTB atomic gradient (Gamma-point). A converged SCC result is
// required.
Eigen::MatrixXd compute_periodic_uscc_dftb_gradient(
    const PeriodicSystem& system,
    const PeriodicUSCCDFTBResult& result,
    const SemiempiricalParameters& params);


// ---------------------------------------------------------------------------
// Periodic stress tensors (SE3a)
// ---------------------------------------------------------------------------

// Periodic DFTB0 analytic stress tensor (Gamma-point).
// Returns (3,3) symmetric matrix σ_{ij} = (1/V)·dE/dε_{ij} in Ha/bohr^3.
Eigen::MatrixXd compute_periodic_dftb0_stress(
    const PeriodicSystem& system,
    const PeriodicDFTB0Result& result,
    const SemiempiricalParameters& params);

// Periodic SCC-DFTB analytic stress tensor (Gamma-point). A converged SCC
// result is required.
Eigen::MatrixXd compute_periodic_scc_dftb_stress(
    const PeriodicSystem& system,
    const PeriodicSCCDFTBResult& result,
    const SemiempiricalParameters& params);

// GFN2-xTB analytic gradient (fixed-charge approximation).
Eigen::MatrixXd compute_gfn2_gradient(
    const Molecule& mol,
    const xtb::GFN2Result& result,
    const xtb::GFN2ParameterSet& params);

// Periodic GFN2-xTB atomic gradient (Gamma-point). Differentiates the
// converged periodic energy/free-energy surface, including AES response.
Eigen::MatrixXd compute_periodic_gfn2_gradient(
    const PeriodicSystem& system,
    const xtb::PeriodicGFN2Result& result,
    const xtb::GFN2ParameterSet& params);

// Periodic GFN2-xTB analytic stress tensor (Gamma-point). Returns
// sigma_ij = (1/V) dE/deps_ij for R' = (I + eps) R.
Eigen::MatrixXd compute_periodic_gfn2_stress(
    const PeriodicSystem& system,
    const xtb::PeriodicGFN2Result& result,
    const xtb::GFN2ParameterSet& params);

}  // namespace semiempirical
}  // namespace vibeqc
