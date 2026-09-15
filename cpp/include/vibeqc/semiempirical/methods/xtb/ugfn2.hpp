// Unrestricted GFN2-xTB energy driver.
//
// Extends GFN2-xTB to open-shell systems with separate alpha/beta
// density matrices.  Follows the same pattern as USCC-DFTB.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

// ---------------------------------------------------------------------------
// Unrestricted GFN2-xTB result
// ---------------------------------------------------------------------------

struct UGFN2Result : ParameterIdentifiedResult {
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

// ---------------------------------------------------------------------------
// Run unrestricted GFN2-xTB SCC calculation
// ---------------------------------------------------------------------------

UGFN2Result run_ugfn2_xtb(
    const Molecule& mol,
    const GFN2ParameterSet& params,
    const XTBSccOptions& opts = {});

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
