// Fail-closed compatibility declarations for the retired periodic OMx driver.
// The prototype omitted the published image-resolved ORT, ECP, and penetration
// Hamiltonian and cannot return OMx-labelled numerical results.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/methods/nddo/omx_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// ---------------------------------------------------------------------------
// Periodic OMx options
// ---------------------------------------------------------------------------

struct PeriodicOMxOptions {
    // Retained for ABI compatibility. All fields are ignored while the route
    // is scientifically gated.
    double cutoff_bohr = 15.0;
    double conv_tol = 1e-7;
    int max_iter = 100;
    bool gamma_only_0 = false;
    int warmup_iters = 3;         // orthogonal iterations before DIIS
    double density_mixing = 0.3;  // mixing during the pre-DIIS warm-up
    double eval_floor = 1e-5;     // eigenvalue floor for Loewdin orthogonalizer
                                   // (was 0.5 — too high, clips physical eigenvalues
                                   // with the correct n-dependent overlaps)
};

// ---------------------------------------------------------------------------
// Periodic OMx result
// ---------------------------------------------------------------------------

struct PeriodicOMxResult : ParameterIdentifiedResult {
    double energy = 0.0;
    double e_electronic = 0.0;
    double e_core = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    Eigen::MatrixXd fock_gamma;
    Eigen::MatrixXd overlap_gamma;
    int n_basis = 0;
    int n_occ = 0;
    int n_cells = 0;
    int n_iter = 0;
    bool converged = false;
};

// ---------------------------------------------------------------------------
// Reject the retired periodic OMx prototype.
// ---------------------------------------------------------------------------

PeriodicOMxResult run_omx_gamma(
    const PeriodicSystem& system,
    const OMxParameterSet& params,
    const PeriodicOMxOptions& opts = {});

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
