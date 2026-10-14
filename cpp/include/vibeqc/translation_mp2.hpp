#pragma once

#include <Eigen/Core>
#include <vector>

namespace vibeqc {

/// One evaluated representative-amplitude snapshot, before any next update.
struct TranslationMP2Progress {
    int iteration;
    double energy;
    double energy_change;
    double maximum_residual;
};

struct TranslationMP2Result {
    std::vector<Eigen::MatrixXd> amplitudes;
    std::vector<Eigen::MatrixXd> residuals;
    std::vector<TranslationMP2Progress> trace;
    bool converged = false;
};

/// Experimental real restricted finite-torus Jacobi solve. Only representative
/// tensors are retained; translated source bases are contracted on demand.
/// The raw source-space residual follows Nejad et al., doi:10.1063/5.0290816,
/// Eqs. (42)-(44). No occupied-Fock cutoff or denominator floor is applied.
TranslationMP2Result translation_mp2_solve(
    const std::vector<int>& mesh, int n_bands,
    const std::vector<int>& pair_i, const std::vector<int>& pair_j,
    const std::vector<Eigen::MatrixXd>& integrals,
    const std::vector<Eigen::MatrixXd>& virtual_fock,
    const std::vector<Eigen::MatrixXd>& seeds,
    const std::vector<Eigen::MatrixXd>& coefficients,
    const Eigen::MatrixXd& occupied_fock, const Eigen::MatrixXd& overlap,
    const std::vector<double>& energy_weights, int maximum_iterations,
    double energy_tolerance, double residual_tolerance, double damping,
    double symmetry_tolerance);

}  // namespace vibeqc
