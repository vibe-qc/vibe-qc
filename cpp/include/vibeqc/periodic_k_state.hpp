// Actual Bloch density and raw-operator residuals for native periodic SCF.
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vibeqc::detail {

// Same boundary as solve_fock_matrix: diagnose an inconsistent builder before
// either a convergence test or a self-adjoint solver can hide it. No projection.
inline void require_hermitian_k_operator(const Eigen::MatrixXcd& matrix) {
    if (matrix.rows() == 0 || matrix.rows() != matrix.cols() || !matrix.allFinite()
        || (matrix - matrix.adjoint()).norm()
            > 1e-10 * std::max(1.0, matrix.norm()))
        throw std::invalid_argument("Periodic k operator must be finite and Hermitian");
}

inline std::vector<Eigen::MatrixXcd> occupied_k_densities(
    const std::vector<Eigen::MatrixXcd>& coefficients, int nocc) {
    std::vector<Eigen::MatrixXcd> densities;
    densities.reserve(coefficients.size());
    for (const auto& c : coefficients) {
        if (nocc < 0 || nocc > c.cols() || !c.allFinite())
            throw std::invalid_argument("Periodic occupied k density: invalid orbitals");
        densities.push_back(2.0 * c.leftCols(nocc) * c.leftCols(nocc).adjoint());
    }
    return densities;
}

// P(g) = sum_k w_k exp(-ik.g) D(k). A finite operator image enclosure is
// not an inverse quadrature for recovering D(k): at Gamma it would multiply
// D by the number of cells. Keep D(k) as SCF state, including its damping.
// Real/imag blocks reproduce sum_k w_k Re tr(error_i(k)^dagger error_j(k)).
inline std::vector<Eigen::MatrixXd> weighted_k_commutators(
    const std::vector<Eigen::MatrixXcd>& fock,
    const std::vector<Eigen::MatrixXcd>& density,
    const std::vector<Eigen::MatrixXcd>& overlap,
    const std::vector<double>& weights) {
    if (fock.empty() || fock.size() != density.size()
        || fock.size() != overlap.size() || fock.size() != weights.size())
        throw std::invalid_argument("Periodic k residual: mismatched state");
    double weight_sum = 0;
    std::vector<Eigen::MatrixXd> blocks;
    blocks.reserve(2 * fock.size());
    for (std::size_t k = 0; k < fock.size(); ++k) {
        require_hermitian_k_operator(fock[k]);
        require_hermitian_k_operator(density[k]);
        require_hermitian_k_operator(overlap[k]);
        if (fock[k].rows() != density[k].rows()
            || fock[k].rows() != overlap[k].rows()
            || !std::isfinite(weights[k]) || weights[k] < 0)
            throw std::invalid_argument("Periodic k residual: invalid dimensions or weight");
        weight_sum += weights[k];
        const Eigen::MatrixXcd error = fock[k] * density[k] * overlap[k]
                                    - overlap[k] * density[k] * fock[k];
        const double scale = std::sqrt(weights[k]);
        blocks.push_back(scale * error.real());
        blocks.push_back(scale * error.imag());
    }
    if (std::abs(weight_sum - 1.0) > 1e-12)
        throw std::invalid_argument("Periodic k residual: weights must sum to one");
    return blocks;
}

inline double block_norm(const std::vector<Eigen::MatrixXd>& blocks) {
    double square = 0;
    for (const auto& block : blocks) square += block.squaredNorm();
    return std::sqrt(square);
}

} // namespace vibeqc::detail
