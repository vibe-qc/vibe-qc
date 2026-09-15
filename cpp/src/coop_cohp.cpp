#include "vibeqc/coop_cohp.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace vibeqc {

// ===================================================================
// Per-k COOP weight
// ===================================================================

Eigen::MatrixXd coop_weights_k(
    const ComplexMatrix& C_k,
    const ComplexMatrix& S_k,
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>&
        pairs_ao)
{
    const Eigen::Index n_bands = C_k.cols();
    const Eigen::Index n_pairs = static_cast<Eigen::Index>(pairs_ao.size());
    Eigen::MatrixXd result(n_pairs, n_bands);

    for (Eigen::Index p = 0; p < n_pairs; ++p) {
        const auto& [ao_a, ao_b] = pairs_ao[static_cast<size_t>(p)];
        const Eigen::Index na = static_cast<Eigen::Index>(ao_a.size());
        const Eigen::Index nb = static_cast<Eigen::Index>(ao_b.size());

        // Extract sub-blocks
        // C_A: (na, n_bands), C_B: (nb, n_bands), S_AB: (na, nb)
        Eigen::MatrixXcd C_A(na, n_bands);
        Eigen::MatrixXcd C_B(nb, n_bands);
        for (Eigen::Index i = 0; i < na; ++i) {
            C_A.row(i) = C_k.row(ao_a[static_cast<size_t>(i)]);
        }
        for (Eigen::Index j = 0; j < nb; ++j) {
            C_B.row(j) = C_k.row(ao_b[static_cast<size_t>(j)]);
        }

        // S_AB: extract sub-matrix
        Eigen::MatrixXcd S_AB(na, nb);
        for (Eigen::Index i = 0; i < na; ++i) {
            for (Eigen::Index j = 0; j < nb; ++j) {
                S_AB(i, j) = S_k(ao_a[static_cast<size_t>(i)],
                                 ao_b[static_cast<size_t>(j)]);
            }
        }

        // w_{AB,n} = Re[ Σ_μ C*_μn · (S_AB · C_B)_μn ]
        //          = Re[ diag( C_A.adjoint() * S_AB * C_B ) ]
        // Or equivalently: Re[ sum( conj(C_A) .* (S_AB * C_B), axis=0 ) ]
        Eigen::MatrixXcd temp = S_AB * C_B;   // (na, n_bands)
        for (Eigen::Index n = 0; n < n_bands; ++n) {
            std::complex<double> acc(0.0, 0.0);
            for (Eigen::Index i = 0; i < na; ++i) {
                acc += std::conj(C_A(i, n)) * temp(i, n);
            }
            result(p, n) = std::real(acc);
        }
    }

    return result;
}

// ===================================================================
// Per-k COHP weight  (identical to COOP but with H_k instead of S_k)
// ===================================================================

Eigen::MatrixXd cohp_weights_k(
    const ComplexMatrix& C_k,
    const ComplexMatrix& H_k,
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>&
        pairs_ao)
{
    const Eigen::Index n_bands = C_k.cols();
    const Eigen::Index n_pairs = static_cast<Eigen::Index>(pairs_ao.size());
    Eigen::MatrixXd result(n_pairs, n_bands);

    for (Eigen::Index p = 0; p < n_pairs; ++p) {
        const auto& [ao_a, ao_b] = pairs_ao[static_cast<size_t>(p)];
        const Eigen::Index na = static_cast<Eigen::Index>(ao_a.size());
        const Eigen::Index nb = static_cast<Eigen::Index>(ao_b.size());

        Eigen::MatrixXcd C_A(na, n_bands);
        Eigen::MatrixXcd C_B(nb, n_bands);
        for (Eigen::Index i = 0; i < na; ++i) {
            C_A.row(i) = C_k.row(ao_a[static_cast<size_t>(i)]);
        }
        for (Eigen::Index j = 0; j < nb; ++j) {
            C_B.row(j) = C_k.row(ao_b[static_cast<size_t>(j)]);
        }

        Eigen::MatrixXcd H_AB(na, nb);
        for (Eigen::Index i = 0; i < na; ++i) {
            for (Eigen::Index j = 0; j < nb; ++j) {
                H_AB(i, j) = H_k(ao_a[static_cast<size_t>(i)],
                                 ao_b[static_cast<size_t>(j)]);
            }
        }

        Eigen::MatrixXcd temp = H_AB * C_B;
        for (Eigen::Index n = 0; n < n_bands; ++n) {
            std::complex<double> acc(0.0, 0.0);
            for (Eigen::Index i = 0; i < na; ++i) {
                acc += std::conj(C_A(i, n)) * temp(i, n);
            }
            result(p, n) = std::real(acc);
        }
    }

    return result;
}

// ===================================================================
// Gaussian broadening accumulator
// ===================================================================

Eigen::MatrixXd gaussian_broaden_projected(
    const Eigen::MatrixXd& energies_per_k,   // (n_k, n_bands)
    const Eigen::MatrixXd& weights_per_k,    // (n_pairs, n_k, n_bands)
    const Eigen::VectorXd& k_weights,         // (n_k,)
    const Eigen::VectorXd& energy_grid,       // (n_e,)
    double sigma)
{
    const Eigen::Index n_pairs = weights_per_k.rows();
    const Eigen::Index n_k     = static_cast<Eigen::Index>(k_weights.size());
    const Eigen::Index n_bands = static_cast<Eigen::Index>(energies_per_k.cols());
    const Eigen::Index n_e     = static_cast<Eigen::Index>(energy_grid.size());

    const double inv_2sigma2 = 0.5 / (sigma * sigma);
    const double norm = 1.0 / (sigma * std::sqrt(2.0 * M_PI));

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_pairs, n_e);

    // weights_per_k is flat: shape (n_pairs, n_k * n_bands)
    // but stored column-major... Actually pybind11 passes numpy arrays
    // as Eigen::Map with the numpy stride, so row/col major depends on
    // the numpy layout. We access via (pair, k, band) indices.
    //
    // For efficiency, we stride through pair as outermost:
    for (Eigen::Index p = 0; p < n_pairs; ++p) {
        for (Eigen::Index k = 0; k < n_k; ++k) {
            const double wk = k_weights(k);
            if (wk == 0.0) continue;
            for (Eigen::Index n = 0; n < n_bands; ++n) {
                const double coeff = wk * weights_per_k(p, k * n_bands + n);
                if (coeff == 0.0) continue;
                const double eps = energies_per_k(k, n);
                for (Eigen::Index e = 0; e < n_e; ++e) {
                    const double de = energy_grid(e) - eps;
                    result(p, e) += coeff * norm *
                                    std::exp(-de * de * inv_2sigma2);
                }
            }
        }
    }

    return result;
}

}  // namespace vibeqc
