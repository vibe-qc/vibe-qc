#include "vibeqc/periodic_bloch.hpp"

#include <complex>
#include <cmath>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibeqc {

namespace {

using Complex = std::complex<double>;

/// Single-k Bloch sum: F(k) = Σ_g exp(+i k·R_g) · block[g].
Eigen::MatrixXcd bloch_sum_one(
    const std::vector<Eigen::MatrixXd>& blocks,
    const std::vector<Eigen::Vector3d>& cell_r,
    const Eigen::Vector3d& k) {

    if (blocks.empty()) return {};
    const Eigen::Index n = blocks[0].rows();
    Eigen::MatrixXcd F_k = Eigen::MatrixXcd::Zero(n, n);

    for (std::size_t g = 0; g < blocks.size(); ++g) {
        const double phase_arg = k.dot(cell_r[g]);
        const Complex phase(std::cos(phase_arg), std::sin(phase_arg));
        F_k.noalias() += phase * blocks[g].cast<Complex>();
    }
    return F_k;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<Eigen::MatrixXcd> bloch_sum_multi_k(
    const std::vector<Eigen::MatrixXd>& blocks,
    const std::vector<Eigen::Vector3d>& cell_r,
    const std::vector<Eigen::Vector3d>& k_vectors) {

    if (blocks.empty() || k_vectors.empty()) return {};

    const std::size_t N_k = k_vectors.size();
    std::vector<Eigen::MatrixXcd> result(N_k);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t ik = 0; ik < N_k; ++ik) {
        result[ik] = bloch_sum_one(blocks, cell_r, k_vectors[ik]);
    }
    return result;
}

std::vector<Eigen::MatrixXcd> assemble_fock_multi_k(
    const std::vector<Eigen::MatrixXd>& f2e_blocks,
    const std::vector<Eigen::Vector3d>& cell_r,
    const std::vector<Eigen::Vector3d>& k_vectors,
    const std::vector<Eigen::MatrixXcd>& hcore_k) {

    auto F_k_list = bloch_sum_multi_k(f2e_blocks, cell_r, k_vectors);

    if (!hcore_k.empty()) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t ik = 0; ik < k_vectors.size(); ++ik) {
            F_k_list[ik].noalias() += hcore_k[ik];
        }
    }
    return F_k_list;
}

std::vector<Eigen::MatrixXcd> inverse_bloch_multi_k(
    const std::vector<Eigen::MatrixXd>& blocks,
    const std::vector<Eigen::Vector3d>& cell_r,
    const std::vector<Eigen::Vector3d>& k_vectors) {

    auto D_k_list = bloch_sum_multi_k(blocks, cell_r, k_vectors);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t ik = 0; ik < k_vectors.size(); ++ik) {
        D_k_list[ik] = 0.5 * (D_k_list[ik] + D_k_list[ik].adjoint());
    }
    return D_k_list;
}

}  // namespace vibeqc
