#include "vibeqc/scf_restart.hpp"

#include <Eigen/QR>
#include <cmath>

namespace vibeqc {

Eigen::MatrixXd random_orthogonal(int n, std::uint64_t seed) {
    if (n <= 0) return Eigen::MatrixXd::Identity(0, 0);
    auto next_u64 = [state = seed]() mutable -> std::uint64_t {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    auto next_gaussian = [&next_u64]() -> double {
        constexpr double scale = 1.0 / (1ULL << 52);
        double u1, u2, s;
        do {
            u1 = 2.0 * (next_u64() >> 12) * scale - 1.0;
            u2 = 2.0 * (next_u64() >> 12) * scale - 1.0;
            s = u1 * u1 + u2 * u2;
        } while (s >= 1.0 || s == 0.0);
        return u1 * std::sqrt(-2.0 * std::log(s) / s);
    };
    Eigen::MatrixXd A(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A(i, j) = next_gaussian();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd Q = qr.householderQ();
    if (Q.determinant() < 0.0) Q.col(n - 1) *= -1.0;
    return Q;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> rotate_orbitals_in_subspaces(
    const Eigen::MatrixXd& C_prev,
    const Eigen::VectorXd& eps_prev,
    int n_occ,
    const Eigen::MatrixXd* F,
    std::uint64_t seed_occ,
    std::uint64_t seed_vir) {

    const Eigen::Index n_kept = C_prev.cols();
    const Eigen::Index n_vir  = n_kept - n_occ;
    if (n_occ <= 0 || n_vir <= 0) {
        Eigen::VectorXd eps_new = eps_prev;
        if (F != nullptr) eps_new = (C_prev.transpose() * (*F) * C_prev).diagonal();
        return {C_prev, eps_new};
    }
    Eigen::MatrixXd Q_occ = random_orthogonal(static_cast<int>(n_occ), seed_occ);
    Eigen::MatrixXd Q_vir = random_orthogonal(static_cast<int>(n_vir), seed_vir);
    Eigen::MatrixXd Q_full = Eigen::MatrixXd::Identity(n_kept, n_kept);
    Q_full.topLeftCorner(n_occ, n_occ) = Q_occ;
    Q_full.bottomRightCorner(n_vir, n_vir) = Q_vir;
    Eigen::MatrixXd C_new = C_prev * Q_full;
    Eigen::VectorXd eps_new = eps_prev;
    if (F != nullptr) {
        Eigen::MatrixXd F_rot = C_new.transpose() * (*F) * C_new;
        F_rot = 0.5 * (F_rot + F_rot.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F_rot);
        if (es.info() == Eigen::Success) {
            C_new = C_new * es.eigenvectors();
            eps_new = es.eigenvalues();
        } else {
            eps_new = F_rot.diagonal();
        }
    }
    return {C_new, eps_new};
}

}  // namespace vibeqc
