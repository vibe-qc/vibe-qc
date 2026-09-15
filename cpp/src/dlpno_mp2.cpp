#include "vibeqc/dlpno_mp2.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>

namespace vibeqc {

namespace {

struct PairKeyHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return static_cast<std::size_t>(p.first) * 65521ULL
             + static_cast<std::size_t>(p.second);
    }
};

/// Closed-shell pair energy: e_ij = (2-d_ij) Σ_ab K_ab (2T_ab - T_ba).
double pair_energy(const Eigen::MatrixXd& K,
                   const Eigen::MatrixXd& T,
                   int i,
                   int j) {
    const double weight = (i == j) ? 1.0 : 2.0;
    return weight * (K.array() * (2.0 * T - T.transpose()).array()).sum();
}

} // anonymous namespace

DLPNOMP2IterResult dlpno_mp2_iterate(
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    const std::vector<Eigen::MatrixXd>& K_blocks,
    const std::vector<Eigen::VectorXd>& eps_blocks,
    const std::vector<Eigen::MatrixXd>& T_blocks,
    const std::vector<Eigen::MatrixXd>& V_blocks,
    const Eigen::MatrixXd& F_oo,
    const Eigen::MatrixXd& S_ao,
    double damping) {

    const int n_pairs = static_cast<int>(pair_i.size());
    const int n_act = F_oo.rows();

    // Build index map: (i,j) -> pair index.
    std::unordered_map<std::pair<int, int>, int, PairKeyHash> pair_idx;
    for (int p = 0; p < n_pairs; ++p) {
        pair_idx[{pair_i[p], pair_j[p]}] = p;
    }

    // Precompute S_link = V_p^T @ S_ao @ V_q for non-diagonal pairs.
    // This is ordered: S_pq is not interchangeable with S_qp when the two
    // PNO spaces have different dimensions. The Python reference caches the
    // ordered key ((i,j),(k,l)); mirror that here.
    std::unordered_map<std::pair<int, int>, Eigen::MatrixXd, PairKeyHash> S_link_cache;

    auto get_S_link = [&](int p_idx, int q_idx) -> const Eigen::MatrixXd& {
        if (p_idx == q_idx) {
            // Identity link when projecting into own basis
            static Eigen::MatrixXd I; // never returned by reference safely
            // Actually we need to handle this case — for same-pair,
            // the projection is identity. The Python code handles this
            // via the 'if q == (i,j): R -= f * T' shortcut.
            // We return a dummy; caller must check p_idx == q_idx.
            static Eigen::MatrixXd dummy;
            return dummy;
        }
        auto key = std::make_pair(p_idx, q_idx);
        auto it = S_link_cache.find(key);
        if (it != S_link_cache.end()) return it->second;
        // Compute and cache.
        Eigen::MatrixXd S = V_blocks[p_idx].transpose()
                          * S_ao
                          * V_blocks[q_idx];
        auto [new_it, _] = S_link_cache.emplace(key, std::move(S));
        return new_it->second;
    };

    auto find_pair = [&](int a, int b) -> std::pair<int, bool> {
        const bool transpose = a > b;
        const auto key = transpose ? std::make_pair(b, a) : std::make_pair(a, b);
        auto it = pair_idx.find(key);
        if (it == pair_idx.end()) return {-1, false};
        return {it->second, transpose};
    };

    // Output.
    std::vector<Eigen::MatrixXd> T_new(n_pairs);
    double max_r = 0.0;
    double e_corr = 0.0;

    for (int p_idx = 0; p_idx < n_pairs; ++p_idx) {
        const int i = pair_i[p_idx];
        const int j = pair_j[p_idx];
        const auto& K = K_blocks[p_idx];
        const auto& eps = eps_blocks[p_idx];
        const auto& T = T_blocks[p_idx];
        const int n_pno = static_cast<int>(eps.size());

        // Residual: R = K + (eps_i + eps_j) * T
        Eigen::MatrixXd R = K;
        for (int a = 0; a < n_pno; ++a) {
            for (int b = 0; b < n_pno; ++b) {
                R(a, b) += (eps[a] + eps[b]) * T(a, b);
            }
        }

        // Occupied-occupied Fock coupling over k.
        for (int k = 0; k < n_act; ++k) {
            // f_ik coupling: R -= f_ik * P(T_kj)
            double f_ik = F_oo(i, k);
            if (std::abs(f_ik) > 1e-14) {
                auto [q_idx, transpose_amp] = find_pair(k, j);
                if (q_idx >= 0) {
                    if (q_idx == p_idx) {
                        // Same pair: T_kj IS T, direct subtract.
                        R.noalias() -= f_ik * T;
                    } else {
                        const auto& S_link = get_S_link(p_idx, q_idx);
                        Eigen::MatrixXd T_kj;
                        if (transpose_amp) {
                            T_kj = T_blocks[q_idx].transpose();
                        } else {
                            T_kj = T_blocks[q_idx];
                        }
                        R.noalias() -= f_ik * (S_link * T_kj * S_link.transpose());
                    }
                }
            }
            // f_kj coupling: R -= f_kj * P(T_ik)
            double f_kj = F_oo(k, j);
            if (std::abs(f_kj) > 1e-14) {
                auto [q_idx, transpose_amp] = find_pair(i, k);
                if (q_idx >= 0) {
                    if (q_idx == p_idx) {
                        R.noalias() -= f_kj * T;
                    } else {
                        const auto& S_link = get_S_link(p_idx, q_idx);
                        Eigen::MatrixXd T_ik;
                        if (transpose_amp) {
                            T_ik = T_blocks[q_idx].transpose();
                        } else {
                            T_ik = T_blocks[q_idx];
                        }
                        R.noalias() -= f_kj * (S_link * T_ik * S_link.transpose());
                    }
                }
            }
        }

        // Max residual.
        double r_max_abs = R.array().abs().maxCoeff();
        if (r_max_abs > max_r) max_r = r_max_abs;

        // Denominator: D = f_ii + f_jj - eps_a - eps_b
        const double fii = F_oo(i, i);
        const double fjj = F_oo(j, j);
        Eigen::MatrixXd Tn = T;
        for (int a = 0; a < n_pno; ++a) {
            for (int b = 0; b < n_pno; ++b) {
                double denom = fii + fjj - eps[a] - eps[b];
                if (std::abs(denom) < 1e-14) denom = 1e-14;
                Tn(a, b) += damping * R(a, b) / denom;
            }
        }

        // Symmetrise diagonal pairs.
        if (i == j) {
            Tn = 0.5 * (Tn + Tn.transpose());
        }

        T_new[p_idx] = std::move(Tn);
        e_corr += pair_energy(K, T_new[p_idx], i, j);
    }

    return {std::move(T_new), max_r, e_corr};
}

}  // namespace vibeqc
