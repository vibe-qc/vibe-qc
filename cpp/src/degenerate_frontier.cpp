#include "vibeqc/degenerate_frontier.hpp"

#include <algorithm>
#include <cmath>

namespace vibeqc {

namespace {

// Sum over atoms of the squared Loewdin population of the subspace spanned by
// the columns of `WU` (already Loewdin-orthonormalised). Depends only on the
// subspace, so it is a function on the Grassmannian.
double atomic_ipr(const Eigen::MatrixXd& WU,
                  const std::vector<int>& ao_atom,
                  int n_atoms) {
    Eigen::VectorXd population = Eigen::VectorXd::Zero(n_atoms);
    for (Eigen::Index mu = 0; mu < WU.rows(); ++mu) {
        population[ao_atom[static_cast<std::size_t>(mu)]] += WU.row(mu).squaredNorm();
    }
    return population.squaredNorm();
}

}  // namespace

Eigen::MatrixXd symmetric_sqrt(const Eigen::MatrixXd& S) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
    Eigen::VectorXd values = solver.eigenvalues();
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        values[i] = std::sqrt(std::max(0.0, values[i]));
    }
    return solver.eigenvectors() * values.asDiagonal()
           * solver.eigenvectors().transpose();
}

Eigen::MatrixXd occupied_block_deterministic(
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& eps,
    int nocc,
    const Eigen::MatrixXd& s_sqrt,
    const std::vector<int>& ao_atom,
    double tol) {
    const Eigen::Index n_mo = C.cols();
    if (nocc <= 0 || nocc >= static_cast<int>(n_mo)) return C.leftCols(nocc);
    // Inputs that do not support the tie-break fall back to plain aufbau, so
    // call sites without a meaningful spectrum keep their old behaviour.
    if (eps.size() != n_mo) return C.leftCols(nocc);
    if (s_sqrt.rows() != C.rows() || s_sqrt.cols() != C.rows()) return C.leftCols(nocc);
    if (static_cast<Eigen::Index>(ao_atom.size()) != C.rows()) return C.leftCols(nocc);

    if (eps[nocc] - eps[nocc - 1] > tol) return C.leftCols(nocc);

    // Widen to the whole degenerate block straddling the frontier.
    Eigen::Index lo = nocc - 1;
    while (lo > 0 && eps[lo] - eps[lo - 1] <= tol) --lo;
    Eigen::Index hi = nocc;
    while (hi + 1 < n_mo && eps[hi + 1] - eps[hi] <= tol) ++hi;
    ++hi;
    const Eigen::Index k = hi - lo;
    const Eigen::Index m = nocc - lo;
    if (k < 2 || m < 1 || m >= k) return C.leftCols(nocc);

    const int n_atoms = *std::max_element(ao_atom.begin(), ao_atom.end()) + 1;
    if (n_atoms < 2) return C.leftCols(nocc);

    // Loewdin-orthonormalised degenerate block.
    const Eigen::MatrixXd W = s_sqrt * C.middleCols(lo, k);

    // Minimise the atomic IPR over Gr(m, k) by Jacobi sweeps: every subspace
    // is reachable by a product of 2x2 rotations mixing an occupied column
    // with a virtual one, and each such rotation is scanned on a fixed grid,
    // so the search itself introduces no dependence on the incoming basis.
    Eigen::MatrixXd U = Eigen::MatrixXd::Identity(k, m);
    double best = atomic_ipr(W * U, ao_atom, n_atoms);
    constexpr int kSweeps = 8;
    constexpr int kSamples = 180;  // 1-degree resolution over a half turn
    for (int sweep = 0; sweep < kSweeps; ++sweep) {
        const double previous = best;
        for (Eigen::Index i = 0; i < m; ++i) {
            for (Eigen::Index j = m; j < k; ++j) {
                const auto plane_rotation = [&](double angle) {
                    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(k, k);
                    G(i, i) = std::cos(angle);
                    G(j, j) = std::cos(angle);
                    G(i, j) = -std::sin(angle);
                    G(j, i) = std::sin(angle);
                    return G;
                };
                double best_angle = 0.0;
                for (int sample = 1; sample < kSamples; ++sample) {
                    const double angle = M_PI * sample / kSamples;
                    const double value =
                        atomic_ipr(W * (plane_rotation(angle) * U), ao_atom, n_atoms);
                    if (value < best - 1e-14) {
                        best = value;
                        best_angle = angle;
                    }
                }
                if (best_angle != 0.0) U = plane_rotation(best_angle) * U;
            }
        }
        if (previous - best <= 1e-14) break;
    }

    Eigen::MatrixXd occupied(C.rows(), nocc);
    if (lo > 0) occupied.leftCols(lo) = C.leftCols(lo);
    occupied.rightCols(m) = C.middleCols(lo, k) * U;
    return occupied;
}

}  // namespace vibeqc
