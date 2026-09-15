// Implementation of canonical orthogonalization — see
// cpp/include/vibeqc/linear_dependence.hpp for the docs.

#include "vibeqc/linear_dependence.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace vibeqc {

CanonicalOrthogonalizer canonical_orthogonalizer(
    const Eigen::MatrixXd& S, double threshold)
{
    if (S.rows() != S.cols()) {
        throw std::runtime_error(
            "canonical_orthogonalizer: S must be square, got "
            + std::to_string(S.rows()) + "x"
            + std::to_string(S.cols()));
    }
    const int n = static_cast<int>(S.rows());
    if (n == 0) {
        throw std::runtime_error(
            "canonical_orthogonalizer: S is empty");
    }
    // Symmetry check — catch obvious misuse. Tolerance is loose on
    // purpose: we want to pass a numerically-symmetric S even when
    // round-off has leaked asymmetry below ~1e-10.
    const double asym = (S - S.transpose()).cwiseAbs().maxCoeff();
    if (asym > 1e-8) {
        throw std::runtime_error(
            "canonical_orthogonalizer: S is not symmetric "
            "(max |S - S^T| = " + std::to_string(asym) + ")");
    }
    if (!(threshold >= 0.0)) {
        throw std::runtime_error(
            "canonical_orthogonalizer: threshold must be >= 0, got "
            + std::to_string(threshold));
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "canonical_orthogonalizer: eigendecomposition of S failed");
    }
    const Eigen::VectorXd& eigs = solver.eigenvalues();        // ascending
    const Eigen::MatrixXd& vecs = solver.eigenvectors();

    // Walk the eigenvalues (ascending) to find the drop/keep boundary.
    // Everything below ``threshold`` is dropped. eigs.size() == n.
    int first_kept = 0;
    while (first_kept < n && eigs[first_kept] < threshold) {
        ++first_kept;
    }
    const int n_kept = n - first_kept;
    const int n_dropped = first_kept;

    CanonicalOrthogonalizer out;
    out.n_kept = n_kept;
    out.n_dropped = n_dropped;
    if (n_dropped > 0) {
        out.max_dropped_eigenvalue = eigs[n_dropped - 1];
    } else {
        out.max_dropped_eigenvalue =
            std::numeric_limits<double>::quiet_NaN();
    }

    if (n_kept == 0) {
        // Pathological — the whole overlap is below threshold. Return
        // an empty X; caller is responsible for noticing and erroring.
        out.X = Eigen::MatrixXd::Zero(n, 0);
        out.min_kept_eigenvalue =
            std::numeric_limits<double>::quiet_NaN();
        return out;
    }
    out.min_kept_eigenvalue = eigs[first_kept];

    // X = V_kept · diag(1 / sqrt(lambda_kept))
    // Columns of X span the range of S and are orthonormal with
    // respect to the S-inner-product (X^T S X = I).
    Eigen::VectorXd inv_sqrt(n_kept);
    for (int i = 0; i < n_kept; ++i) {
        inv_sqrt[i] = 1.0 / std::sqrt(eigs[first_kept + i]);
    }
    out.X = vecs.rightCols(n_kept) * inv_sqrt.asDiagonal();

    return out;
}

}  // namespace vibeqc
