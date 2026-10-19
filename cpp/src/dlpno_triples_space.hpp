#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::detail {

// Orthonormal union of the kept pair spaces, with the caller's existing
// relative singular-value cutoff and zero-rank convention.
inline Eigen::MatrixXd triples_union_space(
    const Eigen::MatrixXd& stacked, double lindep, Eigen::Index minimum_rank) {
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;
    using Index = Eigen::Index;
    const Index kq = std::min(stacked.rows(), stacked.cols());
    if (kq == 0) {
        if (minimum_rank == 0) return Mat(stacked.rows(), 0);
        throw std::runtime_error("DLPNO triples: empty pair-space union");
    }
    Eigen::HouseholderQR<Mat> qr(stacked);
    const Mat Q = qr.householderQ() * Mat::Identity(stacked.rows(), kq);
    Mat R = qr.matrixQR().topRows(kq);
    for (Index c = 0; c < R.cols(); ++c)
        for (Index row = c + 1; row < kq; ++row) R(row, c) = 0.0;
    Eigen::BDCSVD<Mat> svd(R, Eigen::ComputeThinU);
    Vec sv = svd.singularValues();
    Mat left = svd.matrixU();
    Index rank = 0;
    // Rank-deficient unions can produce invalid left singular vectors in
    // Eigen 3.4's BDCSVD despite a Success status. Some are finite and
    // orthogonal but fail to span the input (HCO and separated OH + H2).
    const double roundoff = 64.0 * std::numeric_limits<double>::epsilon()
                          * std::max(R.rows(), R.cols());
    const double rnorm = R.norm();
    const auto valid_space = [&]() {
        if (!sv.allFinite() || !left.allFinite()) return false;
        const double cutoff = lindep * std::max(sv(0), 1e-300);
        rank = 0;
        for (Index c = 0; c < sv.size(); ++c) if (sv(c) > cutoff) ++rank;
        rank = std::max(minimum_rank, rank);
        const auto kept = left.leftCols(rank);
        if ((kept.transpose() * kept - Mat::Identity(rank, rank)).norm() > roundoff)
            return false;
        // Each discarded singular value is at most cutoff, bounding the
        // Frobenius norm of the discarded range by cutoff * sqrt(count).
        const double allowed = cutoff * std::sqrt(double(sv.size() - rank))
                             + roundoff * rnorm;
        return (R - kept * (kept.transpose() * R)).norm() <= allowed;
    };
    if (svd.info() != Eigen::Success || !valid_space()) {
        Eigen::JacobiSVD<Mat> fallback(R, Eigen::ComputeThinU);
        sv = fallback.singularValues();
        left = fallback.matrixU();
        if (fallback.info() != Eigen::Success || !valid_space())
            throw std::runtime_error(
                "DLPNO triples: could not construct an orthonormal triple space");
    }
    return Q * left.leftCols(rank);
}

}  // namespace vibeqc::detail
