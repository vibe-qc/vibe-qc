#include "vibeqc/gplhr.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vibeqc {

namespace {

// Two-pass modified Gram-Schmidt, including vectors accepted in this block.
// A relative cutoff rejects numerical duplicates without discarding a small
// correction merely because the requested eigenpair tolerance is larger.
bool append_orthonormal(Eigen::MatrixXd& basis, Eigen::VectorXd v, int limit)
{
    if (basis.cols() >= limit) return false;
    const double initial_norm = v.norm();
    for (int pass = 0; pass < 2; ++pass)
        for (int j = 0; j < basis.cols(); ++j)
            v -= basis.col(j) * basis.col(j).dot(v);
    const double norm = v.norm();
    if (!std::isfinite(norm) || norm <= 1e-14 * std::max(1.0, initial_norm))
        return false;
    const int col = static_cast<int>(basis.cols());
    basis.conservativeResize(Eigen::NoChange, col + 1);
    basis.col(col) = v / norm;
    return true;
}

GPLHRResult gplhr_kernel(const Eigen::MatrixXd& A,
                         const GPLHROptions& opts)
{
    const int n = static_cast<int>(A.rows());
    if (n == 0 || A.cols() != n || !A.allFinite())
        throw std::invalid_argument("gplhr: A must be a finite nonempty square matrix");
    if (opts.n_eig <= 0 || opts.n_eig > n)
        throw std::invalid_argument("gplhr: n_eig must be between 1 and A.rows()");
    if (opts.max_iter <= 0 || !std::isfinite(opts.tol) || opts.tol <= 0.0)
        throw std::invalid_argument("gplhr: max_iter and tol must be positive");
    if (!std::isfinite(opts.sigma) || !std::isfinite(opts.preshift)
        || opts.preshift < 0.0)
        throw std::invalid_argument("gplhr: sigma must be finite and preshift nonnegative");

    const int n_eig = opts.n_eig;
    const double sigma = opts.sigma;
    const double tol = opts.tol;
    // Expand guard roots too: a single unconverged harmonic pair can favour
    // a neighbouring root before the closest root is represented accurately.
    const int blk = std::min(n, std::max(n_eig, opts.block_size > 0
        ? opts.block_size : std::max(2 * n_eig, 4)));
    const int max_sub = opts.max_subspace > 0
        ? std::min(opts.max_subspace, n) : std::min(6 * blk + 10, n);
    if (max_sub < n_eig || (max_sub == n_eig && n_eig < n))
        throw std::invalid_argument("gplhr: max_subspace must leave room for corrections");
    const int n_work = std::min(blk, max_sub);
    const Eigen::VectorXd diag = A.diagonal();

    // Zuev et al., J. Comput. Chem. 36, 273-284 (2015), GPLHR steps 1, 3-5:
    // diagonal guesses, selection by |harmonic shift|, normalized vectors and
    // Rayleigh-quotient residuals. This experimental implementation retains a
    // growing/restarted harmonic-Davidson space, not the paper's full
    // {V, W, S^(1), ..., S^(m), P} locally harmonic recurrence.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return std::abs(diag[a] - sigma) < std::abs(diag[b] - sigma);
    });
    const int n_guess = std::min(n_work + 5, max_sub);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, n_guess);
    for (int j = 0; j < n_guess; ++j) V(idx[j], j) = 1.0;
    Eigen::MatrixXd AV = A * V;

    GPLHRResult result;
    for (int outer = 0; outer < opts.max_iter; ++outer) {
        const int m_sub = static_cast<int>(V.cols());
        const Eigen::MatrixXd AVs = AV - sigma * V;
        Eigen::MatrixXd H_s = V.transpose() * AVs;
        Eigen::MatrixXd G = AVs.transpose() * AVs;
        H_s = (0.5 * (H_s + H_s.transpose())).eval();
        G = (0.5 * (G + G.transpose())).eval();

        // G y = mu H_s y. H_s is indefinite for an interior shift.
        Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> solver(G, H_s);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("GPLHR harmonic Ritz extraction failed; try solver='jd'");
        const Eigen::VectorXcd mu = solver.eigenvalues();
        const Eigen::MatrixXcd Y_complex = solver.eigenvectors();
        std::vector<int> order;
        for (int i = 0; i < m_sub; ++i) {
            // A 0/0 harmonic value can represent a root exactly at sigma.
            // Dropping it would silently return a more distant eigenpair.
            if (std::isnan(mu(i).real()) || std::isnan(mu(i).imag()))
                throw std::runtime_error("GPLHR singular harmonic Ritz pencil at the target shift; try solver='jd'");
            if (std::isfinite(mu(i).real()) && std::isfinite(mu(i).imag())
                && std::abs(mu(i).imag()) <= 1e-10 * (1.0 + std::abs(mu(i).real()))
                && Y_complex.col(i).allFinite())
                order.push_back(i);
        }
        // Never reorder candidates by residual: it changes the target roots.
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::abs(mu(a).real()) < std::abs(mu(b).real());
        });
        if (static_cast<int>(order.size()) < n_eig)
            throw std::runtime_error("GPLHR has too few finite real harmonic Ritz pairs; try solver='jd'");

        const int n_candidates = static_cast<int>(order.size());
        Eigen::MatrixXd X(n, n_candidates), AX(n, n_candidates);
        Eigen::VectorXd values(n_candidates), norms(n_candidates);
        for (int i = 0; i < n_candidates; ++i) {
            Eigen::VectorXd y = Y_complex.col(order[i]).real();
            Eigen::VectorXd x = V * y;
            const double norm = x.norm();
            if (!std::isfinite(norm) || norm < 1e-14)
                throw std::runtime_error("GPLHR harmonic Ritz vector breakdown; try solver='jd'");
            X.col(i) = x / norm;
            AX.col(i) = AV * (y / norm);
            values(i) = X.col(i).dot(AX.col(i));
            norms(i) = (AX.col(i) - values(i) * X.col(i)).norm();
        }
        result.eigenvalues = values.head(n_eig);
        result.eigenvectors = X.leftCols(n_eig);
        result.n_iter = outer + 1;
        result.subspace_dim = m_sub;
        result.converged = (norms.head(n_eig).array() < tol).all();
        if (result.converged || outer + 1 == opts.max_iter) return result;

        // T approximates (A - sigma I)^(-1), as discussed by Zuev et al.
        // Preserve the sign on both sides of the shift. An absolute-value
        // denominator is not an inverse of the shifted operator.
        Eigen::VectorXd denom = diag.array() - sigma;
        for (int j = 0; j < n; ++j)
            denom(j) = std::copysign(std::max(std::abs(denom(j)),
                std::max(opts.preshift, 1e-14)), denom(j));
        Eigen::MatrixXd expanded = V;
        for (int i = 0; i < std::min(n_work, n_candidates); ++i) {
            if (norms(i) < tol) continue;
            const Eigen::VectorXd r = AX.col(i) - values(i) * X.col(i);
            append_orthonormal(expanded, r.cwiseQuotient(denom), n);
        }
        const int n_corr = static_cast<int>(expanded.cols()) - m_sub;
        if (n_corr == 0) return result;  // Stagnation, explicitly unconverged.

        if (expanded.cols() > max_sub) {
            // Retain the nearest harmonic vectors, not the eigensolver's
            // arbitrary column order, and keep this iteration's corrections.
            const int keep = std::min({n_work + 5, n_candidates,
                                       std::max(n_eig, max_sub - n_corr)});
            Eigen::MatrixXd restarted(n, 0);
            for (int j = 0; j < keep; ++j)
                append_orthonormal(restarted, X.col(j), max_sub);
            for (int j = m_sub; j < expanded.cols(); ++j)
                append_orthonormal(restarted, expanded.col(j), max_sub);
            V = std::move(restarted);
            AV = A * V;
        } else {
            V = std::move(expanded);
            AV.conservativeResize(Eigen::NoChange, V.cols());
            AV.rightCols(n_corr) = A * V.rightCols(n_corr);
        }
    }
    return result;
}

}  // anonymous namespace

GPLHRResult gplhr_solve(const Eigen::MatrixXd& A,
                        const GPLHROptions& opts) {
    return gplhr_kernel(A, opts);
}

GPLHRResult gplhr_solve_matvec(
    int n,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const Eigen::VectorXd& diag,
    const GPLHROptions& opts) {
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
        A.col(i) = matvec(Eigen::VectorXd::Unit(n, i));
    A = 0.5 * (A + A.transpose());
    return gplhr_kernel(A, opts);
}

std::string gplhr_summary(const GPLHRResult& res) {
    std::ostringstream ss;
    ss << "n_eig=" << res.eigenvalues.size()
       << " " << (res.converged ? "converged" : "NOT_CONVERGED")
       << " n_iter=" << res.n_iter
       << " m_sub=" << res.subspace_dim;
    return ss.str();
}

}  // namespace vibeqc
