#include "vibeqc/gplhr.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace vibeqc {

namespace {

GPLHRResult gplhr_kernel(const Eigen::MatrixXd& A,
                         const GPLHROptions& opts)
{
    const int n = static_cast<int>(A.rows());
    if (n == 0) throw std::invalid_argument("gplhr_kernel: A is 0x0");
    if (opts.n_eig <= 0)
        throw std::invalid_argument("gplhr_kernel: n_eig must be > 0");

    const int n_eig = opts.n_eig;
    const double sigma = opts.sigma;
    const double tol = opts.tol;
    const double preshift = opts.preshift;
    const int blk = (opts.block_size > 0)
        ? opts.block_size
        : std::max(2 * n_eig, 4);
    const int max_sub = (opts.max_subspace > 0)
        ? opts.max_subspace
        : std::min(2 * n_eig + 10, n);

    const Eigen::VectorXd diag = A.diagonal();

    // ---- Build initial subspace from diagonal pre-selection ------------
    // Pick unit vectors whose diagonal entries are closest to sigma.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) {
                  return std::abs(diag[a] - sigma) < std::abs(diag[b] - sigma);
              });

    const int n_guess = std::min(n_eig + 5, n);
    Eigen::MatrixXd V(n, n_guess);
    for (int j = 0; j < n_guess; ++j)
        V.col(j) = Eigen::VectorXd::Unit(n, idx[j]);

    int m_sub = n_guess;
    Eigen::MatrixXd AV(n, m_sub);
    AV = A * V.leftCols(m_sub);

    // Shifted operator: A_s = A - sigma*I
    // We store AV and use (AV - sigma*V) implicitly.

    for (int outer = 0; outer < opts.max_iter; ++outer) {
        // ---- Step 1: Build projected matrices ------------------------------
        // AVs = (A - sigma*I) * V
        Eigen::MatrixXd AVs = AV.leftCols(m_sub)
            - sigma * V.leftCols(m_sub);

        // H_s = V^T (A - sigma*I) V  (shifted projected matrix)
        Eigen::MatrixXd H_s = V.leftCols(m_sub).transpose() * AVs;

        // G = V^T (A - sigma*I)^2 V = AVs^T * AVs
        //   = V^T (A-sI)^T (A-sI) V  (equals V^T (A-sI)^2 V since A is symmetric)
        Eigen::MatrixXd G = AVs.transpose() * AVs;

        // Ensure symmetry (may drift slightly due to floating point)
        H_s = 0.5 * (H_s + H_s.transpose());
        G = 0.5 * (G + G.transpose());

        // ---- Step 2: Harmonic Ritz extraction ------------------------------
        // Solve the generalized eigenvalue problem:
        //   G * y = mu * H_s * y
        // where mu are the harmonic Ritz values.
        //
        // H_s = V^T (A - sigma I) V is indefinite when sigma lies inside
        // the spectrum of A.  Use GeneralizedEigenSolver (not SelfAdjoint)
        // to handle the general case correctly.

        Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> solver(
            G, H_s);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("gplhr: harmonic Ritz generalized EVP failed");
        }

        // Extract real parts (complex eigenvalues come in conjugate pairs
        // for symmetric-real pencils; the real ones are what we need).
        Eigen::VectorXd mu(m_sub);
        for (int i = 0; i < m_sub; ++i)
            mu(i) = std::real(solver.eigenvalues()(i));
        Eigen::MatrixXcd Y_cplx = solver.eigenvectors();
        // Take real part of eigenvectors
        Eigen::MatrixXd Y = Y_cplx.real();

        // Harmonic Ritz vectors: X = V * Y
        Eigen::MatrixXd X = V.leftCols(m_sub) * Y;

        // ---- Step 3: Select n_eig eigenvalues (two-stage filter) ---------
        // Stage 1: sort by |mu| = distance to sigma (harmonic Ritz property).
        // Stage 2: within the top candidates, re-sort by residual norm.
        // This avoids both the spurious-near-zero-mu problem and the
        // global-selection problem (picking extremal eigenvalues).

        // Compute residuals for ALL harmonic Ritz pairs
        Eigen::VectorXd all_norms(m_sub);
        for (int i = 0; i < m_sub; ++i) {
            double lambda_i = sigma + mu(i);
            Eigen::VectorXd Ax = AV.leftCols(m_sub) * Y.col(i);
            Eigen::VectorXd x  = X.col(i);
            Eigen::VectorXd r  = Ax - lambda_i * x;
            all_norms(i) = r.norm();
        }

        // Stage 1: sort by |mu|, keep top candidates (2x n_eig as pool)
        const int pool_size = std::min(2 * n_eig, m_sub);
        std::vector<int> order_pool(m_sub);
        std::iota(order_pool.begin(), order_pool.end(), 0);
        std::sort(order_pool.begin(), order_pool.end(),
                  [&](int a, int b) {
                      return std::abs(mu[a]) < std::abs(mu[b]);
                  });

        // Stage 2: within the pool, re-sort by residual norm
        std::vector<int> order(pool_size);
        for (int i = 0; i < pool_size; ++i)
            order[i] = order_pool[i];
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) {
                      return all_norms(a) < all_norms(b);
                  });

        // Take the best n_eig
        order.resize(n_eig);

        // ---- Step 4: Check convergence ------------------------------------
        int n_conv = 0;
        Eigen::VectorXd norms(n_eig);
        for (int i = 0; i < n_eig; ++i) {
            norms(i) = all_norms(order[i]);
            if (norms(i) < tol) ++n_conv;
        }

        if (n_conv == n_eig) {
            GPLHRResult result;
            result.eigenvalues.resize(n_eig);
            result.eigenvectors.resize(n, n_eig);
            for (int i = 0; i < n_eig; ++i) {
                int idx_i = order[i];
                result.eigenvalues(i) = sigma + mu(idx_i);
                result.eigenvectors.col(i) = X.col(idx_i);
            }
            result.n_iter = outer + 1;
            result.subspace_dim = m_sub;
            result.converged = true;
            return result;
        }

        // ---- Step 5: Expand subspace with preconditioned corrections ------
        int n_corr = 0;
        Eigen::MatrixXd corrections(n, n_eig);
        for (int i = 0; i < n_eig; ++i) {
            if (norms(i) < tol) continue;
            int idx_i = order[i];
            double lambda_i = sigma + mu(idx_i);
            Eigen::VectorXd x = X.col(idx_i);
            Eigen::VectorXd Ax = AV.leftCols(m_sub) * Y.col(idx_i);
            Eigen::VectorXd r = Ax - lambda_i * x;

            // GPLHR preconditioner: use sigma as the shift, not lambda_i.
            // The shift-and-project strategy targets (A - sigma I), so the
            // diagonal preconditioner should approximate (diag - sigma)^{-1}.
            Eigen::VectorXd t(n);
            for (int j = 0; j < n; ++j) {
                t(j) = r(j) / std::max(std::abs(diag(j) - sigma + preshift), 1e-14);
            }
            // Orthogonalize against subspace
            for (int k = 0; k < m_sub; ++k) {
                t -= V.col(k) * V.col(k).dot(t);
            }
            double tn = t.norm();
            if (tn > tol) {
                t /= tn;
                corrections.col(n_corr) = t;
                ++n_corr;
            }
        }

        // ---- Step 6: Expand or restart subspace ---------------------------
        const int m_next = m_sub + n_corr;
        if (n_corr == 0) {
            // Stalled — add random vector
            Eigen::VectorXd rnd = Eigen::VectorXd::Random(n);
            for (int k = 0; k < m_sub; ++k)
                rnd -= V.col(k) * V.col(k).dot(rnd);
            double rn = rnd.norm();
            if (rn > 1e-12) {
                rnd /= rn;
                n_corr = 1;
                corrections.resize(n, 1);
                corrections.col(0) = rnd;
            }
        }

        if (m_next >= max_sub) {
            // Restart: keep best harmonic Ritz vectors
            const int keep = std::min(n_eig + 5, n);
            m_sub = keep;
            V.leftCols(m_sub) = X.leftCols(m_sub);
            // Re-orthonormalise
            for (int j = 0; j < m_sub; ++j) {
                for (int k = 0; k < j; ++k)
                    V.col(j) -= V.col(k) * V.col(k).dot(V.col(j));
                double vn = V.col(j).norm();
                if (vn > 1e-14) V.col(j) /= vn;
            }
            AV.leftCols(m_sub) = A * V.leftCols(m_sub);
        } else if (n_corr > 0) {
            int old_m = m_sub;
            m_sub += n_corr;
            V.conservativeResize(Eigen::NoChange, m_sub);
            AV.conservativeResize(Eigen::NoChange, m_sub);
            V.middleCols(old_m, n_corr) = corrections.leftCols(n_corr);
            AV.middleCols(old_m, n_corr) = A * V.middleCols(old_m, n_corr);
        }
    }

    // Exhausted — recompute best harmonic Ritz approximations
    {
        Eigen::MatrixXd AVs = AV.leftCols(m_sub) - sigma * V.leftCols(m_sub);
        Eigen::MatrixXd H_s = V.leftCols(m_sub).transpose() * AVs;
        Eigen::MatrixXd G = AVs.transpose() * AVs;
        H_s = 0.5 * (H_s + H_s.transpose());
        G = 0.5 * (G + G.transpose());

        Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> solver(G, H_s);
        if (solver.info() == Eigen::Success) {
            Eigen::VectorXd mu(m_sub);
            for (int i = 0; i < m_sub; ++i)
                mu(i) = std::real(solver.eigenvalues()(i));
            Eigen::MatrixXd Y = solver.eigenvectors().real();
            Eigen::MatrixXd X = V.leftCols(m_sub) * Y;

            // Two-stage selection: |mu| pool, then residual norm
            Eigen::VectorXd all_norms(m_sub);
            for (int i = 0; i < m_sub; ++i) {
                double lambda_i = sigma + mu(i);
                Eigen::VectorXd Ax = AV.leftCols(m_sub) * Y.col(i);
                Eigen::VectorXd x  = X.col(i);
                Eigen::VectorXd r  = Ax - lambda_i * x;
                all_norms(i) = r.norm();
            }
            const int pool_size = std::min(2 * n_eig, m_sub);
            std::vector<int> order_pool(m_sub);
            std::iota(order_pool.begin(), order_pool.end(), 0);
            std::sort(order_pool.begin(), order_pool.end(),
                      [&](int a, int b) {
                          return std::abs(mu[a]) < std::abs(mu[b]);
                      });
            std::vector<int> order(pool_size);
            for (int i = 0; i < pool_size; ++i)
                order[i] = order_pool[i];
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) {
                          return all_norms(a) < all_norms(b);
                      });
            order.resize(n_eig);

            GPLHRResult result;
            result.eigenvalues.resize(n_eig);
            result.eigenvectors.resize(n, n_eig);
            for (int i = 0; i < n_eig; ++i) {
                result.eigenvalues(i) = sigma + mu(order[i]);
                result.eigenvectors.col(i) = X.col(order[i]);
            }
            result.n_iter = opts.max_iter;
            result.subspace_dim = m_sub;
            result.converged = false;
            return result;
        }
    }

    GPLHRResult result;
    result.n_iter = opts.max_iter;
    result.subspace_dim = m_sub;
    result.converged = false;
    result.eigenvalues.resize(n_eig);
    result.eigenvectors.resize(n, n_eig);
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
