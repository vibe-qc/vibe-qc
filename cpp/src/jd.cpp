#include "vibeqc/jd.hpp"

#include <Eigen/Eigenvalues>  // subspace diagonalization only
#include <Eigen/QR>           // HouseholderQR for orthonormalization
#include <algorithm>
#include <cmath>
#include <numeric>  // std::iota
#include <sstream>
#include <stdexcept>

namespace vibeqc {

namespace {

// ---- MINRES solver for the JD correction equation --------------------------
// Solves (I - u·u^T)(A - theta·I)(I - u·u^T) t = -r
// using split-preconditioned Lanczos-Hestenes-Stiefel MINRES.
//
// Reference: Paige & Saunders, "Solution of Sparse Indefinite Systems
// of Linear Equations", SIAM J. Numer. Anal. 12, 617 (1975).
//
// Algorithm:
//   1. Compute the Davidson diagonal correction
//      t_d = M^{-1} r,  M_i = |diag_i - theta| + preshift.
//      This is exactly the first Krylov direction of the preconditioned
//      MINRES process (t_d = L^{-2} r = L^{-1}(L^{-1} r)).
//   2. Build a Lanczos basis for the split-preconditioned operator
//      B_tilde = L^{-1} (I-uu^T)(A-theta I)(I-uu^T) L^{-1}
//      with L_i = sqrt(max(|diag_i - theta|, preshift) + preshift),
//      starting from v_1 = L^{-1}(-r) / ||L^{-1}(-r)||.
//   3. If the Lanczos process exhausts after one step (the common case
//      for well-conditioned diagonal-dominant matrices), return the
//      Davidson correction directly.
//   4. Otherwise, build the tridiagonal T_k, solve
//      min ||T_k * y - beta_0 * e_1||, reconstruct t = L^{-1} V y,
//      and blend 50:50 with the Davidson correction for robustness.

Eigen::VectorXd minres_correction(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& u,
    double theta,
    const Eigen::VectorXd& r,
    const Eigen::VectorXd& diag,
    const JDOptions& opts)
{
    const int n = static_cast<int>(diag.size());
    const double preshift = opts.preshift;

    // ---- Davidson correction (first Krylov direction) ---------------------
    Eigen::VectorXd t_davidson(n);
    for (int i = 0; i < n; ++i)
        t_davidson(i) = r(i) / std::max(std::abs(diag(i) - theta) + preshift, 1e-14);
    t_davidson -= u * u.dot(t_davidson);

    double rnorm = r.norm();
    if (rnorm < 1e-14)
        return t_davidson;

    // ---- Split preconditioner (clamped to avoid blow-up near |diag-theta|≈0)
    Eigen::VectorXd L_inv(n);
    const double L_inv_max = 1.0 / std::sqrt(2.0 * preshift);
    for (int i = 0; i < n; ++i) {
        double denom = std::max(std::abs(diag(i) - theta), preshift) + preshift;
        L_inv(i) = std::min(1.0 / std::sqrt(denom), L_inv_max);
    }

    // B_tilde(v) = L^{-1}·(I-uu^T)·(A-theta I)·(I-uu^T)·L^{-1}·v
    auto apply_B_tilde = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
        Eigen::VectorXd w = L_inv.array() * v.array();
        w = w - u * u.dot(w);
        w = A * w - theta * w;
        w = w - u * u.dot(w);
        return L_inv.array() * w.array();
    };

    // ---- Lanczos recurrence on B_tilde -----------------------------------
    // Start from v_1 = L^{-1}(-r) / ||L^{-1}(-r)||  (proportional to Davidson).
    Eigen::VectorXd b_prec = -(L_inv.array() * r.array());
    double beta_0 = b_prec.norm();
    if (beta_0 < 1e-14)
        return t_davidson;

    std::vector<Eigen::VectorXd> V;
    std::vector<double> alpha, beta_list;
    Eigen::VectorXd v_k = b_prec / beta_0;
    V.push_back(v_k);
    Eigen::VectorXd v_prev = Eigen::VectorXd::Zero(n);

    const int max_inner = opts.max_inner;
    for (int k = 0; k < max_inner; ++k) {
        Eigen::VectorXd w = apply_B_tilde(v_k);
        double ak = v_k.dot(w);
        alpha.push_back(ak);
        if (k == 0)
            w -= ak * v_k;
        else
            w -= ak * v_k + beta_list[k - 1] * v_prev;
        double bk = w.norm();
        v_prev = v_k;
        if (bk < 1e-14 || k + 1 >= max_inner)
            break;
        v_k = w / bk;
        V.push_back(v_k);
        beta_list.push_back(bk);
    }

    int kk = static_cast<int>(alpha.size());
    // Single-step case: the Krylov subspace is trivial; the Davidson
    // direction already contains all useful information.
    if (kk == 0 || kk == 1)
        return t_davidson;

    // ---- Build tridiagonal T_k and solve least-squares -------------------
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(kk, kk);
    for (int i = 0; i < kk; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < kk) {
            T(i, i + 1) = beta_list[i];
            T(i + 1, i) = beta_list[i];
        }
    }
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(kk);
    rhs(0) = beta_0;
    Eigen::VectorXd y = T.colPivHouseholderQr().solve(rhs);

    // ---- Reconstruct in preconditioned space, then back-transform --------
    Eigen::VectorXd t_prec = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < kk; ++i)
        t_prec += y(i) * V[i];
    Eigen::VectorXd t_minres = L_inv.array() * t_prec.array();
    t_minres -= u * u.dot(t_minres);

    // ---- Blend 50:50 with Davidson for robustness ------------------------
    double tn = t_minres.norm();
    double dn = t_davidson.norm();
    if (tn > 1e-14 && dn > 1e-14) {
        t_minres = (t_minres / tn + t_davidson / dn);
        double blend_n = t_minres.norm();
        if (blend_n > 1e-14)
            t_minres /= blend_n;
        return t_minres;
    }
    return t_davidson;
}

// ---- Single-vector Jacobi-Davidson kernel ----------------------------------

JDResult jd_kernel(const Eigen::MatrixXd& A,
                   const JDOptions& opts)
{
    const int n = static_cast<int>(A.rows());
    if (n == 0) throw std::invalid_argument("jd_kernel: A is 0x0");
    if (A.rows() != A.cols())
        throw std::invalid_argument("jd_kernel: A must be square");
    if (opts.n_eig <= 0)
        throw std::invalid_argument("jd_kernel: n_eig must be > 0");
    if (opts.n_eig > n)
        throw std::invalid_argument("jd_kernel: n_eig > matrix dimension");

    const int n_eig = opts.n_eig;
    const double tol = opts.tol;
    const double sigma = opts.sigma;
    const int n_guess = std::min(std::max(n_eig + 5, std::min(2 * n_eig, n)), n);
    const int max_sub = (opts.max_subspace > 0)
        ? opts.max_subspace
        : std::min(std::max(8 * n_eig, n_guess + 20), n);

    const Eigen::VectorXd diag = A.diagonal();

    // ---- Build initial guess subspace (diagonal pre-selection) ------------
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (sigma == 0.0) {
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return diag[a] < diag[b]; });
    } else {
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) {
                      return std::abs(diag[a] - sigma)
                           < std::abs(diag[b] - sigma);
                  });
    }

    Eigen::MatrixXd V(n, n_guess);
    for (int j = 0; j < n_guess; ++j)
        V.col(j) = Eigen::VectorXd::Unit(n, idx[j]);

    int m_sub = n_guess;
    Eigen::MatrixXd AV(n, m_sub);
    AV = A * V.leftCols(m_sub);

    // ---- Outer JD iteration -------------------------------------------------
    for (int outer = 0; outer < opts.max_iter; ++outer) {
        // Step 1 — Rayleigh-Ritz
        Eigen::MatrixXd H = V.leftCols(m_sub).transpose() * AV.leftCols(m_sub);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("jd_kernel: subspace diag failed at iter "
                                     + std::to_string(outer + 1));
        }
        const Eigen::VectorXd& lambda = solver.eigenvalues();
        const Eigen::MatrixXd& U = solver.eigenvectors();

        Eigen::MatrixXd AX = AV.leftCols(m_sub) * U;
        Eigen::MatrixXd X  = V.leftCols(m_sub) * U;

        // Step 2 — Check convergence of the n_eig target eigenvalues
        // For extremal (sigma=0): use smallest eigenvalues.
        // For interior (sigma!=0): use eigenvalues nearest sigma.
        Eigen::VectorXd target_lambda = lambda;
        Eigen::MatrixXd target_X = X;
        Eigen::MatrixXd target_AX = AX;

        if (sigma != 0.0) {
            // Reorder Ritz pairs by distance to sigma
            std::vector<int> order(m_sub);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) {
                          return std::abs(lambda[a] - sigma)
                               < std::abs(lambda[b] - sigma);
                      });
            // Permute Ritz values, vectors, and AX
            Eigen::VectorXd lambda_sorted(m_sub);
            Eigen::MatrixXd X_sorted(n, m_sub);
            Eigen::MatrixXd AX_sorted(n, m_sub);
            for (int i = 0; i < m_sub; ++i) {
                lambda_sorted(i) = lambda(order[i]);
                X_sorted.col(i) = X.col(order[i]);
                AX_sorted.col(i) = AX.col(order[i]);
            }
            target_lambda = lambda_sorted;
            target_X = X_sorted;
            target_AX = AX_sorted;
        }

        int n_conv_this = 0;
        Eigen::VectorXd norms(n_eig);
        for (int i = 0; i < n_eig; ++i) {
            const Eigen::VectorXd ri = target_AX.col(i)
                - target_lambda(i) * target_X.col(i);
            norms(i) = ri.norm();
            if (norms(i) < tol) ++n_conv_this;
        }

        if (n_conv_this == n_eig) {
            JDResult result;
            result.eigenvalues  = target_lambda.head(n_eig);
            result.eigenvectors = target_X.leftCols(n_eig);
            result.n_iter       = outer + 1;
            result.subspace_dim = m_sub;
            result.converged    = true;
            return result;
        }

        // Step 3 — Compute correction vectors for unconverged roots
        const int n_add = std::min(n_eig, m_sub);
        Eigen::MatrixXd corrections(n, n_add);
        int n_corr = 0;

        for (int i = 0; i < n_eig && n_corr < n_add; ++i) {
            if (norms(i) < tol) continue;

            const Eigen::VectorXd ri = target_AX.col(i)
                - target_lambda(i) * target_X.col(i);
            Eigen::VectorXd u = target_X.col(i);
            double theta = target_lambda(i);

            // JD correction equation: (I-uu^T)(A-theta I)(I-uu^T) t = -r
            Eigen::VectorXd r = ri;
            r -= u * u.dot(r);

            Eigen::VectorXd delta = minres_correction(A, u, theta, r, diag, opts);

            // Orthogonalise against existing subspace V
            for (int k = 0; k < m_sub; ++k) {
                delta -= V.col(k) * V.col(k).dot(delta);
            }
            const double dnorm = delta.norm();
            if (dnorm > tol) {
                delta /= dnorm;
                corrections.col(n_corr) = delta;
                ++n_corr;
            }
        }

        // Step 4 — Expand or collapse subspace
        const int m_next = m_sub + n_corr;
        if (n_corr == 0) {
            // Stalled: add random vector
            Eigen::VectorXd rnd = Eigen::VectorXd::Random(n);
            for (int k = 0; k < m_sub; ++k)
                rnd -= V.col(k) * V.col(k).dot(rnd);
            const double rnorm = rnd.norm();
            if (rnorm > 1e-12) {
                rnd /= rnorm;
                n_corr = 1;
                corrections.resize(n, 1);
                corrections.col(0) = rnd;
            }
        }

        if (m_next >= max_sub) {
            // Collapse: keep best Ritz vectors (nearest sigma if interior)
            const int keep = std::min(n_eig + 5, n);
            m_sub = keep;
            // Use sigma-sorted Ritz vectors for interior problems
            const Eigen::MatrixXd& X_keep = (sigma != 0.0) ? target_X : X;
            V.leftCols(m_sub) = X_keep.leftCols(m_sub);
            AV.leftCols(m_sub) = A * V.leftCols(m_sub);
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

    // Exhausted iterations — recompute best Ritz approximations
    {
        Eigen::MatrixXd H_last = V.leftCols(m_sub).transpose() * AV.leftCols(m_sub);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver_last(H_last);
        const Eigen::VectorXd& lambda_last = solver_last.eigenvalues();
        const Eigen::MatrixXd& U_last = solver_last.eigenvectors();
        Eigen::MatrixXd X_last = V.leftCols(m_sub) * U_last;

        // Apply sigma-aware ordering for interior problems
        if (sigma != 0.0) {
            std::vector<int> order(m_sub);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) {
                          return std::abs(lambda_last[a] - sigma)
                               < std::abs(lambda_last[b] - sigma);
                      });
            Eigen::VectorXd lambda_sorted(m_sub);
            Eigen::MatrixXd X_sorted(n, m_sub);
            for (int i = 0; i < m_sub; ++i) {
                lambda_sorted(i) = lambda_last(order[i]);
                X_sorted.col(i) = X_last.col(order[i]);
            }
            JDResult result;
            result.eigenvalues  = lambda_sorted.head(n_eig);
            result.eigenvectors = X_sorted.leftCols(n_eig);
            result.n_iter       = opts.max_iter;
            result.subspace_dim = m_sub;
            result.converged    = false;
            return result;
        }

        JDResult result;
        result.eigenvalues  = lambda_last.head(n_eig);
        result.eigenvectors = X_last.leftCols(n_eig);
        result.n_iter       = opts.max_iter;
        result.subspace_dim = m_sub;
        result.converged    = false;
        return result;
    }
}

}  // anonymous namespace

JDResult jd_solve(const Eigen::MatrixXd& A,
                  const JDOptions& opts) {
    return jd_kernel(A, opts);
}

JDResult jd_solve_matvec(
    int n,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const Eigen::VectorXd& diag,
    const JDOptions& opts) {
    // Build explicit matrix from matvec (acceptable for the JD kernel
    // since JD needs explicit access for the correction equation).
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        A.col(i) = matvec(Eigen::VectorXd::Unit(n, i));
    }
    A = 0.5 * (A + A.transpose());
    return jd_kernel(A, opts);
}

std::string jd_summary(const JDResult& res) {
    std::ostringstream ss;
    ss << "n_eig=" << res.eigenvalues.size()
       << " " << (res.converged ? "converged" : "NOT_CONVERGED")
       << " n_iter=" << res.n_iter
       << " m_sub=" << res.subspace_dim;
    if (res.eigenvalues.size() > 0) {
        ss << " lambda_0=" << res.eigenvalues(0);
    }
    return ss.str();
}

}  // namespace vibeqc
