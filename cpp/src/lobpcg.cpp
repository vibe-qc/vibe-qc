// LOBPCG eigensolver — implementation.
//
// References
// ----------
// Knyazev, A. V. "Toward the Optimal Preconditioned Eigensolver:
// Locally Optimal Block Preconditioned Conjugate Gradient Method."
// SIAM J. Sci. Comput. 23, 517–541 (2001).
//
// The algorithm extracts the lowest n_eig eigenvalues of a real symmetric
// matrix A using a 3-term block recurrence.  Each outer iteration:
//   1. Rayleigh-Ritz on current X → rotate X, AX, P into eigenbasis
//   2. Compute residuals R = A·X − X·Λ for the first n_eig columns
//   3. Check convergence: ‖r_i‖₂ < tol  for i = 0..n_eig-1
//   4. Precondition W = T⁻¹(R) with Jacobi (diagonal) preconditioner
//   5. Build 3-block subspace Z = [X, W, P]  (all block_size columns)
//   6. Rayleigh-Ritz on Z → (Λ, U)
//   7. Update: X = Z·U_{:,:block_size}, P = Z·U_{:,block_size:2·block_size}
//
// The 3-block subspace always includes the full X and P (block_size columns
// each), plus W computed from n_eig residual columns.  All block_size columns
// of X and P are refreshed each iteration — the extra columns (n_eig+1 ..
// block_size) serve as a dilation buffer that accelerates convergence.

#include "vibeqc/lobpcg.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vibeqc {

namespace {

// ---- Orthonormalise columns via Gram-Schmidt (rank-aware) ----------------
// Uses modified Gram-Schmidt that drops columns with near-zero norm.
// Returns only the non-zero columns. This prevents HouseholderQR from
// producing spurious null-space directions when the input is rank-deficient.

Eigen::MatrixXd orthonormalise(const Eigen::MatrixXd& M, int* n_kept_out = nullptr) {
    const int n = static_cast<int>(M.rows());
    const int k_in = static_cast<int>(M.cols());
    if (k_in == 0 || n == 0) {
        if (n_kept_out) *n_kept_out = 0;
        return Eigen::MatrixXd(n, 0);
    }
    Eigen::MatrixXd Q(n, k_in);
    int kept = 0;
    for (int j = 0; j < k_in; ++j) {
        Eigen::VectorXd v = M.col(j);
        for (int i = 0; i < kept; ++i) {
            const double proj = Q.col(i).dot(v);
            v -= proj * Q.col(i);
        }
        const double vn = v.norm();
        if (vn > 1e-14) {
            Q.col(kept) = v / vn;
            ++kept;
        }
    }
    if (n_kept_out) *n_kept_out = kept;
    return Q.leftCols(kept);
}

// ---- Core LOBPCG kernel --------------------------------------------------

LOBPCGResult lobpcg_kernel(const Eigen::MatrixXd& A,
                           const Eigen::VectorXd& diag,
                           const LOBPCGOptions& opts) {
    const int n = static_cast<int>(A.rows());
    if (n == 0) {
        throw std::invalid_argument("lobpcg: A is 0×0");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("lobpcg: A must be square");
    }
    const int n_eig = opts.n_eig;
    if (n_eig < 1) {
        throw std::invalid_argument("lobpcg: n_eig must be >= 1");
    }
    if (n_eig > n) {
        throw std::invalid_argument(
            "lobpcg: n_eig (" + std::to_string(n_eig)
            + ") > matrix dimension (" + std::to_string(n) + ")");
    }

    // Block size: 2× dilation recommended by Knyazev for fast convergence.
    int block_size = (opts.block_size > 0) ? opts.block_size : 2 * n_eig;
    block_size = std::max(block_size, n_eig);
    if (block_size > n) {
        block_size = n;
    }

    const double tol = opts.tol;
    const double preshift = opts.preshift;

    // ---- Initial guess: random orthonormal vectors -----------------------
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, block_size);
    // Modified Gram-Schmidt orthonormalization.
    for (int j = 0; j < block_size; ++j) {
        for (int k = 0; k < j; ++k) {
            const double proj = X.col(k).dot(X.col(j));
            X.col(j) -= proj * X.col(k);
        }
        const double vn = X.col(j).norm();
        if (vn > 1e-14) {
            X.col(j) /= vn;
        } else {
            X.col(j) = Eigen::VectorXd::Random(n);
            for (int k = 0; k < j; ++k) {
                const double p = X.col(k).dot(X.col(j));
                X.col(j) -= p * X.col(k);
            }
            const double vn2 = X.col(j).norm();
            if (vn2 > 1e-14) X.col(j) /= vn2;
        }
    }

    // Block matvec (single BLAS gemm).
    Eigen::MatrixXd AX = A * X;  // n × block_size

    // Conjugate directions (empty on first iteration).
    Eigen::MatrixXd P;  // n × 0

    // ---- Outer iteration --------------------------------------------------
    for (int iter = 0; iter < opts.max_iter; ++iter) {
        const int n_iter = iter + 1;

        // -- Step 1: Rayleigh-Ritz on current X -----------------------------
        const Eigen::MatrixXd XtAX = X.transpose() * AX;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> rr_solver(XtAX);
        if (rr_solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "lobpcg: subspace diagonalisation failed "
                "at iteration " + std::to_string(n_iter));
        }
        const Eigen::VectorXd Lambda = rr_solver.eigenvalues();  // ascending
        const Eigen::MatrixXd U = rr_solver.eigenvectors();

        // Rotate X, AX, P into the eigenbasis of the projected problem.
        X = X * U;
        AX = AX * U;
        if (P.cols() > 0) {
            P = P * U;
        }

        // -- Step 2: compute residuals, check convergence -------------------
        int n_conv = 0;
        double max_res = 0.0;
        for (int i = 0; i < n_eig; ++i) {
            const Eigen::VectorXd r = AX.col(i) - Lambda(i) * X.col(i);
            const double rnorm = r.norm();
            if (rnorm < tol) {
                ++n_conv;
            }
            if (rnorm > max_res) max_res = rnorm;
        }

        if (opts.verbosity >= 2) {
            fprintf(stderr,
                    "  LOBPCG iter %-3d  block=%-3d  n_conv=%-2d/%-2d  "
                    "max_res=%.6g\n",
                    n_iter, block_size, n_conv, n_eig, max_res);
        }

        if (n_conv == n_eig) {
            LOBPCGResult result;
            result.eigenvalues  = Lambda.head(n_eig);
            result.eigenvectors = X.leftCols(n_eig);
            result.n_iter       = n_iter;
            result.converged    = true;
            return result;
        }

        // -- Step 3: precondition residuals → W ---------------------------
        // Use only the first n_eig residual columns.
        Eigen::MatrixXd W(n, n_eig);
        for (int i = 0; i < n_eig; ++i) {
            const Eigen::VectorXd r = AX.col(i) - Lambda(i) * X.col(i);
            const double la = Lambda(i);
            for (int j = 0; j < n; ++j) {
                const double denom = diag(j) - la + preshift;
                W(j, i) = r(j) / std::max(std::abs(denom), tol);
            }
        }

        // -- Step 4: orthogonalise W against X, then orthonormalise ---------
        W -= X * (X.transpose() * W);
        W = orthonormalise(W);

        // Drop near-zero columns from W (they carry no useful information
        // and would contaminate the 3-block subspace with spurious
        // zero-eigenvalue directions).
        int n_W_active = 0;
        for (int c = 0; c < W.cols(); ++c) {
            if (W.col(c).norm() > tol * 1e-2) {
                if (c != n_W_active)
                    W.col(n_W_active) = W.col(c);
                ++n_W_active;
            }
        }
        const int n_W = n_W_active;

        // -- Step 5: build 3-block subspace Z -------------------------------
        // Z = [X (block_size), W (n_W), P (block_size)]
        const bool has_P = (P.cols() > 0);
        const int zcols = has_P
            ? (block_size + n_W + block_size)
            : (block_size + n_W);
        Eigen::MatrixXd Z(n, zcols);

        // X columns (full block_size).
        Z.leftCols(block_size) = X;
        // W columns.
        if (n_W > 0) Z.middleCols(block_size, n_W) = W.leftCols(n_W);
        // P columns.
        if (has_P) {
            Z.rightCols(block_size) = P;
        }

        // Orthonormalise Z (rank-aware — drops near-zero columns).
        int nz = zcols;
        Z = orthonormalise(Z, &nz);

        // -- Step 6: block matvec (use only the kept columns) ---------------
        const Eigen::MatrixXd AZ = A * Z.leftCols(nz);

        // -- Step 7: Rayleigh-Ritz on the 3-block subspace ------------------
        const Eigen::MatrixXd ZtAZ = Z.leftCols(nz).transpose() * AZ;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> z_solver(ZtAZ);
        if (z_solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "lobpcg: 3-block subspace diagonalisation failed "
                "at iteration " + std::to_string(n_iter));
        }
        const Eigen::MatrixXd Uz = z_solver.eigenvectors();  // nz × nz

        // -- Step 8: extract X and P from subspace eigenvectors -------------
        // First  block_size columns → new X
        // Second block_size columns → new P (conjugate directions)
        const int take_X = std::min(block_size, nz);
        X  = Z.leftCols(nz) * Uz.leftCols(take_X);
        AX = AZ * Uz.leftCols(take_X);
        // If we got fewer than block_size columns from the Ritz step,
        // pad X with random orthonormal directions.
        if (take_X < block_size) {
            X.conservativeResize(n, block_size);
            AX.conservativeResize(n, block_size);
            for (int c = take_X; c < block_size; ++c) {
                X.col(c) = Eigen::VectorXd::Random(n);
                for (int k = 0; k < c; ++k)
                    X.col(c) -= X.col(k) * X.col(k).dot(X.col(c));
                double vn = X.col(c).norm();
                if (vn > 1e-14) X.col(c) /= vn;
                AX.col(c) = A * X.col(c);
            }
        }

        if (has_P) {
            const int p_start = block_size;
            const int p_cols  = std::min(block_size, nz - p_start);
            if (p_cols > 0) {
                P = Z.leftCols(nz) * Uz.middleCols(p_start, p_cols);
                if (p_cols < block_size) {
                    P.conservativeResize(n, block_size);
                }
            }
        } else {
            P.resize(n, block_size);
            const int p_start = block_size;
            const int p_cols  = std::min(block_size, nz - p_start);
            P.setZero();
            if (p_cols > 0) {
                P.leftCols(p_cols) = Z.leftCols(nz) * Uz.middleCols(p_start, p_cols);
            }
        }
    }  // outer iteration

    // ---- Exhausted iterations --------------------------------------------
    {
        const Eigen::MatrixXd XtAX = X.transpose() * AX;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> final_solver(XtAX);
        const Eigen::VectorXd Lambda = final_solver.eigenvalues();
        const Eigen::MatrixXd Uf = final_solver.eigenvectors();
        const Eigen::MatrixXd Xf = X * Uf;

        LOBPCGResult result;
        result.eigenvalues  = Lambda.head(n_eig);
        result.eigenvectors = Xf.leftCols(n_eig);
        result.n_iter       = opts.max_iter;
        result.converged    = false;
        return result;
    }
}

}  // anonymous namespace

// ---- Public entry points -------------------------------------------------

LOBPCGResult lobpcg_solve(const Eigen::MatrixXd& A,
                          const LOBPCGOptions& opts) {
    return lobpcg_kernel(A, A.diagonal(), opts);
}

LOBPCGResult lobpcg_solve_matvec(
    int n,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const Eigen::VectorXd& diag,
    const LOBPCGOptions& opts) {
    if (n < 1) {
        throw std::invalid_argument("lobpcg_solve_matvec: n must be >= 1");
    }
    if (static_cast<int>(diag.size()) != n) {
        throw std::invalid_argument(
            "lobpcg_solve_matvec: diag size (" +
            std::to_string(diag.size()) + ") != n (" +
            std::to_string(n) + ")");
    }
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        A.col(i) = matvec(Eigen::VectorXd::Unit(n, i));
    }
    return lobpcg_kernel(A, diag, opts);
}

std::string lobpcg_summary(const LOBPCGResult& res) {
    std::ostringstream ss;
    ss << "LOBPCG n_eig=" << res.eigenvalues.size()
       << " n_iter=" << res.n_iter
       << " converged=" << (res.converged ? "yes" : "no");
    if (res.eigenvalues.size() > 0) {
        ss << " lambda[0]=" << res.eigenvalues(0);
    }
    return ss.str();
}

}  // namespace vibeqc
