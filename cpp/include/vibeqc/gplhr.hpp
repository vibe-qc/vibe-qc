// GPLHR — Generalized Preconditioned Locally Harmonic Residual method.
//
// An iterative eigensolver for interior eigenvalues of large sparse
// Hermitian matrices, designed for quantum chemistry applications.
// Experimental GPLHR-derived solver using harmonic Ritz extraction and
// signed shifted-diagonal block preconditioning. The growing/restarted
// search space is a harmonic-Davidson variant, not the full locally
// harmonic recurrence with residual-like and previous-direction blocks.
//
// Reference
// ---------
// Zuev, D., Vecharynski, E., Yang, C., Orms, N. & Krylov, A. I.
//   "New algorithms for iterative matrix-free eigensolvers in
//   quantum chemistry." J. Comput. Chem. 36, 273-284 (2015).
//   DOI: 10.1002/jcc.23800
//
// The algorithm finds eigenvalues nearest to a target shift sigma
// by iteratively expanding a subspace with harmonic Ritz extraction.

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <vector>

namespace vibeqc {

struct GPLHROptions {
    // Target shift sigma — eigenvalues nearest sigma are extracted.
    double sigma = 0.0;

    // Number of eigenvalues to extract.
    int n_eig = 1;

    // Number of candidates expanded, including guard roots.
    // 0 = auto (max(2 * n_eig, 4)), capped at the matrix dimension.
    int block_size = 0;

    // Convergence threshold on residual norm.
    double tol = 1.0e-7;

    // Maximum outer iterations.
    int max_iter = 200;

    // Maximum subspace size before restart.
    int max_subspace = 0;  // 0 = auto (min(6 * block_size + 10, n))

    // Minimum magnitude of the signed preconditioner denominator.
    double preshift = 1.0e-6;

    // Verbosity.
    int verbosity = 0;
};

struct GPLHRResult {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
    int n_iter = 0;
    int subspace_dim = 0;
    bool converged = false;
};

// Solve Ax = lambda x for the ``opts.n_eig`` eigenvalues nearest to
// ``opts.sigma`` using the experimental GPLHR-derived method. An unconverged
// result contains the last tested approximations, never uninitialized data.
GPLHRResult gplhr_solve(const Eigen::MatrixXd& A,
                        const GPLHROptions& opts);

// Matrix-free variant.
GPLHRResult gplhr_solve_matvec(
    int n,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const Eigen::VectorXd& diag,
    const GPLHROptions& opts);

std::string gplhr_summary(const GPLHRResult& res);

}  // namespace vibeqc
