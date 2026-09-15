// GPLHR — Generalized Preconditioned Locally Harmonic Residual method.
//
// An iterative eigensolver for interior eigenvalues of large sparse
// Hermitian matrices, designed for quantum chemistry applications.
// Uses harmonic Ritz extraction with a shift-and-project strategy
// and block preconditioned iteration.
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

    // Block size for the subspace. 0 = auto (2 * n_eig).
    int block_size = 0;

    // Convergence threshold on residual norm.
    double tol = 1.0e-7;

    // Maximum outer iterations.
    int max_iter = 200;

    // Maximum subspace size before restart.
    int max_subspace = 0;  // 0 = auto (min(6 * n_eig + 10, n))

    // Preconditioner diagonal shift.
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
// ``opts.sigma`` using the GPLHR method.
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
