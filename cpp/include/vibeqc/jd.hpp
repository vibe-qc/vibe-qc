// Jacobi-Davidson iterative diagonalization.
//
// Unlike the standard Davidson algorithm (which uses a diagonal
// preconditioner on the residual directly), Jacobi-Davidson solves
// a correction equation approximately via an inner iterative solver
// (MINRES or GMRES). This makes JD much more robust for:
//   - Interior eigenvalues (with harmonic Ritz)
//   - Nearly-degenerate eigenvalues
//   - Ill-conditioned matrices where the Davidson diagonal
//     preconditioner fails
//
// References
// ----------
// Sleijpen, G. L. G. & Van der Vorst, H. A. "A Jacobi-Davidson
//   iteration method for linear eigenvalue problems."
//   SIAM J. Matrix Anal. Appl. 17, 401-425 (1996).
//   DOI: 10.1137/S0895479894270427
// Fokkema, D. R., Sleijpen, G. L. G. & Van der Vorst, H. A.
//   "Jacobi-Davidson style QR and QZ algorithms for the reduction
//   of matrix pencils." SIAM J. Sci. Comput. 20, 94-125 (1998).
//   Block variant.

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <string>
#include <vector>

namespace vibeqc {

struct JDOptions {
    // Number of eigenvalues to extract.  Must be > 0.
    int n_eig = 1;

    // Convergence threshold on residual norm ||r_i||_2.
    double tol = 1.0e-7;

    // Maximum outer (JD subspace expansion) iterations.
    int max_iter = 200;

    // Maximum inner (MINRES correction-equation) iterations.
    int max_inner = 30;

    // Inner solver tolerance (relative, on the correction equation).
    double inner_tol = 0.1;

    // Subspace size budget before a collapse reset.
    int max_subspace = 0;  // 0 = auto (min(8 * n_eig, n_basis))

    // Target shift for interior eigenvalues.  When sigma is set,
    // JD targets eigenvalues nearest to sigma instead of the
    // algebraically smallest.  Default 0.0 = no shift (extremal).
    double sigma = 0.0;

    // Preconditioner shift for the correction equation.
    double preshift = 1.0e-6;

    // Verbosity: 0 = silent, 1 = summary, 2 = per-iteration.
    int verbosity = 0;
};

struct JDResult {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
    int n_iter = 0;
    int subspace_dim = 0;
    bool converged = false;
};

// ---- Public API ------------------------------------------------------------

// Solve Ax = lambda x for the lowest ``opts.n_eig`` eigenvalues of
// the real symmetric matrix ``A`` using the Jacobi-Davidson method
// with MINRES correction-equation solver.
JDResult jd_solve(const Eigen::MatrixXd& A,
                  const JDOptions& opts);

// Matrix-free variant.  ``matvec(v)`` returns A·v.  ``diag`` is the
// diagonal of A (needed for the diagonal preconditioner in the
// correction equation).
JDResult jd_solve_matvec(
    int n,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const Eigen::VectorXd& diag,
    const JDOptions& opts);

std::string jd_summary(const JDResult& res);

}  // namespace vibeqc
