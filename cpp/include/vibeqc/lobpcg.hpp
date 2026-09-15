// LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient
// eigensolver for real symmetric matrices.
//
// Reference
// ---------
// Knyazev, A. V. "Toward the Optimal Preconditioned Eigensolver:
// Locally Optimal Block Preconditioned Conjugate Gradient Method."
// SIAM J. Sci. Comput. 23, 517–541 (2001).
// DOI: 10.1137/S1064827500366124
//
// The algorithm finds the lowest n_eig eigenvalues of a real symmetric
// matrix A by iteratively refining a block of trial vectors X using a
// 3-term recurrence: Rayleigh-Ritz on a 3-block subspace [X, W, P]
// where W is a preconditioned residual and P carries conjugate-direction
// momentum from the previous step.  It never forms the full
// eigendecomposition — O(N² · m) per outer iteration where m = block_size
// (≈ 2·n_eig), vs O(N³) for a full diagonalizer.

#pragma once

#include <Eigen/Dense>
#include <functional>

namespace vibeqc {

// ---- Options ---------------------------------------------------------------

struct LOBPCGOptions {
    // Number of eigenvalues to extract.
    int n_eig = 5;

    // Block size (number of trial vectors).  0 = auto (2 * n_eig).
    // Must be >= n_eig.  Larger block sizes typically converge in
    // fewer iterations at the cost of more work per iteration.
    int block_size = 0;

    // Maximum outer iterations.
    int max_iter = 200;

    // Convergence threshold on the residual 2-norm for each eigenpair.
    // Residual r_i = A·x_i − λ_i·x_i; converged when ‖r_i‖₂ < tol.
    double tol = 1.0e-7;

    // Verbosity: 0 = silent, 1 = summary, 2 = per-iteration.
    int verbosity = 0;

    // Preconditioner shift.  The Jacobi (diagonal) preconditioner is
    // T⁻¹ = diag(1 / (diag(A) − λ_i + preshift)).  The shift prevents
    // blow-up when λ_i is close to a diagonal element.
    double preshift = 1.0e-6;
};

// ---- Result ----------------------------------------------------------------

struct LOBPCGResult {
    // Converged eigenvalues in ascending order (size n_eig).
    Eigen::VectorXd eigenvalues;

    // Corresponding eigenvectors (n × n_eig), columnwise orthonormal.
    Eigen::MatrixXd eigenvectors;

    // Number of outer iterations performed.
    int n_iter = 0;

    // True if all n_eig eigenpairs converged to the requested tolerance.
    bool converged = false;
};

// ---- Public API ------------------------------------------------------------

// Solve A x = λ x for the lowest ``opts.n_eig`` eigenpairs of the real
// symmetric matrix ``A``.
//
// Parameters
// ----------
// A       : n × n real symmetric matrix.
// opts    : convergence + performance knobs.
//
// Returns LOBPCGResult with eigenvalues sorted ascending.
LOBPCGResult lobpcg_solve(const Eigen::MatrixXd& A,
                          const LOBPCGOptions& opts);

// Matrix-free variant: ``matvec(v)`` must return A·v.  ``diag`` is the
// diagonal of A (needed for the Jacobi preconditioner).  ``n`` is the
// matrix dimension.
//
// This variant is useful when A is never assembled explicitly, or when
// the cost of storing an n² dense matrix would be prohibitive.
LOBPCGResult lobpcg_solve_matvec(
    int n,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const Eigen::VectorXd& diag,
    const LOBPCGOptions& opts);

// Return a one-line summary string.
std::string lobpcg_summary(const LOBPCGResult& res);

}  // namespace vibeqc
