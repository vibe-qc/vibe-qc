// Davidson iterative diagonalization — symmetric block-Davidson with
// locking, subspace collapse, and diagonal preconditioning.
//
// References
// ----------
// Davidson, E. R. J. Comput. Phys. 17, 87–94 (1975).
//   Original algorithm.
// Liu, B. "The simultaneous expansion method for the iterative solution
//   of several of the lowest eigenvalues and corresponding eigenvectors
//   of large real-symmetric matrices." In NRCC Workshop on Numerical
//   Algorithms in Chemistry, LBL-8158 (1978).  Block generalisation.
// Kresse, G. & Furthmüller, J. Phys. Rev. B 54, 11169 (1996).
//   VASP-style RMM-DIIS / blocked Davidson for Kohn-Sham.
//
// The algorithm finds the lowest n_eig eigenvalues of a real symmetric
// (or Hermitian) matrix A by iteratively expanding a Krylov-like subspace,
// projecting A into it, and refining with a diagonal (Davidson) preconditioner.
// It never forms the full eigendecomposition — O(N² · m) per outer iteration
// where m is the subspace size, vs O(N³) for a full diagonalizer.

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace vibeqc {

// ---- Options ---------------------------------------------------------------

struct DavidsonOptions {
    // Number of eigenvalues to extract (typically n_occ in SCF).
    int n_eig = 0;

    // Initial subspace dimension: the initial guess is built from
    // n_guess vectors (must be >= n_eig). Default 0 = auto,
    // which uses max(n_eig + 5, 2 * n_eig).
    int n_guess = 0;

    // Subspace size budget: when the subspace exceeds this many
    // vectors, a "collapse" resets to the current n_conv converged
    // Ritz vectors + n_eig unconverged + a few extra.
    int max_subspace = 0;  // 0 = auto (min(6 * n_eig, n_basis))

    // Convergence threshold on the residual norm ‖r_i‖₂ for each
    // eigenpair.  Residual r_i = A·x_i − λ_i·x_i.
    double conv_tol = 1.0e-7;

    // Maximum outer iterations (subspace expansion steps).
    int max_iter = 200;

    // Maximum inner iterations for the lock-refinement loop
    // (a locked vector may drift slightly; a small refinement
    //  guarantees the final eigenvectors are fully converged).
    int max_refine = 5;

    // Preconditioner shift (Hartree).  The standard Davidson
    // preconditioner is (diag(A) − λ_i)^(−1).  When λ_i is close
    // to a diagonal element this blows up; ``preshift`` adds a
    // small positive constant so the denominator becomes
    // (diag(A) − λ_i + preshift).  Default 1e-6.
    double preshift = 1.0e-6;

    // Verbosity: 0 = silent, 1 = summary, 2 = per-iteration.
    int verbosity = 0;

    // Optional initial guess vectors (n_basis x n_cols). When non-empty,
    // these columns are orthonormalised and used as the starting subspace
    // instead of the diagonal-sorted unit-vector seeds. This is the key
    // SCF acceleration: pass the previous iteration's eigenvectors so
    // Davidson starts near the solution and converges in 1-3 outer iters.
    // Typical use: guess_vectors = C_prev (the MO coefficients from the
    // last SCF iteration, size n_basis x n_occ in the orthogonalised basis).
    // When empty (default), seed from the smallest diagonal entries of A.
    Eigen::MatrixXd guess_vectors;

    // Complex variant of the same (for Hermitian solver).
    Eigen::MatrixXcd guess_vectors_cplx;
};

// ---- Result ----------------------------------------------------------------

struct DavidsonResult {
    // Converged eigenvalues in ascending order (size n_eig).
    Eigen::VectorXd eigenvalues;

    // Corresponding eigenvectors (n_basis × n_eig).
    Eigen::MatrixXd eigenvectors;

    // Number of subspace expansion iterations performed.
    int n_iter = 0;

    // Subspace dimension at convergence.
    int subspace_dim = 0;

    // How many of the ``n_eig`` requested pairs met ``conv_tol`` (GitLab
    // #123).  ``converged`` is the all-or-nothing summary; this is what a
    // caller needs to use a partial spectrum, and it is the only way to
    // tell a run that found most of its roots from one that found none.
    int n_converged = 0;

    // True if all requested eigenpairs converged.
    bool converged = false;
};

// ---- Complex variant -------------------------------------------------------

struct DavidsonResultComplex {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXcd eigenvectors;  // n_basis × n_eig
    int n_iter = 0;
    int subspace_dim = 0;
    // Count of converged pairs; see DavidsonResult::n_converged (#123).
    int n_converged = 0;
    bool converged = false;
};

// ---- Public API ------------------------------------------------------------

// Solve Ax = λx for the lowest ``opts.n_eig`` eigenpairs of the real
// symmetric matrix ``A``.
//
// Parameters
// ----------
// A       : n × n real symmetric matrix.
// opts    : convergence + performance knobs.
//
// Returns DavidonResult with eigenvalues sorted ascending.
DavidsonResult davidson_solve(const Eigen::MatrixXd& A,
                              const DavidsonOptions& opts);

// Same, but accepts a matrix-vector product function ``matvec`` instead
// of an explicit matrix.  ``n_basis`` is the dimension.  The ``diag``
// vector contains the diagonal elements of the matrix being acted on
// (needed for the preconditioner).
//
// This variant is useful when the matrix is never assembled explicitly,
// or when the cost of storing an n² dense matrix would be prohibitive.
DavidsonResult davidson_solve_matvec(
    int n_basis,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const Eigen::VectorXd& diag,
    const DavidsonOptions& opts);

// Complex Hermitian variant — used for k ≠ Γ periodic Fock matrices.
//
// A is n × n Hermitian.  The returned eigenvectors are column-wise
// orthonormal (U† U = I) and the eigenvalues are real.
DavidsonResultComplex davidson_solve_hermitian(
    const Eigen::MatrixXcd& A_H,
    const DavidsonOptions& opts);

// Complex Hermitian matvec variant.
DavidsonResultComplex davidson_solve_hermitian_matvec(
    int n_basis,
    const std::function<Eigen::VectorXcd(const Eigen::VectorXcd&)>& matvec,
    const Eigen::VectorXd& diag,
    const DavidsonOptions& opts);

// Return a short description string ("n_eig=… tol=… converged=…").
std::string davidson_summary(const DavidsonResult& res);
std::string davidson_summary(const DavidsonResultComplex& res);

}  // namespace vibeqc
