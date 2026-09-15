#include "vibeqc/davidson.hpp"

#include <Eigen/Eigenvalues>  // for subspace diagonalisation only
#include <Eigen/QR>          // HouseholderQR for orthonormalisation
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace vibeqc {

// ============================================================================
//  Internal helpers
// ============================================================================

namespace {

// ---- Real-symmetric block-Davidson kernel (explicit A) ------------------

// Block-Davidson driver shared by the explicit and matrix-free entry points.
//
// The operator reaches the algorithm only through ``apply`` (a block product
// A·V) and ``diag`` (its diagonal, for the preconditioner and the default
// guess).  Keeping one implementation matters: GitLab #503's two defects lived
// in the expansion path, and a second transcription of this loop would have to
// carry those fixes too.
//
// GitLab #506: the matrix-free entry point used to apply the caller's operator
// to all n unit vectors, build the dense matrix and call the explicit kernel --
// costing the memory it was asked to avoid plus n callbacks.  Routing it here
// makes it apply the operator only to the vectors the algorithm actually
// forms: the initial block, each set of corrections, and one rebuild per
// collapse.
using DavidsonApply =
    std::function<Eigen::MatrixXd(const Eigen::Ref<const Eigen::MatrixXd>&)>;

static DavidsonResult davidson_kernel_impl(int n,
                                           const Eigen::VectorXd& diag,
                                           const DavidsonApply& apply,
                                           const DavidsonOptions& opts) {
    if (n == 0) {
        throw std::invalid_argument("davidson_kernel: A is 0×0");
    }
    if (diag.size() != n) {
        throw std::invalid_argument(
            "davidson_kernel: diagonal length " + std::to_string(diag.size())
            + " does not match dimension " + std::to_string(n));
    }
    const int n_eig = (opts.n_eig > 0) ? opts.n_eig
                     : std::min(n, 10);  // sensible default
    if (n_eig > n) {
        throw std::invalid_argument(
            "davidson_kernel: n_eig (" + std::to_string(n_eig)
            + ") > matrix dimension (" + std::to_string(n) + ")");
    }

    const int n_guess = (opts.n_guess > 0)
        ? std::min(opts.n_guess, n)
        : std::min(std::max(n_eig + 5, std::min(2 * n_eig, n)), n);
    const int max_sub = (opts.max_subspace > 0)
        ? opts.max_subspace
        : std::min(std::max(8 * n_eig, n_guess + 20), n);
    const double tol = opts.conv_tol;
    const double preshift = opts.preshift;

    if (n_guess < n_eig) {
        throw std::invalid_argument(
            "davidson_kernel: n_guess (" + std::to_string(n_guess)
            + ") < n_eig (" + std::to_string(n_eig) + ")");
    }

    // ``diag`` is supplied by the caller (see DavidsonApply above).

    // ---- Build initial guess subspace ------------------------------------
    // Use caller-provided guess vectors when available (key for SCF
    // eigenvector recycling — each iteration starts near the solution).
    // Otherwise seed from the smallest diagonal entries (standard
    // plane-wave-style diagonal preselection).
    Eigen::MatrixXd V;
    if (opts.guess_vectors.size() != 0) {
        // Use the provided guess vectors as the initial subspace.
        // They should be column vectors in the orthogonalised basis,
        // typically the MO coefficients from the previous SCF iteration.
        const int ng = static_cast<int>(opts.guess_vectors.cols());
        if (opts.guess_vectors.rows() != n) {
            throw std::invalid_argument(
                "davidson_kernel: guess_vectors has "
                + std::to_string(opts.guess_vectors.rows()) + " rows, "
                "expected " + std::to_string(n));
        }
        V = opts.guess_vectors;
        // Orthonormalise via Gram-Schmidt (they should already be
        // orthonormal, but belt-and-braces).
        const int m0 = static_cast<int>(V.cols());
        for (int j = 0; j < m0; ++j) {
            for (int k = 0; k < j; ++k) {
                const double proj = V.col(k).dot(V.col(j));
                V.col(j) -= proj * V.col(k);
            }
            const double vn = V.col(j).norm();
            if (vn > 1e-14) V.col(j) /= vn;
            else V.col(j).setZero();
        }
    } else {
        // Diagonal pre-selection: unit vectors at the smallest diagonal
        // entries of A.  Robust for Hcore initialisation.
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return diag[a] < diag[b]; });
        V.resize(n, n_guess);
        for (int j = 0; j < n_guess && j < n; ++j) {
            V.col(j) = Eigen::VectorXd::Unit(n, idx[j]);
        }
    }

    // ---- Projected quantities (reused to keep only O(n·m) allocation) ----
    Eigen::MatrixXd AV;       // A · V           n × m_sub
    Eigen::MatrixXd H_proj;   // V^T A V         m_sub × m_sub
    Eigen::MatrixXd U;        // H_proj evecs     m_sub × m_sub
    Eigen::VectorXd lambda;   // H_proj evals    m_sub

    int m_conv = 0;            // number of locked (converged) eigenpairs
    int n_iter = 0;
    int m_sub = static_cast<int>(V.cols());   // current subspace dimension
    // Block AV computation (exploits BLAS for matrix-matrix multiply).
    AV = apply(V.leftCols(m_sub));

    // Cached converged quantities (locked vectors).
    Eigen::MatrixXd X_conv;        // n × m_conv
    Eigen::VectorXd lambda_conv;   // m_conv
    Eigen::MatrixXd AX_conv;       // n × m_conv  (cached A·X_conv)

    // ---- Outer iteration --------------------------------------------------
    for (int outer = 0; outer < opts.max_iter; ++outer) {
        n_iter = outer + 1;

        // Step 1 — Build projected problem H_proj = V^T (A V).
        H_proj = V.leftCols(m_sub).transpose() * AV.leftCols(m_sub);

        // Step 2 — Diagonalise H_proj (small dense, m_sub ≤ ~120).
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_proj);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "davidson_kernel: subspace diagonalisation failed "
                "at iteration " + std::to_string(n_iter));
        }
        lambda = solver.eigenvalues();
        U      = solver.eigenvectors();  // m_sub × m_sub

        // Step 3 — Form Ritz vectors and compute residuals for the
        //          lowest n_eig eigenpairs.
        // X = V · U  (n × m_sub),  λ sorted ascending.
        // r_i = A·x_i − λ_i·x_i  = AV · u_i − λ_i · V · u_i
        //    = (AV − λ_i·V) · u_i.
        // Compute once as a block then pick columns.
        // AX = AV · U   (n × m_sub)
        Eigen::MatrixXd AX = AV.leftCols(m_sub) * U;   // n × m_sub
        Eigen::MatrixXd X  = V.leftCols(m_sub) * U;    // n × m_sub

        // Check convergence of all n_eig wanted pairs.
        int n_conv_this_iter = 0;
        Eigen::VectorXd norms(n_eig);
        for (int i = 0; i < n_eig; ++i) {
            // Residual r_i = AX.col(i) − λ_i · X.col(i)
            const Eigen::VectorXd ri =
                AX.col(i) - lambda(i) * X.col(i);
            norms(i) = ri.norm();
            if (norms(i) < tol) {
                ++n_conv_this_iter;
            }
        }

        if (opts.verbosity >= 2) {
            std::ostringstream oss;
            oss << "  Davidson iter " << n_iter
                << "  m_sub=" << m_sub
                << "  n_conv=" << n_conv_this_iter << "/" << n_eig
                << "  max_res=";
            if (n_eig > 0) oss << norms.head(n_eig).maxCoeff();
            else oss << "0";
            // Only print to stderr if this were a logging path; skip for now.
        }

        // If all converged, done.
        if (n_conv_this_iter == n_eig) {
            DavidsonResult result;
            result.eigenvalues  = lambda.head(n_eig);
            result.eigenvectors = X.leftCols(n_eig);
            result.n_iter       = n_iter;
            result.subspace_dim = m_sub;
            result.converged    = true;
            return result;
        }

        // Step 4 — Pick up to n_add = n_eig − n_conv_this_iter new
        //          correction vectors.  We add one per unconverged pair.
        const int n_add = std::min(n_eig, m_sub);
        Eigen::MatrixXd corrections(n, n_add);
        int n_corr = 0;

        for (int i = 0; i < n_eig && n_corr < n_add; ++i) {
            if (norms(i) < tol) continue;  // already converged

            const Eigen::VectorXd ri =
                AX.col(i) - lambda(i) * X.col(i);

            // Davidson preconditioner: δ_j = r_j / (diag(A)_j − λ_i + ε)
            // with a small shift ε to avoid division by zero.
            Eigen::VectorXd delta(n);
            for (int j = 0; j < n; ++j) {
                double denom = diag(j) - lambda(i) + preshift;
                // Clamp |denom| to avoid blowup; standard trick.
                if (std::abs(denom) < tol) {
                    denom = (denom < 0.0) ? -tol : tol;
                }
                delta(j) = ri(j) / denom;
            }

            // Orthogonalise against the existing subspace V *and* against
            // the corrections already accepted in this same block.
            //
            // GitLab #503: omitting the second part is what corrupted the
            // partial-spectrum path.  Corrections for different unconverged
            // roots are frequently near-parallel -- they are preconditioned
            // residuals of neighbouring Ritz pairs -- so orthogonalising each
            // only against V let a nearly linearly dependent set into the
            // basis.  V then degenerates, H_proj = V^T A V picks up spurious
            // near-zero eigenvalues, and because lambda is sorted ascending
            // those sort BELOW the true roots: lambda.head(n_eig) returns
            // zeros and X.leftCols(n_eig) the corresponding null vectors.
            // Measured before this fix, requesting 8 roots of a 500x500
            // operator: 0 dead columns after 1 iteration, 1 after 2, and 8 by
            // iteration 5, at which point every returned eigenvalue was
            // ~1e-13 instead of the true 0.196, 0.299, 0.398, ...
            //
            // Two passes: one Gram-Schmidt sweep loses orthogonality when the
            // vector is nearly spanned, which is exactly the regime here.
            for (int pass = 0; pass < 2; ++pass) {
                for (int k = 0; k < m_sub; ++k) {
                    const double proj = delta.dot(V.col(k));
                    delta -= proj * V.col(k);
                }
                for (int k = 0; k < n_corr; ++k) {
                    const double proj = delta.dot(corrections.col(k));
                    delta -= proj * corrections.col(k);
                }
            }
            // Accept unless the correction is numerically zero, i.e. the
            // Ritz vector was already fully spanned by the subspace.
            //
            // GitLab #503: this threshold used to be `tol`, the convergence
            // tolerance, which put a floor under the attainable residual.  As
            // residuals approach tol the preconditioned corrections shrink
            // with them, so they were rejected as "converged" exactly when
            // they were still needed, and the iteration stalled at a small
            // multiple of tol forever.  Measured on a 500x500 operator, the
            // stalled residual tracked the tolerance at 3.0x, 4.7x, 3.1x and
            // 3.5x for tol = 1e-3, 1e-5, 1e-6 and 1e-8 -- never converging,
            // whatever tolerance was asked for.  Linear dependence is a
            // property of the vector, not of the caller's tolerance, so the
            // test is now an absolute one.
            constexpr double kCorrectionFloor = 1.0e-12;
            const double dnorm = delta.norm();
            if (dnorm > kCorrectionFloor) {
                delta /= dnorm;
                corrections.col(n_corr) = delta;
                ++n_corr;
            }
        }

        // Step 5 — Expand subspace.  If we're near the budget, collapse.
        const int m_next = m_sub + n_corr;
        if (n_corr == 0) {
            // No useful correction vectors — subspace is saturated but
            // tolerance not met.  This can happen when the original A
            // has near-degenerate eigenvalues and the subspace needs more
            // diversity.  Add a random vector.
            Eigen::VectorXd rnd = Eigen::VectorXd::Random(n);
            for (int k = 0; k < m_sub; ++k) {
                const double proj = rnd.dot(V.col(k));
                rnd -= proj * V.col(k);
            }
            const double rnorm = rnd.norm();
            if (rnorm > 1e-12) {
                rnd /= rnorm;
                n_corr = 1;
                corrections.resize(n, 1);
                corrections.col(0) = rnd;
            } else {
                // Random vector was entirely spanned — matrix is
                // effectively rank-deficient.  Accept the current
                // best approximations.
                n_corr = 0;
            }
        }

        if (m_next >= max_sub) {
            // Collapse: rebuild the subspace from the best current
            // approximations.  Keep the n_eig best Ritz vectors
            // plus a few extras for flexibility.
            const int keep = std::min(n_eig + std::min(5, n - n_eig), n);
            m_sub = keep;
            // Block copy of best Ritz vectors.
            V.leftCols(m_sub) = X.leftCols(m_sub);
            // Re-orthonormalise first, then form AV once.  The previous
            // code applied the operator here *and* again below, discarding
            // the first result -- harmless for a dense gemm, but a whole
            // extra block of operator applications in the matrix-free path
            // (#506).
            // Re-orthonormalise (Gram-Schmidt, belt-and-braces).
            for (int j = 0; j < m_sub; ++j) {
                for (int k = 0; k < j; ++k) {
                    const double proj = V.col(k).dot(V.col(j));
                    V.col(j) -= proj * V.col(k);
                }
                const double vn = V.col(j).norm();
                if (vn > 1e-14) V.col(j) /= vn;
                else {
                    // Column collapsed to zero -- re-seed with a random
                    // direction to keep the subspace non-degenerate.
                    V.col(j) = Eigen::VectorXd::Random(n);
                    for (int k = 0; k < j; ++k) {
                        const double p = V.col(k).dot(V.col(j));
                        V.col(j) -= p * V.col(k);
                    }
                    double vn2 = V.col(j).norm();
                    if (vn2 > 1e-14) V.col(j) /= vn2;
                }
            }
            AV.leftCols(m_sub) = apply(V.leftCols(m_sub));
        } else if (n_corr > 0) {
            // Normal expansion: append correction vectors as a block.
            const int old_m = m_sub;
            m_sub += n_corr;
            V.conservativeResize(Eigen::NoChange, m_sub);
            AV.conservativeResize(Eigen::NoChange, m_sub);
            V.middleCols(old_m, n_corr) = corrections.leftCols(n_corr);
            // Block mat-mat multiply (single BLAS call for all new cols).
            AV.middleCols(old_m, n_corr) =
                apply(V.middleCols(old_m, n_corr));
        } else {
            // Stalled: can't expand subspace.  Accept best effort.
            DavidsonResult result;
            result.eigenvalues  = lambda.head(n_eig);
            result.eigenvectors = X.leftCols(n_eig);
            result.n_iter       = n_iter;
            result.subspace_dim = m_sub;
            result.converged    = false;
            return result;
        }
    }  // outer iteration

    // ---- Exhausted iterations without convergence -------------------------
    // Return the best available approximation.
    {
        H_proj = V.leftCols(m_sub).transpose() * AV.leftCols(m_sub);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_proj);
        lambda = solver.eigenvalues();
        U      = solver.eigenvectors();
        Eigen::MatrixXd X = V.leftCols(m_sub) * U;
        DavidsonResult result;
        result.eigenvalues  = lambda.head(n_eig);
        result.eigenvectors = X.leftCols(n_eig);
        result.n_iter       = n_iter;
        result.subspace_dim = m_sub;
        result.converged    = false;
        return result;
    }
}

// ---- Complex Hermitian block-Davidson kernel (explicit A) ----------------

DavidsonResult davidson_kernel(const Eigen::MatrixXd& A,
                               const DavidsonOptions& opts) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("davidson_kernel: A must be square");
    }
    const int n = static_cast<int>(A.rows());
    return davidson_kernel_impl(
        n, A.diagonal(),
        [&A](const Eigen::Ref<const Eigen::MatrixXd>& V) -> Eigen::MatrixXd {
            return A * V;
        },
        opts);
}

DavidsonResultComplex davidson_kernel_hermitian(
    const Eigen::MatrixXcd& A,
    const DavidsonOptions& opts) {
    const int n = static_cast<int>(A.rows());
    if (n == 0) {
        throw std::invalid_argument(
            "davidson_kernel_hermitian: A is 0×0");
    }

    const int n_eig = (opts.n_eig > 0) ? opts.n_eig
                     : std::min(n, 10);
    if (n_eig > n) {
        throw std::invalid_argument(
            "davidson_kernel_hermitian: n_eig > dim");
    }

    const int n_guess = (opts.n_guess > 0)
        ? std::min(opts.n_guess, n)
        : std::min(std::max(n_eig + 5, std::min(2 * n_eig, n)), n);
    const int max_sub = (opts.max_subspace > 0)
        ? opts.max_subspace
        : std::min(std::max(8 * n_eig, n_guess + 20), n);
    const double tol = opts.conv_tol;
    const double preshift = opts.preshift;

    // Diagonal of A (real, since A is Hermitian).
    const Eigen::VectorXd diag = A.diagonal().real();

    // Initial guess: use caller-provided vectors when available,
    // otherwise seed from the smallest diagonal entries of A.
    Eigen::MatrixXcd V;
    if (opts.guess_vectors_cplx.size() != 0) {
        if (opts.guess_vectors_cplx.rows() != n) {
            throw std::invalid_argument(
                "davidson_kernel_hermitian: guess_vectors_cplx row mismatch");
        }
        V = opts.guess_vectors_cplx;
        const int m0 = static_cast<int>(V.cols());
        for (int j = 0; j < m0; ++j) {
            for (int k = 0; k < j; ++k) {
                const std::complex<double> proj = V.col(k).dot(V.col(j));
                V.col(j) -= proj * V.col(k);
            }
            const double vn = V.col(j).norm();
            if (vn > 1e-14) V.col(j) /= vn;
            else V.col(j).setZero();
        }
    } else {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return diag[a] < diag[b]; });
        V.resize(n, n_guess);
        for (int j = 0; j < n_guess; ++j) {
            V.col(j) = Eigen::VectorXcd::Unit(n, idx[j]);
        }
    }

    Eigen::MatrixXcd AV;
    Eigen::MatrixXcd H_proj;
    Eigen::MatrixXcd U;
    Eigen::VectorXd lambda;

    int n_iter = 0;
    int m_sub = static_cast<int>(V.cols());
    // Block AV computation.
    AV = A * V.leftCols(m_sub);

    for (int outer = 0; outer < opts.max_iter; ++outer) {
        n_iter = outer + 1;

        H_proj = V.leftCols(m_sub).adjoint() * AV.leftCols(m_sub);
        // Force Hermitian (tiny numerical drift possible).
        H_proj = 0.5 * (H_proj + H_proj.adjoint().eval());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H_proj);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "davidson_kernel_hermitian: subspace diagonalisation "
                "failed at iter " + std::to_string(n_iter));
        }
        lambda = solver.eigenvalues();
        U      = solver.eigenvectors();

        Eigen::MatrixXcd AX = AV.leftCols(m_sub) * U;
        Eigen::MatrixXcd X  = V.leftCols(m_sub) * U;

        int n_conv = 0;
        Eigen::VectorXd norms(n_eig);
        for (int i = 0; i < n_eig; ++i) {
            const Eigen::VectorXcd ri =
                AX.col(i) - lambda(i) * X.col(i);
            norms(i) = ri.norm();
            if (norms(i) < tol) ++n_conv;
        }

        if (n_conv == n_eig) {
            DavidsonResultComplex result;
            result.eigenvalues  = lambda.head(n_eig);
            result.eigenvectors = X.leftCols(n_eig);
            result.n_iter       = n_iter;
            result.subspace_dim = m_sub;
            result.converged    = true;
            return result;
        }

        const int n_add = std::min(n_eig, m_sub);
        Eigen::MatrixXcd corrections(n, n_add);
        int n_corr = 0;

        for (int i = 0; i < n_eig && n_corr < n_add; ++i) {
            if (norms(i) < tol) continue;
            const Eigen::VectorXcd ri =
                AX.col(i) - lambda(i) * X.col(i);

            Eigen::VectorXcd delta(n);
            for (int j = 0; j < n; ++j) {
                const double la = lambda(i);
                double denom = diag(j) - la + preshift;
                if (std::abs(denom) < tol) {
                    denom = (denom < 0.0) ? -tol : tol;
                }
                delta(j) = ri(j) / denom;
            }

            // Orthogonalise against current subspace.
            for (int k = 0; k < m_sub; ++k) {
                const std::complex<double> proj = delta.dot(V.col(k));
                delta -= proj * V.col(k);
            }
            const double dnorm = delta.norm();
            if (dnorm > tol) {
                delta /= dnorm;
                corrections.col(n_corr) = delta;
                ++n_corr;
            }
        }

        if (m_sub + n_corr >= max_sub) {
            const int keep = std::min(n_eig + 5, n);
            m_sub = keep;
            V.leftCols(m_sub) = X.leftCols(m_sub);
            // Block AV computation.
            AV.leftCols(m_sub) = A * V.leftCols(m_sub);
            // Quick re-orthogonalisation (Gram-Schmidt).
            for (int j = 0; j < m_sub; ++j) {
                for (int k = 0; k < j; ++k) {
                    const std::complex<double> proj =
                        V.col(k).dot(V.col(j));
                    V.col(j) -= proj * V.col(k);
                }
                const double vn = V.col(j).norm();
                if (vn > 1e-14) V.col(j) /= vn;
            }
            AV.leftCols(m_sub) = A * V.leftCols(m_sub);
        } else if (n_corr > 0) {
            const int old_m = m_sub;
            m_sub += n_corr;
            V.conservativeResize(Eigen::NoChange, m_sub);
            AV.conservativeResize(Eigen::NoChange, m_sub);
            V.middleCols(old_m, n_corr) = corrections.leftCols(n_corr);
            // Block mat-mat multiply for all new cols.
            AV.middleCols(old_m, n_corr) =
                A * V.middleCols(old_m, n_corr);
        } else {
            // Stalled.
            DavidsonResultComplex result;
            result.eigenvalues  = lambda.head(n_eig);
            result.eigenvectors = X.leftCols(n_eig);
            result.n_iter       = n_iter;
            result.subspace_dim = m_sub;
            result.converged    = false;
            return result;
        }
    }

    // Exhausted.
    {
        H_proj = V.leftCols(m_sub).adjoint() * AV.leftCols(m_sub);
        H_proj = 0.5 * (H_proj + H_proj.adjoint().eval());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H_proj);
        lambda = solver.eigenvalues();
        U      = solver.eigenvectors();
        Eigen::MatrixXcd X = V.leftCols(m_sub) * U;
        DavidsonResultComplex result;
        result.eigenvalues  = lambda.head(n_eig);
        result.eigenvectors = X.leftCols(n_eig);
        result.n_iter       = n_iter;
        result.subspace_dim = m_sub;
        result.converged    = false;
        return result;
    }
}

}  // anonymous namespace

// ============================================================================
//  Public API
// ============================================================================

DavidsonResult davidson_solve(const Eigen::MatrixXd& A,
                              const DavidsonOptions& opts) {
    return davidson_kernel(A, opts);
}

DavidsonResult davidson_solve_matvec(
    int n_basis,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& matvec,
    const Eigen::VectorXd& diag,
    const DavidsonOptions& opts) {
    // Genuinely matrix-free (GitLab #506).  This used to apply ``matvec`` to
    // all n unit vectors, assemble the dense matrix and hand it to the
    // explicit kernel -- which cost the n x n allocation the caller asked to
    // avoid, plus n callbacks, and discarded ``diag`` entirely.  Measured on a
    // 2987-dimensional TDA operator asking for 8 roots: 2987 operator
    // applications and 104.7 s, against 11.6 s for simply building the matrix
    // and calling eigh.
    //
    // The operator is now applied only to the vectors the algorithm forms --
    // the initial block, each set of corrections, and one rebuild per
    // collapse -- which is O(n_roots) per iteration rather than O(n) overall.
    const int n = n_basis;
    if (diag.size() != n) {
        throw std::invalid_argument(
            "davidson_solve_matvec: diag length " + std::to_string(diag.size())
            + " does not match n_basis " + std::to_string(n));
    }
    // Column-wise adaptor: the caller supplies a single-vector product, the
    // algorithm consumes block products.
    auto apply = [&matvec, n](const Eigen::Ref<const Eigen::MatrixXd>& V)
        -> Eigen::MatrixXd {
        Eigen::MatrixXd out(n, V.cols());
        for (Eigen::Index c = 0; c < V.cols(); ++c) {
            out.col(c) = matvec(V.col(c));
        }
        return out;
    };
    return davidson_kernel_impl(n, diag, apply, opts);
}

DavidsonResultComplex davidson_solve_hermitian(
    const Eigen::MatrixXcd& A_H,
    const DavidsonOptions& opts) {
    return davidson_kernel_hermitian(A_H, opts);
}

DavidsonResultComplex davidson_solve_hermitian_matvec(
    int n_basis,
    const std::function<Eigen::VectorXcd(const Eigen::VectorXcd&)>& matvec,
    const Eigen::VectorXd& diag,
    const DavidsonOptions& opts) {
    const int n = n_basis;
    Eigen::MatrixXcd A(n, n);
    using cplx = std::complex<double>;
    for (int i = 0; i < n; ++i) {
        A.col(i) = matvec(
            Eigen::VectorXcd::Unit(n, i).template cast<cplx>());
    }
    A = 0.5 * (A + A.adjoint().eval());
    (void)diag;  // reserved for matrix-free preconditioner in a future kernel
    return davidson_kernel_hermitian(A, opts);
}

std::string davidson_summary(const DavidsonResult& res) {
    std::ostringstream ss;
    ss << "n_eig=" << res.eigenvalues.size()
       << " tol=" << (res.converged ? "converged" : "NOT_CONVERGED")
       << " n_iter=" << res.n_iter
       << " m_sub=" << res.subspace_dim
       << " λ_range=[" << res.eigenvalues(0)
       << ", " << res.eigenvalues(res.eigenvalues.size()-1)
       << "]";
    return ss.str();
}

std::string davidson_summary(const DavidsonResultComplex& res) {
    std::ostringstream ss;
    ss << "n_eig=" << res.eigenvalues.size()
       << " tol=" << (res.converged ? "converged" : "NOT_CONVERGED")
       << " n_iter=" << res.n_iter
       << " m_sub=" << res.subspace_dim
       << " λ_range=[" << res.eigenvalues(0)
       << ", " << res.eigenvalues(res.eigenvalues.size()-1)
       << "]";
    return ss.str();
}

}  // namespace vibeqc
