#include "vibeqc/kdiis.hpp"

#include <stdexcept>

namespace vibeqc {

namespace {

// Build the orbital-rotation gradient g_{ai} = F^MO_{ai}, the occ-vir
// block of F in the canonical MO basis. Shape (n_vir, n_occ); identical
// shape contract to soscf_step / newton_step's gradient. The MO basis is
// the one in which ``eps`` is diagonal (i.e. C^T F_diag C = diag(eps) for
// the canonical MOs).
Eigen::MatrixXd orbital_gradient_ov(const Eigen::MatrixXd& F,
                                     const Eigen::MatrixXd& C,
                                     int n_occ) {
    const Eigen::Index n_kept = C.cols();
    const Eigen::Index n_vir = n_kept - n_occ;
    if (n_vir <= 0 || n_occ <= 0) {
        // No virtual or no occupied space; the orbital-rotation manifold
        // is trivial and the gradient is empty by convention.
        return Eigen::MatrixXd();
    }
    // F^MO = C^T F C. We only need the occ-vir block (rows = vir, cols = occ).
    const Eigen::MatrixXd F_mo = C.transpose() * F * C;
    return F_mo.bottomLeftCorner(n_vir, n_occ);
}

// Standard Pulay (n+1)-dim B-matrix solve:
//   [ B_ij  -1 ] [ c ]   [ 0 ]
//   [  -1    0 ] [ λ ] = [-1 ]
// with B_ij = (e_i, e_j)_F (Frobenius inner product).
//
// Returns the c vector (length n). Full-pivot LU absorbs the mild
// ill-conditioning that appears as iterates near-collapse onto a
// low-rank subspace at convergence.
Eigen::VectorXd solve_pulay_coefficients(
    const std::deque<Eigen::MatrixXd>& errors)
{
    const auto n = static_cast<Eigen::Index>(errors.size());
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 1, n + 1);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n + 1);

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j <= i; ++j) {
            const double bij =
                (errors[i].array() * errors[j].array()).sum();
            A(i, j) = bij;
            A(j, i) = bij;
        }
        A(i, n) = -1.0;
        A(n, i) = -1.0;
    }
    A(n, n) = 0.0;
    b(n) = -1.0;

    const Eigen::VectorXd sol = A.fullPivLu().solve(b);
    return sol.head(n);
}

}  // namespace

KDIIS::KDIIS(std::size_t max_subspace) : max_subspace_(max_subspace) {
    if (max_subspace_ < 2) {
        throw std::invalid_argument("KDIIS subspace size must be >= 2");
    }
}

void KDIIS::clear() {
    fock_history_.clear();
    fock_beta_history_.clear();
    error_history_.clear();
    open_shell_ = false;
}

Eigen::MatrixXd KDIIS::extrapolate(const Eigen::MatrixXd& fock,
                                    const Eigen::MatrixXd& C,
                                    const Eigen::VectorXd& eps,
                                    int n_occ) {
    if (open_shell_) {
        throw std::logic_error(
            "KDIIS::extrapolate (closed-shell) called on a history that "
            "was previously fed open-shell iterates; clear() first or "
            "use the open-shell overload consistently.");
    }
    (void)eps;  // Currently unused; reserved for future preconditioning.
    Eigen::MatrixXd error = orbital_gradient_ov(fock, C, n_occ);

    fock_history_.push_back(fock);
    error_history_.push_back(std::move(error));
    while (fock_history_.size() > max_subspace_) {
        fock_history_.pop_front();
        error_history_.pop_front();
    }

    if (fock_history_.size() < 2) {
        return fock;
    }

    const Eigen::VectorXd c = solve_pulay_coefficients(error_history_);

    Eigen::MatrixXd F_extrap = Eigen::MatrixXd::Zero(fock.rows(), fock.cols());
    for (Eigen::Index i = 0; i < c.size(); ++i) {
        F_extrap += c(i) * fock_history_[i];
    }
    return F_extrap;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
KDIIS::extrapolate(const Eigen::MatrixXd& fock_alpha,
                   const Eigen::MatrixXd& fock_beta,
                   const Eigen::MatrixXd& C_alpha,
                   const Eigen::MatrixXd& C_beta,
                   const Eigen::VectorXd& eps_alpha,
                   const Eigen::VectorXd& eps_beta,
                   int n_alpha,
                   int n_beta) {
    if (!fock_history_.empty() && !open_shell_) {
        throw std::logic_error(
            "KDIIS::extrapolate (open-shell) called on a history that "
            "was previously fed closed-shell iterates; clear() first or "
            "use the closed-shell overload consistently.");
    }
    open_shell_ = true;
    (void)eps_alpha; (void)eps_beta;  // Reserved for future use.

    const Eigen::MatrixXd g_alpha =
        orbital_gradient_ov(fock_alpha, C_alpha, n_alpha);
    const Eigen::MatrixXd g_beta =
        orbital_gradient_ov(fock_beta,  C_beta,  n_beta);

    // Concatenate per-spin gradients into a single error matrix for the
    // B-matrix inner products. We stack vertically so the Frobenius inner
    // product of the stacked errors equals (g_α, g_α') + (g_β, g_β'),
    // which is the natural per-spin-coupled metric.
    const Eigen::Index rows_a = g_alpha.size() ? g_alpha.rows() : 0;
    const Eigen::Index cols_a = g_alpha.size() ? g_alpha.cols() : 0;
    const Eigen::Index rows_b = g_beta.size()  ? g_beta.rows()  : 0;
    const Eigen::Index cols_b = g_beta.size()  ? g_beta.cols()  : 0;
    // Pad to a single (rows_a + rows_b, max(cols_a, cols_b)) buffer with
    // zero blocks where the per-spin shapes differ. This keeps the
    // history shape stable across iterations as long as n_alpha / n_beta
    // are stable (which they are within a single SCF run).
    const Eigen::Index cols_max = std::max(cols_a, cols_b);
    Eigen::MatrixXd error_stacked =
        Eigen::MatrixXd::Zero(rows_a + rows_b, cols_max);
    if (g_alpha.size()) {
        error_stacked.topLeftCorner(rows_a, cols_a) = g_alpha;
    }
    if (g_beta.size()) {
        error_stacked.bottomLeftCorner(rows_b, cols_b) = g_beta;
    }

    fock_history_.push_back(fock_alpha);
    fock_beta_history_.push_back(fock_beta);
    error_history_.push_back(std::move(error_stacked));
    while (fock_history_.size() > max_subspace_) {
        fock_history_.pop_front();
        fock_beta_history_.pop_front();
        error_history_.pop_front();
    }

    if (fock_history_.size() < 2) {
        return {fock_alpha, fock_beta};
    }

    const Eigen::VectorXd c = solve_pulay_coefficients(error_history_);

    Eigen::MatrixXd F_alpha_extrap =
        Eigen::MatrixXd::Zero(fock_alpha.rows(), fock_alpha.cols());
    Eigen::MatrixXd F_beta_extrap =
        Eigen::MatrixXd::Zero(fock_beta.rows(),  fock_beta.cols());
    for (Eigen::Index i = 0; i < c.size(); ++i) {
        F_alpha_extrap += c(i) * fock_history_[i];
        F_beta_extrap  += c(i) * fock_beta_history_[i];
    }
    return {F_alpha_extrap, F_beta_extrap};
}

}  // namespace vibeqc
