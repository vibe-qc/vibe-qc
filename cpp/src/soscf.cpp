#include "vibeqc/soscf.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "vibeqc/quadratic_scf.hpp"   // expm_skew

namespace vibeqc {

namespace {

// Occ-vir orbital-rotation gradient g_{ai} = F^MO_{ai} (flattened,
// column-major: index = i*n_vir + a) and the diagonal orbital Hessian
// d_{ai} = ε_a − ε_i (floored for numerical safety). Shared by the
// stateless step and the stateful L-BFGS accelerator.
void orbital_gradient_and_hessian(const Eigen::MatrixXd& F,
                                  const Eigen::MatrixXd& C_prev,
                                  const Eigen::VectorXd& eps_prev,
                                  Eigen::Index n_occ, Eigen::Index n_vir,
                                  Eigen::VectorXd& g_vec,
                                  Eigen::VectorXd& d_vec) {
    Eigen::MatrixXd F_mo = C_prev.transpose() * F * C_prev;
    F_mo = 0.5 * (F_mo + F_mo.transpose());
    const Eigen::MatrixXd g_mat = F_mo.bottomLeftCorner(n_vir, n_occ);
    const Eigen::Index n_pairs = n_vir * n_occ;
    g_vec = Eigen::Map<const Eigen::VectorXd>(g_mat.data(), n_pairs);
    d_vec.resize(n_pairs);
    for (Eigen::Index i = 0; i < n_occ; ++i) {
        const double eps_i = eps_prev(i);
        for (Eigen::Index a = 0; a < n_vir; ++a) {
            d_vec(i * n_vir + a) =
                std::max(eps_prev(n_occ + a) - eps_i, 1e-6);
        }
    }
}

// L-BFGS two-loop recursion returning the quasi-Newton step κ = −H g,
// with H0 = diag(1/d) as the initial inverse Hessian. With an empty
// history this is exactly the diagonal preconditioned step −g/d.
Eigen::VectorXd lbfgs_step(const Eigen::VectorXd& g,
                           const Eigen::VectorXd& d,
                           const std::deque<Eigen::VectorXd>& s_hist,
                           const std::deque<Eigen::VectorXd>& y_hist) {
    const int m = static_cast<int>(s_hist.size());
    Eigen::VectorXd q = g;
    std::vector<double> alpha(m), rho(m);
    for (int j = m - 1; j >= 0; --j) {
        rho[j] = 1.0 / y_hist[j].dot(s_hist[j]);
        alpha[j] = rho[j] * s_hist[j].dot(q);
        q.noalias() -= alpha[j] * y_hist[j];
    }
    Eigen::VectorXd r = q.array() / d.array();  // apply H0 = diag(1/d)
    for (int j = 0; j < m; ++j) {
        const double beta = rho[j] * y_hist[j].dot(r);
        r.noalias() += s_hist[j] * (alpha[j] - beta);
    }
    return -r;
}

// Apply an occ-vir rotation κ (flattened, column-major) to C_prev and
// refresh MO energies from the rotated frame.
void apply_rotation(const Eigen::VectorXd& kappa_vec,
                    const Eigen::MatrixXd& F,
                    const Eigen::MatrixXd& C_prev,
                    Eigen::Index n_occ, Eigen::Index n_vir,
                    Eigen::MatrixXd& C_new, Eigen::VectorXd& eps_new) {
    const Eigen::Index n_kept = C_prev.cols();
    Eigen::MatrixXd kappa_mat(n_vir, n_occ);
    for (Eigen::Index i = 0; i < n_occ; ++i) {
        for (Eigen::Index a = 0; a < n_vir; ++a) {
            kappa_mat(a, i) = kappa_vec(i * n_vir + a);
        }
    }
    Eigen::MatrixXd kappa_full = Eigen::MatrixXd::Zero(n_kept, n_kept);
    kappa_full.bottomLeftCorner(n_vir, n_occ) = kappa_mat;
    kappa_full.topRightCorner(n_occ, n_vir) = -kappa_mat.transpose();
    const Eigen::MatrixXd U = expm_skew(kappa_full);
    C_new = C_prev * U;
    eps_new = (C_new.transpose() * F * C_new).diagonal();
}

// Accept an (s, y) curvature pair into an L-BFGS history if it satisfies
// the curvature condition sᵀy > 0 (keeps the implicit inverse Hessian
// positive definite); cap the history length.
void push_curvature(std::deque<Eigen::VectorXd>& s_hist,
                    std::deque<Eigen::VectorXd>& y_hist,
                    const Eigen::VectorXd& s, const Eigen::VectorXd& y,
                    std::size_t max_history) {
    const double sy = s.dot(y);
    if (sy > 1e-10 * s.norm() * y.norm()) {
        s_hist.push_back(s);
        y_hist.push_back(y);
        while (s_hist.size() > max_history) {
            s_hist.pop_front();
            y_hist.pop_front();
        }
    }
}

}  // namespace

SOSCFStepResult soscf_step(const Eigen::MatrixXd& F,
                            const Eigen::MatrixXd& C_prev,
                            const Eigen::VectorXd& eps_prev,
                            int n_occ,
                            const SOSCFOptions& opts) {
    const Eigen::Index n_kept = C_prev.cols();
    const Eigen::Index n_vir  = n_kept - n_occ;

    SOSCFStepResult out;
    if (n_vir <= 0 || n_occ <= 0) {
        // No occ-vir manifold; rotation is the identity.
        out.C = C_prev;
        out.eps = (C_prev.transpose() * F * C_prev).diagonal();
        return out;
    }

    // F in current MO basis; symmetrise for numerical drift.
    Eigen::MatrixXd F_mo = C_prev.transpose() * F * C_prev;
    F_mo = 0.5 * (F_mo + F_mo.transpose());

    // Orbital-rotation gradient g (occ-vir block of F^MO), shape
    // (n_vir, n_occ); column-major flatten to length n_pairs.
    const Eigen::MatrixXd g_mat = F_mo.bottomLeftCorner(n_vir, n_occ);
    const Eigen::Index n_pairs = n_vir * n_occ;
    const Eigen::Map<const Eigen::VectorXd> g_vec(g_mat.data(), n_pairs);

    // Diagonal orbital Hessian A_{ai} = ε_a − ε_i. Same column-major
    // ordering as g_mat for consistency with the Map view above.
    Eigen::VectorXd d_vec(n_pairs);
    for (Eigen::Index i = 0; i < n_occ; ++i) {
        const double eps_i = eps_prev(i);
        for (Eigen::Index a = 0; a < n_vir; ++a) {
            const double diff = eps_prev(n_occ + a) - eps_i;
            d_vec(i * n_vir + a) = std::max(diff, 1e-6);  // defensive floor
        }
    }

    // Build the augmented Hessian
    //   M = [diag(d_vec), g_vec; g_vec^T, 0]
    // Lowest eigenvalue λ + corresponding eigenvector (κ; α) gives the
    // optimal step κ/α (after normalising by α). Direct dense eigsolve
    // is fine for n_pairs up to ~1000; for larger systems Davidson is
    // a future optimisation (TRAH-style).
    const Eigen::Index n_aug = n_pairs + 1;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_aug, n_aug);
    M.diagonal().head(n_pairs) = d_vec;
    M.col(n_pairs).head(n_pairs) = g_vec;
    M.row(n_pairs).head(n_pairs) = g_vec.transpose();
    // M(n_pairs, n_pairs) is already 0.

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "soscf_step: augmented-Hessian eigsolve failed");
    }

    // Lowest eigenvector / eigenvalue.
    out.lambda_shift = es.eigenvalues()(0);
    const Eigen::VectorXd lowest = es.eigenvectors().col(0);
    const double tail = lowest(n_pairs);
    if (std::abs(tail) < 1e-12) {
        // Eigenvector lies entirely in the κ subspace — happens when g
        // is at convergence already (the AH formulation degenerates).
        // Caller should check grad_norm before invoking soscf_step.
        out.C = C_prev;
        out.eps = F_mo.diagonal();
        return out;
    }
    Eigen::VectorXd kappa_vec = lowest.head(n_pairs) / tail;

    // Trust-region cap on ‖κ‖_F.
    out.kappa_norm = kappa_vec.norm();
    if (out.kappa_norm > opts.trust_radius && out.kappa_norm > 0.0) {
        kappa_vec *= (opts.trust_radius / out.kappa_norm);
    }

    // Reshape κ_vec → κ_mat (n_vir, n_occ) consistent with g_mat
    // column-major layout (i indexes columns, a indexes rows).
    Eigen::MatrixXd kappa_mat(n_vir, n_occ);
    for (Eigen::Index i = 0; i < n_occ; ++i) {
        for (Eigen::Index a = 0; a < n_vir; ++a) {
            kappa_mat(a, i) = kappa_vec(i * n_vir + a);
        }
    }

    // Build full antihermitian rotation generator on the kept-direction
    // subspace.
    Eigen::MatrixXd kappa_full = Eigen::MatrixXd::Zero(n_kept, n_kept);
    kappa_full.bottomLeftCorner(n_vir, n_occ) = kappa_mat;
    kappa_full.topRightCorner(n_occ, n_vir) = -kappa_mat.transpose();

    // C_new = C_prev · exp(κ); ε_new = diag(C_new^T F C_new).
    const Eigen::MatrixXd U = expm_skew(kappa_full);
    out.C = C_prev * U;
    out.eps = (out.C.transpose() * F * out.C).diagonal();
    return out;
}

UHFSOSCFStepResult uhf_soscf_step(const Eigen::MatrixXd& F_alpha,
                                   const Eigen::MatrixXd& F_beta,
                                   const Eigen::MatrixXd& C_alpha_prev,
                                   const Eigen::MatrixXd& C_beta_prev,
                                   const Eigen::VectorXd& eps_alpha_prev,
                                   const Eigen::VectorXd& eps_beta_prev,
                                   int n_alpha,
                                   int n_beta,
                                   const SOSCFOptions& opts) {
    // The diagonal-dominant Hessian approximation has no cross-spin
    // coupling — α and β rotations decouple, so we solve two
    // independent AH eigsolves and combine the results.
    const auto step_a = soscf_step(F_alpha, C_alpha_prev,
                                    eps_alpha_prev, n_alpha, opts);
    const auto step_b = soscf_step(F_beta, C_beta_prev,
                                    eps_beta_prev,  n_beta,  opts);

    UHFSOSCFStepResult out;
    out.C_alpha = step_a.C;
    out.C_beta  = step_b.C;
    out.eps_alpha = step_a.eps;
    out.eps_beta  = step_b.eps;
    out.kappa_norm = std::max(step_a.kappa_norm, step_b.kappa_norm);
    out.lambda_shift = std::min(step_a.lambda_shift, step_b.lambda_shift);
    return out;
}

// ---------------------------------------------------------------------------
// Stateful L-BFGS SOSCF (the production accelerator).
// ---------------------------------------------------------------------------

SOSCF::SOSCF(SOSCFOptions opts) : opts_(opts) {}

void SOSCF::clear() {
    s_hist_.clear();
    y_hist_.clear();
    g_prev_.resize(0);
    kappa_prev_.resize(0);
    have_prev_ = false;
}

SOSCFStepResult SOSCF::step(const Eigen::MatrixXd& F,
                            const Eigen::MatrixXd& C_prev,
                            const Eigen::VectorXd& eps_prev,
                            int n_occ) {
    const Eigen::Index n_kept = C_prev.cols();
    const Eigen::Index n_vir = n_kept - n_occ;

    SOSCFStepResult out;
    if (n_vir <= 0 || n_occ <= 0) {
        out.C = C_prev;
        out.eps = (C_prev.transpose() * F * C_prev).diagonal();
        return out;
    }

    Eigen::VectorXd g_vec, d_vec;
    orbital_gradient_and_hessian(F, C_prev, eps_prev, n_occ, n_vir,
                                 g_vec, d_vec);
    const Eigen::Index n_pairs = n_vir * n_occ;

    // Fold the previous step's outcome into the L-BFGS history: the step we
    // took (kappa_prev_) paired with the resulting gradient change. The
    // dimension guard skips the update if the occ-vir manifold changed size
    // between iterations (e.g. an occupation reshuffle), which would make the
    // stored history vectors inconsistent.
    if (have_prev_ && g_prev_.size() == n_pairs
        && kappa_prev_.size() == n_pairs) {
        push_curvature(s_hist_, y_hist_, kappa_prev_, g_vec - g_prev_,
                       opts_.lbfgs_history);
    }

    Eigen::VectorXd kappa_vec = lbfgs_step(g_vec, d_vec, s_hist_, y_hist_);

    // Descent-direction guard: if L-BFGS step is not a descent direction,
    // clear history and fall back to diagonal-preconditioned step κ = −g/d.
    const double g_dot_kappa = g_vec.dot(kappa_vec);
    if (g_dot_kappa >= 0.0 && !s_hist_.empty()) {
        s_hist_.clear();
        y_hist_.clear();
        kappa_vec = lbfgs_step(g_vec, d_vec, s_hist_, y_hist_);
        if (g_vec.dot(kappa_vec) >= 0.0) {
            out.C = C_prev;
            out.eps = (C_prev.transpose() * F * C_prev).diagonal();
            return out;
        }
    }

    out.kappa_norm = kappa_vec.norm();
    if (out.kappa_norm > opts_.trust_radius && out.kappa_norm > 0.0) {
        kappa_vec *= (opts_.trust_radius / out.kappa_norm);
    }

    g_prev_ = g_vec;
    kappa_prev_ = kappa_vec;
    have_prev_ = true;

    apply_rotation(kappa_vec, F, C_prev, n_occ, n_vir, out.C, out.eps);
    return out;
}

UHFSOSCF::UHFSOSCF(SOSCFOptions opts) : opts_(opts) {}

void UHFSOSCF::clear() {
    s_hist_.clear();
    y_hist_.clear();
    g_prev_.resize(0);
    kappa_prev_.resize(0);
    have_prev_ = false;
}

UHFSOSCFStepResult UHFSOSCF::step(const Eigen::MatrixXd& F_alpha,
                                  const Eigen::MatrixXd& F_beta,
                                  const Eigen::MatrixXd& C_alpha_prev,
                                  const Eigen::MatrixXd& C_beta_prev,
                                  const Eigen::VectorXd& eps_alpha_prev,
                                  const Eigen::VectorXd& eps_beta_prev,
                                  int n_alpha, int n_beta) {
    const Eigen::Index n_vir_a = C_alpha_prev.cols() - n_alpha;
    const Eigen::Index n_vir_b = C_beta_prev.cols() - n_beta;
    const Eigen::Index np_a = (n_vir_a > 0 && n_alpha > 0) ? n_vir_a * n_alpha : 0;
    const Eigen::Index np_b = (n_vir_b > 0 && n_beta > 0) ? n_vir_b * n_beta : 0;

    UHFSOSCFStepResult out;
    out.C_alpha = C_alpha_prev;
    out.C_beta = C_beta_prev;
    out.eps_alpha = (C_alpha_prev.transpose() * F_alpha * C_alpha_prev).diagonal();
    out.eps_beta = (C_beta_prev.transpose() * F_beta * C_beta_prev).diagonal();
    if (np_a + np_b == 0) {
        return out;  // no occ-vir manifold on either spin
    }

    // Stack the per-spin gradients and diagonal Hessians into one vector so a
    // single L-BFGS history couples the spins (the diagonal H0 stays per-spin
    // block-diagonal; the accumulated curvature carries the cross-spin
    // response the diagonal model omits).
    Eigen::VectorXd g_a, d_a, g_b, d_b;
    if (np_a) {
        orbital_gradient_and_hessian(F_alpha, C_alpha_prev, eps_alpha_prev,
                                     n_alpha, n_vir_a, g_a, d_a);
    }
    if (np_b) {
        orbital_gradient_and_hessian(F_beta, C_beta_prev, eps_beta_prev,
                                     n_beta, n_vir_b, g_b, d_b);
    }
    const Eigen::Index np = np_a + np_b;
    Eigen::VectorXd g_vec(np), d_vec(np);
    if (np_a) { g_vec.head(np_a) = g_a; d_vec.head(np_a) = d_a; }
    if (np_b) { g_vec.tail(np_b) = g_b; d_vec.tail(np_b) = d_b; }

    if (have_prev_ && g_prev_.size() == np && kappa_prev_.size() == np) {
        push_curvature(s_hist_, y_hist_, kappa_prev_, g_vec - g_prev_,
                       opts_.lbfgs_history);
    }

    Eigen::VectorXd kappa_vec = lbfgs_step(g_vec, d_vec, s_hist_, y_hist_);

    // Descent-direction guard (same reasoning as closed-shell path).
    const double g_dot_kappa = g_vec.dot(kappa_vec);
    if (g_dot_kappa >= 0.0 && !s_hist_.empty()) {
        s_hist_.clear();
        y_hist_.clear();
        kappa_vec = lbfgs_step(g_vec, d_vec, s_hist_, y_hist_);
        if (g_vec.dot(kappa_vec) >= 0.0) {
            return out;
        }
    }

    out.kappa_norm = kappa_vec.norm();
    if (out.kappa_norm > opts_.trust_radius && out.kappa_norm > 0.0) {
        kappa_vec *= (opts_.trust_radius / out.kappa_norm);
    }

    g_prev_ = g_vec;
    kappa_prev_ = kappa_vec;
    have_prev_ = true;

    if (np_a) {
        apply_rotation(kappa_vec.head(np_a), F_alpha, C_alpha_prev,
                       n_alpha, n_vir_a, out.C_alpha, out.eps_alpha);
    }
    if (np_b) {
        apply_rotation(kappa_vec.tail(np_b), F_beta, C_beta_prev,
                       n_beta, n_vir_b, out.C_beta, out.eps_beta);
    }
    return out;
}

}  // namespace vibeqc
