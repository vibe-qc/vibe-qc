#include "vibeqc/semiempirical/core/charge_mixer.hpp"

#include <algorithm>
#include <cmath>

namespace vibeqc {
namespace semiempirical {

// ===================================================================
// ChargeDIISMixer — Pulay DIIS with normalized error vectors,
// delayed start, and adaptive damping.
//
// Key insight: the SCC charge response dq → dq_new is strongly
// nonlinear far from convergence, so DIIS extrapolation (which
// assumes linearity) can overshoot and diverge.  We fix this by:
//
//   1. Normalised error vectors (correlation-based B-matrix).
//   2. Delayed DIIS start: simple mixing for the first N iterations
//      to enter the linear regime.
//   3. Adaptive damping: blend DIIS with raw input more heavily
//      early, less near convergence.
// ===================================================================

void ChargeDIISMixer::mix(Eigen::VectorXd& dq_out,
                           const Eigen::VectorXd& dq_in,
                           int /*iter*/) {
    const int n = static_cast<int>(dq_in.size());
    if (n == 0) return;

    // Compute residual: e = dq_in - dq_out  (raw SCF output minus mixed input).
    // DIIS stores the mixed inputs and their residuals, then extrapolates
    // a linear combination of mixed inputs that minimises the residual norm.
    Eigen::VectorXd e_raw = dq_in - dq_out;
    double e_norm = e_raw.norm();
    last_error_ = e_norm;

    // Store the MIXED input (dq_out, what SCF used) and its residual.
    Eigen::VectorXd e_hat = (e_norm > 1e-30) ? e_raw / e_norm : e_raw;
    if (q_history_.empty()) {
        q_history_.push_back(dq_out);
        e_history_.push_back(e_hat);
    } else {
        bool should_push = true;
        if (!e_history_.empty()) {
            double dot = std::abs(e_hat.dot(e_history_.back()));
            should_push = (dot < 0.999);
        }
        if (should_push) {
            q_history_.push_back(dq_out);
            e_history_.push_back(e_hat);
            while (q_history_.size() > max_subspace_) {
                q_history_.pop_front(); e_history_.pop_front();
            }
        }
    }

    const auto m = static_cast<Eigen::Index>(q_history_.size());

    // Before enough history: simple damped mixing
    if (m < 2) {
        double alpha = std::max(0.3, std::min(0.95, damping_));
        dq_out = alpha * dq_in + (1.0 - alpha) * dq_out;
        return;
    }

    // ---- DIIS extrapolation ----
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m + 1, m + 1);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(m + 1);
    for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j <= i; ++j) {
            double bij = e_history_[static_cast<std::size_t>(i)].dot(
                         e_history_[static_cast<std::size_t>(j)]);
            A(i, j) = bij; A(j, i) = bij;
        }
        A(i, m) = -1.0; A(m, i) = -1.0;
    }
    A(m, m) = 0.0; b(m) = -1.0;

    Eigen::VectorXd sol = A.fullPivLu().solve(b);

    // Reject wild coefficients (DIIS unconstrained => can be >100).
    bool coeff_ok = sol.allFinite() && sol.head(m).lpNorm<1>() < 1e3;
    if (coeff_ok) {
        for (Eigen::Index i = 0; i < m; ++i)
            if (sol(i) < -1.0 || sol(i) > 2.0) { coeff_ok = false; break; }
    }
    if (!coeff_ok) {
        double alpha = std::max(0.25, damping_);
        dq_out = alpha * dq_in + (1.0 - alpha) * dq_out;
        return;
    }

    // Check coefficients again: wild values indicate ill-conditioned extrapolation.
    bool coeffs_ok = true;
    for (Eigen::Index i = 0; i < m; ++i)
        if (sol(i) < -2.0 || sol(i) > 3.0) { coeffs_ok = false; break; }
    if (!coeffs_ok) {
        double alpha = std::max(0.25, damping_);
        dq_out = alpha * dq_in + (1.0 - alpha) * dq_out;
        return;
    }

    Eigen::VectorXd dq_diis = Eigen::VectorXd::Zero(n);
    for (Eigen::Index i = 0; i < m; ++i)
        dq_diis += sol(i) * q_history_[static_cast<std::size_t>(i)];

    if (!dq_diis.allFinite()) {
        double alpha = std::max(0.25, damping_);
        dq_out = alpha * dq_in + (1.0 - alpha) * dq_out;
        return;
    }

    // Use the DIIS extrapolation directly.  The damping_ parameter
    // controls fallback intensity when DIIS fails; when it succeeds
    // we trust the extrapolation.  (A damped blend of the extrapolated
    // vector was measured to break naphthalene-type SCCs into
    // non-convergence; the GFN2 driver instead guards DIIS with a
    // stall-revert wrapper around the pure extrapolation.)
    dq_out = dq_diis;
}


// ===================================================================
// BroydenMixer — modified Broyden with normalised history vectors
// and adaptive damping.
// ===================================================================

BroydenMixer::BroydenMixer(int memory, double damping)
    : memory_(memory), damping_(damping) {
    if (memory_ < 1) memory_ = 1;
}

void BroydenMixer::allocate(int dim) {
    dim_ = dim;
    U_.resize(dim, memory_);
    U_.setZero();
    V_.resize(dim, memory_);
    V_.setZero();
    A_.resize(0, 0);
    step_ = 0; error_ = 0.0;
    q_last_.resize(dim); q_last_.setZero();
    dq_last_.resize(dim); dq_last_.setZero();
}

void BroydenMixer::reset() {
    step_ = 0; error_ = 0.0;
    if (dim_ > 0) {
        U_.resize(dim_, memory_);
        U_.setZero();
        V_.resize(dim_, memory_);
        V_.setZero();
    }
    A_.resize(0, 0);
    if (dim_ > 0) { q_last_.setZero(); dq_last_.setZero(); }
}

void BroydenMixer::mix(Eigen::VectorXd& dq_out,
                        const Eigen::VectorXd& dq_in,
                        int /*iter*/) {
    int n = static_cast<int>(dq_in.size());
    if (n == 0) return;

    if (dim_ != n) {
        allocate(n);
        double alpha = std::max(0.3, std::min(0.95, damping_));
        dq_out = alpha * dq_in + (1.0 - alpha) * dq_out;
        q_last_ = dq_in;
        return;
    }

    // First step: simple damping as warm-up.
    if (step_ == 0) {
        dq_last_ = dq_in - q_last_;
        double alpha = std::max(0.1, damping_);
        if (dq_last_.norm() > 1e-15) {
            dq_out = alpha * dq_in + (1.0 - alpha) * q_last_;
            step_ = 1;
            q_last_ = dq_in;
            dq_last_ = dq_out - q_last_;
        } else {
            dq_out = dq_in;
        }
        return;
    }

    Eigen::VectorXd dq_in_new = dq_in - q_last_;
    error_ = dq_in_new.norm();

    // Normalised difference vectors.
    double norm_dq = dq_in_new.norm();
    if (norm_dq < 1e-30) { dq_out = dq_in; return; }
    Eigen::VectorXd u = dq_in_new / norm_dq;
    Eigen::VectorXd v = (dq_in_new - dq_last_) / norm_dq;

    // Store history.
    // Number of populated history columns.  Keep this bounded separately
    // from the warm-up marker in step_: every column access below must stay
    // inside the fixed dim x memory allocation.
    int h = std::min(step_ - 1, memory_);
    bool should_push = true;
    if (h > 0) {
        double dot = std::abs(u.dot(U_.col(h - 1)));
        should_push = (dot < 0.999);
    }
    if (should_push) {
        if (h >= memory_) {
            int keep = memory_ - 1;
            U_.leftCols(keep) = U_.middleCols(1, keep);
            V_.leftCols(keep) = V_.middleCols(1, keep);
            h = keep;
        }
        U_.col(h) = u;
        V_.col(h) = v;
        h++;
        step_ = h + 1;
    }

    // Quasi-Newton step.
    if (h < 1) {
        double alpha = std::max(0.1, damping_);
        dq_out = alpha * dq_in + (1.0 - alpha) * q_last_;
    } else {
        A_.resize(h, h);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j <= i; ++j) {
                double a = U_.col(i).dot(V_.col(j));
                A_(i, j) = a; A_(j, i) = a;
            }
            A_(i, i) += w0_ * w0_;
        }

        Eigen::VectorXd rhs = U_.leftCols(h).transpose() * dq_in_new;
        Eigen::VectorXd c = A_.fullPivLu().solve(rhs);
        bool solve_ok = c.allFinite() && c.lpNorm<1>() < 1e6;

        if (solve_ok) {
            Eigen::VectorXd step = dq_in_new - V_.leftCols(h) * c;
            // Fixed damping: beta=0 means full step, beta=1 means no step.
            double beta = std::max(0.01, std::min(0.99, damping_));
            dq_out = q_last_ + (1.0 - beta) * step;
        } else {
            double alpha = std::max(0.1, damping_);
            dq_out = alpha * dq_in + (1.0 - alpha) * q_last_;
            step_ = 1;
            U_.setZero();
            V_.setZero();
        }
    }

    dq_last_ = dq_out - q_last_;
    q_last_ = dq_in;
}

// ===================================================================
// EyertBroydenMixer — the tblite broyden.f90 modified Broyden scheme
// (the mixer the xtb binary uses), translated element for element.
//
// State per iteration (tblite names in comments):
//   q      = dq_out (current input charges)
//   dq     = dq_in - dq_out (the raw charge residual)
//   alpha  = mixing parameter (bromix)
//   omega  = inverse-norm history weight, 0.01/|dq| clamped to [1, 1e5]
//   df     = normalized residual differences (dq - dqlast)/|...|
//   u      = alpha*df + (q - qlast)/|df_raw|
//   beta   = omega_i omega_j <df_i|df_j> + omega0^2 delta, omega0 = 0.01
//   q_new  = q + alpha*dq - sum_i omega_i c_i u_i
// ===================================================================

EyertBroydenMixer::EyertBroydenMixer(int memory, double alpha)
    : memory_(memory), alpha_(alpha) {
    if (memory_ < 1) memory_ = 1;
}

void EyertBroydenMixer::reset() {
    dim_ = 0;
    error_ = 0.0;
    q_last_.resize(0);
    dq_last_.resize(0);
    df_.resize(0, 0);
    u_.resize(0, 0);
    omega_.resize(0);
}

void EyertBroydenMixer::mix(Eigen::VectorXd& dq_out,
                            const Eigen::VectorXd& dq_in,
                            int iter) {
    const int n = static_cast<int>(dq_in.size());
    if (n == 0) return;
    if (dim_ != n) {
        dim_ = n;
        q_last_ = Eigen::VectorXd::Zero(n);
        dq_last_ = Eigen::VectorXd::Zero(n);
        df_.resize(n, memory_);
        df_.setZero();
        u_.resize(n, memory_);
        u_.setZero();
        omega_.resize(memory_);
        omega_.setZero();
    }

    // Residual of the raw charge map.
    const Eigen::VectorXd dq = dq_in - dq_out;
    error_ = dq.norm();

    // Constants from broyden.f90.
    const double omega0 = 0.01;
    const double minw = 1.0;
    const double maxw = 100000.0;
    const double wfac = 0.01;

    if (iter == 1) {
        // First iteration: simple damping.
        dq_last_ = dq;
        q_last_ = dq_out;
        dq_out = dq_out + alpha_ * dq;
        return;
    }

    const int itn = iter - 1;
    const int it1 = (itn - 1) % memory_;  // 0-based slot for the new vector
    const int nhist = std::min(memory_, itn);
    const int first = std::max(1, itn - memory_ + 1);

    // Inverse-norm weight for the current residual (clamped).
    const double dq_norm = dq.norm();
    double w = maxw;
    if (dq_norm > wfac / maxw) {
        w = wfac / dq_norm;
    }
    omega_(it1) = std::max(minw, w);

    // Normalized residual difference df(:, it1).
    Eigen::VectorXd df_new = dq - dq_last_;
    const double df_raw_norm = df_new.norm();
    const double inv = 1.0 / std::max(df_raw_norm, 1.0e-300);
    if (df_raw_norm > 0.0) {
        df_new *= inv;
    }
    df_.col(it1) = df_new;

    // Chronological slot order, oldest first.
    std::vector<int> hist(nhist);
    for (int ih = 0; ih < nhist; ++ih) {
        hist[static_cast<std::size_t>(ih)] = (first + ih - 1) % memory_;
    }

    // beta(i,j) = omega_i omega_j <df_i|df_j> + omega0^2 delta_ij, and the
    // right-hand side omega_i <df_i|dq>.  Both are built over the history
    // slots only.
    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(nhist, nhist);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nhist);
    for (int ih = 0; ih < nhist; ++ih) {
        const int i = hist[static_cast<std::size_t>(ih)];
        rhs(ih) = omega_(i) * df_.col(i).dot(dq);
        for (int jh = 0; jh < nhist; ++jh) {
            const int j = hist[static_cast<std::size_t>(jh)];
            beta(ih, jh) = omega_(i) * omega_(j)
                * df_.col(i).dot(df_.col(j));
        }
        beta(ih, ih) += omega0 * omega0;
    }

    const Eigen::VectorXd c = beta.fullPivLu().solve(rhs);

    // u(:, it1) = alpha*df_normalized + (q - qlast)/|df_raw|.
    Eigen::VectorXd u_new = alpha_ * df_new;
    if (df_raw_norm > 0.0) {
        u_new += inv * (dq_out - q_last_);
    }
    u_.col(it1) = u_new;

    // Save charges and deltas, then take the damped step minus the
    // history correction.
    dq_last_ = dq;
    q_last_ = dq_out;
    dq_out = dq_out + alpha_ * dq;
    for (int ih = 0; ih < nhist; ++ih) {
        const int i = hist[static_cast<std::size_t>(ih)];
        dq_out -= omega_(i) * c(ih) * u_.col(i);
    }
}

}  // namespace semiempirical
}  // namespace vibeqc
