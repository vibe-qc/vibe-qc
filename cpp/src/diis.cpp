#include "vibeqc/diis.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "vibeqc/diagnostics.hpp"

namespace vibeqc {

namespace {

// A Pulay solve whose coefficients exceed this in magnitude is not a useful
// extrapolation: it signals a linearly dependent history (the bordered system
// is near-singular and the solve has amplified round-off). The caller shrinks
// the window from the oldest end and retries.
constexpr double kMaxPulayCoeff = 1.0e6;

// Concatenate a block list into one column vector. Eigen stores column-major,
// and both directions of the round trip use the same order, so the ordering
// itself is immaterial: only the Frobenius inner product, which is invariant
// under any consistent flattening, has to survive.
Eigen::MatrixXd flatten_blocks(const std::vector<Eigen::MatrixXd>& blocks) {
    Eigen::Index total = 0;
    for (const auto& b : blocks) {
        total += b.size();
    }
    Eigen::MatrixXd flat(total, 1);
    Eigen::Index off = 0;
    for (const auto& b : blocks) {
        flat.block(off, 0, b.size(), 1) =
            Eigen::Map<const Eigen::VectorXd>(b.data(), b.size());
        off += b.size();
    }
    return flat;
}

// Inverse of flatten_blocks, taking the block shapes from ``like``.
std::vector<Eigen::MatrixXd> unflatten_like(
    const Eigen::MatrixXd& flat, const std::vector<Eigen::MatrixXd>& like) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(like.size());
    Eigen::Index off = 0;
    for (const auto& b : like) {
        Eigen::MatrixXd blk(b.rows(), b.cols());
        Eigen::Map<Eigen::VectorXd>(blk.data(), blk.size()) =
            flat.block(off, 0, b.size(), 1);
        out.push_back(std::move(blk));
        off += b.size();
    }
    return out;
}

}  // namespace

DIIS::DIIS(std::size_t max_subspace, DIISDepthPolicy policy,
           double adaptive_param)
    : max_subspace_(max_subspace),
      policy_(policy),
      adaptive_param_(adaptive_param) {
    if (max_subspace_ < 2) {
        throw std::invalid_argument("DIIS subspace size must be >= 2");
    }
    if (policy_ != DIISDepthPolicy::FIXED && adaptive_param_ <= 0.0) {
        throw std::invalid_argument(
            "adaptive DIIS parameter (tau/delta) must be > 0");
    }
}

void DIIS::clear() {
    fock_history_.clear();
    error_history_.clear();
    gram_.resize(0, 0);
}

// Append an iterate and extend the cached Gram matrix by one row/column. The
// n new dot products are the only O(d) work; the previous n² entries are
// reused. ``fock`` and ``error`` are taken by value and moved into the
// history, so a caller with a temporary pays no copy at all.
void DIIS::push_history(Eigen::MatrixXd fock, Eigen::MatrixXd error) {
    if (!error_history_.empty()
        && error.size() != error_history_.back().size()) {
        throw std::invalid_argument(
            "DIIS: error vector length changed between iterations ("
            + std::to_string(error_history_.back().size()) + " -> "
            + std::to_string(error.size())
            + "); the block layout must be stable across pushed iterates");
    }
    if (!fock_history_.empty() && fock.size() != fock_history_.back().size()) {
        throw std::invalid_argument(
            "DIIS: Fock length changed between iterations");
    }

    const auto n = static_cast<Eigen::Index>(error_history_.size());
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(n + 1, n + 1);
    if (n > 0) {
        g.topLeftCorner(n, n) = gram_;
    }
    for (Eigen::Index i = 0; i < n; ++i) {
        const double dot =
            (error_history_[static_cast<std::size_t>(i)].array()
             * error.array()).sum();
        g(i, n) = dot;
        g(n, i) = dot;
    }
    g(n, n) = error.squaredNorm();
    gram_.swap(g);

    fock_history_.push_back(std::move(fock));
    error_history_.push_back(std::move(error));
}

void DIIS::pop_oldest() {
    fock_history_.pop_front();
    error_history_.pop_front();
    const Eigen::Index n = gram_.rows();
    gram_ = gram_.bottomRightCorner(n - 1, n - 1).eval();
}

void DIIS::reset_to_newest() {
    Eigen::MatrixXd f_last = std::move(fock_history_.back());
    Eigen::MatrixXd e_last = std::move(error_history_.back());
    const double g_last = gram_(gram_.rows() - 1, gram_.cols() - 1);
    fock_history_.clear();
    error_history_.clear();
    fock_history_.push_back(std::move(f_last));
    error_history_.push_back(std::move(e_last));
    gram_.resize(1, 1);
    gram_(0, 0) = g_last;
}

// Chupin et al. 2021, Algorithm 3 (R-CDIIS). Called with the newest pushed
// pair already at the back of the history. The restart test builds the
// error differences relative to the oldest stored error and asks whether the
// newest difference is (nearly) in the span of the earlier ones; if so, the
// stored set has become linearly dependent and we restart.
bool DIIS::apply_restart_policy() {
    const std::size_t n = error_history_.size();
    // With < 3 iterates the span of stored differences is empty, so the
    // projection residual equals the full difference and the test
    // τ‖s‖ > ‖s‖ can never fire for τ ∈ (0, 1): nothing to do.
    if (n < 3) {
        return false;
    }

    const Eigen::MatrixXd& oldest = error_history_.front();
    const Eigen::Index d = oldest.size();

    // s = r_new − r_oldest, flattened to a column vector.
    const Eigen::MatrixXd s_mat = error_history_.back() - oldest;
    const Eigen::Map<const Eigen::VectorXd> s(s_mat.data(), d);

    // Basis of stored differences {r_i − r_oldest : i = 1 .. n-2}
    // (all stored errors except the oldest and the newest).
    const Eigen::Index n_basis = static_cast<Eigen::Index>(n) - 2;
    Eigen::MatrixXd B(d, n_basis);
    for (Eigen::Index i = 0; i < n_basis; ++i) {
        const Eigen::MatrixXd diff =
            error_history_[static_cast<std::size_t>(i) + 1] - oldest;
        B.col(i) = Eigen::Map<const Eigen::VectorXd>(diff.data(), d);
    }

    // Orthogonal residual of s against span(B) via a least-squares solve
    // (the QR route used in the reference implementation, §5.1).
    const Eigen::VectorXd coeffs = B.colPivHouseholderQr().solve(s);
    const double resid_norm = (s - B * coeffs).norm();

    if (adaptive_param_ * s.norm() > resid_norm) {
        // Restart: keep only the newest iterate (depth resets to 0, so the
        // next extrapolate returns F unchanged — a plain SCF step).
        reset_to_newest();
        return true;
    }
    return false;
}

// Chupin et al. 2021, Algorithm 4 (AD-CDIIS). Called with the newest pushed
// pair at the back. Retain the largest recent run of older iterates whose
// residual r_i satisfies δ‖r_i‖ < ‖r_new‖; drop everything before the first
// violation (older ⇒ typically larger residual ⇒ dropped as convergence
// proceeds).
void DIIS::apply_adaptive_policy() {
    const std::size_t n = error_history_.size();
    if (n < 2) {
        return;
    }
    // Residual norms come straight off the cached Gram diagonal: ‖e_i‖ =
    // √gram_(i, i). No re-traversal of the stored error vectors.
    const auto norm_at = [this](std::size_t i) {
        return std::sqrt(gram_(static_cast<Eigen::Index>(i),
                               static_cast<Eigen::Index>(i)));
    };
    const double r_new = norm_at(n - 1);

    // Walk backwards over the older iterates (indices n-2 .. 0) and count the
    // contiguous recent run that passes the test. ``keep_old`` older iterates
    // are retained, giving a window of ``keep_old + 1`` including the newest.
    std::size_t keep_old = 0;
    for (std::size_t j = n - 1; j-- > 0;) {  // j = n-2, n-3, ..., 0
        if (adaptive_param_ * norm_at(j) < r_new) {
            ++keep_old;
        } else {
            break;
        }
    }

    const std::size_t drop = (n - 1) - keep_old;  // oldest iterates to remove
    for (std::size_t i = 0; i < drop; ++i) {
        pop_oldest();
    }
}

Eigen::MatrixXd DIIS::extrapolate(const Eigen::MatrixXd& fock,
                                  const Eigen::MatrixXd& error) {
    return extrapolate_owned(fock, error);
}

Eigen::MatrixXd DIIS::extrapolate_owned(Eigen::MatrixXd fock,
                                        Eigen::MatrixXd error) {
    const Eigen::Index rows = fock.rows();
    const Eigen::Index cols = fock.cols();
    push_history(std::move(fock), std::move(error));

    // Adaptive depth management (Chupin et al. 2021). Runs before the FIFO
    // safety cap so the cap only ever acts as an upper bound.
    switch (policy_) {
        case DIISDepthPolicy::RESTART:
            apply_restart_policy();
            break;
        case DIISDepthPolicy::ADAPTIVE:
            apply_adaptive_policy();
            break;
        case DIISDepthPolicy::FIXED:
            break;
    }

    while (fock_history_.size() > max_subspace_) {
        pop_oldest();
    }

    VIBEQC_DIAG("diis", vibeqc::DiagLevel::DEBUG,
        "subspace=%zu  max=%zu  policy=%d",
        fock_history_.size(), max_subspace_, static_cast<int>(policy_));

    // ``fock`` and ``error`` have been moved from; the newest iterate now
    // lives at the back of the history. Everything below reads it from there.
    if (fock_history_.size() < 2) {
        return fock_history_.back();  // Need two points to extrapolate.
    }

    // Solve the bordered Pulay system
    //   [ B_ij  -1 ] [ c ]   [ 0 ]
    //   [  -1    0 ] [ λ ] = [-1 ]
    // with B_ij = Tr(e_i^T e_j) read straight off the cached Gram matrix.
    //
    // A history that has gone linearly dependent makes the bordered system
    // near-singular: the solve then returns non-finite entries, or finite
    // coefficients so large that Σ c_i F_i is dominated by cancellation
    // round-off rather than by the Fock matrices. Both are unusable, and both
    // are cured by dropping the oldest iterate and re-solving on the shorter
    // window. If even a two-iterate window fails, we hand back the raw Fock
    // and let the next SCF cycle rebuild the history: a plain SCF step is a
    // slow move, never a wrong one.
    Eigen::VectorXd sol;
    while (true) {
        const auto m = static_cast<Eigen::Index>(fock_history_.size());
        const double gram_scale = gram_.cwiseAbs().maxCoeff();
        if (!(gram_scale > 0.0) || !std::isfinite(gram_scale)) {
            return fock_history_.back();
        }
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m + 1, m + 1);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(m + 1);
        // Pulay 1982, p. 558, requires an ill-conditioned history to be
        // shortened. First keep the rank test itself scale-independent:
        // replacing B by B/max|B| changes only the Lagrange multiplier, not
        // the constrained coefficients c. Without this exact normalization,
        // a converged residual gives B ~ ||e||^2 ~ 1e-12 beside unit border
        // entries, so FullPivLU's numerical rank depends on residual size.
        // PBE/OH/def2-SVP formerly pinned at 1.084e-6 for 120 cycles here.
        A.topLeftCorner(m, m) = gram_ / gram_scale;
        A.col(m).head(m).setConstant(-1.0);
        A.row(m).head(m).setConstant(-1.0);
        b(m) = -1.0;

        // Full-pivot LU handles mild ill-conditioning without regularisation.
        // A rank-deficient solve can nevertheless return finite minimum-norm
        // coefficients and hide a dependent history, so require the bordered
        // system itself to retain full numerical rank before accepting it.
        Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
        const bool full_rank = lu.rank() == A.rows();
        sol = lu.solve(b);
        const bool usable =
            full_rank && sol.allFinite()
            && sol.head(m).cwiseAbs().maxCoeff() <= kMaxPulayCoeff;
        if (usable) {
            break;
        }
        if (m <= 2) {
            // ``pop_oldest`` only ever drops from the front, so the newest
            // iterate is still at the back.
            return fock_history_.back();
        }
        pop_oldest();
    }

    Eigen::MatrixXd F_extrap = Eigen::MatrixXd::Zero(rows, cols);
    for (std::size_t i = 0; i < fock_history_.size(); ++i) {
        F_extrap += sol(static_cast<Eigen::Index>(i)) * fock_history_[i];
    }
    return F_extrap;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> DIIS::extrapolate_spin_coupled(
    const Eigen::MatrixXd& fock_alpha,
    const Eigen::MatrixXd& fock_beta,
    const Eigen::MatrixXd& error_alpha,
    const Eigen::MatrixXd& error_beta) {
    if (fock_alpha.rows() != fock_beta.rows()
        || fock_alpha.cols() != fock_beta.cols()
        || error_alpha.rows() != error_beta.rows()
        || error_alpha.cols() != error_beta.cols()) {
        throw std::invalid_argument(
            "DIIS::extrapolate_spin_coupled: alpha and beta shapes differ");
    }
    // Stack the spins vertically and reuse the scalar extrapolation: the
    // B matrix then accumulates Tr(e_α_i^T e_α_j) + Tr(e_β_i^T e_β_j) and
    // the single coefficient set applies to both Fock blocks.
    Eigen::MatrixXd F_stack(fock_alpha.rows() + fock_beta.rows(),
                            fock_alpha.cols());
    F_stack.topRows(fock_alpha.rows())    = fock_alpha;
    F_stack.bottomRows(fock_beta.rows())  = fock_beta;
    Eigen::MatrixXd e_stack(error_alpha.rows() + error_beta.rows(),
                            error_alpha.cols());
    e_stack.topRows(error_alpha.rows())   = error_alpha;
    e_stack.bottomRows(error_beta.rows()) = error_beta;
    const Eigen::MatrixXd F_ext =
        extrapolate_owned(std::move(F_stack), std::move(e_stack));
    return {F_ext.topRows(fock_alpha.rows()),
            F_ext.bottomRows(fock_beta.rows())};
}

std::vector<Eigen::MatrixXd> DIIS::extrapolate_blocks(
    const std::vector<Eigen::MatrixXd>& fock_blocks,
    const std::vector<Eigen::MatrixXd>& error_blocks) {
    if (fock_blocks.empty() || error_blocks.empty()) {
        throw std::invalid_argument(
            "DIIS::extrapolate_blocks: block lists must be non-empty");
    }
    // Concatenating the blocks turns the per-block Frobenius sum into a plain
    // Euclidean inner product, so the scalar path above (history, Gram cache,
    // depth policies, Pulay solve) applies verbatim. Nothing here is specific
    // to multi-k: it is the same extrapolation on a longer vector.
    //
    // The two flattened vectors are temporaries and are moved into the
    // history rather than copied: at nbf=120 over 16 k-points each one is
    // ~3.7 MB, and copying both on every SCF cycle was the dominant cost of
    // this call.
    return unflatten_like(
        extrapolate_owned(flatten_blocks(fock_blocks),
                          flatten_blocks(error_blocks)),
        fock_blocks);
}

}  // namespace vibeqc
