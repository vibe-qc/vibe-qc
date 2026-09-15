// Pulay's Direct Inversion of Iterative Subspace (DIIS) — SCF accelerator.
//
// Reference: P. Pulay, Chem. Phys. Lett. 73, 393 (1980); J. Comput. Chem. 3,
// 556 (1982).
//
// Maintains a rolling history of (Fock, error) pairs, and on each call solves
// a small (n+1)-dim linear system for coefficients c_i that minimize the
// norm of Σ c_i e_i subject to Σ c_i = 1. The extrapolated Fock matrix
// Σ c_i F_i is used for the next iteration in place of the raw Fock.
//
// The standard error vector is the commutator e = F D S - S D F in the
// AO basis, which vanishes at the SCF fixed point.
//
// Adaptive depth (Chupin, Dupuy, Legendre & Séré, "Convergence analysis of
// adaptive DIIS algorithms", ESAIM: M2AN 55, 2785 (2021),
// doi:10.1051/m2an/2021069; arXiv:2002.12850): the same commutator-DIIS
// extrapolation, but the history *depth* is chosen adaptively each step
// instead of a fixed FIFO window. Two policies are provided — RESTART
// (their Algorithm 3) and ADAPTIVE (their Algorithm 4). The least-squares
// extrapolation is byte-for-byte the fixed-depth one restricted to the
// retained window; only the set of retained (F, error) pairs differs.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

namespace vibeqc {

// History-depth management policy for the DIIS accelerator.
//
//   * FIXED    — Pulay's classic rolling window: keep the last
//                ``max_subspace`` iterates (FIFO). ``adaptive_param``
//                is ignored.
//   * RESTART  — Chupin et al. 2021, Algorithm 3 ("restarted
//                Anderson–Pulay"). Let the depth grow each step; when
//                the newest error *difference* becomes nearly linearly
//                dependent on the stored ones, restart (drop all history
//                but the last iterate). ``adaptive_param`` is the
//                restart parameter τ ∈ (0, 1): restart is triggered when
//                    τ ‖s‖ > ‖(id − Π) s‖,
//                where s = r_new − r_oldest and Π projects onto the span
//                of the stored error differences. Smaller τ ⇒ larger
//                mean depth. Paper default τ = 1e-4.
//   * ADAPTIVE — Chupin et al. 2021, Algorithm 4 ("adaptive-depth
//                Anderson–Pulay"). Keep the largest recent window of
//                iterates whose residual is not much larger than the
//                current one: retain a stored r_i only while
//                    δ ‖r_i‖ < ‖r_new‖,
//                i.e. discard iterates whose residual exceeds 1/δ times
//                the current residual. ``adaptive_param`` is δ > 0.
//                Smaller δ ⇒ larger mean depth. Paper default δ = 1e-4.
//
// Both adaptive policies still honour ``max_subspace`` as a hard upper
// bound on stored iterates (a memory safety cap; the adaptive rule
// usually trims first).
enum class DIISDepthPolicy {
    FIXED,
    RESTART,
    ADAPTIVE,
};

class DIIS {
public:
    explicit DIIS(std::size_t max_subspace = 8,
                  DIISDepthPolicy policy = DIISDepthPolicy::FIXED,
                  double adaptive_param = 1.0e-4);

    // Extend the history with a new (F, error) pair and return an extrapolated
    // Fock. Before we have at least 2 entries, returns F unchanged.
    Eigen::MatrixXd extrapolate(const Eigen::MatrixXd& fock,
                                const Eigen::MatrixXd& error);

    // Spin-coupled extrapolation for unrestricted SCF: one B matrix built
    // from the vertically stacked (e_α; e_β) error and a single coefficient
    // set applied to the stacked (F_α; F_β) pair. The α and β Fock matrices
    // are coupled through the shared Coulomb term J(D_α + D_β); running two
    // independent per-spin Pulay histories instead lets each extrapolation
    // optimise its own residual against a moving other-spin field, which can
    // stall the tail of the SCF just above a tight gradient threshold
    // (OH/def2-TZVP UHF, 2026-07 regression: residual pinned at ~1e-6..1e-4
    // for 90+ iterations with the energy already flat to 1e-13).
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> extrapolate_spin_coupled(
        const Eigen::MatrixXd& fock_alpha,
        const Eigen::MatrixXd& fock_beta,
        const Eigen::MatrixXd& error_alpha,
        const Eigen::MatrixXd& error_beta);

    // Block-vector extrapolation. ``fock_blocks`` and ``error_blocks`` are
    // each treated as one long vector formed by concatenating the blocks, so
    // the Pulay inner product is the sum of the per-block Frobenius products
    //     B_ij = Σ_b (e_i[b] ⊙ e_j[b]).sum().
    // Returns Σ_i c_i F_i block-by-block, and ``fock_blocks`` unchanged when
    // the history holds fewer than two iterates.
    //
    // The two lists are independent histories, so they need not share a shape:
    // KDIIS passes nbf×nbf Fock blocks alongside n_vir×n_occ orbital-gradient
    // error blocks. The block layout must be stable across pushed iterates.
    //
    // This is the entry point multi-k periodic SCF uses. Each per-k Hermitian
    // matrix M(k) is emitted as two real blocks √w_k·Re M(k) and √w_k·Im M(k)
    // (see ``per_k_to_stacked_real_blocks`` in
    // ``python/vibeqc/periodic_scf_accelerators.py``), which makes the
    // Euclidean form above reproduce the k-weighted DIIS inner product
    //     Σ_k w_k Re Tr[e_i(k)† e_j(k)]
    // exactly. Spin-coupled open-shell histories append the β blocks to the
    // same lists, so one B matrix and one coefficient set drive both spins.
    std::vector<Eigen::MatrixXd> extrapolate_blocks(
        const std::vector<Eigen::MatrixXd>& fock_blocks,
        const std::vector<Eigen::MatrixXd>& error_blocks);

    std::size_t subspace_size() const noexcept { return fock_history_.size(); }
    DIISDepthPolicy policy() const noexcept { return policy_; }
    void clear();

    // Read-only access to the stored (Fock, error) histories.  Used by
    // CPCM macro-iteration warm-starting (BUG 100) — the caller saves
    // these after one inner SCF and passes them as warm_fock / warm_error
    // to the next, so the DIIS subspace carries across Hcore changes.
    const std::deque<Eigen::MatrixXd>& fock_history() const noexcept {
        return fock_history_;
    }
    const std::deque<Eigen::MatrixXd>& error_history() const noexcept {
        return error_history_;
    }

private:
    // Apply the RESTART policy (Chupin et al. Algorithm 3) to the current
    // history, which already includes the newest pushed pair. Returns true
    // when a restart was triggered (all but the newest iterate dropped).
    bool apply_restart_policy();

    // Apply the ADAPTIVE policy (Chupin et al. Algorithm 4): drop the
    // oldest iterates whose residual exceeds 1/δ times the newest one.
    void apply_adaptive_policy();

    // The core extrapolation. Takes its arguments by value so a caller
    // holding a temporary (``extrapolate_blocks`` flattens into one) can move
    // it in rather than paying a second copy of the whole error vector on
    // every SCF cycle.
    Eigen::MatrixXd extrapolate_owned(Eigen::MatrixXd fock,
                                      Eigen::MatrixXd error);

    // The three history mutations. All of them go through these so the
    // cached Gram matrix stays in step with the (F, error) deques.
    void push_history(Eigen::MatrixXd fock, Eigen::MatrixXd error);
    void pop_oldest();
    void reset_to_newest();

    std::size_t max_subspace_;
    DIISDepthPolicy policy_;
    double adaptive_param_;  // τ for RESTART, δ for ADAPTIVE; unused for FIXED
    std::deque<Eigen::MatrixXd> fock_history_;
    std::deque<Eigen::MatrixXd> error_history_;

    // Cached Gram matrix of the stored errors: gram_(i, j) = ⟨e_i, e_j⟩,
    // which is the DIIS B-matrix before bordering. Maintained incrementally:
    // a push costs n fresh dot products instead of the n² a full rebuild
    // would need, and the residual norms the adaptive policies test are just
    // √gram_(i, i). Rows/columns are dropped in step with the deques.
    Eigen::MatrixXd gram_;
};

}  // namespace vibeqc
