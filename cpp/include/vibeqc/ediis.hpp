// EDIIS — energy-DIIS extrapolator (Kudin / Scuseria / Cancès 2002).
//
// References:
//   * K. N. Kudin, G. E. Scuseria, E. Cancès, "A black-box self-consistent
//     field convergence algorithm: One step closer", J. Chem. Phys. 116,
//     8255 (2002).
//   * A. J. Garza, G. E. Scuseria, "Comparison of self-consistent field
//     convergence acceleration techniques", J. Chem. Phys. 137, 054110
//     (2012). [Identifies EDIIS+DIIS as the production default.]
//
// Where Pulay's DIIS minimises the orbital-gradient ‖e‖_F (commutator
// e = F D S − S D F), EDIIS minimises a quadratic *energy* functional
// on the convex hull of stored (F_i, D_i) pairs:
//
//   E(c) = Σ_i c_i E_i  −  ¼ Σ_{ij} c_i c_j Tr[(D_i − D_j) (F_i − F_j)]
//
// subject to Σ_i c_i = 1 and c_i ≥ 0  (positive simplex constraints —
// distinct from DIIS, which allows arbitrary signs). At convergence
// the trace term vanishes and EDIIS reproduces the converged density;
// far from convergence EDIIS dominates DIIS — it is provably convergent
// on cases where Pulay's DIIS oscillates.
//
// Storage. Each iterate stores a list of (F, D) blocks plus a scalar
// energy. The single-block case is closed-shell (RHF / RKS); the
// two-block case is open-shell (UHF / UKS) with the trace cross-term
// summed over both spins:
//   Σ_b ½ [Tr(F_{i,b} D_{j,b}) + Tr(F_{j,b} D_{i,b})]
//
// The ¼ factor is for vibe-qc's total/per-spin density convention where
// F = dE/dD; Kudin et al.'s Eq. (8) uses ½ with a pair density and
// F = ½ dE/dD. See cpp/src/ediis.cpp for the convention derivation.
//
// QP solver. Exchange can make this quadratic concave or indefinite, so a
// convex active-set KKT solve is insufficient. Following Kudin et al.
// Eqs. (15)-(16), the global solver visits every non-empty simplex face,
// solves its equality-constrained stationary equations, retains feasible
// candidates, and selects the lowest objective. EDIIS/ADIIS use is therefore
// limited to max_subspace <= 12 (default 8, and the paper used 5-10); deeper
// values remain available to the DIIS-family algorithms that do not use this
// exponential exact solve.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <deque>
#include <cmath>
#include <utility>
#include <vector>

#include "vibeqc/diis.hpp"

namespace vibeqc {

// Selector for the SCF Fock-extrapolation accelerator. Used by
// RHFOptions / UHFOptions / RKSOptions / UKSOptions::scf_accelerator.
//
//   * DIIS         — Pulay's commutator DIIS (the historical default).
//                    Error vector: e = F D S − S D F (AO basis).
//   * KDIIS        — Kollmar 1997 (ORCA's opt-in ``!KDIIS`` keyword).
//                    Same Pulay machinery, different error metric:
//                    g_{ai} = F^MO_{ai} (occ-vir block of F in MO basis).
//                    Often robust where DIIS oscillates on transition-
//                    metal / open-shell cases. See vibeqc/kdiis.hpp.
//   * EDIIS        — energy-DIIS (Kudin/Scuseria/Cancès 2002). More
//                    robust far from convergence; reproduces DIIS at
//                    convergence (the cross term vanishes).
//   * EDIIS_DIIS   — Garza/Scuseria 2012 hybrid. Use EDIIS while
//                    ‖e‖_F > ediis_diis_switch_threshold, then switch
//                    to DIIS for the asymptotic regime.
//   * ADIIS        — augmented-Roothaan-Hall DIIS (Hu & Yang 2010).
//                    Minimises the ARH energy model expanded about the
//                    most recent iterate; same positive-simplex
//                    constraints and QP solver as EDIIS. Garza/Scuseria
//                    2012 found ADIIS and EDIIS near-identical at the
//                    HF level — ADIIS is offered as the sibling option.
//   * ADIIS_DIIS   — ADIIS/DIIS hybrid, the ADIIS analogue of EDIIS_DIIS.
//                    Use ADIIS while ‖e‖_F > ediis_diis_switch_threshold
//                    (robust far from convergence), then switch to DIIS
//                    for the asymptotic regime (which ADIIS, like EDIIS,
//                    cannot reach past the convex hull under its
//                    positive-simplex constraint).
//   * R_CDIIS      — restarted commutator-DIIS (Chupin, Dupuy, Legendre
//                    & Séré, ESAIM: M2AN 55, 2785 (2021)). Same DIIS
//                    extrapolation, but the depth grows until the stored
//                    error differences become nearly linearly dependent,
//                    then the history restarts. Restart aggressiveness is
//                    set by ``diis_restart_tau`` (τ ∈ (0,1); default 1e-4).
//   * AD_CDIIS     — adaptive-depth commutator-DIIS (Chupin et al. 2021).
//                    Same DIIS extrapolation, but the window keeps only
//                    recent iterates whose residual is within 1/δ of the
//                    current one, so the depth shrinks near convergence.
//                    Controlled by ``diis_adaptive_delta`` (δ > 0;
//                    default 1e-4).
enum class SCFAccelerator {
    DIIS,
    KDIIS,
    EDIIS,
    EDIIS_DIIS,
    ADIIS,
    ADIIS_DIIS,
    R_CDIIS,
    AD_CDIIS,
};

inline double ediis_diis_switch_metric(double commutator_norm,
                                       Eigen::Index matrix_dim,
                                       int n_blocks = 1) {
    // The public threshold follows the Garza/Scuseria/PySCF convention and is
    // meant to be intensive. The raw Frobenius commutator norm scales with the
    // AO matrix dimension (and sqrt(number of spin/block matrices)), so compare
    // against an RMS matrix-element norm instead.
    double denom = std::sqrt(static_cast<double>(std::max(1, n_blocks)))
                 * static_cast<double>(matrix_dim);
    if (denom < 1.0) denom = 1.0;
    return commutator_norm / denom;
}

class EDIIS {
public:
    // ``max_subspace`` is validated on first use rather than construction
    // because SCF drivers create all accelerator objects up front. EDIIS use
    // accepts 2..12; a larger shared DIIS history remains valid when EDIIS is
    // not the selected accelerator.
    explicit EDIIS(std::size_t max_subspace = 8);

    // Closed-shell. Stores (F, D, energy) and returns Σ_i c_i F_i. Before
    // the second iterate is appended (n < 2) the input F is returned
    // unchanged.
    Eigen::MatrixXd extrapolate(const Eigen::MatrixXd& fock,
                                const Eigen::MatrixXd& density,
                                double energy);

    // Open-shell. Stores (F_α, F_β, D_α, D_β, energy) and returns the
    // pair (Σ_i c_i F_α,i, Σ_i c_i F_β,i). The same coefficient set
    // applies to both spins — the energy functional couples them via
    // the system energy E_i and the per-spin trace cross-term.
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    extrapolate(const Eigen::MatrixXd& fock_alpha,
                const Eigen::MatrixXd& fock_beta,
                const Eigen::MatrixXd& density_alpha,
                const Eigen::MatrixXd& density_beta,
                double energy);

    // Block-vector generalisation. Stores an (F, D, energy) iterate
    // whose Fock and density are each a vector of nbf × nbf blocks and
    // returns Σ_i c_i F_i block-by-block. The QP cross-term
    // ⟨F_i | D_j⟩ = Σ_b (F_i[b] ⊙ D_j[b]).sum() automatically sums over
    // all blocks, which lets the same kernel cover (a) closed-shell
    // (one block), (b) open-shell α+β (two blocks, the existing UHF
    // overload's contract), and (c) multi-k periodic SCF, where the
    // blocks are the real-space cells of a LatticeMatrixSet and the
    // sum is exactly the periodic energy bilinear form
    //   E_elec = ½ Σ_g (P(g) ⊙ [H(g) + F(g)]).sum().
    // Block layout (count, ordering, dimensions) must match across
    // pushed iterates. Returns ``fock_blocks`` unchanged when n < 2.
    std::vector<Eigen::MatrixXd>
    extrapolate(const std::vector<Eigen::MatrixXd>& fock_blocks,
                const std::vector<Eigen::MatrixXd>& density_blocks,
                double energy);

    // Coefficients from the most recent extrapolate call. Empty before
    // the first call. Useful for testing + trace logging.
    const std::vector<double>& last_coeffs() const noexcept {
        return last_coeffs_;
    }

    std::size_t subspace_size() const noexcept {
        return energy_history_.size();
    }
    // The density whose Fock the most recent extrapolate() call returned:
    // D_tilde[b] = Σ_i c_i D_i[b] per block, combined with the SAME
    // coefficients as the Fock and over the history that SURVIVED the
    // anti-replay guard (finish_extrapolation may erase an interior
    // entry and re-solve, so the coefficient vector need not index the
    // sequence of pushes).
    //
    // At the HF level J and K are linear in D and Σ_i c_i = 1 carries
    // the one-electron term through, so Σ_i c_i F(D_i) == F(D_tilde)
    // exactly: the returned Fock IS the Fock of this density. A driver
    // that couples the extrapolated Fock to density-derived projectors
    // -- the ROHF/ROKS Roothaan effective Fock in python/vibeqc/rohf.py
    // -- must therefore project with D_tilde, not with the current
    // iterate's density. Before this accessor existed the Python driver
    // mirrored the density history itself and could not see the guard's
    // erase(); one guard fire left the mirror one entry too long and
    // the driver silently fell back to the current density, i.e. an
    // inconsistent effective Fock (GitLab #487). Empty before the first
    // extrapolate() call.
    std::vector<Eigen::MatrixXd> last_extrapolated_density() const;
    // Number of history entries the anti-replay guard has erased since
    // construction / clear(). Diagnostic: lets a driver or a test assert
    // that the guard fired (or did not) on a given trajectory.
    std::size_t replay_guard_erasures() const noexcept {
        return replay_guard_erasures_;
    }

    // Tell the accelerator that the Fock returned by the most recent
    // extrapolate() call was NOT handed to the SCF.
    //
    // The anti-replay guard asks "did the SCF already take this step?".
    // It answers that by comparing against last_extrap_, which alone
    // records only what was PRODUCED. The EDIIS_DIIS hybrid calls both
    // branches every cycle and keeps one, so a produced Fock is not
    // necessarily a consumed one: a discarded extrapolate() result was
    // never diagonalised, and reproducing it on a later cycle is a
    // FIRST use, not a replay. Callers that drop a result must say so
    // here, which re-arms the guard from the next genuine consumption.
    void discard_last_extrapolation() noexcept { last_extrap_.clear(); }

    void clear();

private:
    void push_iterate(std::vector<Eigen::MatrixXd>&& fock_blocks,
                      std::vector<Eigen::MatrixXd>&& density_blocks,
                      double energy);
    std::vector<double> solve_qp() const;
    // Solve + combine + anti-replay guard. See the guard note in
    // cpp/src/ediis.cpp: a QP solution with zero weight on the newest
    // iterate that bit-reproduces the previous return would make the
    // deterministic SCF replay a step it has already taken.
    std::vector<Eigen::MatrixXd> finish_extrapolation();

    std::size_t max_subspace_;
    std::deque<std::vector<Eigen::MatrixXd>> fock_history_;
    std::deque<std::vector<Eigen::MatrixXd>> density_history_;
    std::deque<double> energy_history_;
    std::vector<double> last_coeffs_;
    // Extrapolated Fock returned by the previous extrapolate() call
    // (consumed by the anti-replay guard). Reset by clear().
    std::vector<Eigen::MatrixXd> last_extrap_;
    // Anti-replay guard erasures so far (see replay_guard_erasures()).
    std::size_t replay_guard_erasures_ = 0;
};

// ADIIS — augmented-Roothaan-Hall DIIS (Hu & Yang, J. Chem. Phys. 132,
// 054109 (2010)).
//
// Like EDIIS, ADIIS extrapolates on the convex hull of stored (F_i, D_i)
// pairs under positive-simplex constraints (Σ c_i = 1, c_i ≥ 0) and uses
// the *same* exact active-face QP solver. It differs in the objective: where
// EDIIS minimises an energy functional symmetric in all stored
// iterates, ADIIS minimises the augmented-Roothaan-Hall energy model
// expanded about the **most recent** iterate n:
//
//   f(c) = 2 Σ_i c_i ⟨D_i − D_n | F_n⟩
//        +   Σ_{ij} c_i c_j ⟨D_i − D_n | F_j − F_n⟩
//
// with ⟨A|B⟩ = Tr(A B) summed over spin blocks for the open-shell case.
// The reference iterate n carries zero linear and quadratic weight, so
// it absorbs whatever coefficient the simplex constraint leaves over.
//
// ADIIS needs no per-iterate energy — only the (F, D) history — so the
// ``extrapolate`` signatures omit the ``energy`` argument that EDIIS
// takes. Garza & Scuseria (J. Chem. Phys. 137, 054110 (2012)) showed
// ADIIS and EDIIS converge near-identically at the HF level; ADIIS
// ships as the sibling option (roadmap D1b).
class ADIIS {
public:
    explicit ADIIS(std::size_t max_subspace = 8);

    // Closed-shell. Stores (F, D) and returns Σ_i c_i F_i. Before the
    // second iterate is appended (n < 2) the input F is returned
    // unchanged.
    Eigen::MatrixXd extrapolate(const Eigen::MatrixXd& fock,
                                const Eigen::MatrixXd& density);

    // Open-shell. Stores (F_α, F_β, D_α, D_β) and returns the pair
    // (Σ_i c_i F_α,i, Σ_i c_i F_β,i). One coefficient set across both
    // spins; the ARH trace cross-term sums over the two spin blocks.
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    extrapolate(const Eigen::MatrixXd& fock_alpha,
                const Eigen::MatrixXd& fock_beta,
                const Eigen::MatrixXd& density_alpha,
                const Eigen::MatrixXd& density_beta);

    // Block-vector generalisation. Stores an (F, D) iterate whose Fock
    // and density are each a vector of nbf × nbf blocks and returns
    // Σ_i c_i F_i block-by-block. The ARH trace cross-term
    // ⟨D_i − D_n | F_j − F_n⟩ = Σ_b ((D_i − D_n)[b] ⊙ (F_j − F_n)[b]).sum()
    // sums over all blocks, so the same kernel covers closed-shell
    // (one block), open-shell α+β (two blocks), and multi-k periodic
    // SCF (real-space cells of a LatticeMatrixSet). Block layout
    // must match across pushed iterates. Returns ``fock_blocks``
    // unchanged when n < 2.
    std::vector<Eigen::MatrixXd>
    extrapolate(const std::vector<Eigen::MatrixXd>& fock_blocks,
                const std::vector<Eigen::MatrixXd>& density_blocks);

    // Coefficients from the most recent extrapolate call.
    const std::vector<double>& last_coeffs() const noexcept {
        return last_coeffs_;
    }

    std::size_t subspace_size() const noexcept {
        return fock_history_.size();
    }
    // ADIIS sibling of EDIIS::last_extrapolated_density -- see the note
    // there. Same contract: Σ_i c_i D_i over the surviving history, with
    // the coefficients of the most recent extrapolate() return (#487).
    std::vector<Eigen::MatrixXd> last_extrapolated_density() const;
    // ADIIS sibling of EDIIS::replay_guard_erasures.
    std::size_t replay_guard_erasures() const noexcept {
        return replay_guard_erasures_;
    }

    // ADIIS sibling of EDIIS::discard_last_extrapolation — see the note
    // there. The ADIIS_DIIS hybrid has the same produced-vs-consumed
    // distinction as EDIIS_DIIS.
    void discard_last_extrapolation() noexcept { last_extrap_.clear(); }

    void clear();

private:
    void push_iterate(std::vector<Eigen::MatrixXd>&& fock_blocks,
                      std::vector<Eigen::MatrixXd>&& density_blocks);
    std::vector<double> solve_qp() const;
    // Solve + combine + anti-replay guard (same rationale as EDIIS;
    // see the guard note in cpp/src/ediis.cpp).
    std::vector<Eigen::MatrixXd> finish_extrapolation();

    std::size_t max_subspace_;
    std::deque<std::vector<Eigen::MatrixXd>> fock_history_;
    std::deque<std::vector<Eigen::MatrixXd>> density_history_;
    std::vector<double> last_coeffs_;
    // Extrapolated Fock returned by the previous extrapolate() call
    // (consumed by the anti-replay guard). Reset by clear().
    std::vector<Eigen::MatrixXd> last_extrap_;
    // Anti-replay guard erasures so far (see replay_guard_erasures()).
    std::size_t replay_guard_erasures_ = 0;
};

// Construct a commutator-DIIS accelerator whose depth policy is taken from an
// SCF options struct (any of RHFOptions / UHFOptions / RKSOptions /
// UKSOptions / periodic variants — all carry ``scf_accelerator``,
// ``diis_subspace_size``, ``diis_restart_tau`` and ``diis_adaptive_delta``).
// SCFAccelerator::R_CDIIS selects the restart policy (Chupin et al. 2021,
// Algorithm 3) with τ = ``diis_restart_tau``; AD_CDIIS selects the
// adaptive-depth policy (Algorithm 4) with δ = ``diis_adaptive_delta``.
// Every other accelerator (including the plain DIIS used inside the
// EDIIS_DIIS hybrid) yields a fixed-depth history.
template <typename OptsT>
inline DIIS make_diis(const OptsT& opts) {
    DIISDepthPolicy policy = DIISDepthPolicy::FIXED;
    double param = 1.0e-4;
    if (opts.scf_accelerator == SCFAccelerator::R_CDIIS) {
        policy = DIISDepthPolicy::RESTART;
        param = opts.diis_restart_tau;
    } else if (opts.scf_accelerator == SCFAccelerator::AD_CDIIS) {
        policy = DIISDepthPolicy::ADAPTIVE;
        param = opts.diis_adaptive_delta;
    }
    return DIIS(opts.diis_subspace_size, policy, param);
}

}  // namespace vibeqc
