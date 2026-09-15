#include "vibeqc/ediis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vibeqc {

namespace {

// Kudin et al. recommend the exact active-face sweep for small histories and
// used five to ten stored matrices in production. Twelve preserves vibe-qc's
// documented history range while bounding the global solve to 2^12 - 1 =
// 4095 faces (at most 4083 small KKT systems) per extrapolation.
// The public ``diis_subspace_size`` is shared with algorithms that can use a
// deeper history, so EDIIS/ADIIS validate this only when first invoked, not in
// their constructors.
constexpr std::size_t kMaxExactSimplexHistory = 12;

// Sum_b <A_b - Aref_b | B_b>. Forming the difference before the product
// avoids subtracting extensive traces after they have already rounded.
double sum_trace_difference(const std::vector<Eigen::MatrixXd>& A,
                            const std::vector<Eigen::MatrixXd>& Aref,
                            const std::vector<Eigen::MatrixXd>& B) {
    double t = 0.0;
    for (std::size_t b = 0; b < A.size(); ++b) {
        t += ((A[b].array() - Aref[b].array()) * B[b].array()).sum();
    }
    return t;
}

// Sum_b <A_b - Aref_b | B_b - Bref_b>, again differencing elementwise
// before multiplication so the simplex-invariant relative objective keeps
// its small physical curvature beside extensive absolute densities/Focks.
double sum_trace_double_difference(
    const std::vector<Eigen::MatrixXd>& A,
    const std::vector<Eigen::MatrixXd>& Aref,
    const std::vector<Eigen::MatrixXd>& B,
    const std::vector<Eigen::MatrixXd>& Bref) {
    double t = 0.0;
    for (std::size_t b = 0; b < A.size(); ++b) {
        t += ((A[b].array() - Aref[b].array())
              * (B[b].array() - Bref[b].array())).sum();
    }
    return t;
}

// Global solver for the simplex-constrained QP:
//
//   minimise   ½ c^T (2 M) c  +  linear^T c
//   subject to Σ c_i = 1,  c_i ≥ 0.
//
// The EDIIS and ADIIS quadratic forms are not necessarily convex:
// exchange can make the energy concave along a density-history direction.
// Solving just the KKT equations on the full/free set can therefore return
// a maximum or saddle point. Kudin, Scuseria & Cances, J. Chem. Phys. 116,
// 8255 (2002), doi:10.1063/1.1470195, Eqs. (15)-(16), prescribe the exact
// small-history remedy used here: solve the equality-constrained problem on
// every non-empty face of the simplex, retain feasible stationary points,
// and return the candidate with the lowest objective. A global minimiser of
// a quadratic over the compact simplex either is a vertex or is stationary
// in the relative interior of its smallest containing face, so this list is
// complete. EDIIS/ADIIS accept at most kMaxExactSimplexHistory iterates,
// keeping the 2^n - 1 face sweep bounded.
std::vector<double> solve_simplex_qp(const Eigen::VectorXd& linear,
                                     const Eigen::MatrixXd& M) {
    const Eigen::Index n = linear.size();
    if (n == 0) return {};
    if (n == 1) return {1.0};
    if (n > static_cast<Eigen::Index>(kMaxExactSimplexHistory)) {
        throw std::logic_error(
            "EDIIS/ADIIS history exceeded exact simplex solver limit");
    }

    // Remove the affine quadratic gauge before solving. On the simplex,
    // adding 1*a^T + a*1^T to M and subtracting 2*a from ``linear`` leaves
    // the objective unchanged. Raw EDIIS traces naturally contain very large
    // terms of exactly this form; if retained, they can dwarf the physical
    // tangent curvature and make a valid face look numerically singular.
    // Choosing the newest vertex r as reference makes row/column r of M0 and
    // coefficient r of l0 zero, and changes the objective only by a constant.
    const Eigen::Index r = n - 1;
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n);
    const Eigen::VectorXd gauge = M.col(r) - 0.5 * M(r, r) * ones;
    const Eigen::MatrixXd M0 =
        M - ones * gauge.transpose() - gauge * ones.transpose();
    Eigen::VectorXd l0 = linear + 2.0 * gauge;
    l0.array() -= l0(r);

    auto objective = [&l0, &M0](const Eigen::VectorXd& c) {
        return c.dot(M0 * c) + l0.dot(c);
    };

    // Start from the newest pure vertex. Besides supplying a guaranteed
    // feasible candidate, this keeps the newest vertex when it shares the
    // exact minimum; the anti-replay guard remains responsible for non-exact
    // pinned-vertex replays.
    Eigen::VectorXd best = Eigen::VectorXd::Zero(n);
    best(n - 1) = 1.0;
    double best_value = objective(best);

    constexpr double kFeasibilityTol = 1.0e-10;
    const std::uint64_t face_end = std::uint64_t{1}
                                   << static_cast<unsigned>(n);
    for (std::uint64_t face = 1; face < face_end; ++face) {
        std::vector<Eigen::Index> free_idx;
        free_idx.reserve(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            if ((face & (std::uint64_t{1} << static_cast<unsigned>(i))) != 0) {
                free_idx.push_back(i);
            }
        }
        const Eigen::Index nf = static_cast<Eigen::Index>(free_idx.size());

        Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
        if (nf == 1) {
            // A vertex needs no KKT solve. In particular, do not let the
            // rank heuristic discard it when an extensive M_ii dwarfs the
            // unit constraint border: vertices guarantee that ``best`` is
            // never worse than every stored iterate.
            c(free_idx.front()) = 1.0;
        } else {
            // Build the face KKT system
            //
            //   [2 M_FF / s, e_F; e_F^T, 0] [c_F; λ]
            //       = [-linear_F / s; 1].
            //
            // Center again within this face. The global newest-reference
            // gauge stabilises objective comparisons, but a face that omits
            // that vertex can retain a large affine offset of its own.
            Eigen::MatrixXd face_M(nf, nf);
            Eigen::VectorXd face_l(nf);
            for (Eigen::Index i = 0; i < nf; ++i) {
                face_l(i) = l0(free_idx[i]);
                for (Eigen::Index j = 0; j < nf; ++j) {
                    face_M(i, j) = M0(free_idx[i], free_idx[j]);
                }
            }
            const Eigen::Index face_r = nf - 1;
            const Eigen::VectorXd face_ones = Eigen::VectorXd::Ones(nf);
            const Eigen::VectorXd face_gauge =
                face_M.col(face_r)
                - 0.5 * face_M(face_r, face_r) * face_ones;
            const Eigen::MatrixXd centered_face_M =
                face_M - face_ones * face_gauge.transpose()
                - face_gauge * face_ones.transpose();
            face_l += 2.0 * face_gauge;
            face_l.array() -= face_l(face_r);
            const Eigen::MatrixXd face_A = 2.0 * centered_face_M;

            // A common positive objective scale leaves the stationary point
            // unchanged and keeps extensive electronic energies commensurate
            // with the unit simplex-constraint border.
            double qp_scale = 1.0;
            for (Eigen::Index i = 0; i < nf; ++i) {
                qp_scale = std::max(qp_scale, std::abs(face_l(i)));
                for (Eigen::Index j = 0; j < nf; ++j) {
                    qp_scale = std::max(
                        qp_scale, std::abs(face_A(i, j)));
                }
            }

            Eigen::MatrixXd Asub =
                Eigen::MatrixXd::Zero(nf + 1, nf + 1);
            Eigen::VectorXd bsub = Eigen::VectorXd::Zero(nf + 1);
            for (Eigen::Index i = 0; i < nf; ++i) {
                for (Eigen::Index j = 0; j < nf; ++j) {
                    Asub(i, j) = face_A(i, j) / qp_scale;
                }
                Asub(i, nf) = 1.0;
                Asub(nf, i) = 1.0;
                bsub(i) = -face_l(i) / qp_scale;
            }
            bsub(nf) = 1.0;

            const auto decomposition = Asub.fullPivLu();
            // If this bordered system is singular, any feasible stationary
            // set is flat along a simplex tangent and reaches a proper
            // boundary face at the same objective. That face is enumerated
            // separately, so skipping the indeterminate solve loses no
            // global minimum.
            if (decomposition.rank() < Asub.rows()) continue;
            const Eigen::VectorXd sol = decomposition.solve(bsub);
            if (!sol.allFinite()) continue;
            const double residual = (Asub * sol - bsub).norm();
            const double residual_scale =
                1.0 + bsub.norm() + Asub.norm() * sol.norm();
            if (residual > 128.0 * std::numeric_limits<double>::epsilon()
                               * residual_scale) {
                continue;
            }

            bool feasible = true;
            double raw_sum = 0.0;
            for (Eigen::Index i = 0; i < nf; ++i) {
                const double ci = sol(i);
                if (ci < -kFeasibilityTol) {
                    feasible = false;
                    break;
                }
                raw_sum += ci;
            }
            if (!feasible || !std::isfinite(raw_sum)
                || std::abs(raw_sum - 1.0) > kFeasibilityTol) {
                continue;
            }
            double sum = 0.0;
            for (Eigen::Index i = 0; i < nf; ++i) {
                const double clipped = std::max(0.0, sol(i));
                c(free_idx[i]) = clipped;
                sum += clipped;
            }
            if (!(sum > 0.0) || !std::isfinite(sum)) continue;
            c /= sum;
        }

        const double value = objective(c);
        if (!std::isfinite(value)) continue;
        // Strict comparison preserves the pre-seeded newest vertex for exact
        // ties without allowing an unrelated extensive offset to widen a
        // heuristic tie band over a genuinely lower face.
        if (value < best_value) {
            best = std::move(c);
            best_value = value;
        }
    }

    std::vector<double> coeffs(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        coeffs[static_cast<std::size_t>(i)] = best(i);
    }
    return coeffs;
}

// Σ_i c_i F_i, block-by-block. With n < 2 stored iterates the QP is
// trivial (c = {1}) and this reduces to a copy of the newest Fock.
std::vector<Eigen::MatrixXd> combine_blocks(
    const std::deque<std::vector<Eigen::MatrixXd>>& fock_history,
    const std::vector<double>& coeffs) {
    const auto& newest = fock_history.back();
    std::vector<Eigen::MatrixXd> F_extrap(newest.size());
    for (std::size_t b = 0; b < newest.size(); ++b) {
        F_extrap[b] = Eigen::MatrixXd::Zero(newest[b].rows(),
                                            newest[b].cols());
    }
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        const double c = coeffs[i];
        if (c == 0.0) continue;
        const auto& Fi = fock_history[i];
        for (std::size_t b = 0; b < F_extrap.size(); ++b) {
            F_extrap[b].noalias() += c * Fi[b];
        }
    }
    return F_extrap;
}

// Anti-replay guard thresholds. A pinned-vertex replay is bit-exact in
// practice (deterministic Fock build from an identical density), so the
// relative tolerance only needs to absorb non-associative reduction
// noise; at 1e-12 relative the two candidate steps are numerically the
// same step anyway. The coefficient floor treats "the newest iterate
// got (numerically) zero simplex weight" as the semantic trigger.
constexpr double kReplayNewestWeightFloor = 1e-10;
constexpr double kReplayRelTol = 1e-12;

// True when two block stacks agree to ``rel_tol`` in Frobenius norm
// (scale set by the larger stack; layout mismatch => false).
bool blocks_replay_equal(const std::vector<Eigen::MatrixXd>& a,
                         const std::vector<Eigen::MatrixXd>& b,
                         double rel_tol) {
    if (a.size() != b.size()) return false;
    double diff2 = 0.0;
    double scale2 = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].rows() != b[i].rows() || a[i].cols() != b[i].cols()) {
            return false;
        }
        diff2 += (a[i] - b[i]).squaredNorm();
        scale2 = std::max({scale2, a[i].squaredNorm(), b[i].squaredNorm()});
    }
    const double scale = std::max(1.0, std::sqrt(scale2));
    return std::sqrt(diff2) <= rel_tol * scale;
}

// Index of the dominant *old* iterate (excludes the newest, back()).
std::size_t dominant_old_vertex(const std::vector<double>& coeffs) {
    std::size_t pinned = 0;
    for (std::size_t i = 1; i + 1 < coeffs.size(); ++i) {
        if (coeffs[i] > coeffs[pinned]) pinned = i;
    }
    return pinned;
}

}  // namespace

EDIIS::EDIIS(std::size_t max_subspace) : max_subspace_(max_subspace) {
    if (max_subspace_ < 2) {
        throw std::invalid_argument("EDIIS subspace size must be >= 2");
    }
}

void EDIIS::clear() {
    fock_history_.clear();
    density_history_.clear();
    energy_history_.clear();
    last_coeffs_.clear();
    last_extrap_.clear();
    replay_guard_erasures_ = 0;
}

void EDIIS::push_iterate(std::vector<Eigen::MatrixXd>&& fock_blocks,
                         std::vector<Eigen::MatrixXd>&& density_blocks,
                         double energy) {
    if (max_subspace_ > kMaxExactSimplexHistory) {
        throw std::invalid_argument(
            "EDIIS max_subspace must be <= 12 for exact simplex "
            "minimisation");
    }
    if (fock_blocks.size() != density_blocks.size()) {
        throw std::invalid_argument(
            "EDIIS::push_iterate: fock_blocks and density_blocks must "
            "have matching block counts");
    }
    for (std::size_t b = 0; b < fock_blocks.size(); ++b) {
        if (fock_blocks[b].rows() != density_blocks[b].rows() ||
            fock_blocks[b].cols() != density_blocks[b].cols()) {
            throw std::invalid_argument(
                "EDIIS::push_iterate: per-block fock and density "
                "dimensions must match");
        }
    }
    if (!fock_history_.empty()) {
        const auto& prev = fock_history_.back();
        if (prev.size() != fock_blocks.size()) {
            throw std::invalid_argument(
                "EDIIS::push_iterate: block count differs from existing "
                "history (call clear() to reset before changing layout)");
        }
        for (std::size_t b = 0; b < fock_blocks.size(); ++b) {
            if (prev[b].rows() != fock_blocks[b].rows() ||
                prev[b].cols() != fock_blocks[b].cols()) {
                throw std::invalid_argument(
                    "EDIIS::push_iterate: per-block dimensions differ "
                    "from existing history (call clear() to reset)");
            }
        }
    }
    fock_history_.push_back(std::move(fock_blocks));
    density_history_.push_back(std::move(density_blocks));
    energy_history_.push_back(energy);
    while (energy_history_.size() > max_subspace_) {
        fock_history_.pop_front();
        density_history_.pop_front();
        energy_history_.pop_front();
    }
}

// EDIIS energy model — Kudin, Scuseria & Cancès, J. Chem. Phys. 116,
// 8255 (2002), doi:10.1063/1.1470195, Eqs. (7)-(10).
//
// The defining property is Eq. (7): the objective must EQUAL the true
// energy of the hull density, f(c) = E(Σ_i c_i D_i). Eq. (8) states it as
//
//   f^EDIIS(c) = Σ_i c_i E_i − ½ Σ_{ij} c_i c_j Tr((F_i − F_j)(D_i − D_j))
//
// but that ½ is NOT convention-free. Section II of the paper fixes the
// closed-shell PAIR-density convention: D counts electron pairs
// (Tr(SD) = N pairs), E(X) = 2Tr(hX) + Tr(G(X)X) and F(X) = h + G(X), so
// there F = ½ ∂E/∂D. vibe-qc stores the TOTAL (or per-spin) density with
//
//   E = ½ ⟨D | H + F⟩   and   F = ∂E/∂D                             (γ = 1)
//
// at every call site — rhf.cpp/rks.cpp pass D_used with
// E_elec = ½ tr(D(Hcore+F)); uhf.cpp/uks.cpp and the Python Roothaan
// driver pass the (D_α, D_β) pair; periodic_scf.cpp passes the real-space
// cells with E_elec = ½ Σ_g (P(g) ⊙ [H(g)+F(g)]).sum(). Writing
// E(D) = l(D) + q(D,D) with F = γ ∇E, the hull identity is
//
//   E(Σ c_i D_i) = Σ_i c_i E_i
//                  − (1/4γ) Σ_{ij} c_i c_j ⟨F_i − F_j | D_i − D_j⟩
//
// so the paper's ½ is the γ = ½ special case and vibe-qc needs ¼. With
// s_ij = ⟨F_i | D_j⟩ (block-summed) and M_ij = ½(s_ij + s_ji), doubling
// (which leaves the minimiser unchanged) maps it onto solve_simplex_qp's
// "minimise cᵀ M c + linearᵀ c" as
//
//   linear_i = 2 E_i − s_ii,    M_ij = ½ (s_ij + s_ji).
//
// Using the paper's literal `E_i − s_ii` against γ = 1 Focks does not
// merely rescale: it turns the linear term into MINUS the two-electron
// energy where it should be the one-electron energy (E_i − s_ii
// = −E_2e,i + E_nuc, whereas 2E_i − s_ii = ⟨H|D_i⟩ + 2E_nuc). The
// resulting objective equals 2E(D̃) − Σ_i c_i E_i: exact at the pure
// vertices, but wrong everywhere in the interior of the hull — measured
// at up to 0.33 Ha on NH₂/STO-3G ROHF iterates and 10.8 Ha on H₂O/STO-3G
// closed-shell iterates. It is convex-upward relative to the true energy,
// which biases the simplex minimiser onto pure old vertices (the pinned-
// vertex freeze the anti-replay guard below was written to escape) and,
// on NH₂/STO-3G ROHF, drove the DEFAULT ediis_diis route into the excited
// ²A₁ basin, converging at −54.7393101908 Ha instead of −54.8344963786 Ha
// with no warning (issue #119).
//
// For numerical stability the implementation below assembles the equivalent
// newest-relative gauge directly from dF_i = F_i - F_r and
// dD_i = D_i - D_r, rather than forming extensive absolute traces and then
// subtracting them. The closed- and open-shell convention regressions are
// ``test_ediis_convex_hull_uses_vibeqc_total_density_energy_convention`` and
// ``test_ediis_open_shell_hull_uses_the_same_energy_convention`` in
// tests/test_ediis.py.
std::vector<double> EDIIS::solve_qp() const {
    const auto n = static_cast<Eigen::Index>(energy_history_.size());
    if (n == 0) return {};
    if (n == 1) return {1.0};

    const Eigen::Index ref = n - 1;
    const auto& F_ref = fock_history_[ref];
    const auto& D_ref = density_history_[ref];
    const double E_ref = energy_history_[ref];

    Eigen::VectorXd linear(n);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        linear(i) =
            2.0 * (energy_history_[i] - E_ref)
            - sum_trace_double_difference(
                fock_history_[i], F_ref, density_history_[i], D_ref);
    }
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j <= i; ++j) {
            const double a = sum_trace_double_difference(
                fock_history_[i], F_ref, density_history_[j], D_ref);
            const double b = sum_trace_double_difference(
                fock_history_[j], F_ref, density_history_[i], D_ref);
            const double mij = 0.5 * (a + b);
            M(i, j) = mij;
            M(j, i) = mij;
        }
    }
    return solve_simplex_qp(linear, M);
}

// Solve the QP, combine, and apply the anti-replay guard.
//
// The simplex minimiser may put zero weight on the newest iterate and
// return a hull point identical to the previous cycle's return —
// typically a pinned low-energy vertex such as the initial-guess
// iterate (the QP objective at a pure vertex equals that iterate's
// energy E_i, so a guess whose energy undercuts the first Roothaan
// images wins every re-solve). The SCF is deterministic: diagonalising
// the same Fock reproduces the same density, the next cycle pushes a
// duplicate (F, D, E) iterate, and the QP re-selects the same vertex.
// The SCF then freezes (dE = 0 bit-exactly) until the FIFO evicts the
// pinned entry max_subspace-1 cycles later. Observed: c-diamond
// fcc-primitive/STO-3G KRHF-GDF kmesh (2,2,2) froze for 8 cycles once
// EDIIS_DIIS became the periodic default (2026-07-09).
//
// A pinned iterate is fully exploited — its Roothaan image is already
// in the history — so revisiting it carries no new information: drop
// it and re-solve. That is the same eviction the FIFO would perform
// max_subspace-1 cycles later, minus the dead cycles. The guard never
// fires while the extrapolation carries new information (nonzero
// weight on the newest iterate, or a hull point that differs from the
// previous return), so converging trajectories are untouched.
//
// PREMISE — last_extrap_ must hold the previously CONSUMED return, not
// merely the previously produced one. The replay deadlock exists only
// when the SCF actually diagonalised the returned Fock: that is what
// regenerates the duplicate iterate the QP then re-pins. The EDIIS_DIIS
// hybrid calls both branches every cycle and keeps one, so an EDIIS Fock
// produced on a DIIS-branch cycle never reached the SCF, and reproducing
// it later is a FIRST use. Conflating the two made the guard erase live
// history mid-run and fork molecular RKS onto a worse basin (n-octane
// PBE/def2-SVP landed ~10 mHa high from v0.15.31 on; see
// .agents/agentic-loop-state/blockers/
// ediis-antireplay-guard-molecular-basin-regression.md). Every call site
// that drops a return therefore calls discard_last_extrapolation().
std::vector<Eigen::MatrixXd> EDIIS::finish_extrapolation() {
    last_coeffs_ = solve_qp();
    std::vector<Eigen::MatrixXd> F_extrap =
        combine_blocks(fock_history_, last_coeffs_);
    while (energy_history_.size() >= 2
           && last_coeffs_.back() <= kReplayNewestWeightFloor
           && !last_extrap_.empty()
           && blocks_replay_equal(F_extrap, last_extrap_, kReplayRelTol)) {
        const std::size_t pinned = dominant_old_vertex(last_coeffs_);
        fock_history_.erase(fock_history_.begin()
                            + static_cast<std::ptrdiff_t>(pinned));
        density_history_.erase(density_history_.begin()
                               + static_cast<std::ptrdiff_t>(pinned));
        energy_history_.erase(energy_history_.begin()
                              + static_cast<std::ptrdiff_t>(pinned));
        ++replay_guard_erasures_;
        last_coeffs_ = solve_qp();
        F_extrap = combine_blocks(fock_history_, last_coeffs_);
    }
    last_extrap_ = F_extrap;
    return F_extrap;
}

// Σ_i c_i D_i with the coefficients of the most recent return, over the
// history as it stands NOW -- i.e. after any guard erasure above. This is
// the density the returned Fock belongs to (Σ_i c_i F(D_i) == F(Σ_i c_i
// D_i) at the HF level), and the only density a Roothaan-coupled driver
// may project it with (GitLab #487).
std::vector<Eigen::MatrixXd> EDIIS::last_extrapolated_density() const {
    if (last_coeffs_.empty()) return {};
    return combine_blocks(density_history_, last_coeffs_);
}

Eigen::MatrixXd EDIIS::extrapolate(const Eigen::MatrixXd& fock,
                                   const Eigen::MatrixXd& density,
                                   double energy) {
    push_iterate({fock}, {density}, energy);
    return finish_extrapolation()[0];
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
EDIIS::extrapolate(const Eigen::MatrixXd& fock_alpha,
                   const Eigen::MatrixXd& fock_beta,
                   const Eigen::MatrixXd& density_alpha,
                   const Eigen::MatrixXd& density_beta,
                   double energy) {
    push_iterate({fock_alpha, fock_beta},
                 {density_alpha, density_beta}, energy);
    std::vector<Eigen::MatrixXd> F_extrap = finish_extrapolation();
    return {std::move(F_extrap[0]), std::move(F_extrap[1])};
}

std::vector<Eigen::MatrixXd>
EDIIS::extrapolate(const std::vector<Eigen::MatrixXd>& fock_blocks,
                   const std::vector<Eigen::MatrixXd>& density_blocks,
                   double energy) {
    // Copy into rvalue storage for push_iterate (it consumes by && and
    // also validates block-layout consistency against history).
    std::vector<Eigen::MatrixXd> F_copy = fock_blocks;
    std::vector<Eigen::MatrixXd> D_copy = density_blocks;
    push_iterate(std::move(F_copy), std::move(D_copy), energy);
    return finish_extrapolation();
}

// ---------------------------------------------------------------------------
// ADIIS — augmented-Roothaan-Hall DIIS (Hu & Yang 2010). Shares the
// simplex-constrained QP solver and relative block-trace helpers with EDIIS.
// ---------------------------------------------------------------------------

ADIIS::ADIIS(std::size_t max_subspace) : max_subspace_(max_subspace) {
    if (max_subspace_ < 2) {
        throw std::invalid_argument("ADIIS subspace size must be >= 2");
    }
}

void ADIIS::clear() {
    fock_history_.clear();
    density_history_.clear();
    last_coeffs_.clear();
    last_extrap_.clear();
    replay_guard_erasures_ = 0;
}

void ADIIS::push_iterate(std::vector<Eigen::MatrixXd>&& fock_blocks,
                         std::vector<Eigen::MatrixXd>&& density_blocks) {
    if (max_subspace_ > kMaxExactSimplexHistory) {
        throw std::invalid_argument(
            "ADIIS max_subspace must be <= 12 for exact simplex "
            "minimisation");
    }
    if (fock_blocks.size() != density_blocks.size()) {
        throw std::invalid_argument(
            "ADIIS::push_iterate: fock_blocks and density_blocks must "
            "have matching block counts");
    }
    for (std::size_t b = 0; b < fock_blocks.size(); ++b) {
        if (fock_blocks[b].rows() != density_blocks[b].rows() ||
            fock_blocks[b].cols() != density_blocks[b].cols()) {
            throw std::invalid_argument(
                "ADIIS::push_iterate: per-block fock and density "
                "dimensions must match");
        }
    }
    if (!fock_history_.empty()) {
        const auto& prev = fock_history_.back();
        if (prev.size() != fock_blocks.size()) {
            throw std::invalid_argument(
                "ADIIS::push_iterate: block count differs from existing "
                "history (call clear() to reset before changing layout)");
        }
        for (std::size_t b = 0; b < fock_blocks.size(); ++b) {
            if (prev[b].rows() != fock_blocks[b].rows() ||
                prev[b].cols() != fock_blocks[b].cols()) {
                throw std::invalid_argument(
                    "ADIIS::push_iterate: per-block dimensions differ "
                    "from existing history (call clear() to reset)");
            }
        }
    }
    fock_history_.push_back(std::move(fock_blocks));
    density_history_.push_back(std::move(density_blocks));
    while (fock_history_.size() > max_subspace_) {
        fock_history_.pop_front();
        density_history_.pop_front();
    }
}

// Build and solve the ADIIS QP. The augmented-Roothaan-Hall objective,
// expanded about the most recent iterate n = back():
//
//   f(c) = 2 Σ_i c_i ⟨D_i − D_n | F_n⟩
//        +   Σ_{ij} c_i c_j ⟨D_i − D_n | F_j − F_n⟩
//
// maps onto solve_simplex_qp's  minimise cᵀ M c + linearᵀ c  with
//   linear_i = 2 ⟨D_i - D_n|F_n⟩
//   M_{ij}   = ½ (⟨d_i|f_j⟩ + ⟨d_j|f_i⟩)
// All differences are formed elementwise before the block-summed traces;
// subtracting four extensive absolute traces loses the small ARH curvature.
std::vector<double> ADIIS::solve_qp() const {
    const auto n = static_cast<Eigen::Index>(fock_history_.size());
    if (n == 0) return {};
    if (n == 1) return {1.0};

    const Eigen::Index ref = n - 1;   // most recent iterate is the
                                      // ARH expansion point.
    const auto& F_n = fock_history_[ref];
    const auto& D_n = density_history_[ref];
    Eigen::VectorXd linear(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        linear(i) = 2.0 * sum_trace_difference(
            density_history_[i], D_n, F_n);
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j <= i; ++j) {
            const double di_fj = sum_trace_double_difference(
                density_history_[i], D_n, fock_history_[j], F_n);
            const double dj_fi = sum_trace_double_difference(
                density_history_[j], D_n, fock_history_[i], F_n);
            const double mij = 0.5 * (di_fj + dj_fi);
            M(i, j) = mij;
            M(j, i) = mij;
        }
    }
    return solve_simplex_qp(linear, M);
}

// Solve + combine + anti-replay guard. Same deadlock mechanism and
// remedy as EDIIS::finish_extrapolation (see the note there): the ARH
// simplex minimiser can also pin a pure old vertex (the reference
// iterate carries zero linear and quadratic weight, but an old vertex
// with a sufficiently negative model value can still take the whole
// simplex), and a returned Fock identical to the previous cycle's
// return replays a step the deterministic SCF has already taken.
std::vector<Eigen::MatrixXd> ADIIS::finish_extrapolation() {
    last_coeffs_ = solve_qp();
    std::vector<Eigen::MatrixXd> F_extrap =
        combine_blocks(fock_history_, last_coeffs_);
    while (fock_history_.size() >= 2
           && last_coeffs_.back() <= kReplayNewestWeightFloor
           && !last_extrap_.empty()
           && blocks_replay_equal(F_extrap, last_extrap_, kReplayRelTol)) {
        const std::size_t pinned = dominant_old_vertex(last_coeffs_);
        fock_history_.erase(fock_history_.begin()
                            + static_cast<std::ptrdiff_t>(pinned));
        density_history_.erase(density_history_.begin()
                               + static_cast<std::ptrdiff_t>(pinned));
        ++replay_guard_erasures_;
        last_coeffs_ = solve_qp();
        F_extrap = combine_blocks(fock_history_, last_coeffs_);
    }
    last_extrap_ = F_extrap;
    return F_extrap;
}

// ADIIS sibling of EDIIS::last_extrapolated_density (GitLab #487).
std::vector<Eigen::MatrixXd> ADIIS::last_extrapolated_density() const {
    if (last_coeffs_.empty()) return {};
    return combine_blocks(density_history_, last_coeffs_);
}

Eigen::MatrixXd ADIIS::extrapolate(const Eigen::MatrixXd& fock,
                                   const Eigen::MatrixXd& density) {
    push_iterate({fock}, {density});
    return finish_extrapolation()[0];
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
ADIIS::extrapolate(const Eigen::MatrixXd& fock_alpha,
                   const Eigen::MatrixXd& fock_beta,
                   const Eigen::MatrixXd& density_alpha,
                   const Eigen::MatrixXd& density_beta) {
    push_iterate({fock_alpha, fock_beta},
                 {density_alpha, density_beta});
    std::vector<Eigen::MatrixXd> F_extrap = finish_extrapolation();
    return {std::move(F_extrap[0]), std::move(F_extrap[1])};
}

std::vector<Eigen::MatrixXd>
ADIIS::extrapolate(const std::vector<Eigen::MatrixXd>& fock_blocks,
                   const std::vector<Eigen::MatrixXd>& density_blocks) {
    // push_iterate validates block-layout consistency against history.
    std::vector<Eigen::MatrixXd> F_copy = fock_blocks;
    std::vector<Eigen::MatrixXd> D_copy = density_blocks;
    push_iterate(std::move(F_copy), std::move(D_copy));
    return finish_extrapolation();
}

}  // namespace vibeqc
