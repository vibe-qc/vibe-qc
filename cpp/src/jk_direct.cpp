#include "vibeqc/jk_direct.hpp"

#include "vibeqc/init.hpp"
#include "vibeqc/schwarz.hpp"
#include "vibeqc/thread_pool.hpp"

#include <libint2.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace vibeqc {

namespace {

// Per-pair density envelope
//
//   Dpair(s, t) = max |D_μν|  for μ ∈ s, ν ∈ t
//
// Strictly tighter than the per-shell envelope ``Dshell[s] = max_t
// Dpair(s, t)`` (which collapses the pair index to a single shell);
// we read off ``Dpair(s, t)`` for the exact shell-pair block.
// The per-quartet bound for a quartet (s1, s2, s3, s4) is the max
// over the 6 shell-pair Dpair entries that appear in the J + K
// density contractions:
//
//   J reads D(s3, s4) and D(s1, s2)
//   K reads D(s1, s3), D(s1, s4), D(s2, s3), D(s2, s4)
//
// Same memory cost as a copy of Q (n_shells² doubles); free relative
// to libint compute.
//
// Audit (2026-05-18) — no 1.0 floor. The pre-audit version initialised
// ``Dpair`` at ``1.0`` and only wrote ``Dpair(s, t) = mx`` when
// ``mx > 1.0``. The stated rationale was iter-1 robustness (tiny
// guess densities). But the SCF's two-phase Schwarz already handles
// iter-1 (``schwarz_threshold_tighten_at`` opens loose, then tightens),
// and the floor *broke* the incremental-Fock ΔD path: the entire
// point of the ΔD screen in ``DirectJKBuilder::build_g_rhf`` is to
// drop quartets when ``|ΔD|`` is small in the converged region, and
// the 1.0 floor masked exactly those values. Removing the floor lets
// the incremental path tighten as designed.
inline Eigen::MatrixXd density_envelope_per_pair(
    const libint2::BasisSet& shells,
    const Eigen::MatrixXd& D) {
    const auto shell2bf = shells.shell2bf();
    const int n_shells = static_cast<int>(shells.size());
    Eigen::MatrixXd Dpair = Eigen::MatrixXd::Zero(n_shells, n_shells);
    if (D.size() == 0) return Dpair;
    for (int s = 0; s < n_shells; ++s) {
        const auto bf_s = shell2bf[s];
        const auto n_s = shells[s].size();
        for (int t = 0; t <= s; ++t) {
            const auto bf_t = shell2bf[t];
            const auto n_t = shells[t].size();
            double mx = 0.0;
            for (std::size_t i = 0; i < n_s; ++i) {
                for (std::size_t j = 0; j < n_t; ++j) {
                    const double v = std::fabs(
                        D(static_cast<Eigen::Index>(bf_s + i),
                          static_cast<Eigen::Index>(bf_t + j)));
                    if (v > mx) mx = v;
                }
            }
            Dpair(s, t) = mx;
            if (s != t) Dpair(t, s) = mx;
        }
    }
    return Dpair;
}

inline libint2::Engine make_eri_prototype(const libint2::BasisSet& shells) {
    return libint2::Engine(libint2::Operator::coulomb,
                           shells.max_nprim(),
                           shells.max_l(), 0);
}

// erf-attenuated ERI prototype — kernel erf(ω·r₁₂)/r₁₂, used by the
// range-separated-hybrid K build. ω is libint's single scalar
// attenuation parameter (ω = 0 → zero potential, ω → ∞ → bare 1/r).
inline libint2::Engine make_erf_eri_prototype(const libint2::BasisSet& shells,
                                              double omega) {
    libint2::Engine e(libint2::Operator::erf_coulomb,
                      shells.max_nprim(), shells.max_l(), 0);
    e.set_params(omega);
    return e;
}

// K-sorted shell-pair list (ORCA's IPAIR / IPAIRMAX) — see the public
// helper ``build_q_sorted_pairs`` in jk_direct.hpp. Enumerates all
// (s_i, s_j) with s_i ≥ s_j, sorted by descending Q. Lets the
// quartet visitor turn its inner ``continue`` into a ``break`` —
// once we've fallen past the threshold along the Q-descending walk,
// every entry below it is also dropped.
//
// This is the structural difference between ORCA's direct-SCF
// screening (precomputed sort that turns the loop bounds into the
// screen) and a naïve filter-and-skip (every quartet pays the
// test cost). For converged regions on extended systems the win is
// 5-30× in inner-loop iteration count.

// 8-fold-symmetric shell-quartet visitor (rectangular inner loop +
// KPMAXT outer early-out, ORCA audit alignment).
//
// Outer enumeration: parallel for over the K-sorted pair list, so the
// outer (s1, s2) pairs are visited in descending Q order. This makes
// the outer-loop early-out (``q12 * q_max * Dmax_global < thr``,
// ORCA's KPMAXT) fire on a contiguous suffix of low-Q outer pairs.
//
// Inner enumeration kept as the original rectangular triangle
//   for (s3 = 0..s1) for (s4 = 0.. (s1==s3 ? s2 : s3))
// because in benchmark profiling the K-sorted inner walk + lex filter
// adds ~2× iteration count per outer (the break trigger
// ``q12 * q34 * Dshell_max < thr`` is too conservative to compensate
// in the typical SCF regime — see the audit follow-up comment in
// CHANGELOG). The KPMAXT outer-out and the per-shell density
// envelope are the actually-measured wins.
//
// `accumulate` signature unchanged from the prior pre-sort version.
// ``erf_omega`` selects the two-electron kernel: ``erf_omega <= 0`` →
// the bare Coulomb operator 1/r₁₂ (the default for J / K / G builds);
// ``erf_omega > 0`` → the erf-attenuated operator erf(ω·r₁₂)/r₁₂ used
// by the range-separated-hybrid long-range exchange. The Schwarz
// factors ``Q`` are the Coulomb ones either way — since
// erf(ω·r)/r ≤ 1/r pointwise, the Coulomb Q is a valid (conservative)
// screening bound for the erf integrals too.
template <typename Accumulate>
void for_each_unique_quartet(const libint2::BasisSet& shells,
                             const Eigen::MatrixXd& Q,
                             const std::vector<DirectShellPair>& sorted_pairs,
                             const Eigen::MatrixXd& Dpair,
                             double schwarz_threshold,
                             Accumulate&& accumulate,
                             double erf_omega = 0.0) {
    const auto shell2bf = shells.shell2bf();
    auto engines = make_engine_pool(
        erf_omega > 0.0 ? make_erf_eri_prototype(shells, erf_omega)
                        : make_eri_prototype(shells));
    const bool screen = (schwarz_threshold > 0.0);
    const int n_pairs = static_cast<int>(sorted_pairs.size());

    // KPMAX (ORCA's KPMAXT): largest q[s,t]·Dpair[s,t] over all
    // shell pairs. Per-pair bound is strictly tighter than the
    // ``q_max · Dshell_max`` form (which separates the maxes) when
    // the density is localized — a high-Q pair (diagonal shell)
    // and a high-Dpair pair (also diagonal) may not coincide, so
    // their product max < the product of their separate maxes.
    double KP_max = 0.0;
    if (screen) {
        for (const auto& p : sorted_pairs) {
            const double kp = p.q * Dpair(p.s1, p.s2);
            if (kp > KP_max) KP_max = kp;
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int idx12 = 0; idx12 < n_pairs; ++idx12) {
        const auto& outer = sorted_pairs[idx12];
        const int s1 = outer.s1;
        const int s2 = outer.s2;
        const double q12 = outer.q;
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const auto bf2 = shell2bf[s2];
        const auto n2 = shells[s2].size();
        const double s12_deg = (s1 == s2) ? 1.0 : 2.0;

        // Outer-loop early-out (ORCA's KPMAXT test, per-pair-tight).
        // The largest possible inner contribution from any (s3, s4)
        // is q12 · KP_max — if that fails the threshold no inner
        // pair can survive. Outer pairs are iterated in descending
        // Q so this kills a contiguous low-Q suffix.
        if (screen && q12 * KP_max < schwarz_threshold) {
            continue;
        }

        auto& engine = engines[
            static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();

        // Inner enumeration: rectangular orbit-representative
        // triangle (libint2 HF example pattern). Per-quartet
        // continue on the per-pair density envelope — the tightest
        // bound covers all 6 shell-pair Dpair entries that appear
        // in the J + K density contractions.
        for (int s3 = 0; s3 <= s1; ++s3) {
            const auto bf3 = shell2bf[s3];
            const auto n3 = shells[s3].size();
            const int s4_max = (s1 == s3) ? s2 : s3;

            for (int s4 = 0; s4 <= s4_max; ++s4) {
                const auto bf4 = shell2bf[s4];
                const auto n4 = shells[s4].size();
                if (screen) {
                    const double q34 = Q(s3, s4);
                    // 6 Dpair lookups: 2 for J (D[s3,s4], D[s1,s2])
                    // + 4 for K (D[s1,s3], D[s2,s4], D[s1,s4],
                    // D[s2,s3]). Symmetric Dpair lets us read each
                    // pair from a single side.
                    const double D_bound = std::max({
                        Dpair(s1, s2), Dpair(s3, s4),
                        Dpair(s1, s3), Dpair(s2, s4),
                        Dpair(s1, s4), Dpair(s2, s3)});
                    if (q12 * q34 * D_bound < schwarz_threshold) continue;
                }
                engine.compute(shells[s1], shells[s2],
                               shells[s3], shells[s4]);
                const double* block = buf[0];
                if (!block) continue;

                const double s34_deg = (s3 == s4) ? 1.0 : 2.0;
                const double s12_34_deg = (s1 == s3)
                    ? ((s2 == s4) ? 1.0 : 2.0) : 2.0;
                const double deg = s12_deg * s34_deg * s12_34_deg;

                accumulate(s1, s2, s3, s4, deg,
                           bf1, bf2, bf3, bf4,
                           n1, n2, n3, n4, block);
            }
        }
    }
}

}  // namespace

std::vector<DirectShellPair> build_q_sorted_pairs(
    const BasisSet& basis, const Eigen::MatrixXd& Q) {
    const auto& shells = basis.libint();
    const int n_shells = static_cast<int>(shells.size());
    std::vector<DirectShellPair> pairs;
    pairs.reserve(static_cast<std::size_t>(n_shells) * (n_shells + 1) / 2);
    for (int s1 = 0; s1 < n_shells; ++s1) {
        for (int s2 = 0; s2 <= s1; ++s2) {
            pairs.push_back({s1, s2, Q(s1, s2)});
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const DirectShellPair& a, const DirectShellPair& b) {
                  return a.q > b.q;
              });
    return pairs;
}

Eigen::MatrixXd direct_compute_j(const BasisSet& basis,
                                 const Eigen::MatrixXd& D,
                                 const Eigen::MatrixXd& Q,
                                 const std::vector<DirectShellPair>& sorted_pairs,
                                 double schwarz_threshold) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error(
            "direct_compute_j: density shape mismatch");
    }
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> J_tls(
        n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
    const auto Dpair = density_envelope_per_pair(shells, D);

    for_each_unique_quartet(
        shells, Q, sorted_pairs, Dpair, schwarz_threshold,
        [&](int /*s1*/, int /*s2*/, int /*s3*/, int /*s4*/,
            double deg,
            std::size_t bf1, std::size_t bf2,
            std::size_t bf3, std::size_t bf4,
            std::size_t n1, std::size_t n2,
            std::size_t n3, std::size_t n4,
            const double* block) {
            auto& Jm = J_tls[static_cast<std::size_t>(omp_thread_index())];
            for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t j = 0; j < n2; ++j)
            for (std::size_t k = 0; k < n3; ++k)
            for (std::size_t l = 0; l < n4; ++l) {
                const double value =
                    block[((i * n2 + j) * n3 + k) * n4 + l] * deg;
                // Coefficient ½ on each J line + final symmetrization
                // ½·(J + Jᵀ) cancels the orbit-vs-permutation
                // double-count for vibe-qc's spin-density convention.
                Jm(bf1 + i, bf2 + j) += 0.5 * D(bf3 + k, bf4 + l) * value;
                Jm(bf3 + k, bf4 + l) += 0.5 * D(bf1 + i, bf2 + j) * value;
            }
        });

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nbf, nbf);
    for (const auto& m : J_tls) J += m;
    return 0.5 * (J + J.transpose());
}

Eigen::MatrixXd direct_compute_k(const BasisSet& basis,
                                 const Eigen::MatrixXd& D,
                                 const Eigen::MatrixXd& Q,
                                 const std::vector<DirectShellPair>& sorted_pairs,
                                 double schwarz_threshold) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error(
            "direct_compute_k: density shape mismatch");
    }
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> K_tls(
        n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
    const auto Dpair = density_envelope_per_pair(shells, D);

    for_each_unique_quartet(
        shells, Q, sorted_pairs, Dpair, schwarz_threshold,
        [&](int /*s1*/, int /*s2*/, int /*s3*/, int /*s4*/,
            double deg,
            std::size_t bf1, std::size_t bf2,
            std::size_t bf3, std::size_t bf4,
            std::size_t n1, std::size_t n2,
            std::size_t n3, std::size_t n4,
            const double* block) {
            auto& Km = K_tls[static_cast<std::size_t>(omp_thread_index())];
            for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t j = 0; j < n2; ++j)
            for (std::size_t k = 0; k < n3; ++k)
            for (std::size_t l = 0; l < n4; ++l) {
                const double value =
                    block[((i * n2 + j) * n3 + k) * n4 + l] * deg;
                // Four K cells per quartet permutation orbit; quarter
                // coefficient + ½·(K + Kᵀ) symmetrization at the end
                // reproduces K(D)_{μν} = Σ_λσ D_λσ (μλ|νσ).
                Km(bf1 + i, bf3 + k) += 0.25 * D(bf2 + j, bf4 + l) * value;
                Km(bf2 + j, bf4 + l) += 0.25 * D(bf1 + i, bf3 + k) * value;
                Km(bf1 + i, bf4 + l) += 0.25 * D(bf2 + j, bf3 + k) * value;
                Km(bf2 + j, bf3 + k) += 0.25 * D(bf1 + i, bf4 + l) * value;
            }
        });

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nbf, nbf);
    for (const auto& m : K_tls) K += m;
    return 0.5 * (K + K.transpose());
}

Eigen::MatrixXd direct_compute_k_erf(const BasisSet& basis,
                                     const Eigen::MatrixXd& D,
                                     const Eigen::MatrixXd& Q,
                                     const std::vector<DirectShellPair>& sorted_pairs,
                                     double schwarz_threshold,
                                     double omega) {
    ensure_libint_initialized();
    if (omega <= 0.0) {
        throw std::runtime_error(
            "direct_compute_k_erf: omega must be > 0 (the erf "
            "attenuation parameter of a range-separated hybrid)");
    }
    const auto& shells = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error(
            "direct_compute_k_erf: density shape mismatch");
    }
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> K_tls(
        n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
    const auto Dpair = density_envelope_per_pair(shells, D);

    // Identical contraction to ``direct_compute_k`` — the only change
    // is the erf-attenuated two-electron kernel selected by passing
    // ``erf_omega = omega`` to the quartet visitor.
    for_each_unique_quartet(
        shells, Q, sorted_pairs, Dpair, schwarz_threshold,
        [&](int /*s1*/, int /*s2*/, int /*s3*/, int /*s4*/,
            double deg,
            std::size_t bf1, std::size_t bf2,
            std::size_t bf3, std::size_t bf4,
            std::size_t n1, std::size_t n2,
            std::size_t n3, std::size_t n4,
            const double* block) {
            auto& Km = K_tls[static_cast<std::size_t>(omp_thread_index())];
            for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t j = 0; j < n2; ++j)
            for (std::size_t k = 0; k < n3; ++k)
            for (std::size_t l = 0; l < n4; ++l) {
                const double value =
                    block[((i * n2 + j) * n3 + k) * n4 + l] * deg;
                Km(bf1 + i, bf3 + k) += 0.25 * D(bf2 + j, bf4 + l) * value;
                Km(bf2 + j, bf4 + l) += 0.25 * D(bf1 + i, bf3 + k) * value;
                Km(bf1 + i, bf4 + l) += 0.25 * D(bf2 + j, bf3 + k) * value;
                Km(bf2 + j, bf3 + k) += 0.25 * D(bf1 + i, bf4 + l) * value;
            }
        },
        /*erf_omega=*/omega);

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nbf, nbf);
    for (const auto& m : K_tls) K += m;
    return 0.5 * (K + K.transpose());
}

Eigen::MatrixXd direct_compute_g_rhf(const BasisSet& basis,
                                     const Eigen::MatrixXd& D,
                                     const Eigen::MatrixXd& Q,
                                     const std::vector<DirectShellPair>& sorted_pairs,
                                     double schwarz_threshold,
                                     double alpha_hf) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error(
            "direct_compute_g_rhf: density shape mismatch");
    }
    // α_HF = 0 (pure functionals) — skip the K accumulation entirely.
    // direct_compute_j already does just the J piece, no K work.
    if (alpha_hf == 0.0) return direct_compute_j(basis, D, Q, sorted_pairs, schwarz_threshold);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> G_tls(
        n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
    const auto Dpair = density_envelope_per_pair(shells, D);
    const double k_coeff = 0.5 * alpha_hf * 0.25;  // −½·α_HF on K → −0.125·α_HF

    for_each_unique_quartet(
        shells, Q, sorted_pairs, Dpair, schwarz_threshold,
        [&](int /*s1*/, int /*s2*/, int /*s3*/, int /*s4*/,
            double deg,
            std::size_t bf1, std::size_t bf2,
            std::size_t bf3, std::size_t bf4,
            std::size_t n1, std::size_t n2,
            std::size_t n3, std::size_t n4,
            const double* block) {
            auto& Gm = G_tls[static_cast<std::size_t>(omp_thread_index())];
            for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t j = 0; j < n2; ++j)
            for (std::size_t k = 0; k < n3; ++k)
            for (std::size_t l = 0; l < n4; ++l) {
                const double value =
                    block[((i * n2 + j) * n3 + k) * n4 + l] * deg;
                // J piece.
                Gm(bf1 + i, bf2 + j) += 0.5 * D(bf3 + k, bf4 + l) * value;
                Gm(bf3 + k, bf4 + l) += 0.5 * D(bf1 + i, bf2 + j) * value;
                // K piece (subtracted, weighted by ½·α_HF).
                Gm(bf1 + i, bf3 + k) -= k_coeff * D(bf2 + j, bf4 + l) * value;
                Gm(bf2 + j, bf4 + l) -= k_coeff * D(bf1 + i, bf3 + k) * value;
                Gm(bf1 + i, bf4 + l) -= k_coeff * D(bf2 + j, bf3 + k) * value;
                Gm(bf2 + j, bf3 + k) -= k_coeff * D(bf1 + i, bf4 + l) * value;
            }
        });

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nbf, nbf);
    for (const auto& m : G_tls) G += m;
    return 0.5 * (G + G.transpose());
}

}  // namespace vibeqc
