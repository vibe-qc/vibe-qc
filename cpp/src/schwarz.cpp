#include "vibeqc/schwarz.hpp"

#include "vibeqc/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc {

std::vector<libint2::Shell> shift_shells_to_cell(
    const libint2::BasisSet& shells, const Eigen::Vector3d& dr) {
    std::vector<libint2::Shell> out(shells.begin(), shells.end());
    for (auto& s : out) {
        s.O[0] += dr[0];
        s.O[1] += dr[1];
        s.O[2] += dr[2];
    }
    return out;
}

std::vector<std::vector<double>> compute_schwarz_factors_per_cell(
    const libint2::BasisSet& shells_ref,
    const std::vector<std::vector<libint2::Shell>>& shells_at,
    const libint2::Engine& prototype) {
    const std::size_t nshells = shells_ref.size();
    const std::size_t n_c = shells_at.size();
    std::vector<std::vector<double>> Q(n_c,
        std::vector<double>(nshells * nshells, 0.0));

    // Parallelise over the flattened (cell, bra-shell) index. Each
    // (c, s_a) owns the disjoint row Q[c][s_a*nshells + *], so the
    // writes never collide and no reduction is needed. Dynamic schedule
    // because per-pair cost varies with angular momentum (the libint
    // block is ≈ (2l+1)⁴ doubles) and distant image cells screen down to
    // near-zero work — a static split would leave near-cell threads
    // grinding while far-cell threads idle. One libint engine per thread
    // from the pool (Engine is not thread-safe). Mirrors the molecular
    // compute_schwarz_factors() below; this per-cell variant had been
    // left serial. Runs on every periodic Fock / gradient build.
    // These integrals are squared norms, not the mixed ERIs screened by
    // the caller. Primitive screening at machine epsilon can erase a norm
    // whose square root still permits nanohartree-sized contributions.
    // Preserve the norms; the caller applies its requested ERI threshold.
    auto bound_prototype = prototype;
    bound_prototype.set_precision(0.0);
    auto engines = make_engine_pool(bound_prototype);
    const long n_work = static_cast<long>(n_c * nshells);
    #pragma omp parallel for schedule(dynamic)
    for (long idx = 0; idx < n_work; ++idx) {
        const std::size_t c = static_cast<std::size_t>(idx) / nshells;
        const std::size_t s_a = static_cast<std::size_t>(idx) % nshells;
        auto& eng = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = eng.results();
        const auto& shells_c = shells_at[c];
        const auto na = shells_ref[s_a].size();
        for (std::size_t s_b = 0; s_b < nshells; ++s_b) {
            const auto nb = shells_c[s_b].size();
            eng.compute(shells_ref[s_a], shells_c[s_b],
                        shells_ref[s_a], shells_c[s_b]);
            const double* blk = buf[0];
            if (!blk) continue;
            double max_diag = 0.0;
            for (std::size_t i = 0; i < na; ++i) {
                for (std::size_t j = 0; j < nb; ++j) {
                    const double v = blk[
                        ((i * nb + j) * na + i) * nb + j];
                    const double a = std::fabs(v);
                    if (a > max_diag) max_diag = a;
                }
            }
            Q[c][s_a * nshells + s_b] = std::sqrt(max_diag);
        }
    }
    return Q;
}

double density_envelope(const std::vector<Eigen::MatrixXd>& blocks) {
    double D_max = 0.0;
    for (const auto& B : blocks) {
        if (B.size() == 0) continue;
        const double mx = std::max(std::fabs(B.maxCoeff()),
                                   std::fabs(B.minCoeff()));
        if (mx > D_max) D_max = mx;
    }
    if (D_max < 1.0) D_max = 1.0;
    return D_max;
}

Eigen::MatrixXd compute_schwarz_factors(
    const libint2::BasisSet& shells,
    const libint2::Engine& prototype) {
    const auto n_shells = static_cast<int>(shells.size());
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n_shells, n_shells);
    // As in the periodic pre-pass, a rounded-to-zero squared norm is not
    // a safe bound on the mixed integrals that the caller will contract.
    auto bound_prototype = prototype;
    bound_prototype.set_precision(0.0);
    auto engines = make_engine_pool(bound_prototype);

    // Upper triangle (s_b ≤ s_a) only — Q is symmetric. Dynamic
    // scheduling because shells with higher angular momentum have
    // larger blocks (≈ (2l+1)² doubles) so per-pair cost varies.
    #pragma omp parallel for schedule(dynamic)
    for (int s_a = 0; s_a < n_shells; ++s_a) {
        auto& eng = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = eng.results();
        const auto na = shells[s_a].size();
        for (int s_b = 0; s_b <= s_a; ++s_b) {
            const auto nb = shells[s_b].size();
            eng.compute(shells[s_a], shells[s_b],
                        shells[s_a], shells[s_b]);
            const double* blk = buf[0];
            if (!blk) continue;
            // Diagonal-only max — the strict Cauchy-Schwarz bound
            // |⟨μν|λσ⟩| ≤ √|⟨μν|μν⟩| · √|⟨λσ|λσ⟩| uses (ij, ij)
            // elements, not arbitrary block elements.
            double max_diag = 0.0;
            for (std::size_t i = 0; i < na; ++i) {
                for (std::size_t j = 0; j < nb; ++j) {
                    const double v = blk[
                        ((i * nb + j) * na + i) * nb + j];
                    const double a = std::fabs(v);
                    if (a > max_diag) max_diag = a;
                }
            }
            const double q = std::sqrt(max_diag);
            Q(s_a, s_b) = q;
            if (s_a != s_b) Q(s_b, s_a) = q;
        }
    }
    return Q;
}

std::vector<double> max_q_per_cell(
    const std::vector<std::vector<double>>& Q) {
    std::vector<double> out(Q.size(), 0.0);
    for (std::size_t c = 0; c < Q.size(); ++c) {
        const auto& Qc = Q[c];
        double mx = 0.0;
        for (double q : Qc) if (q > mx) mx = q;
        out[c] = mx;
    }
    return out;
}

std::vector<double> compute_shell_radial_cutoffs(
    const libint2::BasisSet& shells, double tol) {
    if (tol <= 0.0) {
        throw std::invalid_argument(
            "compute_shell_radial_cutoffs: tol must be positive");
    }
    const std::size_t n = shells.size();
    std::vector<double> cutoffs(n, 0.0);
    const double ln_tol = std::log(tol);

    for (std::size_t s = 0; s < n; ++s) {
        const auto& shell = shells[s];

        // Most-slowly-decaying primitive (smallest α); largest
        // log-amplitude across primitives (libint stores
        // ``ln |c_p · N(l, α_p)|`` in ``max_ln_coeff``).
        double alpha_min = std::numeric_limits<double>::infinity();
        double max_ln_amp = -std::numeric_limits<double>::infinity();
        const std::size_t n_prim = shell.alpha.size();
        for (std::size_t i = 0; i < n_prim; ++i) {
            const double a = shell.alpha[i];
            if (a < alpha_min) alpha_min = a;
            if (i < shell.max_ln_coeff.size()) {
                const double ln_c = shell.max_ln_coeff[i];
                if (ln_c > max_ln_amp) max_ln_amp = ln_c;
            }
        }
        if (max_ln_amp == -std::numeric_limits<double>::infinity()) {
            // Degenerate shell with no max_ln_coeff data (shouldn't
            // happen for well-formed libint shells, but guard).
            cutoffs[s] = 0.0;
            continue;
        }

        // Maximum angular momentum across the shell's contractions.
        // Generally-contracted SP-blocks expose multiple l's; pick the
        // largest for a conservative polynomial-r prefactor.
        int l_max = 0;
        for (const auto& c : shell.contr) {
            if (c.l > l_max) l_max = c.l;
        }

        // Solve   max_ln_amp + l ln(r) - α_min · r²  =  ln(tol)
        // for r > 0 via fixed-point iteration starting from the l=0
        // solution. Each iteration injects the current best estimate
        // of ``l ln r`` into the RHS and resolves; converges in 2–3
        // steps for typical (α, l). Conservative: always rounds the
        // cutoff *up* by checking the inequality.
        const double rhs0 = max_ln_amp - ln_tol;
        if (rhs0 <= 0.0) {
            // Amplitude is below tol even at r=0 — corner case
            // (e.g. very-diffuse + small-coeff shell at very loose
            // tol). Use a tiny cutoff so screening still works.
            cutoffs[s] = 1.0;  // 1 bohr — generous safety margin
            continue;
        }
        double r = std::sqrt(rhs0 / alpha_min);
        for (int iter = 0; iter < 6; ++iter) {
            const double ln_r = std::log(std::max(r, 1e-12));
            const double rhs = max_ln_amp + l_max * ln_r - ln_tol;
            if (rhs <= 0.0) break;  // monotone falloff already < tol
            const double r_new = std::sqrt(rhs / alpha_min);
            if (std::abs(r_new - r) < 1e-6 * (r + 1e-12)) {
                r = r_new;
                break;
            }
            r = r_new;
        }
        // Safety margin (3×) on top of the converged bound. The
        // bound is theoretically conservative (over-estimates the
        // sum-of-primitives amplitude by using max coefficient with
        // slowest decay), but the cumulative K drift over n_pts ~ 10⁵
        // grid points is sensitive to per-pair drops at the bound
        // boundary; empirically a 3× margin keeps RIJCOSX-vs-direct
        // parity within the 1e-3 Ha gate on the H2O/OH/def2-svp suite
        // while the underlying bound (r* ≈ 2-25 bohr) still gives a
        // useful prune for extended systems where atoms are >5 bohr
        // apart from the bulk of the molecule's grid points.
        cutoffs[s] = 3.0 * r;
    }
    return cutoffs;
}

}  // namespace vibeqc
