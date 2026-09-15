// Direct determinant CAS-CI: string tables, σ builds, Davidson, RDMs.
// Algorithm + conventions documented in include/vibeqc/casci.hpp.

#include "vibeqc/casci.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibeqc {

namespace {

// ── Occupation strings ──────────────────────────────────────────────────

// All C(norb, nelec) occupation masks in itertools.combinations
// (lexicographic) order — matching solvers._determinant.generate_determinants.
std::vector<std::uint64_t> make_strings(int norb, int nelec) {
    std::vector<std::uint64_t> out;
    if (nelec < 0 || nelec > norb) return out;
    std::vector<int> occ(nelec);
    for (int i = 0; i < nelec; ++i) occ[i] = i;
    while (true) {
        std::uint64_t m = 0;
        for (int i : occ) m |= (std::uint64_t(1) << i);
        out.push_back(m);
        // next lexicographic combination
        int i = nelec - 1;
        while (i >= 0 && occ[i] == norb - nelec + i) --i;
        if (i < 0) break;
        ++occ[i];
        for (int j = i + 1; j < nelec; ++j) occ[j] = occ[j - 1] + 1;
    }
    if (nelec == 0) out.assign(1, 0);  // the single empty string
    return out;
}

// One single-excitation table entry: a†_p a_q |str_from> = sign |str_to>.
struct Exc {
    std::uint32_t pair;  // p * n_act + q
    std::int32_t to;     // target string index
    double sign;
};

// Per-spin-sector Jordan-Wigner phases (bits below the target within this
// sector's string) — matching solvers._rdm._apply_aq_adp_spin.
inline int phase_below(std::uint64_t mask, int orb) {
    const std::uint64_t below = mask & ((std::uint64_t(1) << orb) - 1);
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(below) & 1 ? -1 : 1;
#else
    int c = 0;
    for (std::uint64_t b = below; b; b &= b - 1) ++c;
    return (c & 1) ? -1 : 1;
#endif
}

// Excitation table for every string of one spin sector.  offsets[i] ..
// offsets[i+1] delimit string i's entries in ``entries``.
struct ExcTable {
    std::vector<Exc> entries;
    std::vector<std::size_t> offsets;
    int n_per_string = 0;
};

ExcTable make_exc_table(const std::vector<std::uint64_t>& strings, int norb) {
    std::unordered_map<std::uint64_t, std::int32_t> index;
    index.reserve(strings.size() * 2);
    for (std::size_t i = 0; i < strings.size(); ++i)
        index.emplace(strings[i], static_cast<std::int32_t>(i));

    ExcTable t;
    t.offsets.resize(strings.size() + 1, 0);
    for (std::size_t i = 0; i < strings.size(); ++i) {
        const std::uint64_t s = strings[i];
        std::size_t count = 0;
        for (int q = 0; q < norb; ++q) {
            if (!((s >> q) & 1)) continue;
            for (int p = 0; p < norb; ++p) {
                if (p != q && ((s >> p) & 1)) continue;  // p occupied (p≠q)
                ++count;
            }
        }
        t.offsets[i + 1] = t.offsets[i] + count;
    }
    t.entries.resize(t.offsets.back());
    for (std::size_t i = 0; i < strings.size(); ++i) {
        const std::uint64_t s = strings[i];
        std::size_t k = t.offsets[i];
        for (int q = 0; q < norb; ++q) {
            if (!((s >> q) & 1)) continue;
            // annihilate q
            const int sign_q = phase_below(s, q);
            const std::uint64_t s1 = s & ~(std::uint64_t(1) << q);
            for (int p = 0; p < norb; ++p) {
                if (p != q && ((s >> p) & 1)) continue;
                const int sign = sign_q * phase_below(s1, p);
                const std::uint64_t s2 = s1 | (std::uint64_t(1) << p);
                t.entries[k++] = Exc{static_cast<std::uint32_t>(p * norb + q),
                                     index.at(s2), static_cast<double>(sign)};
            }
        }
    }
    return t;
}

// Shared per-problem context.
struct CIContext {
    int n_act = 0;
    int n_alpha = 0;
    int n_beta = 0;
    long long na = 0, nb = 0, ndet = 0;
    std::vector<std::uint64_t> astr, bstr;
    ExcTable atab, btab;
};

CIContext make_context(int n_act, int n_alpha, int n_beta) {
    if (n_act < 1 || n_act > 31)
        throw std::invalid_argument("casci_direct: n_act must be in [1, 31]");
    if (n_alpha < 0 || n_beta < 0 || n_alpha > n_act || n_beta > n_act)
        throw std::invalid_argument("casci_direct: bad electron counts");
    CIContext c;
    c.n_act = n_act;
    c.n_alpha = n_alpha;
    c.n_beta = n_beta;
    c.astr = make_strings(n_act, n_alpha);
    c.bstr = make_strings(n_act, n_beta);
    c.na = static_cast<long long>(c.astr.size());
    c.nb = static_cast<long long>(c.bstr.size());
    c.ndet = c.na * c.nb;
    c.atab = make_exc_table(c.astr, n_act);
    c.btab = make_exc_table(c.bstr, n_act);
    return c;
}

// ── σ = H·c (chunked gather → GEMM → scatter; see header) ──────────────

void sigma_into(const CIContext& ctx,
                const Eigen::MatrixXd& h1p,    // h'_pq (modified 1e)
                const Eigen::MatrixXd& eri_m,  // (pq|rs) as npair × npair
                const double* c,
                double* sigma,
                int chunk) {
    const int n = ctx.n_act;
    const int npair = n * n;
    const long long ndet = ctx.ndet;
    const long long nb = ctx.nb;
    std::fill(sigma, sigma + ndet, 0.0);

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    // Per-thread σ accumulators (scatter targets are global).
    std::vector<std::vector<double>> sig_local(
        nthreads, std::vector<double>(static_cast<std::size_t>(ndet), 0.0));

    const long long nchunks = (ndet + chunk - 1) / chunk;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        Eigen::MatrixXd D(npair, chunk), F(npair, chunk);
        double* sl = sig_local[tid].data();

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long long ic = 0; ic < nchunks; ++ic) {
            const long long c0 = ic * chunk;
            const long long c1 = std::min(ndet, c0 + chunk);
            const int w = static_cast<int>(c1 - c0);
            D.leftCols(w).setZero();

            // Gather D[(q,p), j] = (Ê_qp c)(I) via <I|Ê_qp|T> = sign of the
            // table entry a†_p a_q |I> = sign |T>.
            for (long long I = c0; I < c1; ++I) {
                const long long Ia = I / nb, Ib = I % nb;
                const int j = static_cast<int>(I - c0);
                double* Dj = D.data() + static_cast<std::size_t>(j) * npair;
                for (std::size_t k = ctx.atab.offsets[Ia];
                     k < ctx.atab.offsets[Ia + 1]; ++k) {
                    const Exc& e = ctx.atab.entries[k];
                    const int p = static_cast<int>(e.pair) / n;
                    const int q = static_cast<int>(e.pair) % n;
                    Dj[q * n + p] += e.sign * c[e.to * nb + Ib];
                }
                for (std::size_t k = ctx.btab.offsets[Ib];
                     k < ctx.btab.offsets[Ib + 1]; ++k) {
                    const Exc& e = ctx.btab.entries[k];
                    const int p = static_cast<int>(e.pair) / n;
                    const int q = static_cast<int>(e.pair) % n;
                    Dj[q * n + p] += e.sign * c[Ia * nb + e.to];
                }
            }

            // F = h' c + ½ (pq|rs)·D  (rank-1 + GEMM)
            F.leftCols(w).noalias() = 0.5 * (eri_m * D.leftCols(w));
            Eigen::Map<const Eigen::VectorXd> hvec(h1p.data(), npair);
            for (int j = 0; j < w; ++j)
                F.col(j).noalias() += hvec * c[c0 + j];

            // Scatter σ(T) += sign · F[(p,q), j] for a†_p a_q |I_j> = sign|T>.
            for (long long I = c0; I < c1; ++I) {
                const long long Ia = I / nb, Ib = I % nb;
                const int j = static_cast<int>(I - c0);
                const double* Fj =
                    F.data() + static_cast<std::size_t>(j) * npair;
                for (std::size_t k = ctx.atab.offsets[Ia];
                     k < ctx.atab.offsets[Ia + 1]; ++k) {
                    const Exc& e = ctx.atab.entries[k];
                    sl[e.to * nb + Ib] += e.sign * Fj[e.pair];
                }
                for (std::size_t k = ctx.btab.offsets[Ib];
                     k < ctx.btab.offsets[Ib + 1]; ++k) {
                    const Exc& e = ctx.btab.entries[k];
                    sl[Ia * nb + e.to] += e.sign * Fj[e.pair];
                }
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) {
        const double* sl = sig_local[t].data();
        for (long long i = 0; i < ndet; ++i) sigma[i] += sl[i];
    }
}

// Determinant diagonal ⟨I|H|I⟩ (original h, chemist (pq|rs)).
Eigen::VectorXd h_diagonal(const CIContext& ctx,
                           const Eigen::MatrixXd& h1,
                           const Eigen::MatrixXd& eri_m) {
    const int n = ctx.n_act;
    const long long nb = ctx.nb;
    auto occ_of = [n](std::uint64_t m) {
        std::vector<int> o;
        for (int p = 0; p < n; ++p)
            if ((m >> p) & 1) o.push_back(p);
        return o;
    };
    auto coul = [&](int p, int q) { return eri_m(p * n + p, q * n + q); };
    auto exch = [&](int p, int q) { return eri_m(p * n + q, q * n + p); };

    // Same-spin energies per string + per-string Coulomb field u[r].
    auto sector = [&](const std::vector<std::uint64_t>& strs,
                      std::vector<double>& e1,
                      std::vector<Eigen::VectorXd>& field) {
        e1.resize(strs.size());
        field.resize(strs.size());
        for (std::size_t i = 0; i < strs.size(); ++i) {
            const auto occ = occ_of(strs[i]);
            double e = 0.0;
            Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
            for (int p : occ) {
                e += h1(p, p);
                for (int r = 0; r < n; ++r) u[r] += coul(r, p);
            }
            for (std::size_t a = 0; a < occ.size(); ++a)
                for (std::size_t b = 0; b < occ.size(); ++b)
                    if (a != b)
                        e += 0.5 * (coul(occ[a], occ[b]) - exch(occ[a], occ[b]));
            e1[i] = e;
            field[i] = u;
        }
    };
    std::vector<double> ea, eb;
    std::vector<Eigen::VectorXd> ua, ub;
    sector(ctx.astr, ea, ua);
    sector(ctx.bstr, eb, ub);

    Eigen::VectorXd diag(ctx.ndet);
    for (long long Ia = 0; Ia < ctx.na; ++Ia) {
        const auto& u = ua[Ia];
        for (long long Ib = 0; Ib < nb; ++Ib) {
            double cross = 0.0;
            std::uint64_t m = ctx.bstr[Ib];
            for (int q = 0; q < n; ++q)
                if ((m >> q) & 1) cross += u[q];
            diag[Ia * nb + Ib] = ea[Ia] + eb[Ib] + cross;
        }
    }
    return diag;
}

}  // namespace

Eigen::VectorXd casci_direct_sigma(const Eigen::VectorXd& ci,
                                   const Eigen::MatrixXd& h1,
                                   const std::vector<double>& eri_chem,
                                   int n_act,
                                   int n_alpha,
                                   int n_beta,
                                   int chunk) {
    const CIContext ctx = make_context(n_act, n_alpha, n_beta);
    if (ci.size() != ctx.ndet)
        throw std::invalid_argument("casci_direct_sigma: ci size mismatch");
    const int n = n_act, npair = n * n;
    Eigen::Map<const Eigen::MatrixXd> eri_m(eri_chem.data(), npair, npair);
    // h'_pq = h_pq − ½ Σ_r (pr|rq)
    Eigen::MatrixXd h1p = h1;
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) {
            double s = 0.0;
            for (int r = 0; r < n; ++r)
                s += eri_chem[(static_cast<std::size_t>(p) * n + r) * npair +
                              static_cast<std::size_t>(r) * n + q];
            h1p(p, q) -= 0.5 * s;
        }
    // Row-major (pq|rs) buffer maps directly onto a col-major npair×npair
    // matrix only because the chemist ERI matrix is symmetric.
    Eigen::VectorXd out(ctx.ndet);
    sigma_into(ctx, h1p, eri_m, ci.data(), out.data(), chunk);
    return out;
}

CASCIDirectResult casci_direct_solve(const Eigen::MatrixXd& h1,
                                     const std::vector<double>& eri_chem,
                                     int n_act,
                                     int n_alpha,
                                     int n_beta,
                                     const CASCIDirectOptions& opts,
                                     const Eigen::MatrixXd& guess) {
    const CIContext ctx = make_context(n_act, n_alpha, n_beta);
    const int n = n_act, npair = n * n;
    if (static_cast<long long>(eri_chem.size()) !=
        static_cast<long long>(npair) * npair)
        throw std::invalid_argument("casci_direct: eri size != n_act^4");
    if (h1.rows() != n || h1.cols() != n)
        throw std::invalid_argument("casci_direct: h1 shape != (n_act, n_act)");
    const long long ndet = ctx.ndet;
    const int nroots = std::min<long long>(opts.nroots, ndet);

    Eigen::Map<const Eigen::MatrixXd> eri_m(eri_chem.data(), npair, npair);
    Eigen::MatrixXd h1p = h1;
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) {
            double s = 0.0;
            for (int r = 0; r < n; ++r)
                s += eri_chem[(static_cast<std::size_t>(p) * n + r) * npair +
                              static_cast<std::size_t>(r) * n + q];
            h1p(p, q) -= 0.5 * s;
        }

    CASCIDirectResult res;
    res.n_det = ndet;

    // Tiny spaces: dense diagonalization via explicit σ on unit vectors.
    if (ndet <= 64) {
        Eigen::MatrixXd H(ndet, ndet);
        Eigen::VectorXd e(ndet);
        for (long long j = 0; j < ndet; ++j) {
            e.setZero();
            e[j] = 1.0;
            H.col(j) = casci_direct_sigma(e, h1, eri_chem, n, n_alpha, n_beta,
                                          opts.chunk);
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
            0.5 * (H + H.transpose()));
        res.eigenvalues = es.eigenvalues().head(nroots);
        res.ci = es.eigenvectors().leftCols(nroots);
        res.converged = true;
        res.n_iter = 1;
        return res;
    }

    const Eigen::VectorXd diag = h_diagonal(ctx, h1, eri_m);

    // Block Davidson (Davidson 1975) with diagonal preconditioner.
    const int max_sub = std::max(opts.max_subspace, 2 * nroots + 4);
    Eigen::MatrixXd V(ndet, max_sub), W(ndet, max_sub);
    int m = 0;

    // Seeds: optional warm-start guess columns first, then unit vectors on
    // the lowest-diagonal determinants, each plus a small deterministic
    // random component.  The noise breaks point-group / spin sector
    // confinement: with exact symmetries, pure determinant seeds and their
    // preconditioned residuals stay inside the seeds' symmetry sector and
    // Davidson silently skips roots from other sectors.  Raw mt19937 draws
    // (not std::normal_distribution, which is implementation-defined) keep
    // the run bit-reproducible across platforms.
    {
        const int nseed = std::min<long long>(nroots + 2, ndet);
        std::mt19937 rng(42);
        const double noise = 1e-3;
        auto add_seed = [&](Eigen::VectorXd v, bool with_noise) {
            if (with_noise)
                for (long long i = 0; i < ndet; ++i)
                    v[i] += noise * (static_cast<double>(rng()) /
                                         static_cast<double>(rng.max()) -
                                     0.5);
            for (int pass = 0; pass < 2; ++pass)
                v -= V.leftCols(m) * (V.leftCols(m).transpose() * v);
            const double nrm = v.norm();
            if (nrm > 1e-8) {
                V.col(m) = v / nrm;
                ++m;
            }
        };
        if (guess.size() > 0) {
            if (guess.rows() != ndet)
                throw std::invalid_argument(
                    "casci_direct: guess rows != determinant count");
            // Guess columns are taken as-is (no noise): a converged CI
            // vector from a nearby orbital basis is already a physical
            // full-dimensional vector, and dressing it with noise would
            // re-inflate the starting residual to the noise scale —
            // exactly the warm-start benefit being thrown away.  The
            // symmetry-breaking noise stays on the unit top-up seeds.
            for (int j = 0; j < guess.cols() && m < nseed; ++j)
                add_seed(guess.col(j), false);
        }
        std::vector<long long> order(ndet);
        for (long long i = 0; i < ndet; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + nseed, order.end(),
                          [&](long long a, long long b) {
                              return diag[a] < diag[b];
                          });
        for (int s = 0; s < nseed && m < nseed; ++s) {
            Eigen::VectorXd v = Eigen::VectorXd::Zero(ndet);
            v[order[s]] = 1.0;
            add_seed(std::move(v), true);
        }
        if (m == 0) {  // pathological guess set — fall back to one unit seed
            V.col(0).setZero();
            V.col(0)[order[0]] = 1.0;
            m = 1;
        }
    }
    for (int j = 0; j < m; ++j)
        sigma_into(ctx, h1p, eri_m, V.col(j).data(), W.col(j).data(),
                   opts.chunk);

    Eigen::VectorXd theta(nroots);
    Eigen::MatrixXd X;  // Ritz vectors (ndet × nroots)
    res.converged = false;

    for (int it = 1; it <= opts.max_iter; ++it) {
        res.n_iter = it;
        Eigen::MatrixXd S = V.leftCols(m).transpose() * W.leftCols(m);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
            0.5 * (S + S.transpose()));
        const Eigen::MatrixXd& Y = es.eigenvectors();
        theta = es.eigenvalues().head(nroots);
        X.noalias() = V.leftCols(m) * Y.leftCols(nroots);
        Eigen::MatrixXd WX = W.leftCols(m) * Y.leftCols(nroots);

        // Residuals + preconditioned expansion vectors.
        int n_new = 0;
        bool all_conv = true;
        for (int k = 0; k < nroots; ++k) {
            Eigen::VectorXd r = WX.col(k) - theta[k] * X.col(k);
            const double rn = r.norm();
            if (rn > opts.tol) {
                all_conv = false;
                if (m + n_new < max_sub) {
                    Eigen::VectorXd t = r;
                    for (long long i = 0; i < ndet; ++i) {
                        const double d = theta[k] - diag[i];
                        t[i] /= (std::abs(d) > 1e-8 ? d
                                                    : (d < 0 ? -1e-8 : 1e-8));
                    }
                    // Normalize BEFORE orthogonalizing so the acceptance
                    // floor below is relative; near convergence the
                    // orthogonal complement is tiny but still the right
                    // expansion direction (rejecting it stalls Davidson at
                    // a residual above tol — the 2026-06-10 stagnation bug).
                    t /= t.norm();
                    for (int pass = 0; pass < 2; ++pass)
                        t -= V.leftCols(m + n_new) *
                             (V.leftCols(m + n_new).transpose() * t);
                    double nrm = t.norm();
                    if (nrm < 1e-10) {
                        // Preconditioned direction degenerate with the
                        // subspace — fall back to the raw residual.
                        t = r / rn;
                        for (int pass = 0; pass < 2; ++pass)
                            t -= V.leftCols(m + n_new) *
                                 (V.leftCols(m + n_new).transpose() * t);
                        nrm = t.norm();
                    }
                    if (nrm > 1e-10) {
                        V.col(m + n_new) = t / nrm;
                        ++n_new;
                    }
                }
            }
        }
        if (all_conv) {
            res.converged = true;
            break;
        }
        if (n_new == 0 || m + n_new > max_sub - 1) {
            // Collapse the subspace to the Ritz vectors and re-seed.
            Eigen::MatrixXd Xfull = V.leftCols(m) * Y;
            Eigen::MatrixXd WXfull = W.leftCols(m) * Y;
            const int keep = std::min(m, nroots + 2);
            V.leftCols(keep) = Xfull.leftCols(keep);
            W.leftCols(keep) = WXfull.leftCols(keep);
            m = keep;
            if (n_new == 0) continue;
            // Re-orthogonalize pending expansion vector(s) next round.
            continue;
        }
        for (int j = 0; j < n_new; ++j)
            sigma_into(ctx, h1p, eri_m, V.col(m + j).data(),
                       W.col(m + j).data(), opts.chunk);
        m += n_new;
    }

    res.eigenvalues = theta;
    res.ci = X;
    return res;
}

void casci_direct_rdm12(const Eigen::VectorXd& ci,
                        int n_act,
                        int n_alpha,
                        int n_beta,
                        Eigen::MatrixXd& rdm1,
                        std::vector<double>& rdm2) {
    const CIContext ctx = make_context(n_act, n_alpha, n_beta);
    if (ci.size() != ctx.ndet)
        throw std::invalid_argument("casci_direct_rdm12: ci size mismatch");
    const int n = n_act, npair = n * n;
    const long long ndet = ctx.ndet, nb = ctx.nb;
    const int chunk = 4096;

    // D[(r,s), j] = (Ê_rs c)(I_j) per chunk;
    // rdm1[r,s]   = Σ_I c_I D[(r,s), I]
    // T2[(p,q),(r,s)] = ⟨Ê_pq Ê_rs⟩ = Σ_I D[(q,p), I] D[(r,s), I].
    Eigen::MatrixXd T2 = Eigen::MatrixXd::Zero(npair, npair);
    rdm1 = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd D(npair, chunk), Ds(npair, chunk);
    const double* c = ci.data();

    for (long long c0 = 0; c0 < ndet; c0 += chunk) {
        const long long c1 = std::min(ndet, c0 + chunk);
        const int w = static_cast<int>(c1 - c0);
        D.leftCols(w).setZero();
        for (long long I = c0; I < c1; ++I) {
            const long long Ia = I / nb, Ib = I % nb;
            const int j = static_cast<int>(I - c0);
            double* Dj = D.data() + static_cast<std::size_t>(j) * npair;
            for (std::size_t k = ctx.atab.offsets[Ia];
                 k < ctx.atab.offsets[Ia + 1]; ++k) {
                const Exc& e = ctx.atab.entries[k];
                const int p = static_cast<int>(e.pair) / n;
                const int q = static_cast<int>(e.pair) % n;
                Dj[q * n + p] += e.sign * c[e.to * nb + Ib];
            }
            for (std::size_t k = ctx.btab.offsets[Ib];
                 k < ctx.btab.offsets[Ib + 1]; ++k) {
                const Exc& e = ctx.btab.entries[k];
                const int p = static_cast<int>(e.pair) / n;
                const int q = static_cast<int>(e.pair) % n;
                Dj[q * n + p] += e.sign * c[Ia * nb + e.to];
            }
        }
        for (int j = 0; j < w; ++j) {
            const double* Dj = D.data() + static_cast<std::size_t>(j) * npair;
            double* Dsj = Ds.data() + static_cast<std::size_t>(j) * npair;
            for (int p = 0; p < n; ++p)
                for (int q = 0; q < n; ++q) Dsj[p * n + q] = Dj[q * n + p];
        }
        const Eigen::VectorXd v =
            D.leftCols(w) * Eigen::Map<const Eigen::VectorXd>(c + c0, w);
        for (int r = 0; r < n; ++r)
            for (int s = 0; s < n; ++s) rdm1(r, s) += v[r * n + s];
        T2.noalias() += Ds.leftCols(w) * D.leftCols(w).transpose();
    }

    // rdm2[p,q,r,s] = ⟨Ê_pq Ê_rs⟩ − δ_qr ⟨Ê_ps⟩  (PySCF reorder_dm12).
    rdm2.assign(static_cast<std::size_t>(npair) * npair, 0.0);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s) {
                    double v = T2(p * n + q, r * n + s);
                    if (q == r) v -= rdm1(p, s);
                    rdm2[((static_cast<std::size_t>(p) * n + q) * n + r) * n +
                         s] = v;
                }
}

}  // namespace vibeqc
