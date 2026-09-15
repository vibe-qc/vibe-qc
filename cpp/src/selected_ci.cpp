// Selected CI (CIPSI selection, sparse H, block Davidson, truncated-list
// RDMs) over arbitrary SpinDet bitmask lists.  Conventions documented in
// include/vibeqc/selected_ci.hpp; the Python oracle is
// solvers/_selected_ci.py::selected_casci + solvers/_rdm.py::make_rdm12
// (machine-precision parity pinned in tests/test_selected_casci.py).

#include "vibeqc/selected_ci.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace vibeqc {

namespace {

using std::size_t;
using std::uint64_t;

inline int popcnt(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int c = 0;
    for (; x; x &= x - 1) ++c;
    return c;
#endif
}

inline uint64_t bit(int p) { return uint64_t(1) << p; }
inline uint64_t below(int p) { return bit(p) - 1; }

// Sequential second-quantized ops on one spin sector's mask, tracking the
// Jordan-Wigner parity (bits below the target within this sector), the
// same convention as solvers/_rdm._apply_aq_adp_spin and _mrpt._ann/_cre.
inline bool ann(uint64_t& m, int q, int& par) {
    if (!((m >> q) & 1)) return false;
    par ^= popcnt(m & below(q)) & 1;
    m &= ~bit(q);
    return true;
}

inline bool cre(uint64_t& m, int p, int& par) {
    if ((m >> p) & 1) return false;
    par ^= popcnt(m & below(p)) & 1;
    m |= bit(p);
    return true;
}

// Phase of the single excitation p <- q on mask m (q occupied, p empty):
// (-1)^(number of occupied orbitals strictly between p and q).
inline double single_phase(uint64_t m, int p, int q) {
    const int lo = std::min(p, q), hi = std::max(p, q);
    const uint64_t between = m & (below(hi) & ~below(lo + 1));
    return (popcnt(between) & 1) ? -1.0 : 1.0;
}

struct DetKey {
    uint64_t a, b;
    bool operator==(const DetKey& o) const { return a == o.a && b == o.b; }
};

struct DetHash {
    size_t operator()(const DetKey& k) const {
        // splitmix64-style mix of the two sector masks
        uint64_t x = k.a + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        uint64_t y = k.b + 0x9e3779b97f4a7c15ULL;
        y = (y ^ (y >> 30)) * 0xbf58476d1ce4e5b9ULL;
        return size_t((x ^ (x >> 31)) + 0x9e3779b9ULL * (y ^ (y >> 31)));
    }
};

using DetMap = std::unordered_map<DetKey, std::int32_t, DetHash>;

inline std::vector<int> occ_list(uint64_t m, int n_act) {
    std::vector<int> out;
    out.reserve(popcnt(m));
    for (int p = 0; p < n_act; ++p)
        if ((m >> p) & 1) out.push_back(p);
    return out;
}

inline std::vector<int> vir_list(uint64_t m, int n_act) {
    std::vector<int> out;
    out.reserve(n_act - popcnt(m));
    for (int p = 0; p < n_act; ++p)
        if (!((m >> p) & 1)) out.push_back(p);
    return out;
}

// Chemist (pq|rs) accessor over the row-major n_act^4 block.
struct Eri {
    const double* v;
    int n;
    inline double operator()(int p, int q, int r, int s) const {
        return v[((size_t(p) * n + q) * n + r) * n + s];
    }
};

// <D|H|D> for D = (a, b).
double diag_element(uint64_t a, uint64_t b, const Eigen::MatrixXd& h1,
                    const Eri& g, int n_act) {
    const auto oa = occ_list(a, n_act);
    const auto ob = occ_list(b, n_act);
    double e = 0.0;
    for (int p : oa) e += h1(p, p);
    for (int p : ob) e += h1(p, p);
    for (size_t i = 0; i < oa.size(); ++i)
        for (size_t j = i + 1; j < oa.size(); ++j) {
            const int p = oa[i], q = oa[j];
            e += g(p, p, q, q) - g(p, q, q, p);
        }
    for (size_t i = 0; i < ob.size(); ++i)
        for (size_t j = i + 1; j < ob.size(); ++j) {
            const int p = ob[i], q = ob[j];
            e += g(p, p, q, q) - g(p, q, q, p);
        }
    for (int p : oa)
        for (int q : ob) e += g(p, p, q, q);
    return e;
}

// <D'|H|D> for D' = single excitation p <- q in the alpha sector of D
// (swap roles of the masks for a beta single).
double single_element(uint64_t a, uint64_t b, int p, int q,
                      const Eigen::MatrixXd& h1, const Eri& g, int n_act) {
    double e = h1(p, q);
    uint64_t rest = a & ~bit(q);
    for (int r = 0; r < n_act; ++r)
        if ((rest >> r) & 1) e += g(p, q, r, r) - g(p, r, r, q);
    for (int r = 0; r < n_act; ++r)
        if ((b >> r) & 1) e += g(p, q, r, r);
    return single_phase(a, p, q) * e;
}

// Same-sector double q1,q2 -> p1,p2 (all distinct; q's occupied, p's empty):
// phase from sequential a+_{p1} a+_{p2} a_{q2} a_{q1}, value
// (p1 q1|p2 q2) - (p1 q2|p2 q1).
double double_same_element(uint64_t m, int p1, int p2, int q1, int q2,
                           const Eri& g, double* phase_out = nullptr) {
    uint64_t w = m;
    int par = 0;
    if (!ann(w, q1, par) || !ann(w, q2, par) || !cre(w, p2, par) ||
        !cre(w, p1, par))
        return 0.0;
    const double s = (par & 1) ? -1.0 : 1.0;
    if (phase_out) *phase_out = s;
    return s * (g(p1, q1, p2, q2) - g(p1, q2, p2, q1));
}

// Alpha-beta double (alpha q1->p1, beta q2->p2): independent sector phases.
inline double double_ab_element(uint64_t a, uint64_t b, int p1, int q1,
                                int p2, int q2, const Eri& g,
                                double* phase_out = nullptr) {
    const double s = single_phase(a, p1, q1) * single_phase(b, p2, q2);
    if (phase_out) *phase_out = s;
    return s * g(p1, q1, p2, q2);
}

// ── Heat-bath presorted double-excitation tables ────────────────────────
// Holmes, Tubman & Umrigar, J. Chem. Theory Comput. 12, 3674 (2016),
// Sec. 2.1 (doi:10.1021/acs.jctc.6b00407): a double excitation's matrix
// element magnitude depends only on its four orbital indices, never on
// the host determinant: |H| = |(p1 q1|p2 q2) − (p1 q2|p2 q1)| same-spin,
// |(p1 q1|p2 q2)| opposite-spin, up to the JW sign.  Presorting each
// occupied pair's candidate list by that magnitude lets every selection
// walk stop at the first sub-threshold entry instead of enumerating the
// whole virtual-pair rectangle; entries are skipped (not terminated) when
// a target orbital happens to be occupied in the host determinant.
//
// The walk's keep/drop predicate is the bit-identical expression the
// brute-force prefilter applies per candidate (mag * cmax < select_eps,
// with mag computed by the same arithmetic), and positive-double
// multiplication is monotone, so walk and prefilter accept exactly the
// same candidate set; only the enumeration cost changes.

struct HBEntry {
    double mag;
    std::uint8_t p1, p2;
};

struct HeatBathTables {
    int n = 0;
    // same-spin: row = upper-triangle pair index of q1 < q2; entries are
    // virtual pairs p1 < p2 (both distinct from q1, q2), sorted desc.
    std::vector<std::vector<HBEntry>> same;
    // opposite-spin: row = q1 * n + q2 (q1 alpha-occ, q2 beta-occ);
    // entries (p1 != q1, p2 != q2), sorted desc.
    std::vector<std::vector<HBEntry>> ab;

    inline int pair_idx(int q1, int q2) const {  // requires q1 < q2
        return q1 * n - q1 * (q1 + 1) / 2 + (q2 - q1 - 1);
    }
};

HeatBathTables build_heat_bath_tables(const Eri& g, int n_act) {
    HeatBathTables t;
    t.n = n_act;
    t.same.resize(size_t(n_act) * (n_act - 1) / 2);
    t.ab.resize(size_t(n_act) * n_act);
    const auto desc = [](const HBEntry& x, const HBEntry& y) {
        return x.mag > y.mag;
    };
    for (int q1 = 0; q1 < n_act; ++q1)
        for (int q2 = q1 + 1; q2 < n_act; ++q2) {
            auto& row = t.same[size_t(t.pair_idx(q1, q2))];
            for (int p1 = 0; p1 < n_act; ++p1) {
                if (p1 == q1 || p1 == q2) continue;
                for (int p2 = p1 + 1; p2 < n_act; ++p2) {
                    if (p2 == q1 || p2 == q2) continue;
                    // Same expression as double_same_element's value, so
                    // the stored magnitude is bit-identical to |H|.
                    const double m =
                        std::abs(g(p1, q1, p2, q2) - g(p1, q2, p2, q1));
                    if (m > 0.0)
                        row.push_back(
                            {m, std::uint8_t(p1), std::uint8_t(p2)});
                }
            }
            std::sort(row.begin(), row.end(), desc);
        }
    for (int q1 = 0; q1 < n_act; ++q1)
        for (int q2 = 0; q2 < n_act; ++q2) {
            auto& row = t.ab[size_t(q1) * n_act + q2];
            for (int p1 = 0; p1 < n_act; ++p1) {
                if (p1 == q1) continue;
                for (int p2 = 0; p2 < n_act; ++p2) {
                    if (p2 == q2) continue;
                    const double m = std::abs(g(p1, q1, p2, q2));
                    if (m > 0.0)
                        row.push_back(
                            {m, std::uint8_t(p1), std::uint8_t(p2)});
                }
            }
            std::sort(row.begin(), row.end(), desc);
        }
    return t;
}

// ── Sparse Hamiltonian in the selected space ────────────────────────────

struct SparseH {
    std::vector<std::vector<std::pair<std::int32_t, double>>> rows;
    Eigen::VectorXd diag;
};

// Enumerate every determinant connected to (a, b) by one of the five
// excitation classes; cb(target_key, element) is invoked for hits in the
// map.  The matrix element is exact (no thresholds).
template <typename CB>
void for_connected(uint64_t a, uint64_t b, const Eigen::MatrixXd& h1,
                   const Eri& g, int n_act, const DetMap& index, CB&& cb) {
    const auto oa = occ_list(a, n_act), va = vir_list(a, n_act);
    const auto ob = occ_list(b, n_act), vb = vir_list(b, n_act);

    for (int q : oa)                                       // alpha singles
        for (int p : va) {
            const DetKey k{(a & ~bit(q)) | bit(p), b};
            const auto it = index.find(k);
            if (it == index.end()) continue;
            cb(it->second, single_element(a, b, p, q, h1, g, n_act));
        }
    for (int q : ob)                                       // beta singles
        for (int p : vb) {
            const DetKey k{a, (b & ~bit(q)) | bit(p)};
            const auto it = index.find(k);
            if (it == index.end()) continue;
            cb(it->second, single_element(b, a, p, q, h1, g, n_act));
        }
    for (size_t i = 0; i < oa.size(); ++i)                 // alpha-alpha
        for (size_t j = i + 1; j < oa.size(); ++j)
            for (size_t x = 0; x < va.size(); ++x)
                for (size_t y = x + 1; y < va.size(); ++y) {
                    const int q1 = oa[i], q2 = oa[j];
                    const int p1 = va[x], p2 = va[y];
                    const DetKey k{(a & ~(bit(q1) | bit(q2))) |
                                       bit(p1) | bit(p2),
                                   b};
                    const auto it = index.find(k);
                    if (it == index.end()) continue;
                    cb(it->second,
                       double_same_element(a, p1, p2, q1, q2, g));
                }
    for (size_t i = 0; i < ob.size(); ++i)                 // beta-beta
        for (size_t j = i + 1; j < ob.size(); ++j)
            for (size_t x = 0; x < vb.size(); ++x)
                for (size_t y = x + 1; y < vb.size(); ++y) {
                    const int q1 = ob[i], q2 = ob[j];
                    const int p1 = vb[x], p2 = vb[y];
                    const DetKey k{a, (b & ~(bit(q1) | bit(q2))) |
                                          bit(p1) | bit(p2)};
                    const auto it = index.find(k);
                    if (it == index.end()) continue;
                    cb(it->second,
                       double_same_element(b, p1, p2, q1, q2, g));
                }
    for (int q1 : oa)                                      // alpha-beta
        for (int p1 : va)
            for (int q2 : ob)
                for (int p2 : vb) {
                    const DetKey k{(a & ~bit(q1)) | bit(p1),
                                   (b & ~bit(q2)) | bit(p2)};
                    const auto it = index.find(k);
                    if (it == index.end()) continue;
                    cb(it->second,
                       double_ab_element(a, b, p1, q1, p2, q2, g));
                }
}

// Extend a sparse H in place for determinants [built_n, da.size()): each
// NEW determinant's excitation rectangle is enumerated exactly once over
// the whole solve (the historical per-cycle full rebuild re-enumerated
// every surviving determinant every cycle, the measured wall at 26+
// active orbitals).  New rows are filled by the new determinant's own
// enumeration; connections that land on an already-built row are mirrored
// there symmetrically (H is real symmetric, and the stored coefficient is
// the same H_ij either way).  Entry SET is identical to a full rebuild;
// only the within-row order of mirrored entries differs (matvec rounding
// at machine epsilon).
void extend_sparse_h(SparseH& H, const std::vector<uint64_t>& da,
                     const std::vector<uint64_t>& db,
                     const Eigen::MatrixXd& h1, const Eri& g, int n_act,
                     const DetMap& index, size_t built_n) {
    const size_t n = da.size();
    if (built_n >= n) return;
    H.rows.resize(n);
    H.diag.conservativeResize(static_cast<Eigen::Index>(n));
    for (size_t j = built_n; j < n; ++j) {
        H.diag[static_cast<Eigen::Index>(j)] =
            diag_element(da[j], db[j], h1, g, n_act);
        auto& row = H.rows[j];
        for_connected(da[j], db[j], h1, g, n_act, index,
                      [&](std::int32_t i, double hij) {
                          if (hij == 0.0) return;
                          row.emplace_back(i, hij);
                          // pre-existing rows never re-enumerate: mirror
                          if (size_t(i) < built_n)
                              H.rows[size_t(i)].emplace_back(
                                  std::int32_t(j), hij);
                      });
    }
}

Eigen::VectorXd sparse_matvec(const SparseH& H, const Eigen::VectorXd& x) {
    const Eigen::Index n = x.size();
    Eigen::VectorXd y = H.diag.cwiseProduct(x);
    // sigma = H*x, the Davidson inner kernel (called every iteration).
    // Row j writes only y[j] and reads read-only x / H.rows[j], so the
    // rows are independent — no reduction, no race. Dynamic schedule
    // because determinant connectivity makes per-row length vary widely.
    // A local accumulator keeps the per-row summation order identical to
    // the serial version, so y is bit-identical.
    #pragma omp parallel for schedule(dynamic, 64)
    for (Eigen::Index j = 0; j < n; ++j) {
        double yj = y[j];
        for (const auto& [i, hij] : H.rows[static_cast<size_t>(j)])
            yj += hij * x[i];
        y[j] = yj;
    }
    return y;
}

// Block Davidson with diagonal preconditioning and optional warm start.
// The expansion/restart logic mirrors the proven casci.cpp Davidson
// (incl. its documented 2026-06-10 stagnation fix: normalize the
// preconditioned residual BEFORE orthogonalizing so the acceptance
// floor is relative, and fall back to the raw residual when the
// preconditioned direction degenerates with the subspace).
bool davidson(const SparseH& H, int nroots, int max_iter, double tol,
              const Eigen::MatrixXd& guess, Eigen::VectorXd& evals,
              Eigen::MatrixXd& evecs) {
    const Eigen::Index n = H.diag.size();
    if (n <= 256 || n <= 4 * nroots) {
        // dense fallback for tiny spaces
        Eigen::MatrixXd Hd = Eigen::MatrixXd::Zero(n, n);
        Hd.diagonal() = H.diag;
        for (Eigen::Index j = 0; j < n; ++j)
            for (const auto& [i, hij] : H.rows[static_cast<size_t>(j)])
                Hd(i, j) = hij;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
        evals = es.eigenvalues().head(nroots);
        evecs = es.eigenvectors().leftCols(nroots);
        return true;
    }

    const int max_sub =
        std::min<Eigen::Index>(n, std::max(8 * nroots, 48));
    Eigen::MatrixXd V(n, max_sub), W(n, max_sub);
    int m = 0;

    // Seeds: warm-start columns first, then noise-dressed unit vectors on
    // the lowest-diagonal determinants (the noise breaks symmetry-sector
    // confinement, as in casci.cpp; mt19937 keeps it bit-reproducible).
    {
        std::mt19937 rng(42);
        const double noise = 1e-3;
        auto add_seed = [&](Eigen::VectorXd v, bool with_noise) {
            if (with_noise)
                for (Eigen::Index i = 0; i < n; ++i)
                    v[i] += noise * (static_cast<double>(rng()) /
                                         static_cast<double>(rng.max()) -
                                     0.5);
            for (int pass = 0; pass < 2; ++pass)
                v -= V.leftCols(m) * (V.leftCols(m).transpose() * v);
            const double nrm = v.norm();
            if (nrm > 1e-8 && m < max_sub) {
                V.col(m) = v / nrm;
                ++m;
            }
        };
        if (guess.rows() == n && guess.cols() > 0)
            for (Eigen::Index c = 0; c < guess.cols(); ++c)
                add_seed(guess.col(c), false);
        std::vector<Eigen::Index> order(static_cast<size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i)
            order[static_cast<size_t>(i)] = i;
        const Eigen::Index nseed = std::min<Eigen::Index>(nroots + 2, n);
        std::partial_sort(order.begin(), order.begin() + nseed, order.end(),
                          [&](Eigen::Index x, Eigen::Index y) {
                              return H.diag[x] < H.diag[y];
                          });
        for (Eigen::Index k = 0; m < std::max<int>(nroots, 1) && k < nseed;
             ++k) {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
            e[order[static_cast<size_t>(k)]] = 1.0;
            add_seed(std::move(e), true);
        }
        for (int c = 0; c < m; ++c) W.col(c) = sparse_matvec(H, V.col(c));
    }
    if (m == 0) return false;

    for (int it = 0; it < max_iter; ++it) {
        const Eigen::MatrixXd Hs =
            V.leftCols(m).transpose() * W.leftCols(m);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(
            0.5 * (Hs + Hs.transpose()));
        const int nr = std::min<int>(nroots, m);
        const Eigen::VectorXd theta = es.eigenvalues().head(nr);
        const Eigen::MatrixXd Y = es.eigenvectors();
        evals = theta;
        evecs = V.leftCols(m) * Y.leftCols(nr);
        const Eigen::MatrixXd WX = W.leftCols(m) * Y.leftCols(nr);

        int n_new = 0;
        bool all_conv = (nr == nroots);
        for (int k = 0; k < nr; ++k) {
            Eigen::VectorXd r = WX.col(k) - theta[k] * evecs.col(k);
            const double rn = r.norm();
            if (rn <= tol) continue;
            all_conv = false;
            if (m + n_new >= max_sub) continue;
            Eigen::VectorXd t = r;
            for (Eigen::Index i = 0; i < n; ++i) {
                const double d = theta[k] - H.diag[i];
                t[i] /= (std::abs(d) > 1e-8 ? d : (d < 0 ? -1e-8 : 1e-8));
            }
            t /= t.norm();  // relative acceptance floor (see note above)
            for (int pass = 0; pass < 2; ++pass)
                t -= V.leftCols(m + n_new) *
                     (V.leftCols(m + n_new).transpose() * t);
            double nrm = t.norm();
            if (nrm < 1e-10) {
                t = r / rn;  // raw-residual fallback
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
        if (all_conv) return true;

        if (n_new == 0 || m + n_new > max_sub - 1) {
            // Collapse to the Ritz vectors (keeping their sigma vectors).
            const int keep = std::min(m, nroots + 2);
            const Eigen::MatrixXd Xfull = V.leftCols(m) * Y.leftCols(keep);
            const Eigen::MatrixXd WXfull = W.leftCols(m) * Y.leftCols(keep);
            V.leftCols(keep) = Xfull;
            W.leftCols(keep) = WXfull;
            m = keep;
            continue;
        }
        for (int j = 0; j < n_new; ++j)
            W.col(m + j) = sparse_matvec(H, V.col(m + j));
        m += n_new;
    }
    return false;
}

// Enumerate every external contribution of determinant (a, b) that
// survives the screening |H| * scale < eps (drop), invoking
// cb(target_key, h_element) for survivors NOT in ``index``.  Shared by
// the deterministic and the sampled (stochastic) Epstein-Nesbet PT2:
// singles are enumerated brute-force (cheap, no determinant-independent
// bound exists); doubles walk the presorted heat-bath tables when
// provided, with the bit-identical break predicate.
template <typename CB>
void for_screened_external(uint64_t a, uint64_t b, double scale, double eps,
                           const HeatBathTables* hb,
                           const Eigen::MatrixXd& h1, const Eri& g,
                           int n_act, const DetMap& index, CB&& cb) {
    const auto oa = occ_list(a, n_act), va = vir_list(a, n_act);
    const auto ob = occ_list(b, n_act), vb = vir_list(b, n_act);
    const auto emit = [&](uint64_t ta, uint64_t tb, double h) {
        if (h == 0.0) return;
        if (eps > 0.0 && std::abs(h) * scale < eps) return;
        if (index.count(DetKey{ta, tb})) return;
        cb(DetKey{ta, tb}, h);
    };
    for (int q : oa)
        for (int p : va)
            emit((a & ~bit(q)) | bit(p), b,
                 single_element(a, b, p, q, h1, g, n_act));
    for (int q : ob)
        for (int p : vb)
            emit(a, (b & ~bit(q)) | bit(p),
                 single_element(b, a, p, q, h1, g, n_act));
    if (hb != nullptr && eps > 0.0) {
        for (size_t i = 0; i < oa.size(); ++i)
            for (size_t j = i + 1; j < oa.size(); ++j) {
                const int q1 = oa[i], q2 = oa[j];
                const auto& row = hb->same[size_t(hb->pair_idx(q1, q2))];
                for (const auto& e : row) {
                    if (e.mag * scale < eps) break;
                    if (((a >> e.p1) | (a >> e.p2)) & 1) continue;
                    emit((a & ~(bit(q1) | bit(q2))) | bit(e.p1) | bit(e.p2),
                         b, double_same_element(a, e.p1, e.p2, q1, q2, g));
                }
            }
        for (size_t i = 0; i < ob.size(); ++i)
            for (size_t j = i + 1; j < ob.size(); ++j) {
                const int q1 = ob[i], q2 = ob[j];
                const auto& row = hb->same[size_t(hb->pair_idx(q1, q2))];
                for (const auto& e : row) {
                    if (e.mag * scale < eps) break;
                    if (((b >> e.p1) | (b >> e.p2)) & 1) continue;
                    emit(a,
                         (b & ~(bit(q1) | bit(q2))) | bit(e.p1) | bit(e.p2),
                         double_same_element(b, e.p1, e.p2, q1, q2, g));
                }
            }
        for (int q1 : oa)
            for (int q2 : ob) {
                const auto& row =
                    hb->ab[size_t(q1) * size_t(n_act) + size_t(q2)];
                for (const auto& e : row) {
                    if (e.mag * scale < eps) break;
                    if ((a >> e.p1) & 1) continue;
                    if ((b >> e.p2) & 1) continue;
                    emit((a & ~bit(q1)) | bit(e.p1),
                         (b & ~bit(q2)) | bit(e.p2),
                         double_ab_element(a, b, e.p1, q1, e.p2, q2, g));
                }
            }
    } else {
        for (size_t i = 0; i < oa.size(); ++i)
            for (size_t j = i + 1; j < oa.size(); ++j)
                for (size_t x = 0; x < va.size(); ++x)
                    for (size_t y = x + 1; y < va.size(); ++y)
                        emit((a & ~(bit(oa[i]) | bit(oa[j]))) | bit(va[x]) |
                                 bit(va[y]),
                             b,
                             double_same_element(a, va[x], va[y], oa[i],
                                                 oa[j], g));
        for (size_t i = 0; i < ob.size(); ++i)
            for (size_t j = i + 1; j < ob.size(); ++j)
                for (size_t x = 0; x < vb.size(); ++x)
                    for (size_t y = x + 1; y < vb.size(); ++y)
                        emit(a,
                             (b & ~(bit(ob[i]) | bit(ob[j]))) | bit(vb[x]) |
                                 bit(vb[y]),
                             double_same_element(b, vb[x], vb[y], ob[i],
                                                 ob[j], g));
        for (int q1 : oa)
            for (int p1 : va)
                for (int q2 : ob)
                    for (int p2 : vb)
                        emit((a & ~bit(q1)) | bit(p1),
                             (b & ~bit(q2)) | bit(p2),
                             double_ab_element(a, b, p1, q1, p2, q2, g));
    }
}

// Platform-independent uniform double in [0, 1) from the exactly-specified
// mt19937_64 stream (std::uniform_real_distribution and
// std::discrete_distribution are implementation-defined, which would make
// fixed-seed results differ across standard libraries).
inline double uniform01(std::mt19937_64& rng) {
    return double(rng() >> 11) * 0x1.0p-53;
}

}  // namespace

// ── Public: grow-and-diagonalize selected CI ────────────────────────────

SelectedCIResultCpp selected_ci_solve(
    const Eigen::MatrixXd& h1, const std::vector<double>& eri_chem,
    int n_act, int n_alpha, int n_beta, const SelectedCIOptionsCpp& opts,
    const std::vector<uint64_t>& guess_a,
    const std::vector<uint64_t>& guess_b) {
    if (n_act < 1 || n_act > 64)
        throw std::invalid_argument("selected_ci_solve: n_act must be 1..64");
    if (h1.rows() != n_act || h1.cols() != n_act)
        throw std::invalid_argument("selected_ci_solve: h1 shape mismatch");
    if (eri_chem.size() != size_t(n_act) * n_act * n_act * n_act)
        throw std::invalid_argument("selected_ci_solve: eri size mismatch");
    if (guess_a.size() != guess_b.size())
        throw std::invalid_argument(
            "selected_ci_solve: guess mask arrays must be index-aligned");
    const Eri g{eri_chem.data(), n_act};
    const bool spin_complete = opts.spin_complete && (n_alpha == n_beta);

    std::vector<uint64_t> da, db;
    DetMap index;
    auto add_det = [&](uint64_t a, uint64_t b) {
        const DetKey k{a, b};
        if (index.count(k)) return;
        index.emplace(k, static_cast<std::int32_t>(da.size()));
        da.push_back(a);
        db.push_back(b);
    };

    if (!guess_a.empty()) {
        for (size_t i = 0; i < guess_a.size(); ++i) {
            if (popcnt(guess_a[i]) != n_alpha || popcnt(guess_b[i]) != n_beta)
                throw std::invalid_argument(
                    "selected_ci_solve: guess electron counts mismatch");
            add_det(guess_a[i], guess_b[i]);
        }
    } else {
        const uint64_t ref_a = below(n_alpha), ref_b = below(n_beta);
        add_det(ref_a, ref_b);
        if (opts.nroots > 1) {  // seed singles for multi-root freedom
            for (int q = 0; q < n_alpha; ++q)
                for (int p = n_alpha; p < n_act; ++p)
                    add_det((ref_a & ~bit(q)) | bit(p), ref_b);
            for (int q = 0; q < n_beta; ++q)
                for (int p = n_beta; p < n_act; ++p)
                    add_det(ref_a, (ref_b & ~bit(q)) | bit(p));
        }
    }
    if (spin_complete)
        for (size_t i = 0; i < da.size(); ++i) add_det(db[i], da[i]);
    if (static_cast<int>(da.size()) < opts.nroots)
        throw std::invalid_argument(
            "selected_ci_solve: starting space smaller than nroots");

    SelectedCIResultCpp out;
    Eigen::VectorXd evals;
    Eigen::MatrixXd evecs;
    Eigen::MatrixXd warm;  // previous-cycle CI padded to the new space
    bool have_prev = false;
    Eigen::VectorXd e_prev;

    // Heat-bath walk: only meaningful under a positive prefilter
    // threshold (eps = 0 keeps every candidate, so there is nothing to
    // stop early on); the integrals are fixed, so the tables are built
    // once for all cycles.  use_heat_bath_walk=false forces the brute
    // enumeration at the same eps (the equivalence-test hook).
    const bool hb_walk =
        opts.select_eps > 0.0 && opts.use_heat_bath_walk;
    HeatBathTables hb;
    if (hb_walk) hb = build_heat_bath_tables(g, n_act);

    SparseH H;        // persistent: rows are extended, never rebuilt
    size_t built_n = 0;
    for (int cycle = 0; cycle < opts.max_cycles; ++cycle) {
        out.n_cycles = cycle + 1;
        extend_sparse_h(H, da, db, h1, g, n_act, index, built_n);
        built_n = da.size();
        if (!davidson(H, opts.nroots, opts.davidson_max_iter,
                      opts.davidson_tol, warm, evals, evecs))
            throw std::runtime_error(
                "selected_ci_solve: Davidson did not converge");

        if (have_prev &&
            (evals - e_prev).cwiseAbs().maxCoeff() < opts.conv_tol_energy) {
            out.converged = true;
            break;
        }
        e_prev = evals;
        have_prev = true;

        if (static_cast<int>(da.size()) >= opts.target_size) {
            out.converged = true;
            break;
        }
        if (cycle == opts.max_cycles - 1) break;  // no growth on last cycle

        // ── CIPSI selection ──────────────────────────────────────────
        // coherent numerators per outside candidate, accumulated over all
        // significant variational determinants (matches the Python kernel)
        std::unordered_map<DetKey, std::vector<double>, DetHash> num;
        const int nr = opts.nroots;
        for (size_t jj = 0; jj < da.size(); ++jj) {
            double cmax = 0.0;
            for (int k = 0; k < nr; ++k)
                cmax = std::max(cmax,
                                std::abs(evecs(Eigen::Index(jj), k)));
            if (cmax <= opts.significant_coeff) continue;
            const uint64_t a = da[jj], b = db[jj];
            auto accumulate = [&](uint64_t ta, uint64_t tb, double hdi) {
                if (hdi == 0.0) return;
                if (opts.select_eps > 0.0 &&
                    std::abs(hdi) * cmax < opts.select_eps)
                    return;
                const DetKey k{ta, tb};
                if (index.count(k)) return;
                auto& v = num[k];
                if (v.empty()) v.assign(nr, 0.0);
                for (int r = 0; r < nr; ++r)
                    v[r] += hdi * evecs(Eigen::Index(jj), r);
            };
            const auto oa = occ_list(a, n_act), va = vir_list(a, n_act);
            const auto ob = occ_list(b, n_act), vb = vir_list(b, n_act);
            for (int q : oa)
                for (int p : va)
                    accumulate((a & ~bit(q)) | bit(p), b,
                               single_element(a, b, p, q, h1, g, n_act));
            for (int q : ob)
                for (int p : vb)
                    accumulate(a, (b & ~bit(q)) | bit(p),
                               single_element(b, a, p, q, h1, g, n_act));
            if (hb_walk) {
                // Presorted walks: stop at the first sub-threshold entry
                // (the same predicate accumulate re-applies); skip entries
                // whose target orbitals are occupied in this determinant.
                for (size_t i = 0; i < oa.size(); ++i)
                    for (size_t j2 = i + 1; j2 < oa.size(); ++j2) {
                        const int q1 = oa[i], q2 = oa[j2];
                        const auto& row =
                            hb.same[size_t(hb.pair_idx(q1, q2))];
                        for (const auto& e : row) {
                            if (e.mag * cmax < opts.select_eps) break;
                            if (((a >> e.p1) | (a >> e.p2)) & 1) continue;
                            accumulate(
                                (a & ~(bit(q1) | bit(q2))) | bit(e.p1) |
                                    bit(e.p2),
                                b,
                                double_same_element(a, e.p1, e.p2, q1, q2,
                                                    g));
                        }
                    }
                for (size_t i = 0; i < ob.size(); ++i)
                    for (size_t j2 = i + 1; j2 < ob.size(); ++j2) {
                        const int q1 = ob[i], q2 = ob[j2];
                        const auto& row =
                            hb.same[size_t(hb.pair_idx(q1, q2))];
                        for (const auto& e : row) {
                            if (e.mag * cmax < opts.select_eps) break;
                            if (((b >> e.p1) | (b >> e.p2)) & 1) continue;
                            accumulate(
                                a,
                                (b & ~(bit(q1) | bit(q2))) | bit(e.p1) |
                                    bit(e.p2),
                                double_same_element(b, e.p1, e.p2, q1, q2,
                                                    g));
                        }
                    }
                for (int q1 : oa)
                    for (int q2 : ob) {
                        const auto& row =
                            hb.ab[size_t(q1) * size_t(n_act) + size_t(q2)];
                        for (const auto& e : row) {
                            if (e.mag * cmax < opts.select_eps) break;
                            if ((a >> e.p1) & 1) continue;
                            if ((b >> e.p2) & 1) continue;
                            accumulate(
                                (a & ~bit(q1)) | bit(e.p1),
                                (b & ~bit(q2)) | bit(e.p2),
                                double_ab_element(a, b, e.p1, q1, e.p2, q2,
                                                  g));
                        }
                    }
            } else {
                for (size_t i = 0; i < oa.size(); ++i)
                    for (size_t j2 = i + 1; j2 < oa.size(); ++j2)
                        for (size_t x = 0; x < va.size(); ++x)
                            for (size_t y = x + 1; y < va.size(); ++y)
                                accumulate(
                                    (a & ~(bit(oa[i]) | bit(oa[j2]))) |
                                        bit(va[x]) | bit(va[y]),
                                    b,
                                    double_same_element(a, va[x], va[y],
                                                        oa[i], oa[j2], g));
                for (size_t i = 0; i < ob.size(); ++i)
                    for (size_t j2 = i + 1; j2 < ob.size(); ++j2)
                        for (size_t x = 0; x < vb.size(); ++x)
                            for (size_t y = x + 1; y < vb.size(); ++y)
                                accumulate(
                                    a,
                                    (b & ~(bit(ob[i]) | bit(ob[j2]))) |
                                        bit(vb[x]) | bit(vb[y]),
                                    double_same_element(b, vb[x], vb[y],
                                                        ob[i], ob[j2], g));
                for (int q1 : oa)
                    for (int p1 : va)
                        for (int q2 : ob)
                            for (int p2 : vb)
                                accumulate(
                                    (a & ~bit(q1)) | bit(p1),
                                    (b & ~bit(q2)) | bit(p2),
                                    double_ab_element(a, b, p1, q1, p2, q2,
                                                      g));
            }
        }

        std::vector<std::pair<double, DetKey>> scored;
        scored.reserve(num.size());
        for (const auto& [k, v] : num) {
            const double hdd = diag_element(k.a, k.b, h1, g, n_act);
            double w = 0.0;
            for (int r = 0; r < nr; ++r) {
                double den = std::abs(evals[r] - hdd);
                if (den < 1e-10) den = 1e-10;
                w += v[r] * v[r] / den;
            }
            if (w > opts.pt2_threshold) scored.emplace_back(w, k);
        }
        if (scored.empty()) {
            out.converged = true;
            break;
        }
        const int room = std::min<int>(
            opts.max_new_per_cycle,
            opts.target_size - static_cast<int>(da.size()));
        const size_t keep =
            std::min<size_t>(scored.size(), size_t(std::max(room, 0)));
        std::partial_sort(
            scored.begin(), scored.begin() + keep, scored.end(),
            [](const auto& x, const auto& y) { return x.first > y.first; });
        const size_t old_n = da.size();
        for (size_t i = 0; i < keep; ++i)
            add_det(scored[i].second.a, scored[i].second.b);
        if (spin_complete)
            for (size_t i = old_n; i < da.size(); ++i) add_det(db[i], da[i]);

        // warm-start the next Davidson with the current CI padded by zeros
        warm = Eigen::MatrixXd::Zero(Eigen::Index(da.size()), opts.nroots);
        warm.topRows(evecs.rows()) = evecs;
    }

    out.dets_a = std::move(da);
    out.dets_b = std::move(db);
    out.eigenvalues = evals;
    out.ci = evecs;
    return out;
}

// ── Public: truncated-list spin-summed RDM12 ────────────────────────────

void selected_ci_rdm12(const std::vector<uint64_t>& dets_a,
                       const std::vector<uint64_t>& dets_b,
                       const Eigen::VectorXd& ci, int n_act,
                       Eigen::MatrixXd& rdm1, std::vector<double>& rdm2) {
    const size_t n = dets_a.size();
    if (dets_b.size() != n || size_t(ci.size()) != n)
        throw std::invalid_argument("selected_ci_rdm12: size mismatch");
    rdm1 = Eigen::MatrixXd::Zero(n_act, n_act);
    rdm2.assign(size_t(n_act) * n_act * n_act * n_act, 0.0);
    auto d2 = [&](int p, int q, int r, int s) -> double& {
        return rdm2[((size_t(p) * n_act + q) * n_act + r) * n_act + s];
    };

    DetMap index;
    index.reserve(n * 2);
    for (size_t i = 0; i < n; ++i)
        index.emplace(DetKey{dets_a[i], dets_b[i]},
                      static_cast<std::int32_t>(i));

    for (size_t jj = 0; jj < n; ++jj) {
        const uint64_t a = dets_a[jj], b = dets_b[jj];
        const double cj = ci[Eigen::Index(jj)];
        if (cj == 0.0) continue;
        const auto oa = occ_list(a, n_act), va = vir_list(a, n_act);
        const auto ob = occ_list(b, n_act), vb = vir_list(b, n_act);

        // diagonal (I = J)
        const double cc = cj * cj;
        for (int p : oa) rdm1(p, p) += cc;
        for (int p : ob) rdm1(p, p) += cc;
        for (int p : oa)
            for (int q : oa)
                if (p != q) {
                    d2(p, p, q, q) += cc;   // direct
                    d2(p, q, q, p) -= cc;   // same-sector exchange
                }
        for (int p : ob)
            for (int q : ob)
                if (p != q) {
                    d2(p, p, q, q) += cc;
                    d2(p, q, q, p) -= cc;
                }
        for (int p : oa)
            for (int q : ob) {
                d2(p, p, q, q) += cc;       // alpha-beta, both sector orders
                d2(q, q, p, p) += cc;
            }

        // singles: bra I = (p <- q) applied to ket J, both directions
        // covered because every ordered pair (I, J) is visited once as
        // J runs over the full list.
        auto handle_single = [&](bool alpha_sector, int p, int q) {
            const uint64_t ta = alpha_sector ? ((a & ~bit(q)) | bit(p)) : a;
            const uint64_t tb = alpha_sector ? b : ((b & ~bit(q)) | bit(p));
            const auto it = index.find(DetKey{ta, tb});
            if (it == index.end()) return;
            const double s =
                single_phase(alpha_sector ? a : b, p, q);
            const double w = s * ci[it->second] * cj;  // <I|..|J> c_I c_J
            rdm1(p, q) += w;
            // spectator sums: same-sector occupied r (of the ket, r != q)
            const auto& osame = alpha_sector ? oa : ob;
            const auto& other = alpha_sector ? ob : oa;
            for (int r : osame) {
                if (r == q) continue;
                d2(p, q, r, r) += w;
                d2(r, r, p, q) += w;
                d2(p, r, r, q) -= w;
                d2(r, q, p, r) -= w;
            }
            for (int r : other) {
                d2(p, q, r, r) += w;
                d2(r, r, p, q) += w;
            }
        };
        for (int q : oa)
            for (int p : va) handle_single(true, p, q);
        for (int q : ob)
            for (int p : vb) handle_single(false, p, q);

        // same-sector doubles: bra I = a+_{p1} a+_{p2} a_{q2} a_{q1} |J>
        auto handle_double_same = [&](bool alpha_sector, int p1, int p2,
                                      int q1, int q2) {
            const uint64_t m = alpha_sector ? a : b;
            const uint64_t tm =
                (m & ~(bit(q1) | bit(q2))) | bit(p1) | bit(p2);
            const auto it = index.find(
                alpha_sector ? DetKey{tm, b} : DetKey{a, tm});
            if (it == index.end()) return;
            uint64_t w_ = m;
            int par = 0;
            if (!ann(w_, q1, par) || !ann(w_, q2, par) ||
                !cre(w_, p2, par) || !cre(w_, p1, par))
                return;
            const double s = (par & 1) ? -1.0 : 1.0;
            const double w = s * ci[it->second] * cj;
            // <a+_{p1} a+_{p2} a_{q2} a_{q1}> in dm2[p,q,r,s] = <a+p a+r a_s a_q>:
            d2(p1, q1, p2, q2) += w;
            d2(p2, q2, p1, q1) += w;
            d2(p1, q2, p2, q1) -= w;
            d2(p2, q1, p1, q2) -= w;
        };
        for (size_t i = 0; i < oa.size(); ++i)
            for (size_t j2 = i + 1; j2 < oa.size(); ++j2)
                for (size_t x = 0; x < va.size(); ++x)
                    for (size_t y = x + 1; y < va.size(); ++y)
                        handle_double_same(true, va[x], va[y], oa[i],
                                           oa[j2]);
        for (size_t i = 0; i < ob.size(); ++i)
            for (size_t j2 = i + 1; j2 < ob.size(); ++j2)
                for (size_t x = 0; x < vb.size(); ++x)
                    for (size_t y = x + 1; y < vb.size(); ++y)
                        handle_double_same(false, vb[x], vb[y], ob[i],
                                           ob[j2]);

        // alpha-beta doubles (alpha q1->p1, beta q2->p2)
        for (int q1 : oa)
            for (int p1 : va)
                for (int q2 : ob)
                    for (int p2 : vb) {
                        const DetKey k{(a & ~bit(q1)) | bit(p1),
                                       (b & ~bit(q2)) | bit(p2)};
                        const auto it = index.find(k);
                        if (it == index.end()) continue;
                        const double s = single_phase(a, p1, q1) *
                                         single_phase(b, p2, q2);
                        const double w = s * ci[it->second] * cj;
                        d2(p1, q1, p2, q2) += w;
                        d2(p2, q2, p1, q1) += w;
                    }
    }
}

// ── Public: Epstein-Nesbet PT2 on the selected wavefunction ─────────────
// Sharma, Holmes, Jeanmairet, Alavi & Umrigar, J. Chem. Theory Comput.
// 13, 1595 (2017), doi:10.1021/acs.jctc.6b01028.

double selected_ci_en_pt2_deterministic(
    const Eigen::MatrixXd& h1, const std::vector<double>& eri_chem,
    int n_act, const std::vector<uint64_t>& dets_a,
    const std::vector<uint64_t>& dets_b, const Eigen::VectorXd& ci,
    double e0, double eps2, bool use_heat_bath_walk,
    std::int64_t* n_perturbers_out) {
    const size_t n = dets_a.size();
    if (dets_b.size() != n || size_t(ci.size()) != n)
        throw std::invalid_argument(
            "selected_ci_en_pt2_deterministic: size mismatch");
    const Eri g{eri_chem.data(), n_act};

    DetMap index;
    index.reserve(n * 2);
    for (size_t i = 0; i < n; ++i)
        index.emplace(DetKey{dets_a[i], dets_b[i]},
                      static_cast<std::int32_t>(i));

    HeatBathTables hb;
    const bool walk = use_heat_bath_walk && eps2 > 0.0;
    if (walk) hb = build_heat_bath_tables(g, n_act);

    // Eq. 5: Delta-E2 = sum_a ( sum_i^{(eps2)} H_ai c_i )^2 / (E0 - E_a),
    // the inner sum keeping contributions with |H_ai c_i| above eps2
    // (coherent numerators accumulated over ALL variational generators).
    std::unordered_map<DetKey, double, DetHash> num;
    for (size_t i = 0; i < n; ++i) {
        const double c = ci[Eigen::Index(i)];
        if (c == 0.0) continue;
        for_screened_external(
            dets_a[i], dets_b[i], std::abs(c), eps2, walk ? &hb : nullptr,
            h1, g, n_act, index,
            [&](const DetKey& k, double h) { num[k] += h * c; });
    }

    // Holmes, Tubman & Umrigar, JCTC 12, 3674 (2016), Sec. III.B, Eq. 4
    // (doi:10.1021/acs.jctc.6b00407): the outer sum is over perturbers,
    // independent of the order in which the heat-bath walk discovers them.
    // The walk and brute enumerators insert the same keys into `num` in
    // different orders, and unordered_map traversal would make that
    // implementation detail change the floating-point reduction by an ULP.
    // Reduce in determinant order so both enumeration paths implement one
    // deterministic sum without changing screening or the coherent
    // per-perturber numerators.
    std::vector<std::pair<DetKey, double>> ordered_num;
    ordered_num.reserve(num.size());
    for (const auto& [k, v] : num) ordered_num.emplace_back(k, v);
    std::sort(ordered_num.begin(), ordered_num.end(),
              [](const auto& x, const auto& y) {
                  if (x.first.a != y.first.a) return x.first.a < y.first.a;
                  return x.first.b < y.first.b;
              });
    double e2 = 0.0;
    for (const auto& [k, v] : ordered_num) {
        const double denom = e0 - diag_element(k.a, k.b, h1, g, n_act);
        if (std::abs(denom) < 1e-10) continue;  // intruder guard
        e2 += v * v / denom;
    }
    if (n_perturbers_out) *n_perturbers_out = std::int64_t(num.size());
    return e2;
}

void selected_ci_en_pt2_stochastic(
    const Eigen::MatrixXd& h1, const std::vector<double>& eri_chem,
    int n_act, const std::vector<uint64_t>& dets_a,
    const std::vector<uint64_t>& dets_b, const Eigen::VectorXd& ci,
    double e0, double eps2, double eps2_loose, int n_samples,
    int sample_size, std::uint64_t seed, bool use_heat_bath_walk,
    double* mean_out, double* stderr_out) {
    const size_t n = dets_a.size();
    if (dets_b.size() != n || size_t(ci.size()) != n)
        throw std::invalid_argument(
            "selected_ci_en_pt2_stochastic: size mismatch");
    if (n_samples < 2)
        throw std::invalid_argument(
            "selected_ci_en_pt2_stochastic: n_samples must be >= 2 "
            "(the error bar is the std error over sample batches)");
    if (sample_size < 2)
        throw std::invalid_argument(
            "selected_ci_en_pt2_stochastic: sample_size must be >= 2 "
            "(the N_d(N_d-1) estimator needs two draws)");
    if (!(eps2_loose >= eps2))
        throw std::invalid_argument(
            "selected_ci_en_pt2_stochastic: eps2_loose must be >= eps2");
    const Eri g{eri_chem.data(), n_act};

    DetMap index;
    index.reserve(n * 2);
    for (size_t i = 0; i < n; ++i)
        index.emplace(DetKey{dets_a[i], dets_b[i]},
                      static_cast<std::int32_t>(i));

    HeatBathTables hb;
    const bool walk = use_heat_bath_walk && eps2 > 0.0;
    if (walk) hb = build_heat_bath_tables(g, n_act);

    // Eq. 7: p_i = |c_i| / sum_i |c_i| -- sampled by inverse-CDF binary
    // search over the exactly-specified mt19937_64 stream (the paper uses
    // the Alias method; any exact sampler of p_i gives the same
    // distribution, and CDF search keeps fixed-seed results
    // platform-independent).
    std::vector<double> cdf(n);
    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        norm += std::abs(ci[Eigen::Index(i)]);
        cdf[i] = norm;
    }
    if (norm <= 0.0)
        throw std::invalid_argument(
            "selected_ci_en_pt2_stochastic: zero wavefunction");
    std::mt19937_64 rng(seed);
    const double nd = double(sample_size);

    // Per-perturber accumulators for one batch: coherent w-weighted sums
    // and the diagonal-correction sums of Eq. 10, at the tight (eps2)
    // and loose (eps2_loose) screenings; the SAME sampled determinant
    // set feeds both, which is what cancels the stochastic error in the
    // semistochastic difference (Eq. 11).
    struct Acc {
        double sum_t = 0.0, diag_t = 0.0;
        double sum_l = 0.0, diag_l = 0.0;
    };
    std::unordered_map<DetKey, Acc, DetHash> acc;
    std::unordered_map<std::int32_t, int> counts;

    std::vector<double> batch_vals(static_cast<size_t>(n_samples));
    for (int s = 0; s < n_samples; ++s) {
        // Eq. 8/9: N_d multinomial draws (with replacement) -> w_i.
        counts.clear();
        for (int d = 0; d < sample_size; ++d) {
            const double u = uniform01(rng) * norm;
            const auto it =
                std::lower_bound(cdf.begin(), cdf.end(), u);
            const size_t i = size_t(it - cdf.begin()) < n
                                 ? size_t(it - cdf.begin())
                                 : n - 1;
            ++counts[std::int32_t(i)];
        }
        acc.clear();
        for (const auto& [iidx, w] : counts) {
            const size_t i = size_t(iidx);
            const double c = ci[Eigen::Index(i)];
            const double p = std::abs(c) / norm;
            const double wi = double(w);
            // Eq. 10 ingredient x_i = c_i H_ai; the screening compares
            // |H_ai c_i| (the TRUE coefficient, not the w/p-weighted
            // one) against the thresholds, matching the deterministic
            // sums it estimates.
            for_screened_external(
                dets_a[i], dets_b[i], std::abs(c), eps2,
                walk ? &hb : nullptr, h1, g, n_act, index,
                [&](const DetKey& k, double h) {
                    const double x = c * h;
                    auto& A = acc[k];
                    const double sw = wi * x / p;
                    const double dg =
                        (wi * (nd - 1.0) / p - wi * wi / (p * p)) * x * x;
                    A.sum_t += sw;
                    A.diag_t += dg;
                    if (std::abs(h) * std::abs(c) >= eps2_loose) {
                        A.sum_l += sw;
                        A.diag_l += dg;
                    }
                });
        }
        // Eq. 10 per screening; Eq. 11 batch difference.
        double e2_t = 0.0, e2_l = 0.0;
        for (const auto& [k, A] : acc) {
            const double denom = e0 - diag_element(k.a, k.b, h1, g, n_act);
            if (std::abs(denom) < 1e-10) continue;
            e2_t += (A.sum_t * A.sum_t + A.diag_t) / denom;
            e2_l += (A.sum_l * A.sum_l + A.diag_l) / denom;
        }
        const double scale = 1.0 / (nd * (nd - 1.0));
        batch_vals[size_t(s)] = scale * (e2_t - e2_l);
    }

    double mean = 0.0;
    for (double v : batch_vals) mean += v;
    mean /= double(n_samples);
    double var = 0.0;
    for (double v : batch_vals) var += (v - mean) * (v - mean);
    var /= double(n_samples - 1);
    if (mean_out) *mean_out = mean;
    if (stderr_out) *stderr_out = std::sqrt(var / double(n_samples));
}

}  // namespace vibeqc
