// Closed-shell DLPNO-CCSD amplitude solver contracted in PNO spaces.
//
// Every term below is one of the closed-shell DF-CCSD residual terms of
// python/vibeqc/dlpno/_ccsd_cs.py::cs_ccsd_residual (itself a transcription of
// compute_residuals in ccsd.cpp), evaluated for a target pair ij in that pair's
// PNO space with each contracted virtual index taken in the PNO space of the
// amplitude that carries it (Riplinger and Neese, J. Chem. Phys. 138, 034106
// (2013), Sec. II C). The comments name the term of cs_ccsd_residual they
// implement so the two can be compared line by line.
//
// Notation. For a kept pair q = (k, l), k <= l:
//   U_q      (nv, n_q)        PNO coefficients, orthonormal in the canonical
//                             virtual basis; S^{p,q} = U_p^T U_q.
//   T_q      (n_q, n_q)       t_{kl}^{ab}; the ordered amplitude T(l,k) = T_q^T.
//   Bov_q[m] (naux, n_q)      B^P_{m a} for a in [q], for every occupied m any
//                             target contraction needs (its own coupling set
//                             plus the "off-list" rows of other targets).
//   Bvv_q    (naux, n_q*n_q)  B^P_{ab}, a,b in [q], element (P, a*n_q + b).
// Singles t_i live in [ii]; T1full(i, :) = U_ii t_i is the full-space vector.
//   w_i (naux, no)  w_i(P, m) = sum_v B^P_{mv} T1full(i, v)
//   u_i (naux, nv)  u_i(P, a) = sum_v B^P_{av} T1full(i, v)
// so that sum_f t_i^f (mf|xy) = sum_P w_i(P,m) B^P_{xy} and
//         sum_f t_i^f (af|xy) = sum_P u_i(P,a) B^P_{xy} without truncation.

#include "vibeqc/dlpno_ccsd.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibeqc {

namespace {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using RowMat = DLPNORowMat;
using Index = Eigen::Index;

// Row-major (n, n) view of one P-slice of a (naux, n*n) factor.
using SliceMap = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                                Eigen::Dynamic, Eigen::RowMajor>>;

struct PairData {
    int i = 0;
    int j = 0;
    Index n = 0;
    Mat U;      // (nv, n)
    Vec eps;    // (n)
    Mat fvv;    // (n, n) U^T f_vv U
    Mat K;      // (n, n)
    Mat T;      // (n, n) current doubles t_ij^{ab}
    Mat Bvv;    // (naux, n*n) col-major with P as row index; (P, a*n+b)
    std::vector<int> L;        // coupling set (sorted)
    std::vector<int> rows;     // occupieds with a stored Bov row (superset of L)
    std::vector<int> rowpos;   // occupied -> index into rows / Bov, or -1 (size no)
    std::vector<Mat> Bov;      // Bov[r] (naux, n) for rows[r]
    // Static mixed Coulomb projections per m in L (index lm):
    //   JA1 = U_p^T J_{m j} U_{im}  (n, n_im)   [W2 of half(i,j): -(mj|be), e in [im]]
    //   JA2 = U_p^T J_{m i} U_{mj}  (n, n_mj)   [WX of half(i,j): -(mi|be), e in [mj]]
    //   JA3 = U_p^T J_{m i} U_{jm}  (n, n_jm)   [W2 of half(j,i)]
    //   JA4 = U_p^T J_{m j} U_{mi}  (n, n_mi)   [WX of half(j,i)]
    // with J_{mk} = sum_P B_oo(P, mk) B_vv[P]. Empty where the pair is absent.
    std::vector<Mat> JA1, JA2, JA3, JA4;
    // Dynamic counterparts with Jw_{mk} = sum_P w_k(P, m) B_vv[P], rebuilt
    // every iteration.
    std::vector<Mat> JwA1, JwA2, JwA3, JwA4;
    // 3-external blocks for the singles residual: X3[0](f*n+e, a) = (l f|a e)
    // with a in [kk]; X3[1](f*n+e, a) = (k f|a e) with a in [ll]. For a
    // diagonal pair only X3[0] is filled.
    Mat X3[2];
    // Per-iteration half-contracted amplitude objects (naux, n):
    //   D1(P,b) = sum_f Bov[l](P,f) T(b,f)   D2(P,b) = sum_f Bov[k](P,f) T(f,b)
    //   E1(P,b) = sum_f Bov[k](P,f) T(b,f)   E2(P,b) = sum_f Bov[l](P,f) T(f,b)
    // For the ordered pair (m,n) = (k,l): Dd = D1 (n-row, t_mn^{bf}), Ex = E1
    // (m-row, t_mn^{bf}); for (m,n) = (l,k): Dd = D2, Ex = E2.
    Mat D1, D2, E1, E2;
};

struct Workspace;

// out = Bvv^T C for the (n_aux, n_vir^2) global factor, with the n_vir^2 rows
// of the product split over OpenMP threads. vibe-qc pins the linked BLAS to
// one thread (python/vibeqc/__init__.py::_pin_blas_threads) so that it does
// not oversubscribe the OpenMP loops; a lone GEMM outside a parallel region
// would therefore run on one core, and this one is O(n_vir^2 n_aux n_mk).
void parallel_bvvT_times(const Eigen::Map<const RowMat>& Bvv, const Mat& C,
                         Mat& out) {
    const Index R = Bvv.cols();
    out.resize(R, C.cols());
    const Index blk = 2048;
    #pragma omp parallel for schedule(dynamic)
    for (Index r0 = 0; r0 < R; r0 += blk) {
        const Index rn = std::min(blk, R - r0);
        out.middleRows(r0, rn).noalias() = Bvv.middleCols(r0, rn).transpose() * C;
    }
}

class Solver {
public:
    Solver(const DLPNORowMatCRef& B_ov, const DLPNORowMatCRef& B_oo,
           const DLPNORowMatCRef& B_vv,
           const Mat& f_oo, const Mat& f_ov, const Mat& f_vv,
           std::vector<DLPNOPairInput> pairs, const DLPNOCCSDPNOOptions& opts);

    DLPNOCCSDPNOResult run();

private:
    // ---- static data ----
    const DLPNORowMatCRef B_ov_;
    const DLPNORowMatCRef B_oo_;
    const DLPNORowMatCRef B_vv_;
    const Mat& f_oo_;
    const Mat& f_ov_;
    const Mat& f_vv_;
    const DLPNOCCSDPNOOptions opts_;
    Index no_ = 0, nv_ = 0, naux_ = 0, np_ = 0;
    Vec f_dd_;
    std::vector<PairData> P_;
    std::vector<int> pidx_;   // (min*no + max) -> pair index or -1
    std::vector<int> diag_;   // i -> diagonal pair index or -1
    std::vector<std::vector<int>> pairs_of_occ_;  // i -> pair indices containing i
    std::size_t n_offlist_rows_ = 0;
    std::size_t static_bytes_ = 0;
    double setup_seconds_ = 0.0;

    // ---- per-iteration data ----
    std::vector<Vec> t1_;      // per occupied, in [ii]; empty if no diagonal pair
    Mat T1full_;               // (no, nv)
    std::vector<Mat> w_;       // w_[i] (naux, no)
    std::vector<Mat> u_;       // u_[i] (naux, nv)

    // ---- helpers ----
    int pid(int a, int b) const {
        const int lo = std::min(a, b), hi = std::max(a, b);
        return pidx_[static_cast<std::size_t>(lo) * no_ + hi];
    }
    // Ordered amplitude t_{ab}^{..}: T_q or T_q^T.
    Mat T(int a, int b) const {
        const PairData& q = P_[pid(a, b)];
        return (a <= b) ? q.T : Mat(q.T.transpose());
    }
    const Mat& bov(const PairData& q, int m) const {
        const int r = q.rowpos[m];
        if (r < 0) {
            std::ostringstream msg;
            msg << "dlpno_ccsd_pno_solve: no stored factor row for occupied "
                << m << " of pair (" << q.i << "," << q.j << ")";
            throw std::logic_error(msg.str());
        }
        return q.Bov[r];
    }
    // sum_f Bov_q[row r](P, f) A(b, f)  -> (naux, n_q), A = T(x,y) in [q].
    Mat yobj(const PairData& q, int r, const Mat& A) const {
        return bov(q, r) * A.transpose();
    }
    // Precomputed D/E object for ordered (m,n) when r is one of the pair's
    // own occupieds, else built on the fly.
    // direct:  sum_f Bov[n-row](P,f) t_mn^{bf}
    // exchange: sum_f Bov[m-row](P,f) t_mn^{bf}
    const Mat& dd(const PairData& q, int m, int n) const {
        return (m == q.i && n == q.j) ? q.D1 : q.D2;
    }
    const Mat& ex(const PairData& q, int m, int n) const {
        return (m == q.i && n == q.j) ? q.E1 : q.E2;
    }

    void build_static();
    void build_iteration_data();
    double energy() const;
    void residual_pair(Index p, Workspace& ws, Mat& R2) const;
    void residual_singles(int i, Vec& R1) const;
    std::vector<double> flatten() const;
    void unflatten(const std::vector<double>& v);
};

struct Workspace {
    // Per-target scratch that depends on the target's coupling set.
    std::vector<int> qidx;   // local pair index for (lm, ln) -> pair id or -1
    std::vector<Mat> S;      // S[lm*nL+ln] = U_p^T U_q for the pair q=(m,n)
    std::vector<Vec> t1p;    // singles of m in L, projected into [p]
};

// ---------------------------------------------------------------------------
Solver::Solver(const DLPNORowMatCRef& B_ov, const DLPNORowMatCRef& B_oo,
               const DLPNORowMatCRef& B_vv,
               const Mat& f_oo, const Mat& f_ov, const Mat& f_vv,
               std::vector<DLPNOPairInput> pairs,
               const DLPNOCCSDPNOOptions& opts)
    : B_ov_(B_ov), B_oo_(B_oo), B_vv_(B_vv),
      f_oo_(f_oo), f_ov_(f_ov), f_vv_(f_vv), opts_(opts) {
    no_ = f_oo.rows();
    nv_ = f_vv.rows();
    naux_ = B_ov.rows();
    if (f_oo.cols() != no_ || f_vv.cols() != nv_ || f_ov.rows() != no_
        || f_ov.cols() != nv_) {
        throw std::invalid_argument("dlpno_ccsd_pno_solve: Fock block shapes");
    }
    if (B_ov.cols() != no_ * nv_ || B_oo.rows() != naux_
        || B_oo.cols() != no_ * no_ || B_vv.rows() != naux_
        || B_vv.cols() != nv_ * nv_) {
        throw std::invalid_argument(
            "dlpno_ccsd_pno_solve: DF factor shapes must be (naux, no*nv), "
            "(naux, no*no), (naux, nv*nv)");
    }
    if (B_ov.outerStride() != B_ov.cols() || B_oo.outerStride() != B_oo.cols()
        || B_vv.outerStride() != B_vv.cols()) {
        throw std::invalid_argument(
            "dlpno_ccsd_pno_solve: DF factors must be C-contiguous row-major");
    }
    f_dd_ = f_oo.diagonal();
    np_ = static_cast<Index>(pairs.size());
    P_.resize(static_cast<std::size_t>(np_));
    pidx_.assign(static_cast<std::size_t>(no_ * no_), -1);
    diag_.assign(static_cast<std::size_t>(no_), -1);
    pairs_of_occ_.assign(static_cast<std::size_t>(no_), {});
    for (Index p = 0; p < np_; ++p) {
        DLPNOPairInput& in = pairs[static_cast<std::size_t>(p)];
        PairData& q = P_[static_cast<std::size_t>(p)];
        if (in.i < 0 || in.j < 0 || in.i >= no_ || in.j >= no_ || in.i > in.j) {
            throw std::invalid_argument(
                "dlpno_ccsd_pno_solve: pair indices must satisfy 0 <= i <= j < n_act");
        }
        q.i = in.i;
        q.j = in.j;
        q.n = in.U.cols();
        if (in.U.rows() != nv_ || in.eps.size() != q.n
            || in.K.rows() != q.n || in.K.cols() != q.n
            || in.T2.rows() != q.n || in.T2.cols() != q.n) {
            throw std::invalid_argument(
                "dlpno_ccsd_pno_solve: pair PNO data shapes are inconsistent");
        }
        q.U = std::move(in.U);
        q.eps = std::move(in.eps);
        q.fvv = q.U.transpose() * f_vv_ * q.U;
        q.K = std::move(in.K);
        q.T = std::move(in.T2);
        q.L = std::move(in.L);
        std::sort(q.L.begin(), q.L.end());
        q.L.erase(std::unique(q.L.begin(), q.L.end()), q.L.end());
        if (!std::binary_search(q.L.begin(), q.L.end(), q.i)
            || !std::binary_search(q.L.begin(), q.L.end(), q.j)) {
            throw std::invalid_argument(
                "dlpno_ccsd_pno_solve: a pair's coupling set must contain both "
                "of its occupieds");
        }
        const std::size_t key = static_cast<std::size_t>(q.i) * no_ + q.j;
        if (pidx_[key] >= 0) {
            throw std::invalid_argument("dlpno_ccsd_pno_solve: duplicate pair");
        }
        pidx_[key] = static_cast<int>(p);
        if (q.i == q.j) diag_[static_cast<std::size_t>(q.i)] = static_cast<int>(p);
        pairs_of_occ_[static_cast<std::size_t>(q.i)].push_back(static_cast<int>(p));
        if (q.i != q.j) {
            pairs_of_occ_[static_cast<std::size_t>(q.j)].push_back(static_cast<int>(p));
        }
    }
    t1_.assign(static_cast<std::size_t>(no_), Vec());
    for (Index i = 0; i < no_; ++i) {
        const int d = diag_[static_cast<std::size_t>(i)];
        if (d >= 0) t1_[static_cast<std::size_t>(i)] = Vec::Zero(P_[static_cast<std::size_t>(d)].n);
    }
    const auto t0 = std::chrono::steady_clock::now();
    build_static();
    setup_seconds_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Static factors: Bov rows, Bvv, mixed Coulomb projections, 3-external blocks.
// ---------------------------------------------------------------------------
void Solver::build_static() {
    // 1. Which occupied rows of each pair's Bov does any target need? For a
    //    target p every kept pair q inside L_p x L_p is contracted with rows
    //    m in L_p, so rows_q = union of L_p over the targets p that contain q.
    std::vector<std::vector<char>> need(static_cast<std::size_t>(np_),
                                        std::vector<char>(static_cast<std::size_t>(no_), 0));
    for (Index p = 0; p < np_; ++p) {
        const PairData& t = P_[static_cast<std::size_t>(p)];
        for (int m : t.L)
            for (int n : t.L) {
                if (m > n) continue;
                const int q = pid(m, n);
                if (q < 0) continue;
                for (int r : t.L) need[static_cast<std::size_t>(q)][static_cast<std::size_t>(r)] = 1;
            }
    }
    n_offlist_rows_ = 0;
    static_bytes_ = 0;
    for (Index p = 0; p < np_; ++p) {
        PairData& q = P_[static_cast<std::size_t>(p)];
        q.rowpos.assign(static_cast<std::size_t>(no_), -1);
        q.rows.clear();
        for (int m : q.L) need[static_cast<std::size_t>(p)][static_cast<std::size_t>(m)] = 1;
        for (int m = 0; m < no_; ++m) {
            if (!need[static_cast<std::size_t>(p)][static_cast<std::size_t>(m)]) continue;
            q.rowpos[static_cast<std::size_t>(m)] = static_cast<int>(q.rows.size());
            q.rows.push_back(m);
            if (!std::binary_search(q.L.begin(), q.L.end(), m)) ++n_offlist_rows_;
        }
    }

    // 2. Bov rows and Bvv per pair (parallel over pairs; each is one GEMM
    //    with the global factor).
    #pragma omp parallel for schedule(dynamic)
    for (Index p = 0; p < np_; ++p) {
        PairData& q = P_[static_cast<std::size_t>(p)];
        const Index n = q.n;
        q.Bov.resize(q.rows.size());
        for (std::size_t r = 0; r < q.rows.size(); ++r) {
            const int m = q.rows[r];
            // B_ov(P, m*nv + v) for all P: element (P, m*nv+v) sits at
            // P*cols + m*nv + v, a row-major (naux, nv) block whose rows are
            // `cols` apart.
            Eigen::Map<const RowMat, 0, Eigen::OuterStride<>> Bm(
                B_ov_.data() + static_cast<std::ptrdiff_t>(m) * nv_, naux_, nv_,
                Eigen::OuterStride<>(B_ov_.outerStride()));
            q.Bov[r].noalias() = Bm * q.U;
        }
        // Bvv_half(P*nv + a', b) = sum_v B_vv(P, a'*nv + v) U(v, b): one GEMM on
        // the (naux*nv, nv) row-major view of B_vv.
        Eigen::Map<const RowMat> Bvv_view(B_vv_.data(), naux_ * nv_, nv_);
        Mat half = Bvv_view * q.U;  // (naux*nv, n) column-major
        q.Bvv.resize(naux_, n * n);
        for (Index P = 0; P < naux_; ++P) {
            // slice (nv, n) of half for this P, rows P*nv .. P*nv+nv-1
            const Mat blk = q.U.transpose() * half.middleRows(P * nv_, nv_);  // (n, n): (a, b)
            for (Index a = 0; a < n; ++a)
                for (Index b = 0; b < n; ++b)
                    q.Bvv(P, a * n + b) = blk(a, b);
        }
        // 3-external blocks: (l f|a e) with a in [kk] (X3[0]) and (k f|a e)
        // with a in [ll] (X3[1]), e,f in [q]. Bvv^{kk,q}(P)(a, e) =
        // U_kk^T half_P.
        const int dk = diag_[static_cast<std::size_t>(q.i)];
        const int dl = diag_[static_cast<std::size_t>(q.j)];
        for (int role = 0; role < 2; ++role) {
            if (role == 1 && q.i == q.j) break;
            const int d = (role == 0) ? dk : dl;
            if (d < 0) continue;
            const PairData& dp = P_[static_cast<std::size_t>(d)];
            const int occ_row = (role == 0) ? q.j : q.i;   // (l f|..) or (k f|..)
            const Mat& Bf = bov(q, occ_row);               // (naux, n) over f
            Mat X = Mat::Zero(n * n, dp.n);                // (f*n+e, a)
            for (Index P = 0; P < naux_; ++P) {
                const Mat mixed = dp.U.transpose() * half.middleRows(P * nv_, nv_);  // (n_d, n): (a, e)
                for (Index f = 0; f < n; ++f) {
                    const double bf = Bf(P, f);
                    if (bf == 0.0) continue;
                    // X(f*n+e, a) += bf * mixed(a, e)
                    X.middleRows(f * n, n).noalias() += bf * mixed.transpose();
                }
            }
            q.X3[role] = std::move(X);
        }
    }

    // 3. Mixed Coulomb projections. For each ordered (m, k) with m in the
    //    coupling set of some pair containing k, J_{mk} = sum_P B_oo(P, mk)
    //    B_vv[P]; batch the (m,k) list into GEMMs against B_vv.
    std::vector<std::pair<int, int>> mk_list;
    {
        std::vector<char> seen(static_cast<std::size_t>(no_ * no_), 0);
        for (Index p = 0; p < np_; ++p) {
            const PairData& t = P_[static_cast<std::size_t>(p)];
            for (int m : t.L)
                for (int k : {t.i, t.j}) {
                    const std::size_t key = static_cast<std::size_t>(m) * no_ + k;
                    if (!seen[key]) { seen[key] = 1; mk_list.emplace_back(m, k); }
                }
        }
    }
    for (Index p = 0; p < np_; ++p) {
        PairData& t = P_[static_cast<std::size_t>(p)];
        const std::size_t nL = t.L.size();
        t.JA1.assign(nL, Mat()); t.JA2.assign(nL, Mat());
        t.JA3.assign(nL, Mat()); t.JA4.assign(nL, Mat());
        t.JwA1.assign(nL, Mat()); t.JwA2.assign(nL, Mat());
        t.JwA3.assign(nL, Mat()); t.JwA4.assign(nL, Mat());
    }
    // Coefficient columns: C(P, idx) = B_oo(P, m*no + k).
    const Index n_mk = static_cast<Index>(mk_list.size());
    const Index chunk = 128;
    Eigen::Map<const RowMat> Bvv_view(B_vv_.data(), naux_, nv_ * nv_);
    for (Index c0 = 0; c0 < n_mk; c0 += chunk) {
        const Index cn = std::min(chunk, n_mk - c0);
        Mat C(naux_, cn);
        for (Index c = 0; c < cn; ++c) {
            const auto [m, k] = mk_list[static_cast<std::size_t>(c0 + c)];
            for (Index P = 0; P < naux_; ++P) C(P, c) = B_oo_(P, m * no_ + k);
        }
        // Jflat(v*nv + v', c) = sum_P B_vv(P, v*nv+v') C(P, c)
        Mat Jflat;
        parallel_bvvT_times(Bvv_view, C, Jflat);   // (nv*nv, cn)
        #pragma omp parallel for schedule(dynamic)
        for (Index c = 0; c < cn; ++c) {
            const auto [m, k] = mk_list[static_cast<std::size_t>(c0 + c)];
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>> J(Jflat.col(c).data(), nv_, nv_);
            // Targets p containing k with m in L_p.
            for (int p : pairs_of_occ_[static_cast<std::size_t>(k)]) {
                PairData& t = P_[static_cast<std::size_t>(p)];
                const auto it = std::lower_bound(t.L.begin(), t.L.end(), m);
                if (it == t.L.end() || *it != m) continue;
                const std::size_t lm = static_cast<std::size_t>(it - t.L.begin());
                const Mat UtJ = t.U.transpose() * J;   // (n_p, nv)
                // k == t.j: JA1 = U_p^T J_{mj} U_{im}, JA4 = U_p^T J_{mj} U_{mi}
                if (k == t.j) {
                    const int qim = pid(t.i, m);
                    if (qim >= 0) {
                        t.JA1[lm] = UtJ * P_[static_cast<std::size_t>(qim)].U;
                        t.JA4[lm] = t.JA1[lm];
                    }
                }
                // k == t.i: JA2 = U_p^T J_{mi} U_{mj}, JA3 = U_p^T J_{mi} U_{jm}
                if (k == t.i) {
                    const int qmj = pid(m, t.j);
                    if (qmj >= 0) {
                        t.JA2[lm] = UtJ * P_[static_cast<std::size_t>(qmj)].U;
                        t.JA3[lm] = t.JA2[lm];
                    }
                }
            }
        }
    }
    for (Index p = 0; p < np_; ++p) {
        const PairData& q = P_[static_cast<std::size_t>(p)];
        static_bytes_ += sizeof(double) * static_cast<std::size_t>(naux_) * q.n * q.n;
        static_bytes_ += sizeof(double) * static_cast<std::size_t>(naux_) * q.n * q.rows.size();
    }
}

// ---------------------------------------------------------------------------
// Per-iteration data: singles vectors, D/E objects, Jw projections.
// ---------------------------------------------------------------------------
void Solver::build_iteration_data() {
    T1full_ = Mat::Zero(no_, nv_);
    for (Index i = 0; i < no_; ++i) {
        const int d = diag_[static_cast<std::size_t>(i)];
        if (d >= 0) T1full_.row(i) = (P_[static_cast<std::size_t>(d)].U * t1_[static_cast<std::size_t>(i)]).transpose();
    }
    // w_i(P, m) = sum_v B_ov(P, m*nv+v) T1full(i, v): per P, (no, nv) x (nv, no).
    w_.assign(static_cast<std::size_t>(no_), Mat(naux_, no_));
    u_.assign(static_cast<std::size_t>(no_), Mat(naux_, nv_));
    const Mat T1t = T1full_.transpose();   // (nv, no)
    #pragma omp parallel for schedule(static)
    for (Index P = 0; P < naux_; ++P) {
        SliceMap Bov_P(B_ov_.row(P).data(), no_, nv_);
        const Mat wP = Bov_P * T1t;                 // (no, no): (m, i)
        SliceMap Bvv_P(B_vv_.row(P).data(), nv_, nv_);
        const Mat uP = Bvv_P * T1t;                 // (nv, no): (a, i)
        for (Index i = 0; i < no_; ++i) {
            w_[static_cast<std::size_t>(i)].row(P) = wP.col(i).transpose();
            u_[static_cast<std::size_t>(i)].row(P) = uP.col(i).transpose();
        }
    }
    // D/E objects.
    #pragma omp parallel for schedule(dynamic)
    for (Index p = 0; p < np_; ++p) {
        PairData& q = P_[static_cast<std::size_t>(p)];
        const Mat& Bk = bov(q, q.i);
        const Mat& Bl = bov(q, q.j);
        q.D1.noalias() = Bl * q.T.transpose();   // sum_f Bov[l](P,f) T(b,f)
        q.E1.noalias() = Bk * q.T.transpose();   // sum_f Bov[k](P,f) T(b,f)
        q.D2.noalias() = Bk * q.T;               // sum_f Bov[k](P,f) T(f,b)
        q.E2.noalias() = Bl * q.T;               // sum_f Bov[l](P,f) T(f,b)
    }
    // Jw projections: Jw_{mk} = sum_P w_k(P, m) B_vv[P], same (m,k) list and
    // target bookkeeping as the static J blocks.
    std::vector<std::pair<int, int>> mk_list;
    {
        std::vector<char> seen(static_cast<std::size_t>(no_ * no_), 0);
        for (Index p = 0; p < np_; ++p) {
            const PairData& t = P_[static_cast<std::size_t>(p)];
            for (int m : t.L)
                for (int k : {t.i, t.j}) {
                    const std::size_t key = static_cast<std::size_t>(m) * no_ + k;
                    if (!seen[key]) { seen[key] = 1; mk_list.emplace_back(m, k); }
                }
        }
    }
    const Index n_mk = static_cast<Index>(mk_list.size());
    const Index chunk = 128;
    Eigen::Map<const RowMat> Bvv_view(B_vv_.data(), naux_, nv_ * nv_);
    for (Index c0 = 0; c0 < n_mk; c0 += chunk) {
        const Index cn = std::min(chunk, n_mk - c0);
        Mat C(naux_, cn);
        for (Index c = 0; c < cn; ++c) {
            const auto [m, k] = mk_list[static_cast<std::size_t>(c0 + c)];
            C.col(c) = w_[static_cast<std::size_t>(k)].col(m);
        }
        Mat Jflat;
        parallel_bvvT_times(Bvv_view, C, Jflat);   // (nv*nv, cn)
        #pragma omp parallel for schedule(dynamic)
        for (Index c = 0; c < cn; ++c) {
            const auto [m, k] = mk_list[static_cast<std::size_t>(c0 + c)];
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                           Eigen::RowMajor>> J(Jflat.col(c).data(), nv_, nv_);
            for (int p : pairs_of_occ_[static_cast<std::size_t>(k)]) {
                PairData& t = P_[static_cast<std::size_t>(p)];
                const auto it = std::lower_bound(t.L.begin(), t.L.end(), m);
                if (it == t.L.end() || *it != m) continue;
                const std::size_t lm = static_cast<std::size_t>(it - t.L.begin());
                const Mat UtJ = t.U.transpose() * J;
                if (k == t.j) {
                    const int qim = pid(t.i, m);
                    if (qim >= 0) {
                        t.JwA1[lm] = UtJ * P_[static_cast<std::size_t>(qim)].U;
                        t.JwA4[lm] = t.JwA1[lm];
                    }
                }
                if (k == t.i) {
                    const int qmj = pid(m, t.j);
                    if (qmj >= 0) {
                        t.JwA2[lm] = UtJ * P_[static_cast<std::size_t>(qmj)].U;
                        t.JwA3[lm] = t.JwA2[lm];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Energy with the same convention as the Python solver: singles are
// projected into the pair space before forming tau.
// ---------------------------------------------------------------------------
double Solver::energy() const {
    double e = 0.0;
    for (Index i = 0; i < no_; ++i) {
        const int d = diag_[static_cast<std::size_t>(i)];
        if (d < 0) continue;
        const PairData& dp = P_[static_cast<std::size_t>(d)];
        const Vec f_ovi = (f_ov_.row(i) * dp.U).transpose();
        e += 2.0 * t1_[static_cast<std::size_t>(i)].dot(f_ovi);
    }
    for (Index p = 0; p < np_; ++p) {
        const PairData& q = P_[static_cast<std::size_t>(p)];
        Mat tau = q.T;
        const int di = diag_[static_cast<std::size_t>(q.i)];
        const int dj = diag_[static_cast<std::size_t>(q.j)];
        if (di >= 0 && dj >= 0) {
            const Vec ti = q.U.transpose() * (P_[static_cast<std::size_t>(di)].U * t1_[static_cast<std::size_t>(q.i)]);
            const Vec tj = q.U.transpose() * (P_[static_cast<std::size_t>(dj)].U * t1_[static_cast<std::size_t>(q.j)]);
            tau.noalias() += ti * tj.transpose();
        }
        const double wgt = (q.i == q.j) ? 1.0 : 2.0;
        e += wgt * (tau.array() * (2.0 * q.K - q.K.transpose()).array()).sum();
    }
    return e;
}

// ---------------------------------------------------------------------------
// Doubles residual of target pair p in [p].
// ---------------------------------------------------------------------------
void Solver::residual_pair(Index p, Workspace& ws, Mat& R2) const {
    const PairData& t = P_[static_cast<std::size_t>(p)];
    const int i = t.i, j = t.j;
    const Index n = t.n;
    const std::vector<int>& L = t.L;
    const Index nL = static_cast<Index>(L.size());

    // ---- target-local structures ----
    ws.qidx.assign(static_cast<std::size_t>(nL * nL), -1);
    ws.S.assign(static_cast<std::size_t>(nL * nL), Mat());
    for (Index lm = 0; lm < nL; ++lm)
        for (Index ln = lm; ln < nL; ++ln) {
            const int q = pid(L[static_cast<std::size_t>(lm)], L[static_cast<std::size_t>(ln)]);
            ws.qidx[static_cast<std::size_t>(lm * nL + ln)] = q;
            ws.qidx[static_cast<std::size_t>(ln * nL + lm)] = q;
            if (q < 0) continue;
            Mat S = t.U.transpose() * P_[static_cast<std::size_t>(q)].U;
            ws.S[static_cast<std::size_t>(ln * nL + lm)] = S;
            ws.S[static_cast<std::size_t>(lm * nL + ln)] = std::move(S);
        }
    auto qof = [&](Index lm, Index ln) { return ws.qidx[static_cast<std::size_t>(lm * nL + ln)]; };
    auto Sof = [&](Index lm, Index ln) -> const Mat& { return ws.S[static_cast<std::size_t>(lm * nL + ln)]; };
    auto lpos = [&](int m) {
        return static_cast<Index>(std::lower_bound(L.begin(), L.end(), m) - L.begin());
    };
    const Index li = lpos(i), lj = lpos(j);
    // Singles projected into [p].
    ws.t1p.assign(static_cast<std::size_t>(nL), Vec::Zero(n));
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        const int d = diag_[static_cast<std::size_t>(m)];
        if (d < 0) continue;
        ws.t1p[static_cast<std::size_t>(lm)] = Sof(lm, lm) * t1_[static_cast<std::size_t>(m)];
    }
    auto t1p = [&](Index lm) -> const Vec& { return ws.t1p[static_cast<std::size_t>(lm)]; };
    // Ordered amplitude of (m, n) projected into [p] on both indices.
    auto tproj = [&](Index lm, Index ln) -> Mat {
        const int q = qof(lm, ln);
        if (q < 0) return Mat::Zero(n, n);
        const Mat& S = Sof(lm, ln);
        return S * T(L[static_cast<std::size_t>(lm)], L[static_cast<std::size_t>(ln)]) * S.transpose();
    };
    auto tau_proj = [&](Index lm, Index ln) -> Mat {
        Mat tau = tproj(lm, ln);
        tau.noalias() += t1p(lm) * t1p(ln).transpose();
        return tau;
    };
    // Occupied-index columns of B_oo for the target's occupieds:
    // Boo_x(P, lm) = B_oo(P, m*no + x).
    auto boo_cols = [&](int x) {
        Mat C(naux_, nL);
        for (Index lm = 0; lm < nL; ++lm) {
            const int m = L[static_cast<std::size_t>(lm)];
            for (Index P = 0; P < naux_; ++P) C(P, lm) = B_oo_(P, m * no_ + x);
        }
        return C;
    };
    const Mat Boo_i = boo_cols(i);
    const Mat Boo_j = boo_cols(j);
    // w_x restricted to the coupling set: Wx(P, lm) = w_x(P, m).
    auto w_cols = [&](int x) {
        Mat W(naux_, nL);
        for (Index lm = 0; lm < nL; ++lm) W.col(lm) = w_[static_cast<std::size_t>(x)].col(L[static_cast<std::size_t>(lm)]);
        return W;
    };
    const Mat Wi = w_cols(i);
    const Mat Wj = w_cols(j);
    // u_x projected into [p]: (naux, n).
    auto u_proj = [&](int x) -> Mat { return u_[static_cast<std::size_t>(x)] * t.U; };
    const Mat u_i = u_proj(i);
    const Mat u_j = (i == j) ? u_i : u_proj(j);
    // w_m(P, m) summed over m in L (for sum_m t_m^f (mf|..)).
    Vec wsum = Vec::Zero(naux_);
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        if (diag_[static_cast<std::size_t>(m)] >= 0) wsum += w_[static_cast<std::size_t>(m)].col(m);
    }
    // Bov rows of the target for its coupling set.
    std::vector<const Mat*> Bp(static_cast<std::size_t>(nL));
    for (Index lm = 0; lm < nL; ++lm) Bp[static_cast<std::size_t>(lm)] = &bov(t, L[static_cast<std::size_t>(lm)]);

    // Q_m(P, f) = sum_e Bov_p[m](P, e) T(e, f), and Z(m, n) = sum_{P,f}
    // Q_m(P,f) Bov_p[n](P,f) = sum_ef t_ij^{ef} (me|nf).
    std::vector<Mat> Q(static_cast<std::size_t>(nL));
    for (Index lm = 0; lm < nL; ++lm) Q[static_cast<std::size_t>(lm)].noalias() = (*Bp[static_cast<std::size_t>(lm)]) * t.T;
    Mat Z(nL, nL);
    for (Index lm = 0; lm < nL; ++lm)
        for (Index ln = 0; ln < nL; ++ln)
            Z(lm, ln) = (Q[static_cast<std::size_t>(lm)].array() * Bp[static_cast<std::size_t>(ln)]->array()).sum();
    // Z1(m, n) = sum_P w_i(P,m) w_j(P,n) = sum_ef t_i^e t_j^f (me|nf).
    const Mat Z1 = Wi.transpose() * Wj;

    // ---- R2 = (ia|jb) ----
    R2 = t.K;

    // ---- + sum_mn W_mnij tau_mn (projected) ----
    // W_mnij = (mi|nj) + sum_e t_j^e (mi|ne) + sum_e t_i^e (nj|me)
    //        + 1/2 sum_ef tau_ij^{ef} (me|nf)
    {
        Mat W = Boo_i.transpose() * Boo_j;            // (mi|nj)
        W.noalias() += Boo_i.transpose() * Wj;         // sum_e t_j^e (mi|ne)
        W.noalias() += (Boo_j.transpose() * Wi).transpose();  // sum_e t_i^e (nj|me)
        W += 0.5 * (Z + Z1);
        for (Index lm = 0; lm < nL; ++lm)
            for (Index ln = 0; ln < nL; ++ln) {
                const double wv = W(lm, ln);
                if (wv == 0.0 || qof(lm, ln) < 0) {
                    // No doubles: only the singles product survives.
                    if (wv != 0.0) R2.noalias() += wv * (t1p(lm) * t1p(ln).transpose());
                    continue;
                }
                R2.noalias() += wv * tau_proj(lm, ln);
            }
    }

    // ---- + sum_ef tau_ij^{ef} W_abef  (pair-space ladder, cs_ccsd_pair_ladder) ----
    {
        // Work per P with row-major slices of Bvv.
        Mat Rlad = Mat::Zero(n, n);
        Mat X = Mat::Zero(nL, n);   // sum_{P,f} Bov_p[m](P,f) G(P,a,f)
        Mat Y = Mat::Zero(nL, n);   // sum_{P,e} Bov_p[m](P,e) H(P,b,e)
        Mat G(n, n), H(n, n);
        // Column-major Bvv (naux, n*n): row P holds (a*n+b). Copy slice to a
        // row-major (n, n) matrix per P.
        Mat slice(n, n);
        for (Index P = 0; P < naux_; ++P) {
            for (Index a = 0; a < n; ++a)
                for (Index b = 0; b < n; ++b) slice(a, b) = t.Bvv(P, a * n + b);
            G.noalias() = slice * t.T;                 // (a, f) = sum_e (ae|P) t_ef
            Rlad.noalias() += G * slice.transpose();   // sum_f G(a,f) (bf|P)
            H.noalias() = slice * t.T.transpose();     // (b, e) = sum_f (bf|P) t_ef
            for (Index lm = 0; lm < nL; ++lm) {
                const Mat& Bm = *Bp[static_cast<std::size_t>(lm)];
                X.row(lm).noalias() += Bm.row(P) * G.transpose();   // sum_f Bm(P,f) G(a,f)
                Y.row(lm).noalias() += Bm.row(P) * H.transpose();   // sum_e Bm(P,e) H(b,e)
            }
        }
        for (Index lm = 0; lm < nL; ++lm) {
            Rlad.noalias() -= X.row(lm).transpose() * t1p(lm).transpose();  // -t_m^b (mf|ae)
            Rlad.noalias() -= t1p(lm) * Y.row(lm);                          // -t_m^a (me|bf)
        }
        // +1/2 sum_mn tau_mn^{ab} [sum_ef t_ij^{ef} (me|nf)]  (Z)
        // + t1 x t1 piece contracted in the mixed basis:
        //   (ae|bf) -> u_i^T u_j ; -t_m^b (mf|ae) -> -(w_j^T u_i)(m,a) t_m^b
        //   -t_m^a (me|bf) -> -t_m^a (w_i^T u_j)(m,b) ; +1/2 tau_mn Z1
        Rlad.noalias() += u_i.transpose() * u_j;
        const Mat A = Wj.transpose() * u_i;   // (nL, n): (m, a)
        const Mat Bm2 = Wi.transpose() * u_j; // (nL, n): (m, b)
        for (Index lm = 0; lm < nL; ++lm) {
            Rlad.noalias() -= A.row(lm).transpose() * t1p(lm).transpose();
            Rlad.noalias() -= t1p(lm) * Bm2.row(lm);
        }
        for (Index lm = 0; lm < nL; ++lm)
            for (Index ln = 0; ln < nL; ++ln) {
                const double z = 0.5 * (Z(lm, ln) + Z1(lm, ln));
                if (z == 0.0) continue;
                if (qof(lm, ln) < 0) {
                    Rlad.noalias() += z * (t1p(lm) * t1p(ln).transpose());
                } else {
                    Rlad.noalias() += z * tau_proj(lm, ln);
                }
            }
        R2 += Rlad;
    }

    // ---- F_be in [p] (F_ae of cs_ccsd_residual restricted to b,e in [p]) ----
    // F_ae = f_ae - 1/2 sum_m f_me t_m^a + sum_mf t_m^f [2(mf|ae) - (me|af)]
    //      - sum_mnf taut_mn^{af} [2(me|nf) - (mf|ne)]
    Mat Fbe = t.fvv;   // U^T f_vv U (diagonal to lindep precision)
    {
        const Mat fovU = f_ov_ * t.U;   // (no, n): (m, e)
        for (Index lm = 0; lm < nL; ++lm) {
            const int m = L[static_cast<std::size_t>(lm)];
            Fbe.noalias() -= 0.5 * t1p(lm) * fovU.row(m);   // (b, e)
        }
        // 2 sum_mf t_m^f (mf|be) = 2 sum_P wsum(P) Bvv(P, b*n+e)
        {
            const Vec v = t.Bvv.transpose() * wsum;   // (n*n)
            for (Index b = 0; b < n; ++b)
                for (Index e = 0; e < n; ++e) Fbe(b, e) += 2.0 * v(b * n + e);
        }
        // - sum_mf t_m^f (me|bf) = - sum_m sum_P Bov_p[m](P,e) u_m^{[p]}(P,b)
        for (Index lm = 0; lm < nL; ++lm) {
            const int m = L[static_cast<std::size_t>(lm)];
            if (diag_[static_cast<std::size_t>(m)] < 0) continue;
            const Mat um = u_proj(m);                       // (naux, n): (P, b)
            Fbe.noalias() -= um.transpose() * (*Bp[static_cast<std::size_t>(lm)]);   // (b, e)
        }
        // - sum_mn [2 sum_f taut_mn^{bf} (me|nf) - sum_f taut_mn^{bf} (mf|ne)]
        for (Index lm = 0; lm < nL; ++lm)
            for (Index ln = 0; ln < nL; ++ln) {
                const int m = L[static_cast<std::size_t>(lm)], nn = L[static_cast<std::size_t>(ln)];
                const int q = qof(lm, ln);
                if (q >= 0) {
                    const PairData& qd = P_[static_cast<std::size_t>(q)];
                    const Mat& S = Sof(lm, ln);   // (n_p, n_q)
                    // direct: sum_P Bov_p[m](P,e) Dd_mn(P,b') S(b,b')
                    const Mat Md = Bp[static_cast<std::size_t>(lm)]->transpose() * dd(qd, m, nn);  // (e, b')
                    Fbe.noalias() -= 2.0 * (S * Md.transpose());                                   // (b, e)
                    // exchange: sum_P Ex_mn(P,b') Bov_p[n](P,e)
                    const Mat Mx = Bp[static_cast<std::size_t>(ln)]->transpose() * ex(qd, m, nn);  // (e, b')
                    Fbe.noalias() += S * Mx.transpose();
                }
                // singles part of taut: 1/2 t_m^b t_n^f
                if (diag_[static_cast<std::size_t>(m)] >= 0 && diag_[static_cast<std::size_t>(nn)] >= 0) {
                    const Vec vd = Bp[static_cast<std::size_t>(lm)]->transpose() * w_[static_cast<std::size_t>(nn)].col(nn);  // (e)
                    const Vec vx = Bp[static_cast<std::size_t>(ln)]->transpose() * w_[static_cast<std::size_t>(nn)].col(m);   // (e)
                    Fbe.noalias() -= t1p(lm) * vd.transpose();          // 2 * 1/2
                    Fbe.noalias() += 0.5 * t1p(lm) * vx.transpose();
                }
            }
    }

    // ---- F_me for m in L, e in [p] (sum over n in L, the target's set) ----
    // F_me = f_me + sum_nf t_n^f [2(me|nf) - (mf|ne)]
    Mat Fme(nL, n);
    {
        const Mat fovU = f_ov_ * t.U;
        for (Index lm = 0; lm < nL; ++lm) {
            const int m = L[static_cast<std::size_t>(lm)];
            Vec row = fovU.row(m).transpose();
            row.noalias() += 2.0 * (Bp[static_cast<std::size_t>(lm)]->transpose() * wsum);
            for (Index ln = 0; ln < nL; ++ln) {
                const int nn = L[static_cast<std::size_t>(ln)];
                if (diag_[static_cast<std::size_t>(nn)] < 0) continue;
                row.noalias() -= Bp[static_cast<std::size_t>(ln)]->transpose() * w_[static_cast<std::size_t>(nn)].col(m);
            }
            Fme.row(lm) = row.transpose();
        }
    }
    // Full-space F_me contracted with a singles vector: sum_e t_x^e F_me(m, e).
    auto Fme_dot_t1 = [&](Index lm, int x) -> double {
        const int m = L[static_cast<std::size_t>(lm)];
        if (diag_[static_cast<std::size_t>(x)] < 0) return 0.0;
        double s = f_ov_.row(m).dot(T1full_.row(x));
        // sum_nf t_n^f [2 sum_e t_x^e (me|nf) - sum_e t_x^e (mf|ne)]
        //   = sum_n [2 sum_P w_x(P,m) w_n(P,n) - sum_P w_n(P,m) w_x(P,n)]
        const Mat& wx = w_[static_cast<std::size_t>(x)];
        for (Index ln = 0; ln < nL; ++ln) {
            const int nn = L[static_cast<std::size_t>(ln)];
            if (diag_[static_cast<std::size_t>(nn)] < 0) continue;
            const Mat& wn = w_[static_cast<std::size_t>(nn)];
            s += 2.0 * wx.col(m).dot(wn.col(nn)) - wn.col(m).dot(wx.col(nn));
        }
        return s;
    };
    // Fh_be = F_be - 1/2 sum_m t_m^b F_me
    Mat Fh_be = Fbe;
    for (Index lm = 0; lm < nL; ++lm) Fh_be.noalias() -= 0.5 * t1p(lm) * Fme.row(lm);

    // ---- F_mx for m in L and x in {i, j} (sum over n in L) ----
    // F_mi = f_mi + 1/2 sum_e t_i^e f_me + sum_ne t_n^e [2(mi|ne) - (ni|me)]
    //      + sum_nef taut_in^{ef} [2(me|nf) - (mf|ne)]
    auto Fmx = [&](Index lm, int x, Index lx) -> double {
        const int m = L[static_cast<std::size_t>(lm)];
        double s = f_oo_(m, x) + 0.5 * f_ov_.row(m).dot(T1full_.row(x));
        for (Index ln = 0; ln < nL; ++ln) {
            const int nn = L[static_cast<std::size_t>(ln)];
            if (diag_[static_cast<std::size_t>(nn)] >= 0) {
                // 2 sum_e t_n^e (mx|ne) - sum_e t_n^e (nx|me)
                const Mat& wn = w_[static_cast<std::size_t>(nn)];
                Vec bmx(naux_), bnx(naux_);
                for (Index P = 0; P < naux_; ++P) {
                    bmx(P) = B_oo_(P, m * no_ + x);
                    bnx(P) = B_oo_(P, nn * no_ + x);
                }
                s += 2.0 * bmx.dot(wn.col(nn)) - bnx.dot(wn.col(m));
            }
            const int q = qof(lx, ln);   // pair (x, n)
            if (q >= 0) {
                const PairData& qd = P_[static_cast<std::size_t>(q)];
                const Mat Txn = T(x, nn);                 // (e, f) in [q]
                const Mat& Bm = bov(qd, m);               // (P, e/f)
                const Mat& Bn = bov(qd, nn);
                // 2 sum_ef t^{ef} (me|nf) - sum_ef t^{ef} (mf|ne)
                const Mat BmT = Bm * Txn;                 // (P, f) = sum_e Bm(P,e) T(e,f)
                s += 2.0 * (BmT.array() * Bn.array()).sum();
                const Mat BnT = Bn * Txn;                 // (P, f) = sum_e Bn(P,e) T(e,f)
                s -= (BnT.array() * Bm.array()).sum();
            }
            if (diag_[static_cast<std::size_t>(x)] >= 0 && diag_[static_cast<std::size_t>(nn)] >= 0) {
                // 1/2 t_x^e t_n^f [2(me|nf) - (mf|ne)]
                const Mat& wx = w_[static_cast<std::size_t>(x)];
                const Mat& wn = w_[static_cast<std::size_t>(nn)];
                s += wx.col(m).dot(wn.col(nn)) - 0.5 * wn.col(m).dot(wx.col(nn));
            }
        }
        return s;
    };
    std::vector<double> Fmi(static_cast<std::size_t>(nL)), Fmj(static_cast<std::size_t>(nL));
    for (Index lm = 0; lm < nL; ++lm) {
        Fmi[static_cast<std::size_t>(lm)] = Fmx(lm, i, li);
        Fmj[static_cast<std::size_t>(lm)] = (i == j) ? Fmi[static_cast<std::size_t>(lm)] : Fmx(lm, j, lj);
    }
    // Fh_mj = F_mj + 1/2 sum_e t_j^e F_me
    std::vector<double> Fh_mi(static_cast<std::size_t>(nL)), Fh_mj(static_cast<std::size_t>(nL));
    for (Index lm = 0; lm < nL; ++lm) {
        Fh_mi[static_cast<std::size_t>(lm)] = Fmi[static_cast<std::size_t>(lm)] + 0.5 * Fme_dot_t1(lm, i);
        Fh_mj[static_cast<std::size_t>(lm)] = (i == j) ? Fh_mi[static_cast<std::size_t>(lm)]
                                                       : Fmj[static_cast<std::size_t>(lm)] + 0.5 * Fme_dot_t1(lm, j);
    }

    // ---- the Ph-symmetrised half for orientation (ti, tj) ----
    // ti/tj are the target's occupieds in the given order; a, b in [p].
    auto half = [&](int ti, int tj, Index lti, Index ltj, const Mat& Bti_oo,
                    const Mat& Btj_oo, const Mat& Wti, const Mat& Wtj,
                    const Mat& uti, const Mat& utj, const std::vector<double>& Fh_mtj,
                    const std::vector<Mat>& JAw2, const std::vector<Mat>& JAwx,
                    const std::vector<Mat>& JwAw2, const std::vector<Mat>& JwAwx) -> Mat {
        const Mat tq = (ti <= tj) ? t.T : Mat(t.T.transpose());   // t_{ti tj} in [p]
        Mat h = tq * Fh_be.transpose();                             // sum_e t^{ae} Fh_be
        for (Index lm = 0; lm < nL; ++lm) {
            // - sum_m t_{ti m}^{ab} Fh_{m tj}
            if (qof(lti, lm) >= 0) h.noalias() -= Fh_mtj[static_cast<std::size_t>(lm)] * tproj(lti, lm);
        }
        // ring terms
        for (Index lm = 0; lm < nL; ++lm) {
            const int m = L[static_cast<std::size_t>(lm)];
            const int q1 = qof(lti, lm);     // pair (ti, m)
            const int q2 = qof(lm, ltj);     // pair (m, tj)
            if (q1 >= 0) {
                const PairData& Q1 = P_[static_cast<std::size_t>(q1)];
                const Mat& Bq1m = bov(Q1, m);                    // (P, e), e in [q1]
                const Mat& S1 = Sof(lti, lm);                    // (n_p, n_q1)
                // W1(e, b) and W2(e, b), e in [q1], b in [p]
                const Mat mejb = Bq1m.transpose() * (*Bp[static_cast<std::size_t>(ltj)]);   // (me|jb)
                const Mat mebf_tj = Bq1m.transpose() * utj;                                   // sum_f t_j^f (me|bf)
                // sum_n t_n^b (nj|me): (nj|me)(n, e) = sum_P Boo_tj(P,n) Bq1m(P,e)
                const Mat njme = Btj_oo.transpose() * Bq1m;      // (n, e)
                Mat W1 = mejb + mebf_tj;
                Mat W2 = mejb + mebf_tj;
                // -(mj|be) and - sum_f t_j^f (mf|be): static and dynamic blocks (b, e)
                W2 -= JAw2[static_cast<std::size_t>(lm)].transpose();
                W2 -= JwAw2[static_cast<std::size_t>(lm)].transpose();
                for (Index ln = 0; ln < nL; ++ln) {
                    const int nn = L[static_cast<std::size_t>(ln)];
                    const bool has_t1n = diag_[static_cast<std::size_t>(nn)] >= 0;
                    if (has_t1n) {
                        // - t_n^b (nj|me)  [W1 and W2]
                        W1.noalias() -= njme.row(ln).transpose() * t1p(ln).transpose();
                        W2.noalias() -= njme.row(ln).transpose() * t1p(ln).transpose();
                        // + t_n^b (mj|ne) [W2]: (mj|ne)(e) = sum_P B_oo(P, m tj) Bov_q1[n](P,e)
                        const Mat& Bq1n = bov(Q1, nn);
                        Vec bmj(naux_);
                        for (Index P = 0; P < naux_; ++P) bmj(P) = B_oo_(P, m * no_ + tj);
                        const Vec mjne = Bq1n.transpose() * bmj;
                        W2.noalias() += mjne * t1p(ln).transpose();
                        if (diag_[static_cast<std::size_t>(tj)] >= 0) {
                            // - t_j^f t_n^b (me|nf) [W1]; - t_j^f t_n^b [(me|nf) - (mf|ne)] [W2]
                            const Vec me_nf = Bq1m.transpose() * Wtj.col(ln);   // sum_f t_j^f (me|nf)
                            const Vec mf_ne = Bq1n.transpose() * Wtj.col(lm);   // sum_f t_j^f (mf|ne)
                            W1.noalias() -= me_nf * t1p(ln).transpose();
                            W2.noalias() -= (me_nf - mf_ne) * t1p(ln).transpose();
                        }
                    }
                    const int q3 = qof(ln, ltj);   // pair (n, tj)
                    if (q3 < 0) continue;
                    const PairData& Q3 = P_[static_cast<std::size_t>(q3)];
                    const Mat& S3 = Sof(ln, ltj);            // (n_p, n_q3)
                    // Objects with n's factor row: sum_f Bov_q3[n](P,f) t_{nj}^{fb} = Dd_{jn},
                    // sum_f Bov_q3[n](P,f) t_{jn}^{fb} = Dd_{nj}.
                    const Mat& Dnj_fb = dd(Q3, tj, nn);      // row n, t_{jn}^{bf} = t_{nj}^{fb}
                    const Mat& Djn_fb = ex(Q3, nn, tj);      // row n, t_{nj}^{bf} = t_{jn}^{fb}
                    // (me|nf) contracted: M(e, b') = sum_P Bq1m(P,e) X(P,b'), then S3 on b'.
                    const Mat A1 = Bq1m.transpose() * Dnj_fb;     // sum_f t_nj^{fb'} (me|nf)
                    const Mat A2 = Bq1m.transpose() * Djn_fb;     // sum_f t_jn^{fb'} (me|nf)
                    // (mf|ne) contracted: needs Bov_q3 row m and Bov_q1 row n.
                    const Mat Ym = yobj(Q3, m, T(tj, nn));        // sum_f Bov_q3[m](P,f) t_{jn}^{b'f} = t_nj^{fb'}
                    const Mat Ym2 = yobj(Q3, m, T(nn, tj));       // sum_f Bov_q3[m](P,f) t_{nj}^{b'f} = t_jn^{fb'}
                    const Mat& Bq1n = bov(Q1, nn);
                    const Mat A3 = Bq1n.transpose() * Ym;         // sum_f t_nj^{fb'} (mf|ne)  (e, b')
                    const Mat A4 = Bq1n.transpose() * Ym2;        // sum_f t_jn^{fb'} (mf|ne)
                    // W1 += [A1 - 1/2 A2] S3^T - 1/2 A3 S3^T
                    W1.noalias() += (A1 - 0.5 * A2 - 0.5 * A3) * S3.transpose();
                    // W2 += -tss [(me|nf) - (mf|ne)] + 1/2 t_nj^{fb} (me|nf)
                    //   tss_jn^{fb} = 1/2 (t_jn^{fb} - t_jn^{bf}) (+ singles, handled above)
                    //   -> -1/2 (A2 - A1) + 1/2 (A4 - A3) + 1/2 A1
                    W2.noalias() += (-0.5 * (A2 - A1) + 0.5 * (A4 - A3) + 0.5 * A1) * S3.transpose();
                }
                // contract with the amplitudes of (ti, m): a projected, e in [q1]
                const Mat tim = T(ti, m);                                   // (a', e)
                const Mat tmi = tim.transpose();
                h.noalias() += (S1 * (tim - tmi)) * W1;
                h.noalias() += (S1 * tim) * W2;
            }
            if (q2 >= 0) {
                const PairData& Q2 = P_[static_cast<std::size_t>(q2)];
                const Mat& S2 = Sof(lm, ltj);                    // (n_p, n_q2)
                // WX(e, b), e in [q2], b in [p]:
                //   -(m ti|be) - sum_f t_i^f (mf|be) + sum_n t_n^b (m ti|ne)
                //   + 1/2 sum_nf t_{ti n}^{fb} (mf|ne) + sum_nf t_i^f t_n^b (mf|ne)
                Mat WX = -JAwx[static_cast<std::size_t>(lm)].transpose() - JwAwx[static_cast<std::size_t>(lm)].transpose();
                Vec bmi(naux_);
                for (Index P = 0; P < naux_; ++P) bmi(P) = B_oo_(P, m * no_ + ti);
                for (Index ln = 0; ln < nL; ++ln) {
                    const int nn = L[static_cast<std::size_t>(ln)];
                    const Mat& Bq2n = bov(Q2, nn);               // (P, e)
                    if (diag_[static_cast<std::size_t>(nn)] >= 0) {
                        const Vec mine = Bq2n.transpose() * bmi;             // (m ti|ne)
                        WX.noalias() += mine * t1p(ln).transpose();
                        if (diag_[static_cast<std::size_t>(ti)] >= 0) {
                            const Vec mf_ne = Bq2n.transpose() * Wti.col(lm);   // sum_f t_i^f (mf|ne)
                            WX.noalias() += mf_ne * t1p(ln).transpose();
                        }
                    }
                    const int q4 = qof(lti, ln);   // pair (ti, n)
                    if (q4 < 0) continue;
                    const PairData& Q4 = P_[static_cast<std::size_t>(q4)];
                    const Mat& S4 = Sof(lti, ln);
                    // 1/2 sum_f t_{ti n}^{fb'} (mf|ne): Y = sum_f Bov_q4[m](P,f) t_{n ti}^{b'f}
                    const Mat Y = yobj(Q4, m, T(nn, ti));
                    const Mat A = Bq2n.transpose() * Y;          // (e, b')
                    WX.noalias() += 0.5 * A * S4.transpose();
                }
                const Mat tmj = T(m, tj);                        // (a', e) in [q2]
                h.noalias() += (S2 * tmj) * WX;
            }
        }
        // - sum_me t_ti^e t_m^a (me|tj b) = - sum_m t_m^a [W_ti^T Bov_p[tj]](m, b)
        if (diag_[static_cast<std::size_t>(ti)] >= 0) {
            const Mat A = Wti.transpose() * (*Bp[static_cast<std::size_t>(ltj)]);   // (m, b)
            for (Index lm = 0; lm < nL; ++lm) h.noalias() -= t1p(lm) * A.row(lm);
        }
        // - sum_me t_tj^e t_m^a (m ti|be) = - sum_m t_m^a [Boo_ti^T u_tj](m, b)
        if (diag_[static_cast<std::size_t>(tj)] >= 0) {
            const Mat A = Bti_oo.transpose() * utj;
            for (Index lm = 0; lm < nL; ++lm) h.noalias() -= t1p(lm) * A.row(lm);
        }
        // + sum_e t_ti^e (ae|tj b) = sum_P u_ti(P,a) Bov_p[tj](P,b)
        if (diag_[static_cast<std::size_t>(ti)] >= 0) h.noalias() += uti.transpose() * (*Bp[static_cast<std::size_t>(ltj)]);
        // - sum_m t_m^a (m ti|tj b)
        {
            const Mat A = Bti_oo.transpose() * (*Bp[static_cast<std::size_t>(ltj)]);   // (m, b)
            for (Index lm = 0; lm < nL; ++lm) h.noalias() -= t1p(lm) * A.row(lm);
        }
        return h;
    };

    const Mat h_ij = half(i, j, li, lj, Boo_i, Boo_j, Wi, Wj, u_i, u_j, Fh_mj,
                          t.JA1, t.JA2, t.JwA1, t.JwA2);
    if (i == j) {
        R2 += h_ij + h_ij.transpose();
    } else {
        const Mat h_ji = half(j, i, lj, li, Boo_j, Boo_i, Wj, Wi, u_j, u_i, Fh_mi,
                              t.JA3, t.JA4, t.JwA3, t.JwA4);
        R2 += h_ij + h_ji.transpose();
    }
}

// ---------------------------------------------------------------------------
// Singles residual of occupied i in [ii]; sums over the diagonal pair's set.
// ---------------------------------------------------------------------------
void Solver::residual_singles(int i, Vec& R1) const {
    const int d = diag_[static_cast<std::size_t>(i)];
    const PairData& dp = P_[static_cast<std::size_t>(d)];
    const Index n = dp.n;
    const std::vector<int>& L = dp.L;
    const Index nL = static_cast<Index>(L.size());
    auto lpos = [&](int m) {
        return static_cast<Index>(std::lower_bound(L.begin(), L.end(), m) - L.begin());
    };
    const Index li = lpos(i);
    auto Sd = [&](const PairData& q) -> Mat { return dp.U.transpose() * q.U; };
    // singles projected into [ii]
    std::vector<Vec> t1d(static_cast<std::size_t>(nL), Vec::Zero(n));
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        const int dm = diag_[static_cast<std::size_t>(m)];
        if (dm >= 0) t1d[static_cast<std::size_t>(lm)] = Sd(P_[static_cast<std::size_t>(dm)]) * t1_[static_cast<std::size_t>(m)];
    }
    const Vec& ti = t1_[static_cast<std::size_t>(i)];
    const Mat& wi = w_[static_cast<std::size_t>(i)];
    const Mat ui = u_[static_cast<std::size_t>(i)] * dp.U;          // (P, a)
    const Mat& Bii = bov(dp, i);                                       // (P, a)

    R1 = (f_ov_.row(i) * dp.U).transpose();
    // + sum_e t_i^e F_ae  (F_ae of cs_ccsd_residual, contracted with t_i)
    R1.noalias() += dp.fvv * ti;                                        // f_ae part
    {
        // -1/2 sum_m f_me t_m^a t_i^e
        for (Index lm = 0; lm < nL; ++lm) {
            const int m = L[static_cast<std::size_t>(lm)];
            const double fmt = f_ov_.row(m).dot(T1full_.row(i));
            R1 -= 0.5 * fmt * t1d[static_cast<std::size_t>(lm)];
        }
        // + sum_mf t_m^f [2(mf|ae) - (me|af)] t_i^e
        //   = 2 sum_m sum_P w_m(P,m) u_i(P,a) - sum_m sum_P w_i(P,m) u_m(P,a)
        for (Index lm = 0; lm < nL; ++lm) {
            const int m = L[static_cast<std::size_t>(lm)];
            if (diag_[static_cast<std::size_t>(m)] < 0) continue;
            R1.noalias() += 2.0 * (ui.transpose() * w_[static_cast<std::size_t>(m)].col(m));
            const Mat um = u_[static_cast<std::size_t>(m)] * dp.U;
            R1.noalias() -= um.transpose() * wi.col(m);
        }
        // - sum_mnf taut_mn^{af} [2(me|nf) - (mf|ne)] t_i^e
        for (Index lm = 0; lm < nL; ++lm)
            for (Index ln = 0; ln < nL; ++ln) {
                const int m = L[static_cast<std::size_t>(lm)], nn = L[static_cast<std::size_t>(ln)];
                const int q = pid(m, nn);
                if (q >= 0) {
                    const PairData& qd = P_[static_cast<std::size_t>(q)];
                    const Mat S = Sd(qd);                                          // (n, n_q)
                    // 2 sum_P w_i(P,m) Dd_mn(P,a') - sum_P w_i(P,n) Ex_mn(P,a')
                    const Vec va = 2.0 * (dd(qd, m, nn).transpose() * wi.col(m))
                                 - ex(qd, m, nn).transpose() * wi.col(nn);
                    R1.noalias() -= S * va;
                }
                if (diag_[static_cast<std::size_t>(m)] >= 0 && diag_[static_cast<std::size_t>(nn)] >= 0) {
                    // 1/2 t_m^a t_n^f: 1/2 t_m^a [2 sum_P w_i(P,m) w_n(P,n) - sum_P w_n(P,m) w_i(P,n)]
                    const Mat& wn = w_[static_cast<std::size_t>(nn)];
                    const double s = 2.0 * wi.col(m).dot(wn.col(nn)) - wn.col(m).dot(wi.col(nn));
                    R1 -= 0.5 * s * t1d[static_cast<std::size_t>(lm)];
                }
            }
    }
    // - sum_m F_mi t_m^a, with F_mi summed over n in L (as the target's set)
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        if (diag_[static_cast<std::size_t>(m)] < 0) continue;
        double s = f_oo_(m, i) + 0.5 * f_ov_.row(m).dot(T1full_.row(i));
        for (Index ln = 0; ln < nL; ++ln) {
            const int nn = L[static_cast<std::size_t>(ln)];
            if (diag_[static_cast<std::size_t>(nn)] >= 0) {
                const Mat& wn = w_[static_cast<std::size_t>(nn)];
                Vec bmi(naux_), bni(naux_);
                for (Index P = 0; P < naux_; ++P) {
                    bmi(P) = B_oo_(P, m * no_ + i);
                    bni(P) = B_oo_(P, nn * no_ + i);
                }
                s += 2.0 * bmi.dot(wn.col(nn)) - bni.dot(wn.col(m));
                s += wi.col(m).dot(wn.col(nn)) - 0.5 * wn.col(m).dot(wi.col(nn));
            }
            const int q = pid(i, nn);
            if (q >= 0) {
                const PairData& qd = P_[static_cast<std::size_t>(q)];
                const Mat Tin = T(i, nn);
                const Mat& Bm = bov(qd, m);
                const Mat& Bn = bov(qd, nn);
                const Mat BmT = Bm * Tin;
                s += 2.0 * (BmT.array() * Bn.array()).sum();
                const Mat BnT = Bn * Tin;
                s -= (BnT.array() * Bm.array()).sum();
            }
        }
        R1 -= s * t1d[static_cast<std::size_t>(lm)];
    }
    // + sum_me [2 t_im^{ae} - t_mi^{ae}] F_me, e in [im], F_me summed over n in L
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        const int q = pid(i, m);
        if (q < 0) continue;
        const PairData& qd = P_[static_cast<std::size_t>(q)];
        const Mat& Bm = bov(qd, m);
        Vec Fme = (f_ov_.row(m) * qd.U).transpose();
        for (Index ln = 0; ln < nL; ++ln) {
            const int nn = L[static_cast<std::size_t>(ln)];
            if (diag_[static_cast<std::size_t>(nn)] < 0) continue;
            const Mat& wn = w_[static_cast<std::size_t>(nn)];
            Fme.noalias() += 2.0 * (Bm.transpose() * wn.col(nn));
            Fme.noalias() -= bov(qd, nn).transpose() * wn.col(m);
        }
        const Mat tim = T(i, m);
        R1.noalias() += Sd(qd) * ((2.0 * tim - tim.transpose()) * Fme);
    }
    // + sum_nf t_n^f [2(nf|ia) - (ni|af)]
    for (Index ln = 0; ln < nL; ++ln) {
        const int nn = L[static_cast<std::size_t>(ln)];
        if (diag_[static_cast<std::size_t>(nn)] < 0) continue;
        const Mat& wn = w_[static_cast<std::size_t>(nn)];
        R1.noalias() += 2.0 * (Bii.transpose() * wn.col(nn));
        Vec bni(naux_);
        for (Index P = 0; P < naux_; ++P) bni(P) = B_oo_(P, nn * no_ + i);
        const Mat un = u_[static_cast<std::size_t>(nn)] * dp.U;
        R1.noalias() -= un.transpose() * bni;
    }
    // + sum_mef [2 t_im^{ef} - t_mi^{ef}] (mf|ae): 3-external blocks
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        const int q = pid(i, m);
        if (q < 0) continue;
        const PairData& qd = P_[static_cast<std::size_t>(q)];
        const Mat& X = (i <= m) ? qd.X3[0] : qd.X3[1];   // (f*n_q+e, a)
        if (X.size() == 0) continue;
        const Mat tim = T(i, m);                          // (e, f)
        const Mat A = 2.0 * tim - tim.transpose();        // (e, f)
        // vec(f*n+e) = A(e, f) -> row-major flatten of A^T
        const Index nq = qd.n;
        Vec av(nq * nq);
        for (Index f = 0; f < nq; ++f)
            for (Index e = 0; e < nq; ++e) av(f * nq + e) = A(e, f);
        R1.noalias() += X.transpose() * av;
    }
    // - sum_mne [2 t_mn^{ae} - t_nm^{ae}] (mi|ne), e in [mn]
    for (Index lm = 0; lm < nL; ++lm)
        for (Index ln = 0; ln < nL; ++ln) {
            const int m = L[static_cast<std::size_t>(lm)], nn = L[static_cast<std::size_t>(ln)];
            const int q = pid(m, nn);
            if (q < 0) continue;
            const PairData& qd = P_[static_cast<std::size_t>(q)];
            Vec bmi(naux_);
            for (Index P = 0; P < naux_; ++P) bmi(P) = B_oo_(P, m * no_ + i);
            const Vec mine = bov(qd, nn).transpose() * bmi;   // (e)
            const Mat tmn = T(m, nn);
            R1.noalias() -= Sd(qd) * ((2.0 * tmn - tmn.transpose()) * mine);
        }
    (void)li;
}

// ---------------------------------------------------------------------------
std::vector<double> Solver::flatten() const {
    std::vector<double> v;
    for (Index i = 0; i < no_; ++i) {
        const Vec& t = t1_[static_cast<std::size_t>(i)];
        v.insert(v.end(), t.data(), t.data() + t.size());
    }
    for (Index p = 0; p < np_; ++p) {
        const Mat& T2 = P_[static_cast<std::size_t>(p)].T;
        v.insert(v.end(), T2.data(), T2.data() + T2.size());
    }
    return v;
}

void Solver::unflatten(const std::vector<double>& v) {
    std::size_t off = 0;
    for (Index i = 0; i < no_; ++i) {
        Vec& t = t1_[static_cast<std::size_t>(i)];
        std::copy(v.begin() + static_cast<std::ptrdiff_t>(off),
                  v.begin() + static_cast<std::ptrdiff_t>(off + static_cast<std::size_t>(t.size())), t.data());
        off += static_cast<std::size_t>(t.size());
    }
    for (Index p = 0; p < np_; ++p) {
        Mat& T2 = P_[static_cast<std::size_t>(p)].T;
        std::copy(v.begin() + static_cast<std::ptrdiff_t>(off),
                  v.begin() + static_cast<std::ptrdiff_t>(off + static_cast<std::size_t>(T2.size())), T2.data());
        off += static_cast<std::size_t>(T2.size());
    }
}

// ---------------------------------------------------------------------------
DLPNOCCSDPNOResult Solver::run() {
    DLPNOCCSDPNOResult out;
    out.n_offlist_rows = n_offlist_rows_;
    out.static_factor_bytes = static_bytes_;
    out.setup_seconds = setup_seconds_;
    const auto t_iter0 = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> amp_hist, res_hist;
    double e_prev = energy();
    for (int it = 0; it < opts_.max_iter; ++it) {
        const std::vector<double> x_old = flatten();
        build_iteration_data();
        std::vector<Mat> R2(static_cast<std::size_t>(np_));
        std::vector<Vec> R1(static_cast<std::size_t>(no_));
        #pragma omp parallel
        {
            Workspace ws;
            #pragma omp for schedule(dynamic)
            for (Index p = 0; p < np_; ++p) residual_pair(p, ws, R2[static_cast<std::size_t>(p)]);
            #pragma omp for schedule(dynamic)
            for (Index i = 0; i < no_; ++i) {
                if (diag_[static_cast<std::size_t>(i)] >= 0) residual_singles(static_cast<int>(i), R1[static_cast<std::size_t>(i)]);
            }
        }
        // Jacobi update.
        for (Index p = 0; p < np_; ++p) {
            PairData& q = P_[static_cast<std::size_t>(p)];
            const double fij = f_dd_(q.i) + f_dd_(q.j);
            Mat Tn = q.T;
            for (Index a = 0; a < q.n; ++a)
                for (Index b = 0; b < q.n; ++b)
                    Tn(a, b) += R2[static_cast<std::size_t>(p)](a, b) / (fij - q.eps(a) - q.eps(b));
            if (q.i == q.j) Tn = 0.5 * (Tn + Tn.transpose());
            q.T = std::move(Tn);
        }
        for (Index i = 0; i < no_; ++i) {
            const int d = diag_[static_cast<std::size_t>(i)];
            if (d < 0) continue;
            const PairData& dp = P_[static_cast<std::size_t>(d)];
            Vec& t = t1_[static_cast<std::size_t>(i)];
            for (Index a = 0; a < dp.n; ++a) t(a) += R1[static_cast<std::size_t>(i)](a) / (f_dd_(i) - dp.eps(a));
        }
        // DIIS over the concatenated amplitude vector (as the Python solver).
        amp_hist.push_back(flatten());
        {
            std::vector<double> r(amp_hist.back().size());
            for (std::size_t k = 0; k < r.size(); ++k) r[k] = amp_hist.back()[k] - x_old[k];
            res_hist.push_back(std::move(r));
        }
        if (static_cast<int>(amp_hist.size()) > opts_.diis_size) {
            amp_hist.erase(amp_hist.begin());
            res_hist.erase(res_hist.begin());
        }
        const std::size_t nh = amp_hist.size();
        double r_norm = 0.0;
        for (double v : res_hist.back()) r_norm = std::max(r_norm, std::abs(v));
        if (nh >= 2) {
            Mat Bm = Mat::Constant(static_cast<Index>(nh + 1), static_cast<Index>(nh + 1), -1.0);
            Bm(static_cast<Index>(nh), static_cast<Index>(nh)) = 0.0;
            for (std::size_t a = 0; a < nh; ++a)
                for (std::size_t b = 0; b < nh; ++b) {
                    double s = 0.0;
                    const std::vector<double>& ra = res_hist[a];
                    const std::vector<double>& rb = res_hist[b];
                    for (std::size_t k = 0; k < ra.size(); ++k) s += ra[k] * rb[k];
                    Bm(static_cast<Index>(a), static_cast<Index>(b)) = s;
                }
            Vec rhs = Vec::Zero(static_cast<Index>(nh + 1));
            rhs(static_cast<Index>(nh)) = -1.0;
            // numpy.linalg.solve semantics: extrapolate unless the system is
            // exactly singular (a non-finite solution).
            Eigen::PartialPivLU<Mat> lu(Bm);
            const Vec c = lu.solve(rhs);
            if (c.allFinite()) {
                std::vector<double> mix(amp_hist.back().size(), 0.0);
                for (std::size_t k = 0; k < nh; ++k)
                    for (std::size_t q = 0; q < mix.size(); ++q) mix[q] += c(static_cast<Index>(k)) * amp_hist[k][q];
                unflatten(mix);
            }
        }
        const double e = energy();
        out.trace_e_corr.push_back(e);
        out.trace_r_norm.push_back(r_norm);
        if (it > 0 && std::abs(e - e_prev) < opts_.conv_tol_energy
            && r_norm < opts_.conv_tol_residual) {
            out.converged = true;
            out.n_iter = it + 1;
            e_prev = e;
            break;
        }
        e_prev = e;
        out.n_iter = it + 1;
    }
    out.iteration_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_iter0).count();
    out.e_corr = energy();
    out.T2.reserve(static_cast<std::size_t>(np_));
    for (Index p = 0; p < np_; ++p) out.T2.push_back(P_[static_cast<std::size_t>(p)].T);
    out.t1 = t1_;
    return out;
}

}  // namespace

DLPNOCCSDPNOResult dlpno_ccsd_pno_solve(
    const DLPNORowMatCRef& B_ov, const DLPNORowMatCRef& B_oo,
    const DLPNORowMatCRef& B_vv,
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& f_vv, std::vector<DLPNOPairInput> pairs,
    const DLPNOCCSDPNOOptions& opts) {
    Solver solver(B_ov, B_oo, B_vv, f_oo, f_ov, f_vv, std::move(pairs), opts);
    return solver.run();
}

}  // namespace vibeqc
