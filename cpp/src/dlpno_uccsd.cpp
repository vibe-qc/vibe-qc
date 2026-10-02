// Open-shell DLPNO-CCSD amplitude solver contracted in PNO spaces.
//
// Every term below is one term of the density-fitted spin-orbital residual
// that tests/test_dlpno_uccsd_pno.py checks against
// python/vibeqc/dlpno/_ccsd_ref.py::so_residuals, evaluated for a target
// pair ij in that pair's PNO space with each contracted virtual index taken
// in the PNO space of the amplitude that carries it (Riplinger and Neese,
// J. Chem. Phys. 138, 034106 (2013), Sec. II C). The Fock matrix enters the
// intermediates whole (diagonal included), so the residuals are the true
// residuals and the Jacobi step divides them by the orbital-energy
// denominators.
//
// Notation. For a kept pair q = (k, l), k < l:
//   U_q      (nv, n_q)       PNO coefficients in the canonical spin-virtual
//                            basis; U_q^s its rows of spin s.
//   T_q      (n_q, n_q)      t_{kl}^{ef}, antisymmetric; t_{lk} = -T_q.
//   Bov_q[m] (naux, n_q)     B^P_{m e} for e in [q] (zero unless U_q has
//                            columns of spin(m)); stored for every occupied
//                            m some contraction needs.
//   Bvv_q    blocks (n_b*n_b, naux) of B^P_{ab} over [q], column-major per P:
//            one block per spin when the factor rows are compact (B^P is
//            block-diagonal in spin), else one block over the whole space.
//   El_q, Eh_q (naux, n_q)   Bov_q[k] T_q and Bov_q[l] T_q (per iteration).
// Singles: T1 (no, nv) in canonical coordinates, zero off the spin block.
//   w_x (naux, no)   w_x(P, m) = sum_f B^P_{m f} t_x^f
//   uT_x (nv_s, naux) uT_x(a, P) = sum_f B^P_{a f} t_x^f
// Occupied sums of a target's residual run over its coupling set L.

#include "vibeqc/dlpno_uccsd.hpp"

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
using RowMap = Eigen::Map<const RowMat>;
using StridedRowMap = Eigen::Map<const RowMat, 0, Eigen::OuterStride<>>;

struct UPair {
    int i = 0;
    int j = 0;
    Index n = 0;
    Mat U;          // (nv, n)
    Mat Us[2];      // (nv_s, n) spin blocks of U
    bool has[2] = {false, false};
    Vec eps;
    Mat fvv;        // (n, n)
    Mat K;          // (n, n)
    Mat T;          // (n, n)
    // B^P_{ab} over [q]: per-block (n_b*n_b, naux), column P a slice of the
    // columns [bvv_off[b], bvv_off[b] + bvv_n[b]).
    std::vector<Mat> BvvT;
    std::vector<Index> bvv_off, bvv_n;
    std::vector<int> L;
    std::vector<int> rowpos;     // occupied -> index into Bov, -1 absent
    std::vector<int> rows;
    // Factor row of occupied m (spin s): B^P_{m e} for the columns
    // [off[s], off[s] + ns[s]) of the pair space, the only ones that can be
    // nonzero. When every PNO is spin-pure and the columns are sorted by spin
    // (as the Python builder makes them) that is the spin's own block, so an
    // opposite-spin pair stores half of each row; otherwise both ranges are
    // the whole space. Empty where the pair has no columns of that spin.
    Index off[2] = {0, 0};
    Index ns[2] = {0, 0};
    std::vector<Mat> Bov;        // (naux, ns[spin(m)])
    // Static and dynamic mixed Coulomb blocks per ring orientation o and
    // coupled occupied lm: o = 0 is (x, y) = (i, j), o = 1 is (j, i);
    //   JA[o][lm]  = U^T J_{m y} U_{(x,m)},  J_{m y} = sum_P B^P_{m y} B_vv^P
    //   JwA[o][lm] = U^T Jw_{m y} U_{(x,m)}, Jw_{m y} = sum_P w_y(P, m) B_vv^P
    std::vector<Mat> JA[2];
    std::vector<Mat> JwA[2];
    Mat El, Eh;     // (naux, n)
};

struct USingles {
    std::vector<int> L;
    Mat U;          // (nv_s, n_s) or empty (full canonical)
    Vec eps;        // energies of the basis (n_s) or diag(f_vv_s)
    Vec t;          // amplitudes in the basis
};

struct Workspace {
    std::vector<int> qidx;
    std::vector<Mat> S;
};

// out = V^T C for a row-major (naux, K) factor view V, rows of the product
// split over OpenMP threads (the linked BLAS is pinned to one thread).
void parallel_vT_times(const RowMap& V, const Mat& C, Mat& out) {
    const Index R = V.cols();
    out.resize(R, C.cols());
    const Index blk = 2048;
    #pragma omp parallel for schedule(dynamic)
    for (Index r0 = 0; r0 < R; r0 += blk) {
        const Index rn = std::min(blk, R - r0);
        out.middleRows(r0, rn).noalias() = V.middleCols(r0, rn).transpose() * C;
    }
}

// out = M C for a row-major (R, K) view M with R split over threads.
void parallel_rows_times(const RowMap& M, const Mat& C, Mat& out) {
    const Index R = M.rows();
    out.resize(R, C.cols());
    const Index blk = 4096;
    #pragma omp parallel for schedule(dynamic)
    for (Index r0 = 0; r0 < R; r0 += blk) {
        const Index rn = std::min(blk, R - r0);
        out.middleRows(r0, rn).noalias() = M.middleRows(r0, rn) * C;
    }
}

class USolver {
public:
    USolver(const DLPNORowMatCRef& B_ov_a, const DLPNORowMatCRef& B_ov_b,
            const DLPNORowMatCRef& B_oo_a, const DLPNORowMatCRef& B_oo_b,
            const DLPNORowMatCRef& B_vv_a, const DLPNORowMatCRef& B_vv_b,
            const Mat& f_oo, const Mat& f_ov, const Mat& f_vv_a,
            const Mat& f_vv_b, std::vector<DLPNOUPairInput> pairs,
            std::vector<DLPNOUSinglesInput> singles,
            const DLPNOUCCSDPNOOptions& opts);

    DLPNOUCCSDPNOResult run();
    void residuals_for(const Mat& T1, std::vector<Mat>& R2, Mat& R1);

private:
    // ---- global data ----
    const double* ov_[2];
    const double* oo_[2];
    const double* vv_[2];
    const Mat& f_oo_;
    const Mat& f_ov_;
    const Mat* f_vv_[2];
    const DLPNOUCCSDPNOOptions opts_;
    Index naux_ = 0, no_ = 0, nv_ = 0;
    Index nos_[2] = {0, 0}, nvs_[2] = {0, 0};
    std::vector<UPair> P_;
    std::vector<USingles> S1_;
    std::vector<int> pidx_;
    std::vector<std::vector<int>> pairs_of_occ_;
    std::size_t n_offlist_rows_ = 0;
    std::size_t static_bytes_ = 0;
    double setup_seconds_ = 0.0;

    // ---- per-iteration data ----
    Mat T1_;                    // (no, nv)
    std::vector<Mat> w_;        // (naux, no)
    std::vector<Mat> uT_;       // (nv_s, naux)

    // ---- helpers ----
    int spin(int m) const { return m < nos_[0] ? 0 : 1; }
    Index loc(int m) const { return m < nos_[0] ? m : m - nos_[0]; }
    Index voff(int s) const { return s == 0 ? 0 : nvs_[0]; }
    int pid(int a, int b) const {
        if (a == b) return -1;
        const int lo = std::min(a, b), hi = std::max(a, b);
        return pidx_[static_cast<std::size_t>(lo) * no_ + hi];
    }
    // (naux, nv_s) view of B^P_{m v} over the occupied's own spin block.
    StridedRowMap bov_row(int m) const {
        const int s = spin(m);
        return StridedRowMap(ov_[s] + loc(m) * nvs_[s], naux_, nvs_[s],
                             Eigen::OuterStride<>(nos_[s] * nvs_[s]));
    }
    // B^P_{m x} as a vector; zero for opposite spins.
    Vec boo(int m, int x) const {
        Vec v = Vec::Zero(naux_);
        const int s = spin(m);
        if (s != spin(x)) return v;
        const Index cols = nos_[s] * nos_[s];
        const Index c = loc(m) * nos_[s] + loc(x);
        const double* base = oo_[s];
        for (Index P = 0; P < naux_; ++P) v(P) = base[P * cols + c];
        return v;
    }
    RowMap vv_view(int s) const {  // (naux*nv_s, nv_s): row (P, v), column a
        return RowMap(vv_[s], naux_ * nvs_[s], nvs_[s]);
    }
    RowMap vv_flat(int s) const {  // (naux, nv_s*nv_s)
        return RowMap(vv_[s], naux_, nvs_[s] * nvs_[s]);
    }
    Vec t1s(int m) const {
        const int s = spin(m);
        return T1_.row(m).segment(voff(s), nvs_[s]).transpose();
    }
    // Compact factor row of occupied m in pair q and its column range.
    struct RowView {
        const Mat* B = nullptr;
        Index off = 0;
        Index ns = 0;
        bool empty() const { return B == nullptr || B->size() == 0; }
    };
    RowView bov(const UPair& q, int m) const {
        const int r = q.rowpos[static_cast<std::size_t>(m)];
        if (r < 0) {
            std::ostringstream msg;
            msg << "dlpno_uccsd_pno_solve: no stored factor row for occupied "
                << m << " of pair (" << q.i << "," << q.j << ")";
            throw std::logic_error(msg.str());
        }
        const int s = spin(m);
        return RowView{&q.Bov[static_cast<std::size_t>(r)], q.off[s], q.ns[s]};
    }
    // The row expanded to the pair's whole space, (naux, n).
    Mat full_row(const UPair& q, int m) const {
        Mat out = Mat::Zero(naux_, q.n);
        const RowView r = bov(q, m);
        if (!r.empty()) out.middleCols(r.off, r.ns) = *r.B;
        return out;
    }
    Mat overlap(const UPair& a, const UPair& b) const {
        Mat S = Mat::Zero(a.n, b.n);
        for (int s = 0; s < 2; ++s)
            if (a.has[s] && b.has[s]) S.noalias() += a.Us[s].transpose() * b.Us[s];
        return S;
    }
    // u_x projected into [p]: (naux, n_p).
    Mat u_proj(const UPair& p, int x) const {
        const int s = spin(x);
        if (!p.has[s]) return Mat::Zero(naux_, p.n);
        return uT_[static_cast<std::size_t>(x)].transpose() * p.Us[s];
    }

    void build_static();
    void build_iteration_data();
    void build_mixed_coulomb(bool dynamic);
    double energy() const;
    void residual_pair(Index p, Workspace& ws, Mat& R) const;
    void residual_singles(int i, Vec& r, Mat& Y) const;
    void all_residuals(std::vector<Mat>& R2, std::vector<Vec>& R1full) const;
    std::vector<double> flatten() const;
    void unflatten(const std::vector<double>& v);
    void set_t1_from_singles();
};

// ---------------------------------------------------------------------------
USolver::USolver(const DLPNORowMatCRef& B_ov_a, const DLPNORowMatCRef& B_ov_b,
                 const DLPNORowMatCRef& B_oo_a, const DLPNORowMatCRef& B_oo_b,
                 const DLPNORowMatCRef& B_vv_a, const DLPNORowMatCRef& B_vv_b,
                 const Mat& f_oo, const Mat& f_ov, const Mat& f_vv_a,
                 const Mat& f_vv_b, std::vector<DLPNOUPairInput> pairs,
                 std::vector<DLPNOUSinglesInput> singles,
                 const DLPNOUCCSDPNOOptions& opts)
    : f_oo_(f_oo), f_ov_(f_ov), opts_(opts) {
    const DLPNORowMatCRef* ov[2] = {&B_ov_a, &B_ov_b};
    const DLPNORowMatCRef* oo[2] = {&B_oo_a, &B_oo_b};
    const DLPNORowMatCRef* vv[2] = {&B_vv_a, &B_vv_b};
    f_vv_[0] = &f_vv_a;
    f_vv_[1] = &f_vv_b;
    naux_ = B_vv_a.rows();
    for (int s = 0; s < 2; ++s) {
        nvs_[s] = f_vv_[s]->rows();
        if (f_vv_[s]->cols() != nvs_[s])
            throw std::invalid_argument("dlpno_uccsd_pno_solve: f_vv must be square");
        if (ov[s]->rows() != naux_ || oo[s]->rows() != naux_ || vv[s]->rows() != naux_)
            throw std::invalid_argument(
                "dlpno_uccsd_pno_solve: DF factors must share n_aux rows");
        if (vv[s]->cols() != nvs_[s] * nvs_[s])
            throw std::invalid_argument(
                "dlpno_uccsd_pno_solve: B_vv must have n_vir_s^2 columns");
        if (nvs_[s] == 0) {
            nos_[s] = oo[s]->cols() == 0 ? 0 : static_cast<Index>(std::llround(std::sqrt(static_cast<double>(oo[s]->cols()))));
        } else {
            if (ov[s]->cols() % nvs_[s] != 0)
                throw std::invalid_argument("dlpno_uccsd_pno_solve: B_ov shape");
            nos_[s] = ov[s]->cols() / nvs_[s];
        }
        if (oo[s]->cols() != nos_[s] * nos_[s])
            throw std::invalid_argument("dlpno_uccsd_pno_solve: B_oo shape");
        for (const DLPNORowMatCRef* f : {ov[s], oo[s], vv[s]})
            if (f->cols() > 0 && f->outerStride() != f->cols())
                throw std::invalid_argument(
                    "dlpno_uccsd_pno_solve: DF factors must be C-contiguous row-major");
        ov_[s] = ov[s]->data();
        oo_[s] = oo[s]->data();
        vv_[s] = vv[s]->data();
    }
    no_ = nos_[0] + nos_[1];
    nv_ = nvs_[0] + nvs_[1];
    if (f_oo.rows() != no_ || f_oo.cols() != no_ || f_ov.rows() != no_
        || f_ov.cols() != nv_)
        throw std::invalid_argument("dlpno_uccsd_pno_solve: Fock block shapes");
    if (static_cast<Index>(singles.size()) != no_)
        throw std::invalid_argument(
            "dlpno_uccsd_pno_solve: one singles entry per occupied is required");

    const Index np = static_cast<Index>(pairs.size());
    P_.resize(static_cast<std::size_t>(np));
    pidx_.assign(static_cast<std::size_t>(no_ * no_), -1);
    pairs_of_occ_.assign(static_cast<std::size_t>(no_), {});
    for (Index p = 0; p < np; ++p) {
        DLPNOUPairInput& in = pairs[static_cast<std::size_t>(p)];
        UPair& q = P_[static_cast<std::size_t>(p)];
        if (in.i < 0 || in.j < 0 || in.i >= no_ || in.j >= no_ || in.i >= in.j)
            throw std::invalid_argument(
                "dlpno_uccsd_pno_solve: pair indices must satisfy 0 <= i < j < n_occ");
        q.i = in.i;
        q.j = in.j;
        q.n = in.U.cols();
        if (in.U.rows() != nv_ || in.eps.size() != q.n || in.K.rows() != q.n
            || in.K.cols() != q.n || in.T2.rows() != q.n || in.T2.cols() != q.n)
            throw std::invalid_argument(
                "dlpno_uccsd_pno_solve: pair PNO data shapes are inconsistent");
        q.U = std::move(in.U);
        for (int s = 0; s < 2; ++s) {
            q.Us[s] = q.U.middleRows(voff(s), nvs_[s]);
            q.has[s] = q.Us[s].size() > 0 && q.Us[s].cwiseAbs().maxCoeff() > 0.0;
        }
        {
            // Spin-pure columns sorted alpha first allow compact rows.
            bool compact = true;
            Index n_alpha = 0;
            bool seen_beta = false;
            for (Index c = 0; c < q.n; ++c) {
                const bool a = nvs_[0] > 0 && q.Us[0].col(c).cwiseAbs().maxCoeff() > 0.0;
                const bool b = nvs_[1] > 0 && q.Us[1].col(c).cwiseAbs().maxCoeff() > 0.0;
                if (a && b) { compact = false; break; }
                if (b) seen_beta = true;
                else if (seen_beta) { compact = false; break; }
                else ++n_alpha;
            }
            if (compact) {
                q.off[0] = 0;
                q.ns[0] = n_alpha;
                q.off[1] = n_alpha;
                q.ns[1] = q.n - n_alpha;
            } else {
                for (int s = 0; s < 2; ++s) {
                    q.off[s] = 0;
                    q.ns[s] = q.has[s] ? q.n : 0;
                }
            }
            for (int s = 0; s < 2; ++s)
                if (q.ns[s] == 0) q.has[s] = false;
        }
        q.eps = std::move(in.eps);
        q.fvv = Mat::Zero(q.n, q.n);
        for (int s = 0; s < 2; ++s)
            if (q.has[s]) q.fvv.noalias() += q.Us[s].transpose() * (*f_vv_[s]) * q.Us[s];
        q.K = std::move(in.K);
        q.T = std::move(in.T2);
        q.L = std::move(in.L);
        std::sort(q.L.begin(), q.L.end());
        q.L.erase(std::unique(q.L.begin(), q.L.end()), q.L.end());
        if (!std::binary_search(q.L.begin(), q.L.end(), q.i)
            || !std::binary_search(q.L.begin(), q.L.end(), q.j))
            throw std::invalid_argument(
                "dlpno_uccsd_pno_solve: a pair's coupling set must contain both "
                "of its occupieds");
        for (int m : q.L)
            if (m < 0 || m >= no_)
                throw std::invalid_argument(
                    "dlpno_uccsd_pno_solve: coupling-set index out of range");
        const std::size_t key = static_cast<std::size_t>(q.i) * no_ + q.j;
        if (pidx_[key] >= 0)
            throw std::invalid_argument("dlpno_uccsd_pno_solve: duplicate pair");
        pidx_[key] = static_cast<int>(p);
        pairs_of_occ_[static_cast<std::size_t>(q.i)].push_back(static_cast<int>(p));
        pairs_of_occ_[static_cast<std::size_t>(q.j)].push_back(static_cast<int>(p));
    }
    S1_.resize(static_cast<std::size_t>(no_));
    for (Index i = 0; i < no_; ++i) {
        DLPNOUSinglesInput& in = singles[static_cast<std::size_t>(i)];
        USingles& sg = S1_[static_cast<std::size_t>(i)];
        const int s = spin(static_cast<int>(i));
        sg.L = std::move(in.L);
        std::sort(sg.L.begin(), sg.L.end());
        sg.L.erase(std::unique(sg.L.begin(), sg.L.end()), sg.L.end());
        if (!std::binary_search(sg.L.begin(), sg.L.end(), static_cast<int>(i)))
            throw std::invalid_argument(
                "dlpno_uccsd_pno_solve: a singles coupling set must contain its "
                "occupied");
        for (int m : sg.L)
            if (m < 0 || m >= no_)
                throw std::invalid_argument(
                    "dlpno_uccsd_pno_solve: coupling-set index out of range");
        if (in.U.size() > 0) {
            if (in.U.rows() != nvs_[s] || in.eps.size() != in.U.cols())
                throw std::invalid_argument(
                    "dlpno_uccsd_pno_solve: singles basis shape");
            sg.U = std::move(in.U);
            sg.eps = std::move(in.eps);
        } else {
            sg.eps = f_vv_[s]->diagonal();
        }
        sg.t = Vec::Zero(sg.eps.size());
    }
    T1_ = Mat::Zero(no_, nv_);
    const auto t0 = std::chrono::steady_clock::now();
    build_static();
    setup_seconds_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Static factors: Bov rows, pair Bvv, mixed Coulomb blocks.
// ---------------------------------------------------------------------------
void USolver::build_static() {
    const Index np = static_cast<Index>(P_.size());
    // 1. Rows. Pair q = (k, l) needs its own occupieds and coupling set; a
    //    ring or occupied-Fock contraction of target p reads rows L_p of the
    //    pairs (x, m), x in p and m in L_p; the singles residual of i reads
    //    rows L_i of the pairs (i, m), m in L_i.
    std::vector<std::vector<char>> need(static_cast<std::size_t>(np),
                                        std::vector<char>(static_cast<std::size_t>(no_), 0));
    auto mark = [&](int q, const std::vector<int>& L) {
        std::vector<char>& nq = need[static_cast<std::size_t>(q)];
        for (int r : L) nq[static_cast<std::size_t>(r)] = 1;
    };
    for (Index p = 0; p < np; ++p) {
        const UPair& t = P_[static_cast<std::size_t>(p)];
        mark(static_cast<int>(p), t.L);
        for (int x : {t.i, t.j})
            for (int m : t.L) {
                const int q = pid(x, m);
                if (q >= 0) mark(q, t.L);
            }
    }
    for (Index i = 0; i < no_; ++i) {
        const USingles& sg = S1_[static_cast<std::size_t>(i)];
        for (int m : sg.L) {
            const int q = pid(static_cast<int>(i), m);
            if (q >= 0) mark(q, sg.L);
        }
    }
    n_offlist_rows_ = 0;
    for (Index p = 0; p < np; ++p) {
        UPair& q = P_[static_cast<std::size_t>(p)];
        q.rowpos.assign(static_cast<std::size_t>(no_), -1);
        q.rows.clear();
        for (int m = 0; m < no_; ++m) {
            if (!need[static_cast<std::size_t>(p)][static_cast<std::size_t>(m)]) continue;
            q.rowpos[static_cast<std::size_t>(m)] = static_cast<int>(q.rows.size());
            q.rows.push_back(m);
            if (!std::binary_search(q.L.begin(), q.L.end(), m) && q.has[spin(m)])
                ++n_offlist_rows_;
        }
    }

    // 2. Bov rows and BvvT per pair.
    #pragma omp parallel for schedule(dynamic)
    for (Index p = 0; p < np; ++p) {
        UPair& q = P_[static_cast<std::size_t>(p)];
        const Index n = q.n;
        q.Bov.assign(q.rows.size(), Mat());
        for (std::size_t r = 0; r < q.rows.size(); ++r) {
            const int m = q.rows[r];
            const int s = spin(m);
            if (!q.has[s]) continue;   // identically zero row
            q.Bov[r].noalias() = bov_row(m) * q.Us[s].middleCols(q.off[s], q.ns[s]);
        }
        // Compact layout: B^P has no cross-spin block, so each spin's block
        // is stored alone; otherwise one block over the whole pair space.
        const bool compact_layout = (q.off[1] == q.ns[0] && q.off[0] == 0
                                     && q.ns[0] + q.ns[1] == n);
        q.BvvT.clear();
        q.bvv_off.clear();
        q.bvv_n.clear();
        if (compact_layout) {
            for (int s = 0; s < 2; ++s)
                if (q.ns[s] > 0) {
                    q.bvv_off.push_back(q.off[s]);
                    q.bvv_n.push_back(q.ns[s]);
                }
        } else {
            q.bvv_off.push_back(0);
            q.bvv_n.push_back(n);
        }
        for (std::size_t b = 0; b < q.bvv_off.size(); ++b) {
            const Index bo = q.bvv_off[b], bn = q.bvv_n[b];
            Mat blockT = Mat::Zero(bn * bn, naux_);
            for (int s = 0; s < 2; ++s) {
                if (!q.has[s] || nvs_[s] == 0) continue;
                const Mat Usb = q.Us[s].middleCols(bo, bn);
                if (Usb.cwiseAbs().maxCoeff() == 0.0) continue;
                const RowMap V = vv_view(s);
                const Index nvs = nvs_[s];
                const Index pblk = std::max<Index>(1, std::min<Index>(naux_, 32));
                Mat half;
                for (Index P0 = 0; P0 < naux_; P0 += pblk) {
                    const Index pn = std::min(pblk, naux_ - P0);
                    half.noalias() = V.middleRows(P0 * nvs, pn * nvs) * Usb;  // (pn*nvs, bn)
                    for (Index dp = 0; dp < pn; ++dp) {
                        Eigen::Map<Mat> slice(blockT.col(P0 + dp).data(), bn, bn);
                        slice.noalias() += Usb.transpose() * half.middleRows(dp * nvs, nvs);
                    }
                }
            }
            q.BvvT.push_back(std::move(blockT));
        }
    }

    // 3. Static mixed Coulomb blocks JA. (m y|b e) needs spin(m) = spin(y) for
    //    the occupied charge distribution only; b and e may be of either
    //    spin, so every virtual spin block of B_vv contributes.
    for (Index p = 0; p < np; ++p) {
        UPair& t = P_[static_cast<std::size_t>(p)];
        for (int o = 0; o < 2; ++o) {
            t.JA[o].assign(t.L.size(), Mat());
            t.JwA[o].assign(t.L.size(), Mat());
        }
    }
    build_mixed_coulomb(false);
    static_bytes_ = 0;
    for (Index p = 0; p < np; ++p) {
        const UPair& q = P_[static_cast<std::size_t>(p)];
        for (const Mat& Bb : q.BvvT) static_bytes_ += sizeof(double) * static_cast<std::size_t>(Bb.size());
        for (const Mat& B : q.Bov) static_bytes_ += sizeof(double) * static_cast<std::size_t>(B.size());
        for (int o = 0; o < 2; ++o)
            for (const Mat& J : q.JA[o]) static_bytes_ += sizeof(double) * static_cast<std::size_t>(J.size());
    }
}


// Mixed Coulomb blocks for the ring term. For every target p, ring
// orientation o ((x, y) = (i, j) or (j, i)) and coupled m with the pair
// (x, m) kept:
//   static  (dynamic = false): JA[o][lm]  = U_p^T J_{m y} U_{(x,m)}
//   dynamic (dynamic = true):  JwA[o][lm] = U_p^T Jw_{m y} U_{(x,m)}
// with J_{m y} = sum_P B^P_{m y} B_vv^P and Jw_{m y} = sum_P w_y(P, m) B_vv^P,
// both zero unless spin(m) = spin(y), and summed over the virtual spin
// blocks of B_vv.
void USolver::build_mixed_coulomb(bool dynamic) {
    const Index np = static_cast<Index>(P_.size());
    std::vector<std::pair<int, int>> my;
    {
        std::vector<char> seen(static_cast<std::size_t>(no_ * no_), 0);
        for (Index p = 0; p < np; ++p) {
            const UPair& t = P_[static_cast<std::size_t>(p)];
            for (int y : {t.i, t.j})
                for (int m : t.L) {
                    if (spin(m) != spin(y)) continue;
                    const std::size_t key = static_cast<std::size_t>(m) * no_ + y;
                    if (!seen[key]) { seen[key] = 1; my.emplace_back(m, y); }
                }
        }
    }
    const Index nmy = static_cast<Index>(my.size());
    const Index chunk = 64;
    for (int sv = 0; sv < 2; ++sv) {
        const Index nvs = nvs_[sv];
        if (nvs == 0) continue;
        const RowMap Vflat = vv_flat(sv);
        for (Index c0 = 0; c0 < nmy; c0 += chunk) {
            const Index cn = std::min(chunk, nmy - c0);
            Mat C(naux_, cn);
            for (Index c = 0; c < cn; ++c) {
                const auto [m, y] = my[static_cast<std::size_t>(c0 + c)];
                if (dynamic) C.col(c) = w_[static_cast<std::size_t>(y)].col(m);
                else C.col(c) = boo(m, y);
            }
            Mat Jflat;
            parallel_vT_times(Vflat, C, Jflat);   // (nvs*nvs, cn)
            // Targets are visited per (m, y) column; one column may touch the
            // same target block from two columns only if (m, y) repeats, which
            // the de-duplicated list excludes, so writes do not race.
            #pragma omp parallel for schedule(dynamic)
            for (Index c = 0; c < cn; ++c) {
                const auto [m, y] = my[static_cast<std::size_t>(c0 + c)];
                Eigen::Map<const RowMat> J(Jflat.col(c).data(), nvs, nvs);
                for (int pp : pairs_of_occ_[static_cast<std::size_t>(y)]) {
                    UPair& t = P_[static_cast<std::size_t>(pp)];
                    if (!t.has[sv]) continue;
                    const auto it = std::lower_bound(t.L.begin(), t.L.end(), m);
                    if (it == t.L.end() || *it != m) continue;
                    const std::size_t lm = static_cast<std::size_t>(it - t.L.begin());
                    const int x = (y == t.j) ? t.i : t.j;
                    const int o = (y == t.j) ? 0 : 1;
                    const int q1 = pid(x, m);
                    if (q1 < 0) continue;
                    const UPair& Q1 = P_[static_cast<std::size_t>(q1)];
                    if (!Q1.has[sv]) continue;
                    Mat& dst = dynamic ? t.JwA[o][lm] : t.JA[o][lm];
                    if (dst.size() == 0) dst = Mat::Zero(t.n, Q1.n);
                    dst.noalias() += t.Us[sv].transpose() * (J * Q1.Us[sv]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
void USolver::set_t1_from_singles() {
    T1_.setZero();
    for (Index i = 0; i < no_; ++i) {
        const USingles& sg = S1_[static_cast<std::size_t>(i)];
        const int s = spin(static_cast<int>(i));
        if (sg.U.size() > 0)
            T1_.row(i).segment(voff(s), nvs_[s]) = (sg.U * sg.t).transpose();
        else
            T1_.row(i).segment(voff(s), nvs_[s]) = sg.t.transpose();
    }
}

// Per-iteration data: w, u, E objects, dynamic mixed Coulomb blocks.
void USolver::build_iteration_data() {
    w_.assign(static_cast<std::size_t>(no_), Mat::Zero(naux_, no_));
    uT_.assign(static_cast<std::size_t>(no_), Mat());
    for (int s = 0; s < 2; ++s) {
        const Index nos = nos_[s], nvs = nvs_[s];
        if (nos == 0) continue;
        const Index off = (s == 0) ? 0 : nos_[0];
        if (nvs == 0) {
            for (Index xl = 0; xl < nos; ++xl) uT_[static_cast<std::size_t>(off + xl)] = Mat::Zero(0, naux_);
            continue;
        }
        const Mat T1sT = T1_.block(off, voff(s), nos, nvs).transpose();   // (nvs, nos)
        // w: rows (P, m) of the row-major (naux*nos, nvs) view of B_ov^s.
        const RowMap Vov(ov_[s], naux_ * nos, nvs);
        Mat Wall;
        parallel_rows_times(Vov, T1sT, Wall);   // (naux*nos, nos): ((P,m), x)
        // u: rows (P, a) of the (naux*nvs, nvs) view of B_vv^s.
        Mat Uall;
        parallel_rows_times(vv_view(s), T1sT, Uall);   // (naux*nvs, nos)
        #pragma omp parallel for schedule(static)
        for (Index xl = 0; xl < nos; ++xl) {
            Mat& w = w_[static_cast<std::size_t>(off + xl)];
            Eigen::Map<const Mat> wmP(Wall.col(xl).data(), nos, naux_);   // (m, P)
            w.middleCols(off, nos) = wmP.transpose();
            Eigen::Map<const Mat> uaP(Uall.col(xl).data(), nvs, naux_);   // (a, P)
            uT_[static_cast<std::size_t>(off + xl)] = uaP;
        }
    }
    const Index np = static_cast<Index>(P_.size());
    #pragma omp parallel for schedule(dynamic)
    for (Index p = 0; p < np; ++p) {
        UPair& q = P_[static_cast<std::size_t>(p)];
        const RowView Bk = bov(q, q.i);
        const RowView Bl = bov(q, q.j);
        if (!Bk.empty()) q.El.noalias() = (*Bk.B) * q.T.middleRows(Bk.off, Bk.ns);
        else q.El = Mat::Zero(naux_, q.n);
        if (!Bl.empty()) q.Eh.noalias() = (*Bl.B) * q.T.middleRows(Bl.off, Bl.ns);
        else q.Eh = Mat::Zero(naux_, q.n);
    }
    // Dynamic mixed Coulomb blocks, same bookkeeping as the static ones.
    for (UPair& t : P_)
        for (Mat& J : t.JwA[0]) J.resize(0, 0);
    for (UPair& t : P_)
        for (Mat& J : t.JwA[1]) J.resize(0, 0);
    build_mixed_coulomb(true);
}

// ---------------------------------------------------------------------------
// E = sum_i f_i . t_i + sum_{i<j} sum_ab <ij||ab> [1/2 t_ij^ab + t_i^a t_j^b],
// singles projected into the pair space for the last term.
// ---------------------------------------------------------------------------
double USolver::energy() const {
    double e = 0.0;
    for (Index i = 0; i < no_; ++i) e += f_ov_.row(i).dot(T1_.row(i));
    for (const UPair& q : P_) {
        Vec ti = Vec::Zero(q.n), tj = Vec::Zero(q.n);
        const int si = spin(q.i), sj = spin(q.j);
        if (q.has[si]) ti = q.Us[si].transpose() * t1s(q.i);
        if (q.has[sj]) tj = q.Us[sj].transpose() * t1s(q.j);
        e += 0.5 * (q.K.array() * q.T.array()).sum() + ti.dot(q.K * tj);
    }
    return e;
}

// ---------------------------------------------------------------------------
// Doubles residual of target pair p in [p].
// ---------------------------------------------------------------------------
void USolver::residual_pair(Index pidx, Workspace& ws, Mat& R) const {
    const UPair& t = P_[static_cast<std::size_t>(pidx)];
    const int i = t.i, j = t.j;
    const Index n = t.n;
    const std::vector<int>& L = t.L;
    const Index nL = static_cast<Index>(L.size());
    auto lpos = [&](int m) {
        return static_cast<Index>(std::lower_bound(L.begin(), L.end(), m) - L.begin());
    };
    const Index li = lpos(i), lj = lpos(j);

    ws.qidx.assign(static_cast<std::size_t>(nL * nL), -1);
    ws.S.assign(static_cast<std::size_t>(nL * nL), Mat());
    for (Index a = 0; a < nL; ++a)
        for (Index b = a + 1; b < nL; ++b) {
            const int q = pid(L[static_cast<std::size_t>(a)], L[static_cast<std::size_t>(b)]);
            if (q < 0) continue;
            ws.qidx[static_cast<std::size_t>(a * nL + b)] = q;
            ws.qidx[static_cast<std::size_t>(b * nL + a)] = q;
            ws.S[static_cast<std::size_t>(a * nL + b)] =
                (q == static_cast<int>(pidx)) ? Mat(Mat::Identity(n, n))
                                              : overlap(t, P_[static_cast<std::size_t>(q)]);
        }
    auto qof = [&](Index a, Index b) { return ws.qidx[static_cast<std::size_t>(a * nL + b)]; };
    auto Sof = [&](Index a, Index b) -> const Mat& {
        return (a < b) ? ws.S[static_cast<std::size_t>(a * nL + b)]
                       : ws.S[static_cast<std::size_t>(b * nL + a)];
    };
    // Ordered amplitude t_{L[a] L[b]} projected into [p] on both indices.
    auto tproj = [&](Index a, Index b) -> Mat {
        const int q = qof(a, b);
        const Mat& S = Sof(a, b);
        const double sg = (L[static_cast<std::size_t>(a)] < L[static_cast<std::size_t>(b)]) ? 1.0 : -1.0;
        return sg * (S * P_[static_cast<std::size_t>(q)].T * S.transpose());
    };

    // Singles projected into [p]: column lm.
    Mat T1p = Mat::Zero(n, nL);
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        const int s = spin(m);
        if (t.has[s]) T1p.col(lm) = t.Us[s].transpose() * t1s(m);
    }
    std::vector<RowView> Bp(static_cast<std::size_t>(nL));
    for (Index lm = 0; lm < nL; ++lm) Bp[static_cast<std::size_t>(lm)] = bov(t, L[static_cast<std::size_t>(lm)]);
    auto has_row = [&](Index lm) { return !Bp[static_cast<std::size_t>(lm)].empty(); };
    // The target's own rows, expanded (zero where the pair kept no PNOs of
    // that spin).
    const Mat Bi = full_row(t, i);
    const Mat Bj = full_row(t, j);
    Mat Boo_i(naux_, nL), Boo_j(naux_, nL), Wi(naux_, nL), Wj(naux_, nL);
    Vec XL = Vec::Zero(naux_);
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        Boo_i.col(lm) = boo(m, i);
        Boo_j.col(lm) = boo(m, j);
        Wi.col(lm) = w_[static_cast<std::size_t>(i)].col(m);
        Wj.col(lm) = w_[static_cast<std::size_t>(j)].col(m);
        XL += w_[static_cast<std::size_t>(m)].col(m);
    }
    const Mat ui = u_proj(t, i);
    const Mat uj = u_proj(t, j);

    // ---- <ij||ab> ----
    R = t.K;

    // ---- sum_{m<n} 2 Wt(m,n) tau_mn: the W_mnij ladder and the quadratic
    //      tau-tau term of W_abef ----
    std::vector<Mat> Q(static_cast<std::size_t>(nL));
    for (Index lm = 0; lm < nL; ++lm)
        if (has_row(lm)) {
            const RowView& r = Bp[static_cast<std::size_t>(lm)];
            Q[static_cast<std::size_t>(lm)].noalias() = (*r.B) * t.T.middleRows(r.off, r.ns);
        }
    Mat Z = Mat::Zero(nL, nL);
    for (Index a = 0; a < nL; ++a) {
        if (!has_row(a)) continue;
        for (Index b = 0; b < nL; ++b) {
            if (!has_row(b)) continue;
            const RowView& r = Bp[static_cast<std::size_t>(b)];
            Z(a, b) = (Q[static_cast<std::size_t>(a)].middleCols(r.off, r.ns).array() * r.B->array()).sum();
        }
    }
    const Mat Z1 = Wi.transpose() * Wj;
    Mat Wt = 0.5 * (Boo_i.transpose() * Boo_j - Boo_j.transpose() * Boo_i);
    Wt.noalias() += 0.5 * (Boo_i.transpose() * Wj);
    Wt.noalias() -= 0.5 * (Wj.transpose() * Boo_i);
    Wt.noalias() -= 0.5 * (Boo_j.transpose() * Wi);
    Wt.noalias() += 0.5 * (Wi.transpose() * Boo_j);
    Wt += 0.5 * (Z + Z1 - Z1.transpose());
    for (Index a = 0; a < nL; ++a)
        for (Index b = a + 1; b < nL; ++b) {
            const int q = qof(a, b);
            if (q < 0) continue;
            const double wv = Wt(a, b);
            if (wv == 0.0) continue;
            const Mat& S = Sof(a, b);
            R.noalias() += (2.0 * wv) * (S * P_[static_cast<std::size_t>(q)].T * S.transpose());
        }
    R.noalias() += 2.0 * (T1p * Wt * T1p.transpose());

    // ---- sum_ef tau_ij^ef (ae|bf) and -P(ab) sum_m t_m^b sum_ef tau_ij^ef (ae|mf) ----
    {
        Mat Rlad = Mat::Zero(n, n);
        Mat Xm = Mat::Zero(nL, n);
        Mat G = Mat::Zero(n, n);
        const std::size_t nblk = t.BvvT.size();
        for (Index P = 0; P < naux_; ++P) {
            // G(a, f) = sum_e B_ae T(e, f): row block b reads block b of B.
            for (std::size_t b = 0; b < nblk; ++b) {
                const Index bo = t.bvv_off[b], bn = t.bvv_n[b];
                Eigen::Map<const Mat> slice(t.BvvT[b].col(P).data(), bn, bn);
                G.middleRows(bo, bn).noalias() = slice * t.T.middleRows(bo, bn);
            }
            // sum_f G(a, f) B_bf: column block b of Rlad reads block b of B.
            for (std::size_t b = 0; b < nblk; ++b) {
                const Index bo = t.bvv_off[b], bn = t.bvv_n[b];
                Eigen::Map<const Mat> slice(t.BvvT[b].col(P).data(), bn, bn);
                Rlad.middleCols(bo, bn).noalias() += G.middleCols(bo, bn) * slice.transpose();
            }
            for (Index lm = 0; lm < nL; ++lm) {
                if (!has_row(lm)) continue;
                const RowView& r = Bp[static_cast<std::size_t>(lm)];
                Xm.row(lm).noalias() += r.B->row(P) * G.middleCols(r.off, r.ns).transpose();
            }
        }
        R += Rlad;
        R.noalias() += ui.transpose() * uj;
        R.noalias() -= uj.transpose() * ui;
        Mat Gfull = Xm;
        Gfull.noalias() += Wj.transpose() * ui;
        Gfull.noalias() -= Wi.transpose() * uj;
        const Mat M = Gfull.transpose() * T1p.transpose();   // (a, b)
        R -= M - M.transpose();
    }

    // ---- F_me for m in L, e in [p] ----
    Mat Fme(nL, n);
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        const int s = spin(m);
        if (t.has[s])
            Fme.row(lm) = f_ov_.row(m).segment(voff(s), nvs_[s]) * t.Us[s];
        else
            Fme.row(lm).setZero();
        if (has_row(lm)) {
            const RowView& r = Bp[static_cast<std::size_t>(lm)];
            Fme.row(lm).segment(r.off, r.ns).noalias() += (r.B->transpose() * XL).transpose();
        }
    }
    {
        // - sum_n Bp[n]^T w_n(:, m): (n, nL) per n.
        Mat Wn(naux_, nL);
        for (Index ln = 0; ln < nL; ++ln) {
            if (!has_row(ln)) continue;
            const RowView& r = Bp[static_cast<std::size_t>(ln)];
            const Mat& wn = w_[static_cast<std::size_t>(L[static_cast<std::size_t>(ln)])];
            for (Index lm = 0; lm < nL; ++lm) Wn.col(lm) = wn.col(L[static_cast<std::size_t>(lm)]);
            Fme.middleCols(r.off, r.ns).noalias() -= (r.B->transpose() * Wn).transpose();
        }
    }

    // ---- P(ab) sum_e t_ij^ae Fh_be ----
    {
        Mat Fh = t.fvv;
        {
            for (std::size_t b = 0; b < t.BvvT.size(); ++b) {
                const Index bo = t.bvv_off[b], bn = t.bvv_n[b];
                const Vec v = t.BvvT[b] * XL;
                Fh.block(bo, bo, bn, bn) += Eigen::Map<const Mat>(v.data(), bn, bn).transpose();
            }
        }
        for (Index lm = 0; lm < nL; ++lm) {
            if (!has_row(lm)) continue;
            const RowView& r = Bp[static_cast<std::size_t>(lm)];
            const Mat um = u_proj(t, L[static_cast<std::size_t>(lm)]);
            Fh.middleCols(r.off, r.ns).noalias() -= um.transpose() * (*r.B);
        }
        for (Index a = 0; a < nL; ++a)
            for (Index b = a + 1; b < nL; ++b) {
                const int q = qof(a, b);
                if (q < 0) continue;
                const UPair& Qd = P_[static_cast<std::size_t>(q)];
                const Mat& S = Sof(a, b);
                Mat A = Mat::Zero(Qd.n, n);
                if (has_row(a)) {
                    const RowView& r = Bp[static_cast<std::size_t>(a)];
                    A.middleCols(r.off, r.ns).noalias() += Qd.Eh.transpose() * (*r.B);
                }
                if (has_row(b)) {
                    const RowView& r = Bp[static_cast<std::size_t>(b)];
                    A.middleCols(r.off, r.ns).noalias() -= Qd.El.transpose() * (*r.B);
                }
                Fh.noalias() += S * A;
            }
        Fh.noalias() -= T1p * Fme;
        const Mat M = t.T * Fh.transpose();
        R += M - M.transpose();
    }

    // ---- -P(ij) sum_m t_im^ab Fh_mj ----
    {
        auto Fh_m = [&](int x, Index lx, const Mat& Boo_x) {
            const Mat& wx = w_[static_cast<std::size_t>(x)];
            Vec out(nL);
            Mat Wm(naux_, nL);   // w_n(:, m) over n
            Mat Wx(naux_, nL);   // w_x(:, n) over n
            for (Index ln = 0; ln < nL; ++ln) Wx.col(ln) = wx.col(L[static_cast<std::size_t>(ln)]);
            for (Index lm = 0; lm < nL; ++lm) {
                const int m = L[static_cast<std::size_t>(lm)];
                for (Index ln = 0; ln < nL; ++ln)
                    Wm.col(ln) = w_[static_cast<std::size_t>(L[static_cast<std::size_t>(ln)])].col(m);
                double s = f_oo_(m, x) + f_ov_.row(m).dot(T1_.row(x));
                s += wx.col(m).dot(XL) - (Wm.array() * Wx.array()).sum();
                s += Boo_x.col(lm).dot(XL) - (Wm.array() * Boo_x.array()).sum();
                // sum_n sum_ef t_xn^{ef} (me|nf)
                for (Index ln = 0; ln < nL; ++ln) {
                    const int nn = L[static_cast<std::size_t>(ln)];
                    if (nn == x) continue;
                    const int q = qof(lx, ln);
                    if (q < 0) continue;
                    const UPair& Qd = P_[static_cast<std::size_t>(q)];
                    const RowView Bm = bov(Qd, m);
                    if (Bm.empty()) continue;
                    const double sg = (x < nn) ? 1.0 : -1.0;
                    const Mat& E = (x < nn) ? Qd.Eh : Qd.El;
                    s -= sg * (Bm.B->array() * E.middleCols(Bm.off, Bm.ns).array()).sum();
                }
                out(lm) = s;
            }
            return out;
        };
        const Vec Fh_mj = Fh_m(j, lj, Boo_j);
        const Vec Fh_mi = Fh_m(i, li, Boo_i);
        for (Index lm = 0; lm < nL; ++lm) {
            if (lm != li && qof(li, lm) >= 0) R.noalias() -= Fh_mj(lm) * tproj(li, lm);
            if (lm != lj && qof(lj, lm) >= 0) R.noalias() += Fh_mi(lm) * tproj(lj, lm);
        }
    }

    // ---- ring: P(ij) P(ab) [sum_me t_im^ae W_mbej - t_i^e t_m^a <mb||ej>] ----
    auto half = [&](int x, Index lx, int y, Index ly, const Mat& ux, const Mat& uy,
                    const Mat& Wx, const Mat& Wy, const Mat& Boo_y, int o) -> Mat {
        Mat h = Mat::Zero(n, n);
        const Mat& By = (y == i) ? Bi : Bj;
        Mat base = By + uy;
        base.noalias() -= (Boo_y + Wy) * T1p.transpose();   // - Vs_y
        {
            Mat V = Mat::Zero(naux_, n);   // sum_n (Bq3[n] t_yn) S3^T
            for (Index ln = 0; ln < nL; ++ln) {
                const int nn = L[static_cast<std::size_t>(ln)];
                if (nn == y) continue;
                const int q3 = qof(ly, ln);
                if (q3 < 0) continue;
                const UPair& Q3 = P_[static_cast<std::size_t>(q3)];
                const double sg = (y < nn) ? 1.0 : -1.0;
                V.noalias() += sg * ((y < nn ? Q3.Eh : Q3.El) * Sof(ly, ln).transpose());
            }
            base.noalias() -= 0.5 * V;
        }
        for (Index lm = 0; lm < nL; ++lm) {
            const int m = L[static_cast<std::size_t>(lm)];
            if (m == x) continue;
            const int q1 = qof(lx, lm);
            if (q1 < 0) continue;
            const UPair& Q1 = P_[static_cast<std::size_t>(q1)];
            const RowView B1m = bov(Q1, m);
            Mat W = Mat::Zero(Q1.n, n);
            if (!B1m.empty()) W.middleRows(B1m.off, B1m.ns).noalias() = B1m.B->transpose() * base;
            if (t.JA[o][static_cast<std::size_t>(lm)].size() > 0)
                W -= t.JA[o][static_cast<std::size_t>(lm)].transpose();
            if (t.JwA[o][static_cast<std::size_t>(lm)].size() > 0)
                W -= t.JwA[o][static_cast<std::size_t>(lm)].transpose();
            // + sum_n (Bq1[n]^T v_m) t1p_n^T and + 1/2 sum_n Bq1[n]^T (Bq3[m] t_yn) S3^T
            const Vec vm = Boo_y.col(lm) + Wy.col(lm);
            Mat C = Mat::Zero(Q1.n, nL);
            for (Index ln = 0; ln < nL; ++ln) {
                const int nn = L[static_cast<std::size_t>(ln)];
                const RowView B1n = bov(Q1, nn);
                if (B1n.empty()) continue;
                C.col(ln).segment(B1n.off, B1n.ns).noalias() = B1n.B->transpose() * vm;
                if (nn == y) continue;
                const int q3 = qof(ly, ln);
                if (q3 < 0) continue;
                const UPair& Q3 = P_[static_cast<std::size_t>(q3)];
                const RowView B3m = bov(Q3, m);
                if (B3m.empty()) continue;
                const double sg = (y < nn) ? 0.5 : -0.5;
                const Mat Y = (*B3m.B) * Q3.T.middleRows(B3m.off, B3m.ns);   // (naux, n_q3)
                W.middleRows(B1n.off, B1n.ns).noalias() +=
                    sg * ((B1n.B->transpose() * Y) * Sof(ly, ln).transpose());
            }
            W.noalias() += C * T1p.transpose();
            const double sgx = (x < m) ? 1.0 : -1.0;
            h.noalias() += (sgx * (Sof(lx, lm) * Q1.T)) * W;
        }
        // - sum_m t_m^a [ (Bp[y]^T w_x(:,m))(b) - (u_x^T boo(m,y))(b) ]
        Mat A = Wx.transpose() * By;
        A.noalias() -= Boo_y.transpose() * ux;
        h.noalias() -= T1p * A;
        return h;
    };
    {
        const Mat h_ij = half(i, li, j, lj, ui, uj, Wi, Wj, Boo_j, 0);
        const Mat h_ji = half(j, lj, i, li, uj, ui, Wj, Wi, Boo_i, 1);
        const Mat Hd = h_ij - h_ji;
        R += Hd - Hd.transpose();
    }

    // ---- P(ij) sum_e t_i^e <ab||ej> ----
    R.noalias() += ui.transpose() * Bj;
    R.noalias() -= Bj.transpose() * ui;
    R.noalias() -= uj.transpose() * Bi;
    R.noalias() += Bi.transpose() * uj;

    // ---- -P(ab) sum_m t_m^a <mb||ij> ----
    {
        Mat A = Boo_i.transpose() * Bj;
        A.noalias() -= Boo_j.transpose() * Bi;
        const Mat M = T1p * A;
        R -= M - M.transpose();
    }
}

// ---------------------------------------------------------------------------
// Singles residual of occupied i over its spin's full canonical virtuals; the
// 3-external term is returned as Y (nv_s, naux) for one batched contraction
// with B_vv: r -= sum_P B^P Y(:, P).
// ---------------------------------------------------------------------------
void USolver::residual_singles(int i, Vec& r, Mat& Y) const {
    const int s = spin(i);
    const Index nvs = nvs_[s];
    const USingles& sg = S1_[static_cast<std::size_t>(i)];
    const std::vector<int>& L = sg.L;
    const Index nL = static_cast<Index>(L.size());
    const Vec ti = t1s(i);
    const Mat& wi = w_[static_cast<std::size_t>(i)];

    r = f_ov_.row(i).segment(voff(s), nvs).transpose();
    r.noalias() += (*f_vv_[s]) * ti;
    Vec XL = Vec::Zero(naux_);
    for (int m : L) XL += w_[static_cast<std::size_t>(m)].col(m);

    // c(m, n) = w_i(:,m).w_n(:,n) - w_n(:,m).w_i(:,n); csum(m) = sum_n c(m,n)
    Vec csum = Vec::Zero(nL);
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        double acc = 0.0;
        for (int nn : L) {
            const Mat& wn = w_[static_cast<std::size_t>(nn)];
            acc += wi.col(m).dot(wn.col(nn)) - wn.col(m).dot(wi.col(nn));
        }
        csum(lm) = acc;
    }

    // Same-spin coupled singles carry the t_m^a terms.
    r.noalias() += uT_[static_cast<std::size_t>(i)] * XL;
    for (Index lm = 0; lm < nL; ++lm) {
        const int m = L[static_cast<std::size_t>(lm)];
        if (spin(m) != s) continue;
        const Vec tm = t1s(m);
        // -1/2 sum_m (f_m . t_i) t_m
        const double fmt = f_ov_.row(m).dot(T1_.row(i));
        // -sum_m u_m^T w_i(:, m)
        r.noalias() -= uT_[static_cast<std::size_t>(m)] * wi.col(m);
        // -sum_m F_mi t_m and -1/2 sum_m csum(m) t_m
        double Fmi = f_oo_(m, i) + 0.5 * fmt;
        const Vec bmi = boo(m, i);
        Fmi += bmi.dot(XL);
        for (int nn : L) Fmi -= w_[static_cast<std::size_t>(nn)].col(m).dot(boo(nn, i));
        for (int nn : L) {
            if (nn == i) continue;
            const int q = pid(i, nn);
            if (q < 0) continue;
            const UPair& Qd = P_[static_cast<std::size_t>(q)];
            const RowView Bm = bov(Qd, m);
            if (Bm.empty()) continue;
            const double sgn = (i < nn) ? 1.0 : -1.0;
            const Mat& E = (i < nn) ? Qd.Eh : Qd.El;
            Fmi -= sgn * (Bm.B->array() * E.middleCols(Bm.off, Bm.ns).array()).sum();
        }
        Fmi += 0.5 * csum(lm);
        r -= (0.5 * fmt + 0.5 * csum(lm) + Fmi) * tm;
    }

    // Doubles of F_ae: + U_q^s (Eh^T w_i(:,m) - El^T w_i(:,n)), q = (m<n) in L.
    // - U_q^s T_q [Bq[n]^T boo(m,i) - Bq[m]^T boo(n,i)].
    for (Index a = 0; a < nL; ++a)
        for (Index b = a + 1; b < nL; ++b) {
            const int m = L[static_cast<std::size_t>(a)], nn = L[static_cast<std::size_t>(b)];
            const int q = pid(m, nn);
            if (q < 0) continue;
            const UPair& Qd = P_[static_cast<std::size_t>(q)];
            if (!Qd.has[s]) continue;
            Vec v = Qd.Eh.transpose() * wi.col(m);
            v.noalias() -= Qd.El.transpose() * wi.col(nn);
            Vec z = Vec::Zero(Qd.n);
            const RowView Bn = bov(Qd, nn);
            const RowView Bm = bov(Qd, m);
            if (!Bn.empty()) z.segment(Bn.off, Bn.ns).noalias() += Bn.B->transpose() * boo(m, i);
            if (!Bm.empty()) z.segment(Bm.off, Bm.ns).noalias() -= Bm.B->transpose() * boo(nn, i);
            v.noalias() -= Qd.T * z;
            r.noalias() += Qd.Us[s] * v;
        }

    // + sum_m U_q^s (t_im F_me[q]) and the 3-external Y, q = pair(i, m).
    Y = Mat::Zero(nvs, naux_);
    for (int m : L) {
        if (m == i) continue;
        const int q = pid(i, m);
        if (q < 0) continue;
        const UPair& Qd = P_[static_cast<std::size_t>(q)];
        const int sm = spin(m);
        const double sgn = (i < m) ? 1.0 : -1.0;
        Vec F = Vec::Zero(Qd.n);
        if (Qd.has[sm]) F = (f_ov_.row(m).segment(voff(sm), nvs_[sm]) * Qd.Us[sm]).transpose();
        const RowView Bm = bov(Qd, m);
        if (!Bm.empty()) F.segment(Bm.off, Bm.ns).noalias() += Bm.B->transpose() * XL;
        for (int nn : L) {
            const RowView Bn = bov(Qd, nn);
            if (Bn.empty()) continue;
            F.segment(Bn.off, Bn.ns).noalias() -= Bn.B->transpose() * w_[static_cast<std::size_t>(nn)].col(m);
        }
        if (!Qd.has[s]) continue;
        r.noalias() += Qd.Us[s] * (sgn * (Qd.T * F));
        // Bq[m] t_im = sgn * (Bq[m] T): Eh when m is the pair's second index.
        const Mat& E = (i < m) ? Qd.Eh : Qd.El;
        Y.noalias() += Qd.Us[s] * (sgn * E).transpose();
    }

    // + B_ov[i]^T X_L - sum_n u_n^T boo(n, i)
    r.noalias() += bov_row(i).transpose() * XL;
    for (int nn : L) {
        if (spin(nn) != s) continue;
        r.noalias() -= uT_[static_cast<std::size_t>(nn)] * boo(nn, i);
    }
}

// All residuals for the current amplitudes; singles in full canonical
// coordinates of their spin.
void USolver::all_residuals(std::vector<Mat>& R2, std::vector<Vec>& R1full) const {
    const Index np = static_cast<Index>(P_.size());
    R2.assign(static_cast<std::size_t>(np), Mat());
    R1full.assign(static_cast<std::size_t>(no_), Vec());
    std::vector<Mat> Y(static_cast<std::size_t>(no_));
    #pragma omp parallel
    {
        Workspace ws;
        #pragma omp for schedule(dynamic)
        for (Index p = 0; p < np; ++p) residual_pair(p, ws, R2[static_cast<std::size_t>(p)]);
        #pragma omp for schedule(dynamic)
        for (Index i = 0; i < no_; ++i)
            residual_singles(static_cast<int>(i), R1full[static_cast<std::size_t>(i)], Y[static_cast<std::size_t>(i)]);
    }
    // Batched 3-external contraction per spin: r_i -= M^T vec(Y_i) with
    // M[(P,v), a] = B^P_{v a}.
    for (int s = 0; s < 2; ++s) {
        const Index nos = nos_[s], nvs = nvs_[s];
        if (nos == 0 || nvs == 0) continue;
        const Index off = (s == 0) ? 0 : nos_[0];
        Mat Ystack(naux_ * nvs, nos);
        for (Index xl = 0; xl < nos; ++xl)
            Ystack.col(xl) = Eigen::Map<const Vec>(Y[static_cast<std::size_t>(off + xl)].data(), naux_ * nvs);
        const RowMap M = vv_view(s);
        Mat out(nvs, nos);
        const Index blk = 64;
        #pragma omp parallel for schedule(dynamic)
        for (Index a0 = 0; a0 < nvs; a0 += blk) {
            const Index an = std::min(blk, nvs - a0);
            out.middleRows(a0, an).noalias() = M.middleCols(a0, an).transpose() * Ystack;
        }
        for (Index xl = 0; xl < nos; ++xl) R1full[static_cast<std::size_t>(off + xl)] -= out.col(xl);
    }
}

// ---------------------------------------------------------------------------
std::vector<double> USolver::flatten() const {
    std::vector<double> v;
    for (const USingles& sg : S1_) v.insert(v.end(), sg.t.data(), sg.t.data() + sg.t.size());
    for (const UPair& q : P_) v.insert(v.end(), q.T.data(), q.T.data() + q.T.size());
    return v;
}

void USolver::unflatten(const std::vector<double>& v) {
    std::size_t off = 0;
    for (USingles& sg : S1_) {
        std::copy(v.begin() + static_cast<std::ptrdiff_t>(off),
                  v.begin() + static_cast<std::ptrdiff_t>(off + static_cast<std::size_t>(sg.t.size())), sg.t.data());
        off += static_cast<std::size_t>(sg.t.size());
    }
    for (UPair& q : P_) {
        std::copy(v.begin() + static_cast<std::ptrdiff_t>(off),
                  v.begin() + static_cast<std::ptrdiff_t>(off + static_cast<std::size_t>(q.T.size())), q.T.data());
        off += static_cast<std::size_t>(q.T.size());
    }
}

void USolver::residuals_for(const Mat& T1, std::vector<Mat>& R2, Mat& R1) {
    if (T1.rows() != no_ || T1.cols() != nv_)
        throw std::invalid_argument("dlpno_uccsd_pno_residuals: T1 shape");
    T1_ = T1;
    build_iteration_data();
    std::vector<Vec> R1full;
    all_residuals(R2, R1full);
    R1 = Mat::Zero(no_, nv_);
    for (Index i = 0; i < no_; ++i) {
        const int s = spin(static_cast<int>(i));
        R1.row(i).segment(voff(s), nvs_[s]) = R1full[static_cast<std::size_t>(i)].transpose();
    }
}

// ---------------------------------------------------------------------------
DLPNOUCCSDPNOResult USolver::run() {
    DLPNOUCCSDPNOResult out;
    out.n_offlist_rows = n_offlist_rows_;
    out.static_factor_bytes = static_bytes_;
    out.setup_seconds = setup_seconds_;
    const auto t_iter0 = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> amp_hist, res_hist;
    set_t1_from_singles();
    double e_prev = energy();
    const Index np = static_cast<Index>(P_.size());
    for (int it = 0; it < opts_.max_iter; ++it) {
        const std::vector<double> x_old = flatten();
        build_iteration_data();
        std::vector<Mat> R2;
        std::vector<Vec> R1;
        all_residuals(R2, R1);
        for (Index p = 0; p < np; ++p) {
            UPair& q = P_[static_cast<std::size_t>(p)];
            const double fij = f_oo_(q.i, q.i) + f_oo_(q.j, q.j);
            Mat Tn = q.T;
            const Mat& Rq = R2[static_cast<std::size_t>(p)];
            for (Index b = 0; b < q.n; ++b)
                for (Index a = 0; a < q.n; ++a)
                    Tn(a, b) += Rq(a, b) / (fij - q.eps(a) - q.eps(b));
            q.T = 0.5 * (Tn - Tn.transpose());
        }
        for (Index i = 0; i < no_; ++i) {
            USingles& sg = S1_[static_cast<std::size_t>(i)];
            const Vec rs = (sg.U.size() > 0) ? Vec(sg.U.transpose() * R1[static_cast<std::size_t>(i)])
                                             : R1[static_cast<std::size_t>(i)];
            const double fi = f_oo_(i, i);
            for (Index a = 0; a < sg.t.size(); ++a) sg.t(a) += rs(a) / (fi - sg.eps(a));
        }
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
            Eigen::PartialPivLU<Mat> lu(Bm);
            const Vec c = lu.solve(rhs);
            if (c.allFinite()) {
                std::vector<double> mix(amp_hist.back().size(), 0.0);
                for (std::size_t k = 0; k < nh; ++k)
                    for (std::size_t q = 0; q < mix.size(); ++q) mix[q] += c(static_cast<Index>(k)) * amp_hist[k][q];
                unflatten(mix);
            }
        }
        set_t1_from_singles();
        const double e = energy();
        out.trace_e_corr.push_back(e);
        out.trace_r_norm.push_back(r_norm);
        out.n_iter = it + 1;
        if (it > 0 && std::abs(e - e_prev) < opts_.conv_tol_energy
            && r_norm < opts_.conv_tol_residual) {
            out.converged = true;
            e_prev = e;
            break;
        }
        e_prev = e;
    }
    out.iteration_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_iter0).count();
    out.e_corr = energy();
    out.T2.reserve(P_.size());
    for (const UPair& q : P_) out.T2.push_back(q.T);
    out.T1 = T1_;
    return out;
}

}  // namespace

DLPNOUCCSDPNOResult dlpno_uccsd_pno_solve(
    const DLPNORowMatCRef& B_ov_a, const DLPNORowMatCRef& B_ov_b,
    const DLPNORowMatCRef& B_oo_a, const DLPNORowMatCRef& B_oo_b,
    const DLPNORowMatCRef& B_vv_a, const DLPNORowMatCRef& B_vv_b,
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& f_vv_a, const Eigen::MatrixXd& f_vv_b,
    std::vector<DLPNOUPairInput> pairs, std::vector<DLPNOUSinglesInput> singles,
    const DLPNOUCCSDPNOOptions& opts) {
    USolver solver(B_ov_a, B_ov_b, B_oo_a, B_oo_b, B_vv_a, B_vv_b, f_oo, f_ov,
                   f_vv_a, f_vv_b, std::move(pairs), std::move(singles), opts);
    return solver.run();
}

void dlpno_uccsd_pno_residuals(
    const DLPNORowMatCRef& B_ov_a, const DLPNORowMatCRef& B_ov_b,
    const DLPNORowMatCRef& B_oo_a, const DLPNORowMatCRef& B_oo_b,
    const DLPNORowMatCRef& B_vv_a, const DLPNORowMatCRef& B_vv_b,
    const Eigen::MatrixXd& f_oo, const Eigen::MatrixXd& f_ov,
    const Eigen::MatrixXd& f_vv_a, const Eigen::MatrixXd& f_vv_b,
    std::vector<DLPNOUPairInput> pairs, std::vector<DLPNOUSinglesInput> singles,
    const Eigen::MatrixXd& T1, std::vector<Eigen::MatrixXd>& R2,
    Eigen::MatrixXd& R1) {
    USolver solver(B_ov_a, B_ov_b, B_oo_a, B_oo_b, B_vv_a, B_vv_b, f_oo, f_ov,
                   f_vv_a, f_vv_b, std::move(pairs), std::move(singles),
                   DLPNOUCCSDPNOOptions{});
    solver.residuals_for(T1, R2, R1);
}

// ===========================================================================
// Local open-shell (T0).
// ===========================================================================
namespace {

// Kept eigenvectors of a PSD density with occupation above `threshold`
// (all at threshold 0; the largest alone if none), semicanonicalised against
// `f`. Mirrors python/vibeqc/dlpno/uccsd_local_solver.py::_domain_from_density.
void domain_from_density(const Mat& density, const Mat& f, double threshold,
                         Mat& vectors, Vec& eps) {
    Eigen::SelfAdjointEigenSolver<Mat> es(0.5 * (density + density.transpose()));
    const Vec occ = es.eigenvalues();          // ascending
    const Mat& vec = es.eigenvectors();
    const Index n = occ.size();
    std::vector<Index> keep;
    for (Index k = n - 1; k >= 0; --k)
        if (threshold <= 0.0 || occ(k) > threshold) keep.push_back(k);
    if (keep.empty() && n > 0) keep.push_back(n - 1);
    Mat kept(density.rows(), static_cast<Index>(keep.size()));
    for (std::size_t c = 0; c < keep.size(); ++c) kept.col(static_cast<Index>(c)) = vec.col(keep[c]);
    const Mat fd = kept.transpose() * f * kept;
    Eigen::SelfAdjointEigenSolver<Mat> es2(0.5 * (fd + fd.transpose()));
    vectors = kept * es2.eigenvectors();
    eps = es2.eigenvalues();
}

class UTriples {
public:
    UTriples(const DLPNORowMatCRef* ov[2], const DLPNORowMatCRef* oo[2],
             const DLPNORowMatCRef* vv[2], const Vec& eps_o, const Mat* fvv[2],
             const std::vector<int>& pair_i, const std::vector<int>& pair_j,
             const std::vector<Mat>& U, const std::vector<Mat>& T2, const Mat& T1,
             double tcut_tno, double lindep)
        : eps_o_(eps_o), U_(U), T2_(T2), T1_(T1), tcut_tno_(tcut_tno), lindep_(lindep) {
        naux_ = vv[0]->rows();
        for (int s = 0; s < 2; ++s) {
            fvv_[s] = fvv[s];
            nvs_[s] = fvv[s]->rows();
            nos_[s] = (nvs_[s] > 0) ? ov[s]->cols() / nvs_[s]
                                    : static_cast<Index>(std::llround(std::sqrt(static_cast<double>(oo[s]->cols()))));
            ov_[s] = ov[s]->data();
            oo_[s] = oo[s]->data();
            vv_[s] = vv[s]->data();
            if (ov[s]->rows() != naux_ || oo[s]->rows() != naux_)
                throw std::invalid_argument("dlpno_uccsd_local_triples: DF factor rows");
            for (const DLPNORowMatCRef* fct : {ov[s], oo[s], vv[s]})
                if (fct->cols() > 0 && fct->outerStride() != fct->cols())
                    throw std::invalid_argument(
                        "dlpno_uccsd_local_triples: DF factors must be C-contiguous row-major");
        }
        no_ = nos_[0] + nos_[1];
        nv_ = nvs_[0] + nvs_[1];
        if (eps_o.size() != no_ || T1.rows() != no_ || T1.cols() != nv_)
            throw std::invalid_argument("dlpno_uccsd_local_triples: eps_o / T1 shapes");
        const std::size_t np = pair_i.size();
        if (pair_j.size() != np || U.size() != np || T2.size() != np)
            throw std::invalid_argument("dlpno_uccsd_local_triples: per-pair lists");
        pidx_.assign(static_cast<std::size_t>(no_ * no_), -1);
        partners_.assign(static_cast<std::size_t>(no_), {});
        for (std::size_t p = 0; p < np; ++p) {
            const int i = pair_i[p], j = pair_j[p];
            if (i < 0 || j < 0 || i >= no_ || j >= no_ || i >= j)
                throw std::invalid_argument("dlpno_uccsd_local_triples: pair indices");
            if (U[p].rows() != nv_ || T2[p].rows() != U[p].cols() || T2[p].cols() != U[p].cols())
                throw std::invalid_argument("dlpno_uccsd_local_triples: pair shapes");
            pidx_[static_cast<std::size_t>(i) * no_ + j] = static_cast<int>(p);
            partners_[static_cast<std::size_t>(i)].push_back(j);
            partners_[static_cast<std::size_t>(j)].push_back(i);
        }
        for (auto& v : partners_) std::sort(v.begin(), v.end());
    }

    double triple(int i, int j, int k, int& n_tno) const;

private:
    const Vec& eps_o_;
    const std::vector<Mat>& U_;
    const std::vector<Mat>& T2_;
    const Mat& T1_;
    double tcut_tno_, lindep_;
    const Mat* fvv_[2];
    const double* ov_[2];
    const double* oo_[2];
    const double* vv_[2];
    Index naux_ = 0, no_ = 0, nv_ = 0, nos_[2] = {0, 0}, nvs_[2] = {0, 0};
    std::vector<int> pidx_;
    std::vector<std::vector<int>> partners_;

    int spin(int m) const { return m < nos_[0] ? 0 : 1; }
    Index loc(int m) const { return m < nos_[0] ? m : m - nos_[0]; }
    Index voff(int s) const { return s == 0 ? 0 : nvs_[0]; }
    int pid(int a, int b) const {
        if (a == b) return -1;
        return pidx_[static_cast<std::size_t>(std::min(a, b)) * no_ + std::max(a, b)];
    }
    Vec boo(int m, int x) const {
        Vec v = Vec::Zero(naux_);
        const int s = spin(m);
        if (s != spin(x)) return v;
        const Index cols = nos_[s] * nos_[s];
        const Index c = loc(m) * nos_[s] + loc(x);
        for (Index P = 0; P < naux_; ++P) v(P) = oo_[s][P * cols + c];
        return v;
    }
    StridedRowMap bov_row(int m) const {
        const int s = spin(m);
        return StridedRowMap(ov_[s] + loc(m) * nvs_[s], naux_, nvs_[s],
                             Eigen::OuterStride<>(nos_[s] * nvs_[s]));
    }
    // Ordered amplitude t_{a b} (sign * T of the unordered pair).
    double sign(int a, int b) const { return a < b ? 1.0 : -1.0; }
};

double UTriples::triple(int i, int j, int k, int& n_tno) const {
    const int q_ij = pid(i, j), q_ik = pid(i, k), q_jk = pid(j, k);
    if (q_ij < 0 || q_ik < 0 || q_jk < 0)
        throw std::invalid_argument(
            "dlpno_uccsd_local_triples: every pair of an evaluated triple must be kept");
    // ---- TNO space: orthonormalised union of the three pair spaces ----
    const int qs[3] = {q_ij, q_ik, q_jk};
    Index ncol = 0;
    for (int q : qs) ncol += U_[static_cast<std::size_t>(q)].cols();
    Mat stacked(nv_, ncol);
    {
        Index c0 = 0;
        for (int q : qs) {
            const Mat& Uq = U_[static_cast<std::size_t>(q)];
            stacked.middleCols(c0, Uq.cols()) = Uq;
            c0 += Uq.cols();
        }
    }
    Mat precursor;
    {
        Eigen::HouseholderQR<Mat> qr(stacked);
        const Index kq = std::min(nv_, ncol);
        const Mat Q = qr.householderQ() * Mat::Identity(nv_, kq);
        Mat R = qr.matrixQR().topRows(kq);
        for (Index c = 0; c < R.cols(); ++c)
            for (Index row = c + 1; row < kq; ++row) R(row, c) = 0.0;
        Eigen::BDCSVD<Mat> svd(R, Eigen::ComputeThinU);
        const Vec sv = svd.singularValues();
        const double cutoff = lindep_ * std::max(sv.size() ? sv(0) : 0.0, 1e-300);
        Index rank = 0;
        for (Index c = 0; c < sv.size(); ++c) if (sv(c) > cutoff) ++rank;
        rank = std::max<Index>(1, rank);
        precursor = Q * svd.matrixU().leftCols(rank);
    }
    const Index r = precursor.cols();
    Mat f_pre = Mat::Zero(r, r);
    for (int s = 0; s < 2; ++s)
        if (nvs_[s] > 0) {
            const auto Ps = precursor.middleRows(voff(s), nvs_[s]);
            f_pre.noalias() += Ps.transpose() * (*fvv_[s]) * Ps;
        }
    Mat density = Mat::Zero(r, r);
    if (tcut_tno_ > 0.0) {
        for (int q : qs) {
            const Mat S = precursor.transpose() * U_[static_cast<std::size_t>(q)];
            const Mat A = S * T2_[static_cast<std::size_t>(q)] * S.transpose();
            density.noalias() += A * A.transpose();
            density.noalias() += A.transpose() * A;
        }
    }
    Mat vectors;
    Vec eps_t;
    domain_from_density(density, f_pre, tcut_tno_, vectors, eps_t);
    const Mat V = precursor * vectors;                 // (nv, nt)
    const Index nt = V.cols();
    n_tno = static_cast<int>(nt);
    if (nt < 3) return 0.0;   // a triple excitation needs three virtuals
    Mat Vs[2];
    bool has[2];
    for (int s = 0; s < 2; ++s) {
        Vs[s] = V.middleRows(voff(s), nvs_[s]);
        has[s] = Vs[s].size() > 0 && Vs[s].cwiseAbs().maxCoeff() > 0.0;
    }

    // ---- DF blocks in the TNO space ----
    const int occ3[3] = {i, j, k};
    Mat BT[3];   // (naux, nt): B^P_{x a}
    for (int a = 0; a < 3; ++a) {
        const int x = occ3[a];
        const int s = spin(x);
        BT[a] = has[s] ? Mat(bov_row(x) * Vs[s]) : Mat::Zero(naux_, nt);
    }
    // Bstack(e, b + P*nt) = B^P_{e b}.
    Mat Bstack = Mat::Zero(nt, nt * naux_);
    for (int s = 0; s < 2; ++s) {
        if (!has[s]) continue;
        const Index nvs = nvs_[s];
        const RowMap Vview(vv_[s], naux_ * nvs, nvs);
        const Index pblk = std::max<Index>(1, std::min<Index>(naux_, 32));
        Mat half;
        for (Index P0 = 0; P0 < naux_; P0 += pblk) {
            const Index pn = std::min(pblk, naux_ - P0);
            half.noalias() = Vview.middleRows(P0 * nvs, pn * nvs) * Vs[s];   // (pn*nvs, nt)
            for (Index dp = 0; dp < pn; ++dp)
                Bstack.middleCols((P0 + dp) * nt, nt).noalias() +=
                    Vs[s].transpose() * half.middleRows(dp * nvs, nvs);
        }
    }
    auto tproj = [&](int a, int b) -> Mat {   // ordered t_{ab} in the TNO basis
        const int q = pid(a, b);
        const Mat S = V.transpose() * U_[static_cast<std::size_t>(q)];
        return sign(a, b) * (S * T2_[static_cast<std::size_t>(q)] * S.transpose());
    };
    Mat t1T(nt, 3);
    for (int a = 0; a < 3; ++a) t1T.col(a) = V.transpose() * T1_.row(occ3[a]).transpose();

    const std::size_t nt3 = static_cast<std::size_t>(nt) * nt * nt;
    auto at = [nt](Index a, Index b, Index c) {
        return static_cast<std::size_t>(a) + static_cast<std::size_t>(nt) * (static_cast<std::size_t>(b) + static_cast<std::size_t>(nt) * c);
    };
    // X(p,q,r)(a,b,c) = sum_e t_qr^{ae} <ep||bc> - sum_m t_pm^{bc} <ma||qr>
    auto X = [&](int ap, int aq, int ar, std::vector<double>& out) {
        const int p = occ3[ap], q = occ3[aq], rr = occ3[ar];
        out.assign(nt3, 0.0);
        // term 1 via (e b|p c) - (e c|p b)
        const Mat tqr = tproj(q, rr);                          // (a, e)
        const Mat G = tqr * Bstack;                           // (a, b + P*nt)
        Eigen::Map<const Mat> Gm(G.data(), nt * nt, naux_);    // ((a,b), P)
        const Mat M1 = Gm * BT[ap];                           // ((a,b), c)
        for (Index c = 0; c < nt; ++c)
            for (Index b = 0; b < nt; ++b)
                for (Index a = 0; a < nt; ++a)
                    out[at(a, b, c)] = M1(a + nt * b, c) - M1(a + nt * c, b);
        // term 2 over m with a kept pair (p, m)
        const std::vector<int>& ms = partners_[static_cast<std::size_t>(p)];
        if (ms.empty()) return;
        const Index nm = static_cast<Index>(ms.size());
        Mat Tstack(nt * nt, nm);
        Mat Gmat(nm, nt);
        for (Index c = 0; c < nm; ++c) {
            const int m = ms[static_cast<std::size_t>(c)];
            const Mat tpm = tproj(p, m);
            Tstack.col(c) = Eigen::Map<const Vec>(tpm.data(), nt * nt);
            Gmat.row(c) = boo(m, q).transpose() * BT[ar] - boo(m, rr).transpose() * BT[aq];
        }
        const Mat M2 = Tstack * Gmat;                          // ((b,c), a)
        for (Index c = 0; c < nt; ++c)
            for (Index b = 0; b < nt; ++b)
                for (Index a = 0; a < nt; ++a)
                    out[at(a, b, c)] -= M2(b + nt * c, a);
    };
    std::vector<double> wc(nt3), work(nt3), wd(nt3, 0.0);
    X(0, 1, 2, wc);
    X(1, 0, 2, work);
    for (std::size_t z = 0; z < nt3; ++z) wc[z] -= work[z];
    X(2, 1, 0, work);
    for (std::size_t z = 0; z < nt3; ++z) wc[z] -= work[z];
    auto permute_abc = [&](std::vector<double>& w) {
        for (Index c = 0; c < nt; ++c)
            for (Index b = 0; b < nt; ++b)
                for (Index a = 0; a < nt; ++a)
                    work[at(a, b, c)] = w[at(a, b, c)] - w[at(b, a, c)] - w[at(c, b, a)];
        std::swap(w, work);
    };
    permute_abc(wc);
    // Disconnected: Y(p,q,r) = t_p^a <qr||bc>.
    auto add_y = [&](int ap, int aq, int ar, double sg) {
        const Mat Kqr = BT[aq].transpose() * BT[ar];           // (b, c) = (qb|rc)
        const Mat Kas = Kqr - Kqr.transpose();
        for (Index c = 0; c < nt; ++c)
            for (Index b = 0; b < nt; ++b) {
                const double v = sg * Kas(b, c);
                for (Index a = 0; a < nt; ++a) wd[at(a, b, c)] += v * t1T(a, ap);
            }
    };
    add_y(0, 1, 2, 1.0);
    add_y(1, 0, 2, -1.0);
    add_y(2, 1, 0, -1.0);
    permute_abc(wd);
    const double d_occ = eps_o_(i) + eps_o_(j) + eps_o_(k);
    double e = 0.0;
    for (Index c = 0; c < nt; ++c)
        for (Index b = 0; b < nt; ++b)
            for (Index a = 0; a < nt; ++a) {
                const std::size_t z = at(a, b, c);
                e += wc[z] * (wc[z] + wd[z]) / (d_occ - eps_t(a) - eps_t(b) - eps_t(c));
            }
    return e / 6.0;
}

}  // namespace

DLPNOUTriplesResult dlpno_uccsd_local_triples(
    const DLPNORowMatCRef& B_ov_a, const DLPNORowMatCRef& B_ov_b,
    const DLPNORowMatCRef& B_oo_a, const DLPNORowMatCRef& B_oo_b,
    const DLPNORowMatCRef& B_vv_a, const DLPNORowMatCRef& B_vv_b,
    const Eigen::VectorXd& eps_o, const Eigen::MatrixXd& f_vv_a,
    const Eigen::MatrixXd& f_vv_b, const std::vector<int>& pair_i,
    const std::vector<int>& pair_j, const std::vector<Eigen::MatrixXd>& U,
    const std::vector<Eigen::MatrixXd>& T2, const Eigen::MatrixXd& T1,
    const std::vector<std::vector<int>>& triples, double tcut_tno,
    double lindep) {
    const DLPNORowMatCRef* ov[2] = {&B_ov_a, &B_ov_b};
    const DLPNORowMatCRef* oo[2] = {&B_oo_a, &B_oo_b};
    const DLPNORowMatCRef* vv[2] = {&B_vv_a, &B_vv_b};
    const Mat* fvv[2] = {&f_vv_a, &f_vv_b};
    const auto t0 = std::chrono::steady_clock::now();
    UTriples engine(ov, oo, vv, eps_o, fvv, pair_i, pair_j, U, T2, T1, tcut_tno, lindep);
    const Index n = static_cast<Index>(triples.size());
    DLPNOUTriplesResult out;
    out.n_tno.assign(static_cast<std::size_t>(n), 0);
    out.e_triple.assign(static_cast<std::size_t>(n), 0.0);
    for (const auto& t : triples)
        if (t.size() != 3 || !(t[0] < t[1] && t[1] < t[2]))
            throw std::invalid_argument(
                "dlpno_uccsd_local_triples: triples must be sorted i < j < k");
    #pragma omp parallel for schedule(dynamic)
    for (Index z = 0; z < n; ++z) {
        const auto& t = triples[static_cast<std::size_t>(z)];
        int nt = 0;
        out.e_triple[static_cast<std::size_t>(z)] = engine.triple(t[0], t[1], t[2], nt);
        out.n_tno[static_cast<std::size_t>(z)] = nt;
    }
    for (double e : out.e_triple) out.e_t += e;
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace vibeqc
