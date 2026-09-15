// Open-shell UCCSD and perturbative (T) on a UHF reference, density-fitted
// by default (CCSDOptions.density_fit = true) or with exact four-index
// integrals on the canonical conventional route (density_fit = false); the
// amplitude equations are shared between the two.
//
// This is the unrestricted sibling of cpp/src/ccsd.cpp. Where the closed-
// shell kernel spin-integrates the SGWB-1991 working equations to spatial
// orbitals, this kernel evaluates them directly over SPIN ORBITALS, so it
// runs unmodified on a UHF reference (the alpha and beta spin orbitals carry
// distinct spatial MOs). The equation set transcribed here is exactly the
// spin-orbital reference kernel python/vibeqc/dlpno/_ccsd_ref.py
// (so_energy / so_residuals / so_triples_correction, used through
// run_ref_uccsd); that reference is FCI-anchored and validated against PySCF
// cc.UCCSD/UCCSD(T) to < 1 uHa, and this C++ port is pinned to it by
// tests/test_uccsd_anchor.py.
//
// Integrals (DePrince-Sherrill 2013 DF): a block-diagonal spin-orbital
// B-tensor is assembled from the alpha/beta MO-transformed DF tensors
// (mo_transform); the antisymmetrised spin-orbital integrals follow from
//   <pq||rs> = (pr|qs) - (ps|qr),  (pq|rs) = sum_P B^P_pq B^P_rs,
// where B^P_pq is non-zero only for same-spin (p,q) pairs. The four-index
// antisymmetrised tensor is materialised over the correlated spin-orbital
// window (dense O(N^6) small-molecule pilot, matching ccsd.cpp's nv^4 block).
// On the canonical route (density_fit = false) the same <pq||rs> tensor is
// assembled from exact per-spin-channel MO integrals (eri_mo_pair_transform)
// instead of the RI-fitted B factorisation.
//
// References: Purvis-Bartlett 1982 (CCSD); Stanton-Gauss-Watts-Bartlett 1991
// (working equations); DePrince-Sherrill 2013 (DF-CCSD); Raghavachari-
// Trucks-Pople-Head-Gordon 1989 ((T)); Pulay 1980 (DIIS). Full provenance in
// cpp/include/vibeqc/uccsd.hpp.
//
// Conventions: spin orbitals are ordered occupied-first,
// [alpha-occ, beta-occ, alpha-vir, beta-vir]; n = no + nv. i,j,k,l,m,n index
// occupied spin orbitals [0,no); a,b,c,d,e,f index virtual spin orbitals
// [0,nv) (combined index no + a). The Fock diagonal is kept in the MP
// denominators D1/D2 and the residuals carry the explicit "- t * D" term, so
// the returned R1/R2 are true residuals that vanish at the CCSD fixed point
// (matching _ccsd_ref.py).

#include "vibeqc/uccsd.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/correlation_conventions.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vibeqc/df.hpp"
#include "vibeqc/integrals.hpp"

namespace vibeqc {

namespace {

using Mat = Eigen::MatrixXd;
using RowMat =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// =========================================================================
// Antisymmetrised spin-orbital integrals over the correlated window.
//
// eri(p,q,r,s) = <pq||rs>, stored dense row-major over the combined
// spin-orbital space of size n = no + nv.
// =========================================================================
struct SpinOrbitalIntegrals {
    std::vector<double> eri;  // n^4, <pq||rs>
    Eigen::Index n = 0;

    double operator()(Eigen::Index p, Eigen::Index q, Eigen::Index r,
                      Eigen::Index s) const noexcept {
        return eri[(((static_cast<std::size_t>(p) * n + q) * n + r) * n + s)];
    }
};

// Scatter a same-spin MO-transformed DF block B_full (n_aux, nt*nt) for one
// spin channel into the combined spin-orbital B-tensor B_so (n_aux, n*n).
// `to_combined[x]` maps a spin-channel orbital index x in [0, nt) to its
// position in the combined occupied-first ordering.
void scatter_spin_block(const RowMat& B_full, Eigen::Index nt,
                        const std::vector<Eigen::Index>& to_combined,
                        Eigen::Index n, RowMat& B_so) {
    const Eigen::Index n_aux = B_full.rows();
    for (Eigen::Index x = 0; x < nt; ++x) {
        const Eigen::Index cx = to_combined[static_cast<std::size_t>(x)];
        for (Eigen::Index y = 0; y < nt; ++y) {
            const Eigen::Index cy = to_combined[static_cast<std::size_t>(y)];
            for (Eigen::Index P = 0; P < n_aux; ++P)
                B_so(P, cx * n + cy) = B_full(P, x * nt + y);
        }
    }
}

// Build <pq||rs> over the combined window from the block-diagonal
// spin-orbital B-tensor: chem = B_so^T B_so gives (pq|rs); antisymmetrise.
SpinOrbitalIntegrals build_so_integrals(const RowMat& B_so, Eigen::Index n) {
    // chem((p*n+r),(q*n+s)) = (pr|qs) = B_so^T B_so, antisymmetrised into
    // V.eri[p,q,r,s] = <pq||rs>. chem is (n^2 x n^2) = n^4, the SAME size as
    // V.eri, so forming it whole alongside V.eri made the build-time peak 2*n^4
    // -- the largest allocation in the whole open-shell CC run (the iteration
    // holds V.eri plus the smaller Wabef ~ nv^4). Block chem over the first
    // index p: each p-tile forms only chem's rows {p*n+r : p in tile} =
    // B_so columns [p0*n, (p0+pc)*n), fills V.eri for that tile, then frees the
    // tile. Peak drops to V.eri + one chem row-tile. Bit-identical: a chem tile
    // is a sub-block of the same GEMM.
    SpinOrbitalIntegrals V;
    V.n = n;
    V.eri.assign(static_cast<std::size_t>(n) * n * n * n, 0.0);

    // Tile width: bound the chem tile to ~kSoChemTileRows combined rows. Small
    // windows (n < ~45) use a single tile == the original (one GEMM).
    constexpr Eigen::Index kSoChemTileRows = 2048;
    const Eigen::Index pblk = std::max<Eigen::Index>(
        1, std::min<Eigen::Index>(n, kSoChemTileRows /
                                       std::max<Eigen::Index>(1, n)));

    for (Eigen::Index p0 = 0; p0 < n; p0 += pblk) {
        const Eigen::Index pc = std::min(pblk, n - p0);
        // chem_tile(pl*n + r, q*n + s) = chem((p0+pl)*n + r, q*n + s).
        const Mat chem_tile =
            B_so.middleCols(p0 * n, pc * n).transpose() * B_so;  // (pc*n, n^2)
        #pragma omp parallel for collapse(2) schedule(static)
        for (Eigen::Index pl = 0; pl < pc; ++pl)
            for (Eigen::Index q = 0; q < n; ++q) {
                const Eigen::Index p = p0 + pl;
                for (Eigen::Index r = 0; r < n; ++r)
                    for (Eigen::Index s = 0; s < n; ++s)
                        V.eri[(((static_cast<std::size_t>(p) * n + q) * n + r)
                               * n) + s] =
                            chem_tile(pl * n + r, q * n + s)
                          - chem_tile(pl * n + s, q * n + r);
            }
    }
    return V;
}

// Canonical (non-DF) counterpart of the B-tensor construction: build
// <pq||rs> over the combined spin-orbital window from exact four-index MO
// integrals, one chemists' pair transform per spin channel (aa, bb, ab; the
// ba channel is the ab transpose).  Bit-exact integrals instead of the RI
// factorisation -- the conventional route for parity against conventional
// UCCSD(T) in other programs.  Small-molecule by design: holds the AO
// tensor (nao^4) plus the three spatial channel tensors (~3 nt^4)
// transiently alongside the spin-orbital V.eri (n^4).
SpinOrbitalIntegrals build_so_integrals_canonical(
    const BasisSet& basis,
    const Mat& C_full_a, const Mat& C_full_b,
    const std::vector<Eigen::Index>& map_a,
    const std::vector<Eigen::Index>& map_b,
    Eigen::Index n) {
    const Eigen::Index nta = C_full_a.cols();
    const Eigen::Index ntb = C_full_b.cols();

    Mat W_aa, W_bb, W_ab;
    {
        const Eri4D eri = compute_eri(basis);
        W_aa = eri_mo_pair_transform(eri, C_full_a, C_full_a);
        W_bb = eri_mo_pair_transform(eri, C_full_b, C_full_b);
        W_ab = eri_mo_pair_transform(eri, C_full_a, C_full_b);
    }

    // Invert the occupied-first combined maps: spin + channel index per
    // combined spin orbital.  map_a/map_b are a disjoint cover of [0, n).
    std::vector<int> spin_of(static_cast<std::size_t>(n));
    std::vector<Eigen::Index> idx_of(static_cast<std::size_t>(n));
    for (Eigen::Index x = 0; x < nta; ++x) {
        const auto c = static_cast<std::size_t>(map_a[static_cast<std::size_t>(x)]);
        spin_of[c] = 0;
        idx_of[c] = x;
    }
    for (Eigen::Index x = 0; x < ntb; ++x) {
        const auto c = static_cast<std::size_t>(map_b[static_cast<std::size_t>(x)]);
        spin_of[c] = 1;
        idx_of[c] = x;
    }

    // chem(c1, c2, c3, c4) = (c1 c2 | c3 c4): non-zero only when each
    // charge-distribution pair is same-spin.
    const auto chem = [&](Eigen::Index c1, Eigen::Index c2,
                          Eigen::Index c3, Eigen::Index c4) -> double {
        const int s12 = spin_of[static_cast<std::size_t>(c1)];
        if (s12 != spin_of[static_cast<std::size_t>(c2)]
            || spin_of[static_cast<std::size_t>(c3)]
                   != spin_of[static_cast<std::size_t>(c4)])
            return 0.0;
        const int s34 = spin_of[static_cast<std::size_t>(c3)];
        const Eigen::Index x = idx_of[static_cast<std::size_t>(c1)];
        const Eigen::Index y = idx_of[static_cast<std::size_t>(c2)];
        const Eigen::Index z = idx_of[static_cast<std::size_t>(c3)];
        const Eigen::Index w = idx_of[static_cast<std::size_t>(c4)];
        if (s12 == 0 && s34 == 0) return W_aa(x * nta + y, z * nta + w);
        if (s12 == 1 && s34 == 1) return W_bb(x * ntb + y, z * ntb + w);
        if (s12 == 0) return W_ab(x * nta + y, z * ntb + w);
        // (beta beta | alpha alpha): the ab tensor with the pairs swapped.
        return W_ab(z * nta + w, x * ntb + y);
    };

    SpinOrbitalIntegrals V;
    V.n = n;
    V.eri.assign(static_cast<std::size_t>(n) * n * n * n, 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index p = 0; p < n; ++p)
        for (Eigen::Index q = 0; q < n; ++q)
            for (Eigen::Index r = 0; r < n; ++r)
                for (Eigen::Index s = 0; s < n; ++s)
                    V.eri[(((static_cast<std::size_t>(p) * n + q) * n + r) * n)
                          + s] = chem(p, r, q, s) - chem(p, s, q, r);
    return V;
}

// MO-basis Fock block for one spin channel over the active window, assembled
// occupied-first into the combined spin-orbital Fock f_so.
void scatter_fock_block(const Mat& f_mo, Eigen::Index nt,
                        const std::vector<Eigen::Index>& to_combined,
                        Mat& f_so) {
    for (Eigen::Index x = 0; x < nt; ++x) {
        const Eigen::Index cx = to_combined[static_cast<std::size_t>(x)];
        for (Eigen::Index y = 0; y < nt; ++y)
            f_so(cx, to_combined[static_cast<std::size_t>(y)]) = f_mo(x, y);
    }
}

// =========================================================================
// CCSD correlation energy (spin orbital, _ccsd_ref.so_energy):
//   E = sum_ia f_ia t_i^a + 1/4 sum_ijab <ij||ab> t_ij^ab
//                         + 1/2 sum_ijab <ij||ab> t_i^a t_j^b
// =========================================================================
double so_energy(const SpinOrbitalIntegrals& V, const Mat& f_ov,
                 const Mat& T1, const std::vector<double>& T2,
                 Eigen::Index no, Eigen::Index nv) {
    double e = (f_ov.array() * T1.array()).sum();
    #pragma omp parallel for collapse(2) reduction(+ : e) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    const double oovv = V(i, j, no + a, no + b);
                    const double t2 =
                        T2[(((static_cast<std::size_t>(i) * no + j) * nv + a)
                            * nv) + b];
                    e += 0.25 * oovv * t2 + 0.5 * oovv * T1(i, a) * T1(j, b);
                }
    return e;
}

// T2 flat index helper.
inline std::size_t t2idx(Eigen::Index i, Eigen::Index j, Eigen::Index a,
                         Eigen::Index b, Eigen::Index no, Eigen::Index nv) {
    return (((static_cast<std::size_t>(i) * no + j) * nv + a) * nv) + b;
}

// =========================================================================
// CCSD residuals (spin orbital, _ccsd_ref.so_residuals).  Returns true
// residuals R1, R2 (they include the explicit "- t * D" term); the caller's
// Jacobi step T += R / D then reduces to the standard RHS/D update.
//
// f_so is the full spin-orbital Fock (diagonal kept); f_od is f_so with the
// diagonal zeroed; D1/D2 are the MP denominators from eps = diag(f_so).
// =========================================================================
void so_residuals(const SpinOrbitalIntegrals& V, const Mat& f_so,
                  const Mat& f_od, const Mat& T1, const std::vector<double>& T2,
                  const Mat& D1, const std::vector<double>& D2, Eigen::Index no,
                  Eigen::Index nv, Mat& R1, std::vector<double>& R2) {
    auto t2 = [&](Eigen::Index i, Eigen::Index j, Eigen::Index a,
                  Eigen::Index b) { return T2[t2idx(i, j, a, b, no, nv)]; };
    // f_ov(m,e) = f_so(m, no+e); f_od_vv(a,e) = f_od(no+a, no+e); etc.

    // ---- tau-tilde and tau-full (stored full no*no*nv*nv) ----
    const std::size_t n2 = static_cast<std::size_t>(no) * no * nv * nv;
    std::vector<double> tt(n2), tf(n2);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    const std::size_t idx = t2idx(i, j, a, b, no, nv);
                    const double dd = T1(i, a) * T1(j, b) - T1(i, b) * T1(j, a);
                    tt[idx] = T2[idx] + 0.5 * dd;
                    tf[idx] = T2[idx] + dd;
                }

    // ---- one-body intermediates ----
    // Fae(a,e) = f_od_vv(a,e) - 1/2 sum_m f_ov(m,e) t_m^a
    //          + sum_mf t_m^f <ma||fe> - 1/2 sum_mnf tt_mn^af <mn||ef>
    Mat Fae = Mat::Zero(nv, nv);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index e = 0; e < nv; ++e) {
            double s = f_od(no + a, no + e);
            for (Eigen::Index m = 0; m < no; ++m) {
                s -= 0.5 * f_so(m, no + e) * T1(m, a);
                for (Eigen::Index f = 0; f < nv; ++f)
                    s += T1(m, f) * V(m, no + a, no + f, no + e);
            }
            for (Eigen::Index m = 0; m < no; ++m)
                for (Eigen::Index nn = 0; nn < no; ++nn)
                    for (Eigen::Index f = 0; f < nv; ++f)
                        s -= 0.5 * tt[t2idx(m, nn, a, f, no, nv)]
                           * V(m, nn, no + e, no + f);
            Fae(a, e) = s;
        }

    // Fmi(m,i) = f_od_oo(m,i) + 1/2 sum_e t_i^e f_ov(m,e)
    //          + sum_ne t_n^e <mn||ie> + 1/2 sum_nef tt_in^ef <mn||ef>
    Mat Fmi = Mat::Zero(no, no);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index i = 0; i < no; ++i) {
            double s = f_od(m, i);
            for (Eigen::Index e = 0; e < nv; ++e)
                s += 0.5 * T1(i, e) * f_so(m, no + e);
            for (Eigen::Index nn = 0; nn < no; ++nn)
                for (Eigen::Index e = 0; e < nv; ++e)
                    s += T1(nn, e) * V(m, nn, i, no + e);
            for (Eigen::Index nn = 0; nn < no; ++nn)
                for (Eigen::Index e = 0; e < nv; ++e)
                    for (Eigen::Index f = 0; f < nv; ++f)
                        s += 0.5 * tt[t2idx(i, nn, e, f, no, nv)]
                           * V(m, nn, no + e, no + f);
            Fmi(m, i) = s;
        }

    // Fme(m,e) = f_ov(m,e) + sum_nf t_n^f <mn||ef>
    Mat Fme = Mat::Zero(no, nv);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index e = 0; e < nv; ++e) {
            double s = f_so(m, no + e);
            for (Eigen::Index nn = 0; nn < no; ++nn)
                for (Eigen::Index f = 0; f < nv; ++f)
                    s += T1(nn, f) * V(m, nn, no + e, no + f);
            Fme(m, e) = s;
        }

    // ---- T1 residual ----
    // r_i^a = f_ia + sum_e t_i^e Fae(a,e) - sum_m t_m^a Fmi(m,i)
    //       + sum_me t_im^ae Fme(m,e) - sum_nf t_n^f <na||if>
    //       - 1/2 sum_mef t_im^ef <ma||ef> - 1/2 sum_mne t_mn^ae <nm||ei>
    //       - t_i^a D1
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a) {
            double s = f_so(i, no + a);
            for (Eigen::Index e = 0; e < nv; ++e) s += T1(i, e) * Fae(a, e);
            for (Eigen::Index m = 0; m < no; ++m) s -= T1(m, a) * Fmi(m, i);
            for (Eigen::Index m = 0; m < no; ++m)
                for (Eigen::Index e = 0; e < nv; ++e)
                    s += t2(i, m, a, e) * Fme(m, e);
            for (Eigen::Index nn = 0; nn < no; ++nn)
                for (Eigen::Index f = 0; f < nv; ++f)
                    s -= T1(nn, f) * V(nn, no + a, i, no + f);
            for (Eigen::Index m = 0; m < no; ++m)
                for (Eigen::Index e = 0; e < nv; ++e)
                    for (Eigen::Index f = 0; f < nv; ++f)
                        s -= 0.5 * t2(i, m, e, f) * V(m, no + a, no + e, no + f);
            for (Eigen::Index m = 0; m < no; ++m)
                for (Eigen::Index nn = 0; nn < no; ++nn)
                    for (Eigen::Index e = 0; e < nv; ++e)
                        s -= 0.5 * t2(m, nn, a, e) * V(nn, m, no + e, i);
            R1(i, a) = s - T1(i, a) * D1(i, a);
        }

    // ---- two-body intermediates ----
    // Wmnij(m,n,i,j) = <mn||ij> + sum_e t_j^e <mn||ie> - sum_e t_i^e <mn||je>
    //               + 1/4 sum_ef tf_ij^ef <mn||ef>
    std::vector<double> Wmnij(static_cast<std::size_t>(no) * no * no * no);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index nn = 0; nn < no; ++nn)
            for (Eigen::Index i = 0; i < no; ++i)
                for (Eigen::Index j = 0; j < no; ++j) {
                    double s = V(m, nn, i, j);
                    for (Eigen::Index e = 0; e < nv; ++e) {
                        s += T1(j, e) * V(m, nn, i, no + e);
                        s -= T1(i, e) * V(m, nn, j, no + e);
                    }
                    for (Eigen::Index e = 0; e < nv; ++e)
                        for (Eigen::Index f = 0; f < nv; ++f)
                            s += 0.25 * tf[t2idx(i, j, e, f, no, nv)]
                               * V(m, nn, no + e, no + f);
                    Wmnij[(((static_cast<std::size_t>(m) * no + nn) * no + i)
                           * no) + j] = s;
                }

    // Wabef(a,b,e,f) = <ab||ef> - sum_m t_m^b <am||ef> + sum_m t_m^a <bm||ef>
    //               + 1/4 sum_mn tf_mn^ab <mn||ef>
    std::vector<double> Wabef(static_cast<std::size_t>(nv) * nv * nv * nv);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index b = 0; b < nv; ++b)
            for (Eigen::Index e = 0; e < nv; ++e)
                for (Eigen::Index f = 0; f < nv; ++f) {
                    double s = V(no + a, no + b, no + e, no + f);
                    for (Eigen::Index m = 0; m < no; ++m) {
                        s -= T1(m, b) * V(no + a, m, no + e, no + f);
                        s += T1(m, a) * V(no + b, m, no + e, no + f);
                    }
                    for (Eigen::Index m = 0; m < no; ++m)
                        for (Eigen::Index nn = 0; nn < no; ++nn)
                            s += 0.25 * tf[t2idx(m, nn, a, b, no, nv)]
                               * V(m, nn, no + e, no + f);
                    Wabef[(((static_cast<std::size_t>(a) * nv + b) * nv + e)
                           * nv) + f] = s;
                }

    // Wmbej(m,b,e,j) = <mb||ej> + sum_f t_j^f <mb||ef> - sum_n t_n^b <mn||ej>
    //               - sum_nf (1/2 t_jn^fb + t_j^f t_n^b) <mn||ef>
    std::vector<double> Wmbej(static_cast<std::size_t>(no) * nv * nv * no);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index b = 0; b < nv; ++b)
            for (Eigen::Index e = 0; e < nv; ++e)
                for (Eigen::Index j = 0; j < no; ++j) {
                    double s = V(m, no + b, no + e, j);
                    for (Eigen::Index f = 0; f < nv; ++f)
                        s += T1(j, f) * V(m, no + b, no + e, no + f);
                    for (Eigen::Index nn = 0; nn < no; ++nn)
                        s -= T1(nn, b) * V(m, nn, no + e, j);
                    for (Eigen::Index nn = 0; nn < no; ++nn)
                        for (Eigen::Index f = 0; f < nv; ++f)
                            s -= (0.5 * t2(j, nn, f, b) + T1(j, f) * T1(nn, b))
                               * V(m, nn, no + e, no + f);
                    Wmbej[(((static_cast<std::size_t>(m) * nv + b) * nv + e)
                           * no) + j] = s;
                }

    auto wmbej = [&](Eigen::Index m, Eigen::Index b, Eigen::Index e,
                     Eigen::Index j) -> double {
        return Wmbej[(((static_cast<std::size_t>(m) * nv + b) * nv + e) * no)
                     + j];
    };

    // ---- T2 residual ----
    // Fae_h(b,e) = Fae(b,e) - 1/2 sum_m t_m^b Fme(m,e)
    // Fmi_h(m,j) = Fmi(m,j) + 1/2 sum_e t_j^e Fme(m,e)
    Mat Fae_h = Fae;
    Fae_h.noalias() -= 0.5 * T1.transpose() * Fme;
    Mat Fmi_h = Fmi;
    Fmi_h.noalias() += 0.5 * Fme * T1.transpose();

    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    double s = V(i, j, no + a, no + b);
                    for (Eigen::Index e = 0; e < nv; ++e) {
                        s += t2(i, j, a, e) * Fae_h(b, e);
                        s -= t2(i, j, b, e) * Fae_h(a, e);
                    }
                    for (Eigen::Index m = 0; m < no; ++m) {
                        s -= t2(i, m, a, b) * Fmi_h(m, j);
                        s += t2(j, m, a, b) * Fmi_h(m, i);
                    }
                    for (Eigen::Index m = 0; m < no; ++m)
                        for (Eigen::Index nn = 0; nn < no; ++nn)
                            s += 0.5 * tf[t2idx(m, nn, a, b, no, nv)]
                               * Wmnij[(((static_cast<std::size_t>(m) * no + nn)
                                         * no + i) * no) + j];
                    for (Eigen::Index e = 0; e < nv; ++e)
                        for (Eigen::Index f = 0; f < nv; ++f)
                            s += 0.5 * tf[t2idx(i, j, e, f, no, nv)]
                               * Wabef[(((static_cast<std::size_t>(a) * nv + b)
                                         * nv + e) * nv) + f];
                    // ring: P(ij) P(ab) [ t_im^ae Wmbej(m,b,e,j)
                    //                     - t_i^e t_m^a <mb||ej> ]
                    for (Eigen::Index m = 0; m < no; ++m)
                        for (Eigen::Index e = 0; e < nv; ++e) {
                            s += t2(i, m, a, e) * wmbej(m, b, e, j)
                               - T1(i, e) * T1(m, a) * V(m, no + b, no + e, j);
                            s -= t2(j, m, a, e) * wmbej(m, b, e, i)
                               - T1(j, e) * T1(m, a) * V(m, no + b, no + e, i);
                            s -= t2(i, m, b, e) * wmbej(m, a, e, j)
                               - T1(i, e) * T1(m, b) * V(m, no + a, no + e, j);
                            s += t2(j, m, b, e) * wmbej(m, a, e, i)
                               - T1(j, e) * T1(m, b) * V(m, no + a, no + e, i);
                        }
                    for (Eigen::Index e = 0; e < nv; ++e) {
                        s += T1(i, e) * V(no + a, no + b, no + e, j);
                        s -= T1(j, e) * V(no + a, no + b, no + e, i);
                    }
                    for (Eigen::Index m = 0; m < no; ++m) {
                        s -= T1(m, a) * V(m, no + b, i, j);
                        s += T1(m, b) * V(m, no + a, i, j);
                    }
                    R2[t2idx(i, j, a, b, no, nv)] =
                        s - T2[t2idx(i, j, a, b, no, nv)]
                              * D2[t2idx(i, j, a, b, no, nv)];
                }
}

// =========================================================================
// (T) correction (Raghavachari 1989), spin-orbital
// (_ccsd_ref.so_triples_correction).  Evaluated triple-by-triple over
// distinct (i<j<k); the t3 tensor is never materialised.
// =========================================================================
int uccsd_triples_runtime_max_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

double so_triples(const SpinOrbitalIntegrals& V, const Mat& T1,
                  const std::vector<double>& T2, const Eigen::VectorXd& eps,
                  Eigen::Index no, Eigen::Index nv,
                  int requested_threads = 0) {
    if (no < 1 || nv < 1) return 0.0;
    const auto nv3 = static_cast<std::size_t>(nv) * nv * nv;
    auto t2 = [&](Eigen::Index i, Eigen::Index j, Eigen::Index a,
                  Eigen::Index b) { return T2[t2idx(i, j, a, b, no, nv)]; };

    double e_t = 0.0;
    const int n_threads = requested_threads > 0
        ? requested_threads : uccsd_triples_runtime_max_threads();
    #pragma omp parallel num_threads(n_threads) reduction(+ : e_t)
    {
        std::vector<double> wc(nv3), wd(nv3), x(nv3);

        // X(i_,j_,k_)[a,b,c] = sum_e t_{j_ k_}^{a e} <e i_||b c>
        //                    - sum_m t_{i_ m}^{b c} <m a||j_ k_}
        auto build_X = [&](Eigen::Index i_, Eigen::Index j_, Eigen::Index k_) {
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    for (Eigen::Index c = 0; c < nv; ++c) {
                        double s = 0.0;
                        for (Eigen::Index e = 0; e < nv; ++e)
                            s += t2(j_, k_, a, e)
                               * V(no + e, i_, no + b, no + c);
                        for (Eigen::Index m = 0; m < no; ++m)
                            s -= t2(i_, m, b, c) * V(m, no + a, j_, k_);
                        x[(static_cast<std::size_t>(a) * nv + b) * nv + c] = s;
                    }
        };

        #pragma omp for collapse(2) schedule(dynamic)
        for (Eigen::Index i = 0; i < no; ++i)
            for (Eigen::Index j = 0; j < no; ++j) {
                if (j <= i) continue;
                for (Eigen::Index k = j + 1; k < no; ++k) {
                    // Wc = X(i,j,k) - X(j,i,k) - X(k,j,i), occupied P(i/jk)
                    std::fill(wc.begin(), wc.end(), 0.0);
                    build_X(i, j, k);
                    for (std::size_t t = 0; t < nv3; ++t) wc[t] += x[t];
                    build_X(j, i, k);
                    for (std::size_t t = 0; t < nv3; ++t) wc[t] -= x[t];
                    build_X(k, j, i);
                    for (std::size_t t = 0; t < nv3; ++t) wc[t] -= x[t];
                    // virtual P(a/bc): Wc[a,b,c] -= Wc[b,a,c] + Wc[c,b,a]
                    for (Eigen::Index a = 0; a < nv; ++a)
                        for (Eigen::Index b = 0; b < nv; ++b)
                            for (Eigen::Index c = 0; c < nv; ++c) {
                                const std::size_t abc =
                                    (static_cast<std::size_t>(a) * nv + b) * nv
                                    + c;
                                const std::size_t bac =
                                    (static_cast<std::size_t>(b) * nv + a) * nv
                                    + c;
                                const std::size_t cba =
                                    (static_cast<std::size_t>(c) * nv + b) * nv
                                    + a;
                                x[abc] = wc[abc] - wc[bac] - wc[cba];
                            }
                    std::swap(wc, x);

                    // Wd from disconnected Y(i_,j_,k_)[a,b,c]
                    //   = t_{i_}^a <j_ k_||b c>
                    auto build_Y = [&](Eigen::Index i_, Eigen::Index j_,
                                       Eigen::Index k_, double sign) {
                        for (Eigen::Index a = 0; a < nv; ++a)
                            for (Eigen::Index b = 0; b < nv; ++b)
                                for (Eigen::Index c = 0; c < nv; ++c)
                                    wd[(static_cast<std::size_t>(a) * nv + b)
                                       * nv + c] +=
                                        sign * T1(i_, a)
                                        * V(j_, k_, no + b, no + c);
                    };
                    std::fill(wd.begin(), wd.end(), 0.0);
                    build_Y(i, j, k, +1.0);
                    build_Y(j, i, k, -1.0);
                    build_Y(k, j, i, -1.0);
                    for (Eigen::Index a = 0; a < nv; ++a)
                        for (Eigen::Index b = 0; b < nv; ++b)
                            for (Eigen::Index c = 0; c < nv; ++c) {
                                const std::size_t abc =
                                    (static_cast<std::size_t>(a) * nv + b) * nv
                                    + c;
                                const std::size_t bac =
                                    (static_cast<std::size_t>(b) * nv + a) * nv
                                    + c;
                                const std::size_t cba =
                                    (static_cast<std::size_t>(c) * nv + b) * nv
                                    + a;
                                x[abc] = wd[abc] - wd[bac] - wd[cba];
                            }
                    std::swap(wd, x);

                    const double d_occ = eps(i) + eps(j) + eps(k);
                    double s = 0.0;
                    for (Eigen::Index a = 0; a < nv; ++a)
                        for (Eigen::Index b = 0; b < nv; ++b)
                            for (Eigen::Index c = 0; c < nv; ++c) {
                                const std::size_t abc =
                                    (static_cast<std::size_t>(a) * nv + b) * nv
                                    + c;
                                const double d = d_occ - eps(no + a)
                                               - eps(no + b) - eps(no + c);
                                s += wc[abc] * (wc[abc] + wd[abc]) / d;
                            }
                    e_t += s / 6.0;
                }
            }
    }
    return e_t;
}

// Scalar form of the same spin-orbital (T) equations. Each (a,b,c) moment is
// formed, contracted into the energy, and discarded, eliminating the
// historical 3*nv^3 doubles per active OpenMP worker. The dense n^4
// SpinOrbitalIntegrals object remains a known open-shell floor; this fallback
// bounds triples workspace without claiming to solve integral storage.
double so_triples_direct(const SpinOrbitalIntegrals& V, const Mat& T1,
                         const std::vector<double>& T2,
                         const Eigen::VectorXd& eps, Eigen::Index no,
                         Eigen::Index nv, int requested_threads = 0) {
    if (no < 3 || nv < 1) return 0.0;
    const auto t2 = [&](Eigen::Index i, Eigen::Index j, Eigen::Index a,
                        Eigen::Index b) {
        return T2[t2idx(i, j, a, b, no, nv)];
    };
    const auto x = [&](Eigen::Index i, Eigen::Index j, Eigen::Index k,
                       Eigen::Index a, Eigen::Index b, Eigen::Index c) {
        double value = 0.0;
        for (Eigen::Index e = 0; e < nv; ++e)
            value += t2(j, k, a, e) * V(no + e, i, no + b, no + c);
        for (Eigen::Index m = 0; m < no; ++m)
            value -= t2(i, m, b, c) * V(m, no + a, j, k);
        return value;
    };
    const auto occupied_x = [&](Eigen::Index i, Eigen::Index j,
                                Eigen::Index k, Eigen::Index a,
                                Eigen::Index b, Eigen::Index c) {
        return x(i, j, k, a, b, c) - x(j, i, k, a, b, c)
             - x(k, j, i, a, b, c);
    };
    const auto y = [&](Eigen::Index i, Eigen::Index j, Eigen::Index k,
                       Eigen::Index a, Eigen::Index b, Eigen::Index c) {
        return T1(i, a) * V(j, k, no + b, no + c)
             - T1(j, a) * V(i, k, no + b, no + c)
             - T1(k, a) * V(j, i, no + b, no + c);
    };

    const int n_threads = requested_threads > 0
        ? requested_threads : uccsd_triples_runtime_max_threads();
    double e_t = 0.0;
    #pragma omp parallel for collapse(2) num_threads(n_threads) \
        reduction(+ : e_t) schedule(dynamic)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j) {
            if (j <= i) continue;
            double pair_energy = 0.0;
            for (Eigen::Index k = j + 1; k < no; ++k) {
                const double d_occ = eps(i) + eps(j) + eps(k);
                for (Eigen::Index a = 0; a < nv; ++a)
                    for (Eigen::Index b = 0; b < nv; ++b)
                        for (Eigen::Index c = 0; c < nv; ++c) {
                            const double wc =
                                occupied_x(i, j, k, a, b, c)
                                - occupied_x(i, j, k, b, a, c)
                                - occupied_x(i, j, k, c, b, a);
                            const double wd = y(i, j, k, a, b, c)
                                - y(i, j, k, b, a, c)
                                - y(i, j, k, c, b, a);
                            const double denominator = d_occ - eps(no + a)
                                - eps(no + b) - eps(no + c);
                            pair_energy += wc * (wc + wd) / denominator;
                        }
            }
            e_t += pair_energy / 6.0;
        }
    return e_t;
}

struct UCCSDTriplesMemoryPlan {
    bool direct = false;
    std::string name = "fast";
    int tile_size = 0;
    int threads = 1;
    std::size_t peak_bytes = 0;
};

std::size_t uccsd_saturating_add(std::size_t a, std::size_t b) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    return b > maximum - a ? maximum : a + b;
}

std::size_t uccsd_saturating_mul(std::size_t a, std::size_t b) {
    if (a == 0 || b == 0) return 0;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    return a > maximum / b ? maximum : a * b;
}

std::string normalise_uccsd_triples_mode(const std::string& input) {
    std::string mode;
    for (const char c : input) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        mode.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    if (mode.empty()) mode = "auto";
    if (mode == "low") mode = "blocked";
    if (mode != "auto" && mode != "fast" && mode != "blocked"
        && mode != "direct" && mode != "disk") {
        throw std::invalid_argument(
            "run_uccsd: unknown triples_memory_mode '" + input
            + "'; supported: auto, fast, blocked (legacy: low), direct, disk");
    }
    return mode;
}

UCCSDTriplesMemoryPlan plan_uccsd_triples_memory(
    const SpinOrbitalIntegrals& V, const Mat& T1,
    const std::vector<double>& T2, const Eigen::VectorXd& eps,
    Eigen::Index no, Eigen::Index nv, const CCSDOptions& opts) {
    if (opts.triples_tile_size < 0)
        throw std::invalid_argument(
            "run_uccsd: triples_tile_size must be >= 0");
    if (opts.triples_max_threads < 0)
        throw std::invalid_argument(
            "run_uccsd: triples_max_threads must be >= 0");
    const std::string mode = normalise_uccsd_triples_mode(
        opts.triples_memory_mode);
    if (mode == "disk")
        throw std::runtime_error(
            "run_uccsd: triples_memory_mode='disk' cannot spill the current "
            "dense spin-orbital n^4 integral representation; use 'direct' "
            "to bound triples workspace");

    const std::size_t retained = uccsd_saturating_add(
        uccsd_saturating_add(
            uccsd_saturating_mul(V.eri.size(), sizeof(double)),
            uccsd_saturating_mul(static_cast<std::size_t>(T1.size()),
                                 sizeof(double))),
        uccsd_saturating_add(
            uccsd_saturating_mul(T2.size(), sizeof(double)),
            uccsd_saturating_mul(static_cast<std::size_t>(eps.size()),
                                 sizeof(double))));
    const std::size_t triples_count = no >= 3
        ? static_cast<std::size_t>(no) * static_cast<std::size_t>(no - 1)
              * static_cast<std::size_t>(no - 2) / 6
        : 1;
    int max_threads = uccsd_triples_runtime_max_threads();
    if (opts.triples_max_threads > 0)
        max_threads = std::min(max_threads, opts.triples_max_threads);
    max_threads = std::max(1, std::min(
        max_threads, static_cast<int>(std::min<std::size_t>(
                         triples_count,
                         static_cast<std::size_t>(
                             std::numeric_limits<int>::max())))));
    const std::size_t nv3 = uccsd_saturating_mul(
        uccsd_saturating_mul(static_cast<std::size_t>(nv),
                             static_cast<std::size_t>(nv)),
        static_cast<std::size_t>(nv));
    const std::size_t fast_per_thread = uccsd_saturating_mul(
        uccsd_saturating_mul(3, nv3), sizeof(double));
    const bool bounded = opts.requested_memory_bytes > 0;
    const auto fits = [&](std::size_t bytes) {
        return !bounded || bytes <= opts.requested_memory_bytes;
    };
    const auto fast_plan = [&](int threads) {
        UCCSDTriplesMemoryPlan plan;
        plan.name = "fast";
        plan.tile_size = static_cast<int>(nv);
        plan.threads = threads;
        plan.peak_bytes = uccsd_saturating_add(
            retained, uccsd_saturating_mul(
                          static_cast<std::size_t>(threads),
                          fast_per_thread));
        return plan;
    };
    const auto direct_plan = [&] {
        UCCSDTriplesMemoryPlan plan;
        plan.direct = true;
        plan.name = "direct";
        plan.threads = max_threads;
        plan.peak_bytes = retained;
        return plan;
    };

    if (mode == "fast") {
        for (int threads = max_threads; threads >= 1; --threads) {
            const auto plan = fast_plan(threads);
            if (fits(plan.peak_bytes)) return plan;
        }
    } else if (mode == "direct" || mode == "blocked") {
        const auto plan = direct_plan();
        if (fits(plan.peak_bytes)) return plan;
    } else {
        const auto fast = fast_plan(max_threads);
        if (fits(fast.peak_bytes)) return fast;
        const auto direct = direct_plan();
        if (fits(direct.peak_bytes)) return direct;
    }

    std::ostringstream message;
    message << "run_uccsd: requested triples memory budget of "
            << opts.requested_memory_bytes
            << " bytes cannot hold the selected '" << mode
            << "' strategy; dense spin-orbital integral floor is "
            << retained << " bytes";
    throw std::runtime_error(message.str());
}

// =========================================================================
// DIIS over the joint (T1, T2) amplitude/residual vectors (Pulay 1980),
// mirroring CCSD_DIIS in ccsd.cpp.
// =========================================================================
class UCCSD_DIIS {
public:
    explicit UCCSD_DIIS(std::size_t max_subspace) : max_(max_subspace) {}

    void extrapolate(Mat& T1, std::vector<double>& T2, const Mat& R1,
                     const std::vector<double>& R2) {
        const std::size_t n1 = static_cast<std::size_t>(T1.size());
        const std::size_t n2 = T2.size();
        Eigen::VectorXd amp(n1 + n2), err(n1 + n2);
        for (std::size_t k = 0; k < n1; ++k) amp[k] = T1.data()[k];
        for (std::size_t k = 0; k < n2; ++k) amp[n1 + k] = T2[k];
        for (std::size_t k = 0; k < n1; ++k) err[k] = R1.data()[k];
        for (std::size_t k = 0; k < n2; ++k) err[n1 + k] = R2[k];

        amp_.push_back(amp);
        err_.push_back(err);
        if (amp_.size() > max_) {
            amp_.pop_front();
            err_.pop_front();
        }
        used_ = amp_.size();
        if (used_ < 2) return;

        const auto N = static_cast<Eigen::Index>(used_);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N + 1, N + 1);
        for (Eigen::Index k = 0; k < N; ++k)
            for (Eigen::Index l = 0; l < N; ++l)
                B(k, l) = err_[static_cast<std::size_t>(k)].dot(
                    err_[static_cast<std::size_t>(l)]);
        B.row(N).setConstant(-1.0);
        B.col(N).setConstant(-1.0);
        B(N, N) = 0.0;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N + 1);
        rhs(N) = -1.0;
        Eigen::VectorXd c = B.fullPivLu().solve(rhs);
        if (!c.head(N).allFinite()) return;

        Eigen::VectorXd mix = Eigen::VectorXd::Zero(n1 + n2);
        for (Eigen::Index k = 0; k < N; ++k)
            mix += c(k) * amp_[static_cast<std::size_t>(k)];
        for (std::size_t k = 0; k < n1; ++k) T1.data()[k] = mix[k];
        for (std::size_t k = 0; k < n2; ++k) T2[k] = mix[n1 + k];
    }

    std::size_t subspace_size() const noexcept { return used_; }

private:
    std::size_t max_;
    std::size_t used_ = 0;
    std::deque<Eigen::VectorXd> amp_;
    std::deque<Eigen::VectorXd> err_;
};

}  // namespace

// =========================================================================
// Shared implementation: DF-UCCSD(T) on given MOs + per-spin Fock.
// =========================================================================
namespace {
CCSDResult run_uccsd_impl(const Molecule& mol, const BasisSet& basis,
                          const Eigen::MatrixXd& C_alpha,
                          const Eigen::MatrixXd& C_beta,
                          const Eigen::MatrixXd& Fock_alpha,
                          const Eigen::MatrixXd& Fock_beta, double e_hf,
                          int ecp_total_ncore,
                          const CCSDOptions& opts) {
    // Coupled-pair variants (CCD / LCCD / LCCSD / CEPA) are implemented in
    // the closed-shell kernel only.
    {
        std::string k;
        for (const char c : opts.cc_variant)
            if (c != ' ' && c != '_' && c != '-')
                k.push_back(static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c))));
        if (!k.empty() && k != "ccsd")
            throw std::invalid_argument(
                "run_uccsd: cc_variant='" + opts.cc_variant +
                "' is only available in the closed-shell kernel; "
                "open-shell coupled-pair variants are not implemented");
    }
    if (opts.compute_triples) {
        const std::string mode = normalise_uccsd_triples_mode(
            opts.triples_memory_mode);
        if (opts.triples_tile_size < 0)
            throw std::invalid_argument(
                "run_uccsd: triples_tile_size must be >= 0");
        if (opts.triples_max_threads < 0)
            throw std::invalid_argument(
                "run_uccsd: triples_max_threads must be >= 0");
        if (mode == "disk")
            throw std::runtime_error(
                "run_uccsd: triples_memory_mode='disk' cannot spill the "
                "current dense spin-orbital n^4 integral representation; "
                "use 'direct' to bound triples workspace");
    }
    const auto n_orb = static_cast<Eigen::Index>(C_alpha.cols());
    const int n_elec =
        effective_electron_count(mol, ecp_total_ncore, "run_uccsd");
    const int two_s = mol.multiplicity() - 1;  // 2S = n_alpha - n_beta
    const int n_alpha = (n_elec + two_s) / 2;
    const int n_beta = n_elec - n_alpha;
    if (n_alpha + n_beta != n_elec || n_alpha < n_beta)
        throw std::runtime_error(
            "run_uccsd: inconsistent electron count / multiplicity");

    const int n_frozen = resolve_native_frozen_core(
        mol, opts.n_frozen_core, ecp_total_ncore, "run_uccsd");
    if (n_frozen > n_beta)
        throw std::runtime_error(
            "run_uccsd: n_frozen_core exceeds the beta-occupied orbitals");
    if (n_frozen >= n_alpha)
        throw std::runtime_error(
            "run_uccsd: n_frozen_core freezes all alpha-occupied orbitals");

    const Eigen::Index no_a = n_alpha - n_frozen;
    const Eigen::Index no_b = n_beta - n_frozen;
    const Eigen::Index nv_a = n_orb - n_alpha;
    const Eigen::Index nv_b = n_orb - n_beta;
    const Eigen::Index no = no_a + no_b;
    const Eigen::Index nv = nv_a + nv_b;
    const Eigen::Index n = no + nv;
    if (nv < 1) throw std::runtime_error("run_uccsd: no virtual orbitals");

    // Active MO columns per spin: [active occ | vir].
    Mat C_occ_a = C_alpha.middleCols(n_frozen, no_a);
    Mat C_vir_a = C_alpha.middleCols(n_alpha, nv_a);
    Mat C_occ_b = C_beta.middleCols(n_frozen, no_b);
    Mat C_vir_b = C_beta.middleCols(n_beta, nv_b);
    Mat C_full_a(n_orb, no_a + nv_a);
    C_full_a << C_occ_a, C_vir_a;
    Mat C_full_b(n_orb, no_b + nv_b);
    C_full_b << C_occ_b, C_vir_b;

    // Combined occupied-first index maps for each spin channel.
    std::vector<Eigen::Index> map_a(static_cast<std::size_t>(no_a + nv_a));
    for (Eigen::Index x = 0; x < no_a; ++x)
        map_a[static_cast<std::size_t>(x)] = x;
    for (Eigen::Index x = 0; x < nv_a; ++x)
        map_a[static_cast<std::size_t>(no_a + x)] = no + x;
    std::vector<Eigen::Index> map_b(static_cast<std::size_t>(no_b + nv_b));
    for (Eigen::Index x = 0; x < no_b; ++x)
        map_b[static_cast<std::size_t>(x)] = no_a + x;
    for (Eigen::Index x = 0; x < nv_b; ++x)
        map_b[static_cast<std::size_t>(no_b + x)] = no + nv_a + x;

    // Spin-orbital integrals <pq||rs>.  DF (default): block-diagonal
    // spin-orbital B-tensor.  Canonical (density_fit = false): exact
    // four-index per-spin-channel MO integrals.  Either way the
    // construction scratch is freed as soon as the dense integral set V is
    // built -- the iteration + triples below hold V.eri (~O(n^4)), the
    // memory wall.
    const SpinOrbitalIntegrals V = [&] {
        if (opts.density_fit) {
            const BasisSet aux(mol, opts.aux_basis);
            DensityFitting df(basis, aux);
            const RowMat B_a = df.mo_transform(C_full_a, C_full_a);
            const RowMat B_b = df.mo_transform(C_full_b, C_full_b);
            const auto n_aux = static_cast<Eigen::Index>(df.n_aux());
            RowMat B_so = RowMat::Zero(n_aux, n * n);
            scatter_spin_block(B_a, no_a + nv_a, map_a, n, B_so);
            scatter_spin_block(B_b, no_b + nv_b, map_b, n, B_so);
            return build_so_integrals(B_so, n);
        }
        return build_so_integrals_canonical(basis, C_full_a, C_full_b,
                                            map_a, map_b, n);
    }();

    // Spin-orbital Fock: f_mo per spin = C^T F_ao C over the active window.
    Mat f_so = Mat::Zero(n, n);
    {
        const Mat f_mo_a = C_full_a.transpose() * Fock_alpha * C_full_a;
        const Mat f_mo_b = C_full_b.transpose() * Fock_beta * C_full_b;
        scatter_fock_block(f_mo_a, no_a + nv_a, map_a, f_so);
        scatter_fock_block(f_mo_b, no_b + nv_b, map_b, f_so);
    }
    C_occ_a.resize(0, 0);
    C_vir_a.resize(0, 0);
    C_occ_b.resize(0, 0);
    C_vir_b.resize(0, 0);
    C_full_a.resize(0, 0);
    C_full_b.resize(0, 0);
    std::vector<Eigen::Index>().swap(map_a);
    std::vector<Eigen::Index>().swap(map_b);
    Mat f_od = f_so;
    for (Eigen::Index p = 0; p < n; ++p) f_od(p, p) = 0.0;
    const Eigen::VectorXd eps = f_so.diagonal();

    // f_ov block (no x nv) for the energy.
    Mat f_ov(no, nv);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a) f_ov(i, a) = f_so(i, no + a);

    const std::size_t n2 = static_cast<std::size_t>(no) * no * nv * nv;
    Mat T1(no, nv);
    std::vector<double> T2(n2);
    CCSDResult result;
    result.e_hf = e_hf;

    // Denominators, residuals, and DIIS histories are iteration-only and must
    // be destroyed before the requested-budget triples phase begins.
    {
    // MP denominators D1(i,a) = eps_i - eps_a; D2 = eps_i+eps_j-eps_a-eps_b.
    Mat D1(no, nv);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a)
            D1(i, a) = eps(i) - eps(no + a);
    std::vector<double> D2(n2);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    D2[t2idx(i, j, a, b, no, nv)] =
                        eps(i) + eps(j) - eps(no + a) - eps(no + b);

    // Amplitudes; MP1 guess t1 = f_ov / D1, t2 = <ij||ab> / D2.
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a)
            T1(i, a) = f_ov(i, a) / D1(i, a);
    std::vector<double> R2(n2);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    const std::size_t idx = t2idx(i, j, a, b, no, nv);
                    T2[idx] = V(i, j, no + a, no + b) / D2[idx];
                }
    Mat R1(no, nv);

    UCCSD_DIIS diis(opts.diis_subspace_size);
    double e_prev = so_energy(V, f_ov, T1, T2, no, nv);

    for (int iter = 0; iter < opts.max_iter; ++iter) {
        so_residuals(V, f_so, f_od, T1, T2, D1, D2, no, nv, R1, R2);
        const double r1n = R1.norm();
        double r2n = 0.0;
        for (double x : R2) r2n += x * x;
        r2n = std::sqrt(r2n);
        if (!std::isfinite(r1n) || !std::isfinite(r2n)) break;

        // Jacobi step T += R / D, then DIIS over the post-step amplitudes.
        for (Eigen::Index i = 0; i < no; ++i)
            for (Eigen::Index a = 0; a < nv; ++a)
                T1(i, a) += R1(i, a) / D1(i, a);
        for (std::size_t k = 0; k < n2; ++k) T2[k] += R2[k] / D2[k];
        if (opts.diis_subspace_size > 0) diis.extrapolate(T1, T2, R1, R2);

        const double e_corr = so_energy(V, f_ov, T1, T2, no, nv);
        const double delta_e = e_corr - e_prev;
        result.cc_trace.push_back({iter + 1, result.e_hf + e_corr, delta_e, r1n,
                                   r2n, static_cast<int>(diis.subspace_size())});
        e_prev = e_corr;
        if (std::abs(delta_e) < opts.conv_tol_energy
            && (r1n + r2n) < opts.conv_tol_residual) {
            result.converged = true;
            result.n_iter = iter + 1;
            break;
        }
    }
    }

    const double e_corr = so_energy(V, f_ov, T1, T2, no, nv);
    result.e_ccsd_correlation = e_corr;
    result.e_ccsd = result.e_hf + e_corr;
    result.t1_norm = T1.norm();
    double t2n = 0.0;
    for (double x : T2) t2n += x * x;
    result.t2_norm = std::sqrt(t2n);
    if (!result.converged) result.n_iter = opts.max_iter;

    // These solve-only matrices are no longer inputs to either triples mode.
    f_so.resize(0, 0);
    f_od.resize(0, 0);
    f_ov.resize(0, 0);

    if (opts.compute_triples && result.converged) {
        // The spin-orbital open-shell kernel implements the standard
        // Raghavachari (T) only; the bracket [T] split lives in the
        // closed-shell kernel.
        if (opts.triples_variant != "(t)" && !opts.triples_variant.empty())
            throw std::invalid_argument(
                "run_uccsd: triples_variant='" + opts.triples_variant +
                "' is only available in the closed-shell kernel; the "
                "open-shell (T) is the standard Raghavachari correction");
        const UCCSDTriplesMemoryPlan plan = plan_uccsd_triples_memory(
            V, T1, T2, eps, no, nv, opts);
        result.triples_memory_mode_used = plan.name;
        result.triples_tile_size_used = plan.tile_size;
        result.triples_threads_used = plan.threads;
        result.triples_workspace_bytes = plan.peak_bytes;
        result.triples_disk_bytes = 0;
        result.e_t = plan.direct
            ? so_triples_direct(V, T1, T2, eps, no, nv, plan.threads)
            : so_triples(V, T1, T2, eps, no, nv, plan.threads);
    }

    result.e_ccsd_t = result.e_ccsd + result.e_t;
    result.e_total = result.e_ccsd_t;
    return result;
}
}  // namespace

// =========================================================================
// Public entry point: caller-supplied local spin-orbital UCCSD residual
// =========================================================================
void dlpno_uccsd_pair_residual(
    const Mat& T1, const Mat& T2_flat, const RowMat& B_so,
    const Mat& f_so, Mat& R1, Mat& R2_flat) {
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();
    if (no < 1 || nv < 1)
        throw std::invalid_argument(
            "dlpno_uccsd_pair_residual: T1 must have non-zero shape "
            "(n_occ, n_vir)");

    const Eigen::Index n = no + nv;
    if (T2_flat.rows() != no * no || T2_flat.cols() != nv * nv)
        throw std::invalid_argument(
            "dlpno_uccsd_pair_residual: T2_flat must have shape "
            "(n_occ*n_occ, n_vir*n_vir)");
    if (B_so.rows() < 1 || B_so.cols() != n * n)
        throw std::invalid_argument(
            "dlpno_uccsd_pair_residual: B_so must have shape "
            "(n_aux, (n_occ+n_vir)*(n_occ+n_vir)) with n_aux > 0");
    if (f_so.rows() != n || f_so.cols() != n)
        throw std::invalid_argument(
            "dlpno_uccsd_pair_residual: f_so must have shape "
            "(n_occ+n_vir, n_occ+n_vir)");

    // The caller supplies one bounded local spin-orbital domain. Reuse the
    // validated dense spin-orbital integral and residual kernels inside that
    // domain; no global-system tensor is formed here.
    const SpinOrbitalIntegrals V = build_so_integrals(B_so, n);
    Mat f_od = f_so;
    for (Eigen::Index p = 0; p < n; ++p) f_od(p, p) = 0.0;

    const Eigen::VectorXd eps = f_so.diagonal();
    Mat D1(no, nv);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a)
            D1(i, a) = eps(i) - eps(no + a);

    const std::size_t n2 = static_cast<std::size_t>(no) * no * nv * nv;
    std::vector<double> T2(n2), D2(n2), R2(n2);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    const std::size_t idx = t2idx(i, j, a, b, no, nv);
                    // Explicit indexing is required: Eigen::MatrixXd storage
                    // is column-major, while t2idx is the spin-orbital
                    // row-major convention shared with the Python oracle.
                    T2[idx] = T2_flat(i * no + j, a * nv + b);
                    D2[idx] = eps(i) + eps(j) - eps(no + a) - eps(no + b);
                }

    R1 = Mat::Zero(no, nv);
    so_residuals(V, f_so, f_od, T1, T2, D1, D2, no, nv, R1, R2);

    R2_flat = Mat::Zero(no * no, nv * nv);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    R2_flat(i * no + j, a * nv + b) =
                        R2[t2idx(i, j, a, b, no, nv)];
}

// =========================================================================
// Public entry point: caller-supplied local spin-orbital (T) triple
// =========================================================================
double dlpno_spin_orbital_triple_energy(
    Eigen::Index i, Eigen::Index j, Eigen::Index k,
    const Mat& T1, const Mat& T2_flat,
    const RowMat& eri_vovv, const RowMat& eri_ovoo,
    const RowMat& eri_oovv, const Eigen::VectorXd& eps_o,
    const Eigen::VectorXd& eps_v) {
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();
    if (no < 3 || nv < 1)
        throw std::invalid_argument(
            "dlpno_spin_orbital_triple_energy: T1 must have shape "
            "(n_occ, n_vir) with n_occ >= 3 and n_vir >= 1");
    if (i < 0 || i >= no || j <= i || j >= no || k <= j || k >= no)
        throw std::invalid_argument(
            "dlpno_spin_orbital_triple_energy: occupied indices must satisfy "
            "0 <= i < j < k < n_occ");
    if (T2_flat.rows() != no * no || T2_flat.cols() != nv * nv)
        throw std::invalid_argument(
            "dlpno_spin_orbital_triple_energy: T2_flat must have shape "
            "(n_occ*n_occ, n_vir*n_vir)");
    if (eri_vovv.rows() != nv * no || eri_vovv.cols() != nv * nv)
        throw std::invalid_argument(
            "dlpno_spin_orbital_triple_energy: eri_vovv must have shape "
            "(n_vir*n_occ, n_vir*n_vir)");
    if (eri_ovoo.rows() != no * nv || eri_ovoo.cols() != no * no)
        throw std::invalid_argument(
            "dlpno_spin_orbital_triple_energy: eri_ovoo must have shape "
            "(n_occ*n_vir, n_occ*n_occ)");
    if (eri_oovv.rows() != no * no || eri_oovv.cols() != nv * nv)
        throw std::invalid_argument(
            "dlpno_spin_orbital_triple_energy: eri_oovv must have shape "
            "(n_occ*n_occ, n_vir*n_vir)");
    if (eps_o.size() != no || eps_v.size() != nv)
        throw std::invalid_argument(
            "dlpno_spin_orbital_triple_energy: orbital-energy vector sizes "
            "must match n_occ and n_vir");

    const auto nv3 = static_cast<std::size_t>(nv) * nv * nv;
    std::vector<double> wc(nv3), wd(nv3), work(nv3);
    const auto index3 = [nv](Eigen::Index a, Eigen::Index b,
                             Eigen::Index c) {
        return (static_cast<std::size_t>(a) * nv + b) * nv + c;
    };
    const auto t2 = [&](Eigen::Index p, Eigen::Index q,
                        Eigen::Index a, Eigen::Index b) {
        return T2_flat(p * no + q, a * nv + b);
    };

    const auto build_x = [&](Eigen::Index p, Eigen::Index q,
                             Eigen::Index r) {
        for (Eigen::Index a = 0; a < nv; ++a)
            for (Eigen::Index b = 0; b < nv; ++b)
                for (Eigen::Index c = 0; c < nv; ++c) {
                    double value = 0.0;
                    for (Eigen::Index e = 0; e < nv; ++e)
                        value += t2(q, r, a, e)
                               * eri_vovv(e * no + p, b * nv + c);
                    for (Eigen::Index m = 0; m < no; ++m)
                        value -= t2(p, m, b, c)
                               * eri_ovoo(m * nv + a, q * no + r);
                    work[index3(a, b, c)] = value;
                }
    };

    build_x(i, j, k);
    wc = work;
    build_x(j, i, k);
    for (std::size_t pos = 0; pos < nv3; ++pos) wc[pos] -= work[pos];
    build_x(k, j, i);
    for (std::size_t pos = 0; pos < nv3; ++pos) wc[pos] -= work[pos];
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index b = 0; b < nv; ++b)
            for (Eigen::Index c = 0; c < nv; ++c)
                work[index3(a, b, c)] =
                    wc[index3(a, b, c)] - wc[index3(b, a, c)]
                    - wc[index3(c, b, a)];
    std::swap(wc, work);

    const auto add_y = [&](Eigen::Index p, Eigen::Index q,
                           Eigen::Index r, double sign) {
        for (Eigen::Index a = 0; a < nv; ++a)
            for (Eigen::Index b = 0; b < nv; ++b)
                for (Eigen::Index c = 0; c < nv; ++c)
                    wd[index3(a, b, c)] +=
                        sign * T1(p, a)
                        * eri_oovv(q * no + r, b * nv + c);
    };
    add_y(i, j, k, +1.0);
    add_y(j, i, k, -1.0);
    add_y(k, j, i, -1.0);
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index b = 0; b < nv; ++b)
            for (Eigen::Index c = 0; c < nv; ++c)
                work[index3(a, b, c)] =
                    wd[index3(a, b, c)] - wd[index3(b, a, c)]
                    - wd[index3(c, b, a)];
    std::swap(wd, work);

    const double d_occ = eps_o(i) + eps_o(j) + eps_o(k);
    double energy = 0.0;
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index b = 0; b < nv; ++b)
            for (Eigen::Index c = 0; c < nv; ++c) {
                const auto abc = index3(a, b, c);
                const double denominator =
                    d_occ - eps_v(a) - eps_v(b) - eps_v(c);
                energy += wc[abc] * (wc[abc] + wd[abc]) / denominator;
            }
    return energy / 6.0;
}

// =========================================================================
// Public entry point: run_uccsd (via UHFResult)
// =========================================================================
CCSDResult run_uccsd(const Molecule& mol, const BasisSet& basis,
                     const UHFResult& uhf, const CCSDOptions& opts) {
    if (!uhf.converged)
        throw std::runtime_error("run_uccsd: UHF reference is not converged");
    if (opts.density_fit && opts.aux_basis.empty())
        throw std::invalid_argument(
            "run_uccsd: density_fit requires aux_basis");
    return run_uccsd_impl(mol, basis,
                          uhf.mo_coeffs_alpha, uhf.mo_coeffs_beta,
                          uhf.fock_alpha, uhf.fock_beta,
                          uhf.energy, uhf.ecp_total_ncore, opts);
}

// =========================================================================
// Public entry point: run_uccsd_from_mos (explicit arrays)
// =========================================================================
CCSDResult run_uccsd_from_mos(const Molecule& mol, const BasisSet& basis,
                              const Eigen::MatrixXd& C_alpha,
                              const Eigen::MatrixXd& C_beta,
                              const Eigen::MatrixXd& F_alpha,
                              const Eigen::MatrixXd& F_beta, double e_hf,
                              int ecp_total_ncore,
                              const CCSDOptions& opts) {
    (void)effective_electron_count(mol, ecp_total_ncore, "run_uccsd_from_mos");
    if (opts.density_fit && opts.aux_basis.empty())
        throw std::invalid_argument(
            "run_uccsd_from_mos: density_fit requires aux_basis");
    if (C_alpha.rows() == 0 || C_alpha.cols() == 0
        || C_beta.rows() == 0 || C_beta.cols() == 0)
        throw std::invalid_argument(
            "run_uccsd_from_mos: MO coefficient matrices must be non-empty");
    if (C_alpha.rows() != C_beta.rows() || C_alpha.cols() != C_beta.cols())
        throw std::invalid_argument(
            "run_uccsd_from_mos: alpha and beta MO coefficient matrices must "
            "have the same dimensions");

    return run_uccsd_impl(mol, basis, C_alpha, C_beta, F_alpha, F_beta, e_hf,
                          ecp_total_ncore,
                          opts);
}

}  // namespace vibeqc
