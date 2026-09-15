// Per-shell-pair moment shift to adjoined-Gaussian product centres.
//
// Shifts global-origin Cartesian multipole moments to per-pair
// product centres using the standard polynomial-shift formula. The periodic
// expansion source, Pisani-Dovesi-Roetti (1988), Ch. II.4c, calls for
// product-distribution centroids. Pisani-Dovesi (1980), Sec. 4, supplies the
// adjoined diffuse s-Gaussian convention, and the standard Gaussian product
// theorem supplies its approximate centre.
//
// Processes ALL shell pairs (both triangles) identically to the Python
// reference, avoiding the complexity of upper-triangle + transpose.
// OpenMP-parallelised over cells.

#include "vibeqc/bipole_pair_moments.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace vibeqc {

namespace {

// Minimum primitive exponent per shell.
std::vector<double> shell_min_exponents(const BasisSet& basis) {
    std::vector<double> result;
    for (const auto& sh : basis.libint()) {
        double m = 1e300;
        for (std::size_t p = 0; p < sh.nprim(); ++p)
            m = std::min(m, static_cast<double>(sh.alpha[p]));
        result.push_back(m);
    }
    return result;
}

// Shell AO index ranges.
std::vector<std::pair<int, int>> build_shell_slices(const BasisSet& basis) {
    const auto& s2bf = basis.libint().shell2bf();
    std::vector<std::pair<int, int>> r;
    for (std::size_t s = 0; s < basis.libint().size(); ++s)
        r.emplace_back(static_cast<int>(s2bf[s]),
                       static_cast<int>(basis.libint()[s].size()));
    return r;
}

int n_cart(int L) {
    if (L == 2) return 10;
    if (L == 3) return 20;
    if (L == 4) return 35;
    throw std::invalid_argument("L_target must be 2, 3, or 4");
}

// Cartesian component indices — same order as libint and Python _COMP_* dicts.
// Overlap S = 0
// Dipole:  d[x]=1, d[y]=2, d[z]=3
// Quadrupole: xx=4, xy=5, xz=6, yy=7, yz=8, zz=9
// Octupole:  xxx=10, xxy=11, xxz=12, xyy=13, xyz=14, xzz=15,
//            yyy=16, yyz=17, yzz=18, zzz=19
// Hexadecapole: xxxx=20, xxxy=21, xxxz=22, xxyy=23, xxyz=24, xxzz=25,
//               xyyy=26, xyyz=27, xyzz=28, xzzz=29,
//               yyyy=30, yyyz=31, yyzz=32, yzzz=33, zzzz=34

// Map sorted axis tuple (a,b) with a<=b to quadrupole component index.
int quad_comp(int a, int b) {
    static const int m[3][3] = {{4,5,6},{5,7,8},{6,8,9}};
    return m[a][b];
}

// Map sorted axis tuple (a,b,c) with a<=b<=c to octupole component.
int oct_comp(int a, int b, int c) {
    // Returns the libint octupole index for the monomial with the given
    // axis power counts.  The tuple IS the power counts in lex order.
    if (a==0 && b==0 && c==0) return 10;
    if (a==0 && b==0 && c==1) return 11;
    if (a==0 && b==0 && c==2) return 12;
    if (a==0 && b==1 && c==1) return 13;
    if (a==0 && b==1 && c==2) return 14;
    if (a==0 && b==2 && c==2) return 15;
    if (a==1 && b==1 && c==1) return 16;
    if (a==1 && b==1 && c==2) return 17;
    if (a==1 && b==2 && c==2) return 18;
    return 19;  // (2,2,2)
}

// Map sorted axis tuple (a,b,c,d) with a<=b<=c<=d to hexadecapole component.
int hex_comp(int a, int b, int c, int d) {
    if (a==0 && b==0 && c==0 && d==0) return 20;
    if (a==0 && b==0 && c==0 && d==1) return 21;
    if (a==0 && b==0 && c==0 && d==2) return 22;
    if (a==0 && b==0 && c==1 && d==1) return 23;
    if (a==0 && b==0 && c==1 && d==2) return 24;
    if (a==0 && b==0 && c==2 && d==2) return 25;
    if (a==0 && b==1 && c==1 && d==1) return 26;
    if (a==0 && b==1 && c==1 && d==2) return 27;
    if (a==0 && b==1 && c==2 && d==2) return 28;
    if (a==0 && b==2 && c==2 && d==2) return 29;
    if (a==1 && b==1 && c==1 && d==1) return 30;
    if (a==1 && b==1 && c==1 && d==2) return 31;
    if (a==1 && b==1 && c==2 && d==2) return 32;
    if (a==1 && b==2 && c==2 && d==2) return 33;
    return 34;  // (2,2,2,2)
}

}  // namespace

// ---------------------------------------------------------------------------
// Adjoined-Gaussian product centres
// ---------------------------------------------------------------------------

std::vector<std::vector<std::array<double, 3>>> compute_adjoined_pair_centres(
    const BasisSet& basis,
    const std::vector<LatticeCell>& cells) {

    const auto& shells = basis.libint();
    const int n_sh = static_cast<int>(shells.size());
    const auto a_min = shell_min_exponents(basis);

    std::vector<std::array<double, 3>> origins(n_sh);
    for (int s = 0; s < n_sh; ++s)
        origins[s] = {shells[s].O[0], shells[s].O[1], shells[s].O[2]};

    const int n_cells = static_cast<int>(cells.size());
    std::vector<std::vector<std::array<double, 3>>> result(n_cells);

    for (int c = 0; c < n_cells; ++c) {
        result[c].resize(n_sh * n_sh);
        double gx = cells[c].r_cart[0];
        double gy = cells[c].r_cart[1];
        double gz = cells[c].r_cart[2];
        for (int s1 = 0; s1 < n_sh; ++s1) {
            double a1 = a_min[s1];
            for (int s2 = 0; s2 < n_sh; ++s2) {
                double a2 = a_min[s2];
                double denom = a1 + a2;
                double w1 = a1 / denom;
                double w2 = a2 / denom;
                result[c][s1 * n_sh + s2] = {
                    w1 * origins[s1][0] + w2 * (origins[s2][0] + gx),
                    w1 * origins[s1][1] + w2 * (origins[s2][1] + gy),
                    w1 * origins[s1][2] + w2 * (origins[s2][2] + gz)
                };
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Standard polynomial shift; kept identical to the Python implementation
// ---------------------------------------------------------------------------

PairMultipoleMomentsCpp shift_multipole_moments_to_pair_centres(
    const LatticeMultipoleSet& M_lat,
    const BasisSet& basis,
    int L_target,
    const std::array<double, 3>& origin) {

    if (M_lat.spherical)
        throw std::runtime_error("shift_multipole_moments_to_pair_centres: "
                                 "input must be Cartesian (spherical=false)");
    if (M_lat.L_max < L_target)
        throw std::runtime_error("shift_multipole_moments_to_pair_centres: "
                                 "input L_max < L_target");

    const int nbf = M_lat.nbf;
    const int n_cells = static_cast<int>(M_lat.cells.size());
    const int n_comp = n_cart(L_target);
    const int n_sh = static_cast<int>(basis.libint().size());

    const auto slices = build_shell_slices(basis);
    const auto all_centres = compute_adjoined_pair_centres(basis, M_lat.cells);

    PairMultipoleMomentsCpp result;
    result.nbf = nbf;
    result.L_max = L_target;
    result.cells = M_lat.cells;
    result.blocks.resize(n_cells);
    result.shell_slices = slices;
    result.centres_flat = all_centres;

    // Number of available components per source cell.
    std::vector<int> src_ncomp(n_cells);
    for (int c = 0; c < n_cells; ++c)
        src_ncomp[c] = static_cast<int>(M_lat.blocks[c].size());

    // Allocate output: start as copy of source, zero-pad if needed.
    for (int c = 0; c < n_cells; ++c) {
        result.blocks[c].resize(n_comp);
        for (int comp = 0; comp < n_comp; ++comp) {
            if (comp < src_ncomp[c])
                result.blocks[c][comp] = M_lat.blocks[c][comp];
            else
                result.blocks[c][comp] = Eigen::MatrixXd::Zero(nbf, nbf);
        }
    }

    const double ox = origin[0], oy = origin[1], oz = origin[2];

    // Loop over all cells (OpenMP-parallel) and ALL shell pairs.
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        for (int s1 = 0; s1 < n_sh; ++s1) {
            int b1 = slices[s1].first, n1 = slices[s1].second;
            for (int s2 = 0; s2 < n_sh; ++s2) {
                int b2 = slices[s2].first, n2 = slices[s2].second;

                // Shift vector D = centre - origin.
                const auto& ctr = all_centres[c][s1 * n_sh + s2];
                double Dx = ctr[0] - ox, Dy = ctr[1] - oy, Dz = ctr[2] - oz;

                // Read source moment sub-blocks (unshifted, origin moments).
                auto read_src = [&](int comp) -> Eigen::MatrixXd {
                    if (comp >= src_ncomp[c])
                        return Eigen::MatrixXd::Zero(n1, n2);
                    return M_lat.blocks[c][comp].block(b1, b2, n1, n2);
                };

                // Overlap.
                Eigen::MatrixXd S_src = read_src(0);
                // Dipoles (unshifted).
                Eigen::MatrixXd d_src[3] = {read_src(1), read_src(2), read_src(3)};
                // Quadrupoles (unshifted).
                Eigen::MatrixXd Q_src[3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = a; b < 3; ++b)
                        Q_src[a][b] = read_src(quad_comp(a, b));

                // Shifted dipoles.
                double Dv[3] = {Dx, Dy, Dz};
                Eigen::MatrixXd ds[3];
                for (int a = 0; a < 3; ++a)
                    ds[a] = d_src[a] - Dv[a] * S_src;

                // ----- Write shifted overlap and dipoles -----
                result.blocks[c][0].block(b1, b2, n1, n2) = S_src;
                for (int a = 0; a < 3; ++a)
                    result.blocks[c][1 + a].block(b1, b2, n1, n2) = ds[a];

                if (L_target < 2) continue;

                // ----- Shifted quadrupoles -----
                Eigen::MatrixXd Q_dst[3][3];
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        Q_dst[a][b] = Q_src[a][b]
                            - Dv[a] * d_src[b] - Dv[b] * d_src[a]
                            + Dv[a] * Dv[b] * S_src;
                        result.blocks[c][quad_comp(a,b)].block(b1,b2,n1,n2)
                            = Q_dst[a][b];
                    }
                }

                if (L_target < 3) continue;

                // ----- Shifted octupoles -----
                // The shift formulas expand the multi-index binomial theorem
                //   M'_p = sum_{q <= p} binom(p, q) (-D)^(p-q) M_q,
                // so every lower-order moment on the right-hand side is the
                // UNSHIFTED (src) moment about the original origin. Feeding
                // already-shifted lower moments into the same symmetric slot
                // pattern double-counts the translation.
                // Octupole source blocks.
                Eigen::MatrixXd O_src[3][3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = a; b < 3; ++b)
                        for (int ci = b; ci < 3; ++ci)
                            O_src[a][b][ci] = read_src(oct_comp(a,b,ci));

                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        for (int ci = b; ci < 3; ++ci) {
                            Eigen::MatrixXd O_dst = O_src[a][b][ci]
                                - Dv[a] * Q_src[b][ci]
                                - Dv[b] * Q_src[a][ci]
                                - Dv[ci] * Q_src[a][b]
                                + Dv[a]*Dv[b] * d_src[ci]
                                + Dv[a]*Dv[ci] * d_src[b]
                                + Dv[b]*Dv[ci] * d_src[a]
                                - Dv[a]*Dv[b]*Dv[ci] * S_src;
                            result.blocks[c][oct_comp(a,b,ci)]
                                .block(b1,b2,n1,n2) = O_dst;
                        }
                    }
                }

                if (L_target < 4) continue;

                // ----- Shifted hexadecapoles -----
                // Hexadecapole source blocks.
                Eigen::MatrixXd H_src[3][3][3][3];
                for (int a = 0; a < 3; ++a)
                    for (int b = a; b < 3; ++b)
                        for (int ci = b; ci < 3; ++ci)
                            for (int d = ci; d < 3; ++d)
                                H_src[a][b][ci][d] = read_src(hex_comp(a,b,ci,d));

                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        for (int ci = b; ci < 3; ++ci) {
                            for (int d = ci; d < 3; ++d) {
                                Eigen::MatrixXd H_dst = H_src[a][b][ci][d]
                                    - Dv[a] * O_src[b][ci][d]
                                    - Dv[b] * O_src[a][ci][d]
                                    - Dv[ci] * O_src[a][b][d]
                                    - Dv[d] * O_src[a][b][ci]
                                    + Dv[a]*Dv[b] * Q_src[ci][d]
                                    + Dv[a]*Dv[ci] * Q_src[b][d]
                                    + Dv[a]*Dv[d] * Q_src[b][ci]
                                    + Dv[b]*Dv[ci] * Q_src[a][d]
                                    + Dv[b]*Dv[d] * Q_src[a][ci]
                                    + Dv[ci]*Dv[d] * Q_src[a][b]
                                    - Dv[a]*Dv[b]*Dv[ci] * d_src[d]
                                    - Dv[a]*Dv[b]*Dv[d] * d_src[ci]
                                    - Dv[a]*Dv[ci]*Dv[d] * d_src[b]
                                    - Dv[b]*Dv[ci]*Dv[d] * d_src[a]
                                    + Dv[a]*Dv[b]*Dv[ci]*Dv[d] * S_src;
                                result.blocks[c][hex_comp(a,b,ci,d)]
                                    .block(b1,b2,n1,n2) = H_dst;
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

}  // namespace vibeqc
