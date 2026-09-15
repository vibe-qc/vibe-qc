// MSINDO STO integral kernel — faithful C++ port of the general Slater machinery.
//
// Re-implements MSINDO's analytic Slater-type-orbital integrals so the INDO
// engine generalizes to s/p/d without libint (which is Gaussian-only).  Each
// function is a faithful translation of the named MSINDO routine, mirroring the
// validated reference implementation in
// python/vibeqc/semiempirical/methods/msindo_integrals.py (oracle-exact).
//
// Header-only (all inline, function-local static tables) so the engine and the
// pybind layer can include it without a separate translation unit.  Orbital
// order: 1=s, 2..4 = px,py,pz, 5..9 = d (HARMTR column convention).
//
// References: Mulliken-Rieke-Orloff-Orloff; Harris, J. Chem. Phys. 51, 4777
// (1969); Kuppermann-Karplus-Isaacson, Z. Naturforsch. 14A, 311 (1959).
// © Mulliken Center for Theoretical Chemistry, University of Bonn (method);
// independent vibe-qc re-implementation (no MSINDO source copied).

#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace vibeqc {
namespace semiempirical {
namespace indo {

// MSINDO derives this value from its 1986-CODATA ``const.f`` literals with
// default REAL*8. Geometry is part of the parametrized reference convention.
constexpr double MSINDO_BOHR_ANGSTROM = 0.5291772575069162;
constexpr double ANGSTROM_TO_BOHR = 1.0 / MSINDO_BOHR_ANGSTROM;

// --------------------------------------------------------------------------- //
// Factorials / binomials (const.f).
// --------------------------------------------------------------------------- //
inline double facs(int n) {
    static const std::array<double, 40> F = [] {
        std::array<double, 40> f{};
        f[0] = 1.0;
        for (int i = 1; i < 40; ++i) f[i] = f[i - 1] * i;
        return f;
    }();
    return F[n];
}

inline double binom(int n, int k) {
    static const std::array<std::array<double, 40>, 40> B = [] {
        std::array<std::array<double, 40>, 40> b{};
        for (int n = 0; n < 40; ++n) {
            b[n][0] = 1.0;
            for (int k = 1; k <= n; ++k)
                b[n][k] = b[n - 1][k - 1] + b[n - 1][k];
        }
        return b;
    }();
    if (k < 0 || k > n) return 0.0;
    return B[n][k];
}

// XSAVE(l,m) = 2^-l sqrt((2l+1) BIN(l+m,l)/(2 BIN(l,m))) (const.f).
inline double xsave(int l, int m) {
    static const std::array<std::array<double, 3>, 9> X = [] {
        std::array<std::array<double, 3>, 9> x{};
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 9; ++j)
                x[j][i] = std::pow(2.0, -j) *
                          std::sqrt((2 * j + 1) * binom(j + i, j) /
                                    (2.0 * binom(j, i)));
        return x;
    }();
    return X[l][m];
}

// --------------------------------------------------------------------------- //
// Wigner 3j (L1=L2=L) and Coulomb angular A0 = F_A (wig.f / f_a.f / const.f).
// --------------------------------------------------------------------------- //
inline double wig(int L, int J, int m1, int m2) {
    int M = m1 + m2;
    if (J < std::abs(M)) return 0.0;
    int ku = std::max(std::max(0, L - J - m1), L - J + m2);
    int ko = std::min(std::min(2 * L - J, L - m1), L + m2);
    double s = 0.0;
    for (int k = ku; k <= ko; ++k) {
        double denom = facs(k) * facs(2 * L - J - k) * facs(L - m1 - k) *
                       facs(L + m2 - k) * facs(J - L + m1 + k) *
                       facs(J - L - m2 + k);
        s += ((k % 2 == 0) ? 1.0 : -1.0) / denom;
    }
    double pref = std::sqrt(
        facs(2 * L - J) * facs(J) * facs(J) * facs(L + m1) * facs(L - m1) *
        facs(L + m2) * facs(L - m2) * facs(J + M) * facs(J - M) /
        facs(2 * L + J + 1));
    return ((M % 2 == 0) ? 1.0 : -1.0) * pref * s;
}

inline double f_a(int L, int J, int m1, int m2) {
    double v = std::sqrt((2 * L + 1) * (2 * L + 1) * (2 * J + 1) / 4.0 / M_PI) *
               wig(L, J, m1, m2) * wig(L, J, 0, 0);
    if (m1 > 0) v *= (m1 % 2 == 0) ? 1.0 : -1.0;
    if (m2 > 0) v *= (m2 % 2 == 0) ? 1.0 : -1.0;
    return v;
}

// A0[L][J][M] = F_A(L,J,M,-M), L,M=0..2, J=0..2L (const.f).
inline double a0(int L, int J, int M) {
    static const std::array<std::array<std::array<double, 3>, 5>, 3> A = [] {
        std::array<std::array<std::array<double, 3>, 5>, 3> a{};
        for (int L = 0; L < 3; ++L)
            for (int J = 0; J <= 2 * L; ++J)
                for (int Mm = 0; Mm <= L; ++Mm)
                    a[L][J][Mm] = f_a(L, J, Mm, -Mm);
        return a;
    }();
    if (L < 0 || L > 2 || J < 0 || J > 4 || M < 0 || M > 2) return 0.0;
    return A[L][J][M];
}

// --------------------------------------------------------------------------- //
// BISSIG -> BSSIG angular overlap coefficients (bissig.f).
// --------------------------------------------------------------------------- //
inline std::vector<double> bissig(int n1, int n2, int l1, int l2, int m) {
    int sigu = n1 + n2 - l1 - l2;
    int sigo = n1 + n2 + l1 + l2;
    int i1o = (l1 - m) / 2;
    int i2o = (l2 - m) / 2;
    std::vector<double> out;
    for (int sig = sigu; sig <= sigo; sig += 2) {
        int su = std::max(0, sig - n1 - n2);
        int so = su + (std::min(sig, n1 + n2) - su) / 2;
        for (int s = su; s <= so; ++s) {
            double val = 0.0;
            for (int tau = 0; tau <= m; ++tau)
                for (int lam = 0; lam <= m; ++lam)
                    for (int i1 = 0; i1 <= i1o; ++i1)
                        for (int i2 = 0; i2 <= i2o; ++i2) {
                            int ko = l1 - m - 2 * i1;
                            for (int k = 0; k <= ko; ++k) {
                                int yu = (sig - n1 - n2 + l1 + l2) / 2 - tau -
                                         lam - i1 - i2 - k;
                                int yo = l2 - m - 2 * i2;
                                if (yu < 0 || yu > yo) continue;
                                int xo = n2 - l2 + 2 * i2;
                                int po = n1 - l1 + 2 * i1 + k;
                                int x = -(sig - n1 - n2 + l1 + l2) / 2 + s -
                                        tau + lam + i1 + i2 + k;
                                for (int p = k; p <= po; ++p) {
                                    int xu = x - p;
                                    if (xu < 0 || xu > xo) continue;
                                    int sgn = (tau + lam + i1 + i2 + p) % 2;
                                    val += ((sgn == 0) ? 1.0 : -1.0) *
                                           binom(m, tau) * binom(m, lam) *
                                           binom(l1, i1) * binom(l2, i2) *
                                           binom(2 * (l1 - i1), l1 + m) *
                                           binom(2 * (l2 - i2), l2 + m) *
                                           binom(n1 - l1 + 2 * i1, p - k) *
                                           binom(l1 - m - 2 * i1, k) *
                                           binom(xo, xu) * binom(yo, yu);
                                }
                            }
                        }
            out.push_back(val * ((n1 % 2 == 0) ? 1.0 : -1.0));
        }
    }
    return out;
}

inline double calsum(const std::vector<double>& bss, const std::vector<double>& a_int,
                     const std::vector<double>& b_int, int n1, int n2, int l1, int l2) {
    int sigu = n1 - l1 + n2 - l2;
    int sigo = n1 + l1 + n2 + l2;
    double vz = ((n1 + l1) % 2 == 0) ? 1.0 : -1.0;
    double total = 0.0;
    int z = 0;
    for (int sig = sigu; sig <= sigo; sig += 2) {
        int su = std::max(0, sig - n1 - n2);
        int so = su + (std::min(sig, n1 + n2) - su) / 2;
        for (int s = su; s <= so; ++s) {
            if (sigu % 2 == 0 && sig == 2 * s) {
                total += bss[z] * a_int[s] * b_int[s];
            } else {
                total += bss[z] * (a_int[s] * b_int[sig - s] +
                                   vz * a_int[sig - s] * b_int[s]);
            }
            ++z;
        }
    }
    return total;
}

// --------------------------------------------------------------------------- //
// A_k / B_k auxiliary integrals (aintgs.f / bintgs.f).
// --------------------------------------------------------------------------- //
inline std::vector<double> aintgs(double x, int kmax) {
    std::vector<double> a(kmax + 1, 0.0);
    a[0] = 1.0 / x;
    for (int k = 1; k <= kmax; ++k) a[k] = (a[k - 1] * k + 1.0) * (1.0 / x);
    return a;
}

inline std::vector<double> bintgs(double x, double y, int kmax) {
    std::vector<double> b(kmax + 1, 0.0);
    double absy = std::abs(y);
    double expmx = std::exp(-x);
    if (absy < 1e-10) {
        for (int k = 0; k <= kmax; ++k)
            b[k] = ((k % 2 == 0) ? 2.0 / (k + 1.0) : 0.0) * expmx;
    } else if (absy < 1.5) {
        double y2 = y * y;
        for (int k = 0; k <= kmax; k += 2) {
            double tot = 0.0, yy = 1.0;
            for (int nu = 0; nu < 18; nu += 2) {
                double term = (2.0 / (facs(nu) * (k + nu + 1))) * yy;
                tot += term;
                if (std::abs(term) < 1e-10) break;
                yy *= y2;
            }
            b[k] = expmx * tot;
        }
        for (int k = 1; k <= kmax; k += 2) {
            double tot = 0.0, yy = y;
            for (int nu = 1; nu < 18; nu += 2) {
                double term = (2.0 / (facs(nu) * (k + nu + 1))) * yy;
                tot -= term;
                if (std::abs(term) < 1e-10) break;
                yy *= y2;
            }
            b[k] = expmx * tot;
        }
    } else {
        double fak = 1.0 / y;
        double expy = std::pow(std::exp((y - x) / 2.0), 2);
        double expmy = std::pow(std::exp((-y - x) / 2.0), 2);
        b[0] = (expy - expmy) * fak;
        for (int k = 1; k <= kmax; ++k) {
            double sgn = (k % 2 == 0) ? 1.0 : -1.0;
            b[k] = (k * b[k - 1] + sgn * expy - expmy) * fak;
        }
    }
    return b;
}

// --------------------------------------------------------------------------- //
// OVERLP — generalized reduced overlap (overlp.f).  S2INT adds normalization.
// --------------------------------------------------------------------------- //
inline double overlp(int n2, int l2, double z2, int n1, int l1, double z1,
                     int m, double R) {
    int mb = std::abs(m);
    if (l2 < 0 || l1 < 0 || l2 < mb || l1 < mb) return 0.0;
    int kmax = n1 + n2 + 1;
    double rm12 = 0.5 * R;
    double alpha = rm12 * (z1 + z2);
    double beta = rm12 * (z2 - z1);
    std::vector<double> a_int = aintgs(alpha, kmax);
    std::vector<double> b_int = bintgs(alpha, beta, kmax);
    std::vector<double> bss = bissig(n1, n2, l1, l2, mb);
    double s = calsum(bss, a_int, b_int, n1, n2, l1, l2);
    return std::pow(rm12, kmax) * xsave(l1, mb) * xsave(l2, mb) * s;
}

inline double s2int(int n1, int l1, int m1, double z1, int n2, int l2, int m2,
                    double z2, double R) {
    if (m1 != m2) return 0.0;
    double fak = std::sqrt(
        std::pow(z1 + z1, n1 + n1 + 1) * std::pow(z2 + z2, n2 + n2 + 1) /
        (facs(n1 + n1) * facs(n2 + n2)));
    return fak * overlp(n1, l1, z1, n2, l2, z2, m1, R);
}

// --------------------------------------------------------------------------- //
// Nuclear V2INT (v2int.f), Coulomb C2INT/HARRIS (c2int.f/harris.f),
// Slater-Condon RADINT/SK0 (radint.f/sk0.f).
// --------------------------------------------------------------------------- //
inline double v2int(int n1, int l1, int m1, double z1, double R) {
    int m = std::abs(m1);
    int nh = 2 * n1 - 1;
    double zeth = 2 * z1;
    double fak = std::pow(zeth, nh + 2) / facs(2 * n1) * 2.0 * std::sqrt(M_PI);
    double s = 0.0;
    for (int i = 0; i <= 2 * l1; i += 2)
        s += a0(l1, i, m) * overlp(nh, i, zeth, 0, 0, 0.0, 0, R);
    return fak * s;
}

inline double harris(int n1, int l1, double z1, int n2, int l2, double z2,
                     int m, double R) {
    int l1l2m = l1 + l2 - m;
    double vf1 = 4.0 * M_PI / (z1 * z1);
    double vf2 = 4.0 * M_PI / (z2 * z2);
    double form17 = vf1 * (overlp(0, 0, 0.0, l1l2m, l1l2m, z2, 0, R) -
                           overlp(0, 0, z1, l1l2m, l1l2m, z2, 0, R));
    double summ = 0.0;
    for (int j = 0; j < m; ++j)
        summ += std::pow(-1.0 / z1 / z1 / R, m - j) / (2 * j + 2) *
                std::sqrt(facs(2 * m + 1) * facs(l1l2m + m) * facs(l1l2m - j) /
                          facs(2 * j + 1) / facs(l1l2m - m) / facs(l1l2m + j)) *
                overlp(j + 1, j, z1, l1l2m, l1l2m, z2, j, R);
    double form30 = std::pow(-1.0 / z1 / z1 / R, m) *
                        std::sqrt(facs(2 * m + 1) * facs(l1l2m + m) /
                                  facs(l1l2m - m)) * form17 -
                    vf1 * z1 * summ;
    double form26 = form30;
    for (int j = 1; j <= l1 - m; ++j) {
        int L = m + j - 1;
        int ls = l1l2m - j + 1;
        double w1 = std::sqrt((double)(L - m + 1) * (L + m + 1) /
                              ((2 * L + 1) * (2 * L + 3)));
        double w2 = std::sqrt((double)(ls - m) * (ls + m) /
                              ((2 * ls - 1) * (2 * ls + 1)));
        double fak1 = z1 / z2 * 2 * ls * w1;
        double fak2 = z2 / z1 * (2 * L + 2) * w2;
        summ = vf1 * std::sqrt((double)(L + m) * (L - m) /
                               ((2 * L - 1) * (2 * L + 1))) *
                   (z1 * overlp(L + 1, L - 1, z1, ls, ls - 1, z2, m, R) +
                    overlp(L, L - 1, z1, ls, ls - 1, z2, m, R)) +
               vf2 * std::sqrt((double)(ls + m - 1) * (ls - m - 1) /
                               ((2 * ls - 3) * (2 * ls - 1))) *
                   (z2 * overlp(L + 1, L, z1, ls, ls - 2, z2, m, R) +
                    overlp(L + 1, L, z1, ls - 1, ls - 2, z2, m, R)) +
               vf2 * z1 * w1 * overlp(L + 1, L + 1, z1, ls, ls - 1, z2, m, R) +
               vf1 * z2 * w2 * overlp(L + 1, L, z1, ls, ls, z2, m, R);
        form26 = (summ - fak2 * form26) / fak1;
    }
    double vf1b = facs(n1 + l1 + 1) / facs(2 * l1 + 1);
    double summ1 = 0.0, summ2 = 0.0;
    for (int j = n2; j > l2; --j)
        summ1 += std::pow(z2, j - n2 - 2) *
                 (facs(n2 + l2 + 1) / facs(j + l2) -
                  facs(n2 - l2) / facs(j - l2 - 1)) *
                 overlp(l1, l1, z1, j, l2, z2, m, R);
    for (int j = n1; j > l1; --j)
        summ2 += std::pow(z1, j - n1 - 2) *
                 (facs(n1 + l1 + 1) / facs(j + l1) -
                  facs(n1 - l1) / facs(j - l1 - 1)) *
                 overlp(j, l1, z1, n2, l2, z2, m, R);
    return vf1b * facs(n2 + l2 + 1) / facs(2 * l2 + 1) * std::pow(z1, l1 - n1) *
               std::pow(z2, l2 - n2) * form26 -
           4.0 * M_PI / (2 * l2 + 1) * vf1b * std::pow(z1, l1 - n1) * summ1 -
           4.0 * M_PI / (2 * l1 + 1) * summ2;
}

inline double c2int(int n1, int l1, int m1, double z1, int n2, int l2, int m2,
                    double z2, double R) {
    int m = std::abs(m1);
    int ms = std::abs(m2);
    int nh1 = 2 * n1 - 1;
    int nh2 = 2 * n2 - 1;
    double zeth1 = 2 * z1;
    double zeth2 = 2 * z2;
    double fak = std::pow(zeth1, nh1 + 2) * std::pow(zeth2, nh2 + 2) /
                 facs(2 * n1) / facs(2 * n2);
    double s = 0.0;
    for (int j = 0; j <= 2 * l1; j += 2)
        for (int js = 0; js <= 2 * l2; js += 2)
            s += a0(l1, j, m) * a0(l2, js, ms) *
                 harris(nh1, j, zeth1, nh2, js, zeth2, 0, R);
    return fak * s;
}

inline double sk0(int k, double p) { return facs(k - 1) / std::pow(p + 1.0, k); }

inline double radint(int lamb, int na, int nb, int nc, int nd, double za,
                     double zb, double zc, double zd) {
    int n1 = na + nc;
    int n2 = nb + nd;
    double z1 = za + zc;
    double z2 = zb + zd;
    if (z1 == 0.0 || z2 == 0.0) return 0.0;
    int k = n1 + n2;
    int l1 = n2 - lamb - 1;
    int l2 = n1 - lamb - 1;
    double p1 = z1 / z2;
    double p2 = z2 / z1;
    int expo = -n1 - n2 - 1;

    auto skl = [](int L, int kk, double p) -> double {
        if (L == 0) return sk0(kk, p);
        if (L == 1) return sk0(kk - 1, p) + sk0(kk, p);
        double val = sk0(kk - L, p) + sk0(kk - L + 1, p);
        double fak = 2.0;
        for (int i = 2; i <= L; ++i) {
            val = fak * val + sk0(kk - L + i, p);
            fak += 1.0;
        }
        return val;
    };

    double skl1 = skl(l1, k, p1);
    double skl2 = skl(l2, k, p2);
    double rad = std::pow(z2, expo) * skl1 + std::pow(z1, expo) * skl2;
    double norfak = std::pow(2.0 * za, 2 * na + 1) * std::pow(2.0 * zb, 2 * nb + 1) *
                    std::pow(2.0 * zc, 2 * nc + 1) * std::pow(2.0 * zd, 2 * nd + 1) /
                    (facs(2 * na) * facs(2 * nb) * facs(2 * nc) * facs(2 * nd));
    return rad * std::sqrt(norfak);
}

// --------------------------------------------------------------------------- //
// Integral derivatives — cdsum.f, ds2int.f, dharri.f, dc2int.f, dv2int.f     //
// --------------------------------------------------------------------------- //

// Derivative of calsum w.r.t. R (cdsum.f).  Product rule on (A_i * B_j) terms.
// dadr[k] = d/dR a_int[k], dbdr[k] = d/dR b_int[k]
inline double cdsum(const std::vector<double>& bss, const std::vector<double>& a_int,
                    const std::vector<double>& b_int, int n1, int n2, int l1, int l2,
                    const std::vector<double>& dadr, const std::vector<double>& dbdr) {
    int sigu = n1 - l1 + n2 - l2;
    int sigo = n1 + l1 + n2 + l2;
    double vz = ((n1 + l1) % 2 == 0) ? 1.0 : -1.0;
    double total = 0.0;
    int z = 0;
    if (sigu % 2 == 0) {
        for (int sig = sigu; sig <= sigo; sig += 2) {
            int su = std::max(0, sig - n1 - n2);
            int so = su + (std::min(sig, n1 + n2) - su) / 2;
            for (int s = su; s <= so; ++s) {
                if (sig == 2 * s)
                    total += bss[z] * (dadr[s] * b_int[s] + a_int[s] * dbdr[s]);
                else
                    total += bss[z] * (dadr[s] * b_int[sig - s] +
                                       a_int[s] * dbdr[sig - s] +
                                       vz * (dadr[sig - s] * b_int[s] +
                                             a_int[sig - s] * dbdr[s]));
                ++z;
            }
        }
    } else {
        for (int sig = sigu; sig <= sigo; sig += 2) {
            int su = std::max(0, sig - n1 - n2);
            int so = su + (std::min(sig, n1 + n2) - su) / 2;
            for (int s = su; s <= so; ++s) {
                total += bss[z] * (dadr[s] * b_int[sig - s] +
                                   a_int[s] * dbdr[sig - s] +
                                   vz * (dadr[sig - s] * b_int[s] +
                                         a_int[sig - s] * dbdr[s]));
                ++z;
            }
        }
    }
    return total;
}

// overlp with gradient: when grad==true returns d/dR of the overlap.
inline double overlp_grad(int n2, int l2, double z2, int n1, int l1, double z1,
                          int m, double R) {
    int mb = std::abs(m);
    if (l2 < 0 || l1 < 0 || l2 < mb || l1 < mb) return 0.0;
    int kmax = n1 + n2 + 1;
    double rm12 = 0.5 * R;
    double alpha = rm12 * (z1 + z2);
    double beta = rm12 * (z2 - z1);
    // Need a_int/b_int up to kmax+1 for derivative setup
    std::vector<double> a_int = aintgs(alpha, kmax + 1);
    std::vector<double> b_int = bintgs(alpha, beta, kmax + 1);
    std::vector<double> bss = bissig(n1, n2, l1, l2, mb);
    double s = calsum(bss, a_int, b_int, n1, n2, l1, l2);

    // d/dR of (R/2)^kmax = kmax * (R/2)^(kmax-1) * 1/2
    double dxdr = 0.5 * kmax * std::pow(rm12, kmax - 1);
    // DADR[i] = -A_int[i+1] * alpha/R, DBDR[i] = -B_int[i+1] * beta/R
    double vdra = alpha / R;
    double vdrb = beta / R;
    std::vector<double> dadr(kmax + 1), dbdr(kmax + 1);
    for (int i = 0; i <= kmax; ++i) {
        dadr[i] = -a_int[i + 1] * vdra;
        dbdr[i] = -b_int[i + 1] * vdrb;
    }
    double dsum = cdsum(bss, a_int, b_int, n1, n2, l1, l2, dadr, dbdr);
    double xs = xsave(l1, mb) * xsave(l2, mb);
    return xs * (dxdr * s + std::pow(rm12, kmax) * dsum);
}

// ds2int — derivative d/dR of two-center overlap <mu_a|nu_b> (ds2int.f).
inline double ds2int(int n1, int l1, int m1, double z1, int n2, int l2, int m2,
                     double z2, double R) {
    if (m1 != m2) return 0.0;
    double fak = std::sqrt(
        std::pow(z1 + z1, n1 + n1 + 1) * std::pow(z2 + z2, n2 + n2 + 1) /
        (facs(n1 + n1) * facs(n2 + n2)));
    return fak * overlp_grad(n1, l1, z1, n2, l2, z2, m1, R);
}

// _dharris — derivative d/dR of the Harris auxiliary function (dharri.f).
inline double dharris(int n1, int l1, double z1, int n2, int l2, double z2,
                      int m, double R) {
    int l1l2m = l1 + l2 - m;
    double vf1 = 4.0 * M_PI / (z1 * z1);
    double vf2 = 4.0 * M_PI / (z2 * z2);

    // Formula 17 derivative
    double dfor17 = vf1 * (overlp_grad(0, 0, 0.0, l1l2m, l1l2m, z2, 0, R) -
                           overlp_grad(0, 0, z1, l1l2m, l1l2m, z2, 0, R));
    double form17 = vf1 * (overlp(0, 0, 0.0, l1l2m, l1l2m, z2, 0, R) -
                           overlp(0, 0, z1, l1l2m, l1l2m, z2, 0, R));

    // Formula 30 derivative
    double dsum = 0.0;
    for (int j = 0; j < m; ++j) {
        double ds_val =
            (j - m) * std::pow(R, j - m - 1) *
                overlp(j + 1, j, z1, l1l2m, l1l2m, z2, j, R) +
            std::pow(R, j - m) *
                overlp_grad(j + 1, j, z1, l1l2m, l1l2m, z2, j, R);
        double prefac = std::pow(-z1 * z1, j - m) / (2 * j + 2);
        double fac0 = std::sqrt(
            facs(2 * m + 1) * facs(l1l2m + m) * facs(l1l2m - j) /
            (facs(2 * j + 1) * facs(l1l2m - m) * facs(l1l2m + j)));
        dsum += prefac * fac0 * ds_val;
    }
    double sqrt_factor = std::sqrt(facs(2 * m + 1) * facs(l1l2m + m) /
                                    facs(l1l2m - m));
    double dfor30 =
        std::pow(-1.0 / z1 / z1, m) * sqrt_factor *
            (-m * std::pow(R, -m - 1) * form17 +
             std::pow(R, -m) * dfor17) -
        vf1 * z1 * dsum;

    // Formula 26 derivative (L-M recursion)
    double dfor26 = dfor30;
    for (int j = 1; j <= l1 - m; ++j) {
        int L = m + j - 1;
        int ls = l1l2m - j + 1;
        double w1 = std::sqrt((double)(L - m + 1) * (L + m + 1) /
                              ((2 * L + 1) * (2 * L + 3)));
        double w2 = std::sqrt((double)(ls - m) * (ls + m) /
                              ((2 * ls - 1) * (2 * ls + 1)));
        double fak1 = z1 / z2 * 2 * ls * w1;
        double fak2 = z2 / z1 * (2 * L + 2) * w2;
        double dsum_term =
            vf1 * std::sqrt((double)(L + m) * (L - m) /
                            ((2 * L - 1) * (2 * L + 1))) *
                (z1 * overlp_grad(L + 1, L - 1, z1, ls, ls - 1, z2, m, R) +
                 overlp_grad(L, L - 1, z1, ls, ls - 1, z2, m, R)) +
            vf2 * std::sqrt((double)(ls + m - 1) * (ls - m - 1) /
                            ((2 * ls - 3) * (2 * ls - 1))) *
                (z2 * overlp_grad(L + 1, L, z1, ls, ls - 2, z2, m, R) +
                 overlp_grad(L + 1, L, z1, ls - 1, ls - 2, z2, m, R)) +
            vf2 * z1 * w1 *
                overlp_grad(L + 1, L + 1, z1, ls, ls - 1, z2, m, R) +
            vf1 * z2 * w2 * overlp_grad(L + 1, L, z1, ls, ls, z2, m, R);
        dfor26 = (dsum_term - fak2 * dfor26) / fak1;
    }

    // Formula 23 (final assembly)
    double vf1_n = facs(n1 + l1 + 1) / facs(2 * l1 + 1);
    double dsum1 = 0.0, dsum2 = 0.0;
    for (int j = n2; j > l2; --j)
        dsum1 += std::pow(z2, j - n2 - 2) *
                 (facs(n2 + l2 + 1) / facs(j + l2) -
                  facs(n2 - l2) / facs(j - l2 - 1)) *
                 overlp_grad(l1, l1, z1, j, l2, z2, m, R);
    for (int j = n1; j > l1; --j)
        dsum2 += std::pow(z1, j - n1 - 2) *
                 (facs(n1 + l1 + 1) / facs(j + l1) -
                  facs(n1 - l1) / facs(j - l1 - 1)) *
                 overlp_grad(j, l1, z1, n2, l2, z2, m, R);
    return vf1_n * facs(n2 + l2 + 1) / facs(2 * l2 + 1) *
               std::pow(z1, l1 - n1) * std::pow(z2, l2 - n2) * dfor26 -
           4.0 * M_PI / (2 * l2 + 1) * vf1_n * std::pow(z1, l1 - n1) * dsum1 -
           4.0 * M_PI / (2 * l1 + 1) * dsum2;
}

// dc2int — derivative d/dR of two-center Coulomb integral (dc2int.f).
inline double dc2int(int n1, int l1, int m1, double z1, int n2, int l2, int m2,
                     double z2, double R) {
    int m = std::abs(m1);
    int ms = std::abs(m2);
    int nh1 = 2 * n1 - 1;
    int nh2 = 2 * n2 - 1;
    double zeth1 = 2 * z1;
    double zeth2 = 2 * z2;
    double fak = std::pow(zeth1, nh1 + 2) * std::pow(zeth2, nh2 + 2) /
                 facs(2 * n1) / facs(2 * n2);
    double s = 0.0;
    for (int j = 0; j <= 2 * l1; j += 2)
        for (int js = 0; js <= 2 * l2; js += 2)
            s += a0(l1, j, m) * a0(l2, js, ms) *
                 dharris(nh1, j, zeth1, nh2, js, zeth2, 0, R);
    return fak * s;
}

// dv2int — derivative d/dR of one-center nuclear attraction (dv2int.f).
inline double dv2int(int n1, int l1, int m1, double z1, double R) {
    int m = std::abs(m1);
    int nh = 2 * n1 - 1;
    double zeth = 2 * z1;
    double fak = std::pow(zeth, nh + 2) / facs(2 * n1) * 2.0 * std::sqrt(M_PI);
    double s = 0.0;
    for (int i = 0; i <= 2 * l1; i += 2)
        s += a0(l1, i, m) * overlp_grad(nh, i, zeth, 0, 0, 0.0, 0, R);
    return fak * s;
}

// --------------------------------------------------------------------------- //
// HARMTR — local->global rotation matrix from direction cosine E (harmtr.f).
// Fills T[9][9] (Eigen-free).  Orbital order: 0=s, 1..3 = px,py,pz, 4..8 = d.
// --------------------------------------------------------------------------- //
inline void harmtr(int maxkl, const double E[3], double T[9][9]) {
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j) T[i][j] = 0.0;
    T[0][0] = 1.0;
    if (maxkl <= 1) return;
    double cost = E[2];
    double sint, cosp, sinp;
    if (std::abs(cost) == 1.0) {
        sint = 0.0; cosp = 1.0; sinp = 0.0;
    } else if (std::abs(cost) == 0.0) {
        sint = 1.0; cosp = E[0]; sinp = E[1];
    } else {
        sint = std::sqrt(1.0 - cost * cost);
        cosp = E[0] / sint; sinp = E[1] / sint;
    }
    T[1][1] = sint * cosp;  T[2][1] = sint * sinp;  T[3][1] = cost;
    T[1][2] = cost * cosp;  T[2][2] = cost * sinp;  T[3][2] = -sint;
    T[1][3] = -sinp;        T[2][3] = cosp;         T[3][3] = 0.0;
    if (maxkl <= 2) return;
    double cos2t = cost * cost - sint * sint;
    double sin2t = 2.0 * sint * cost;
    double cos2p = cosp * cosp - sinp * sinp;
    double sin2p = 2.0 * sinp * cosp;
    double s3 = std::sqrt(3.0);
    T[4][4] = (3.0 * cost * cost - 1.0) * 0.5;
    T[5][4] = s3 * sin2t * cosp * 0.5;
    T[6][4] = s3 * sin2t * sinp * 0.5;
    T[7][4] = s3 * sint * sint * cos2p * 0.5;
    T[8][4] = s3 * sint * sint * sin2p * 0.5;
    T[4][5] = -s3 * sin2t * 0.5;
    T[5][5] = cos2t * cosp;
    T[6][5] = cos2t * sinp;
    T[7][5] = sin2t * cos2p * 0.5;
    T[8][5] = sin2t * sin2p * 0.5;
    T[4][6] = 0.0;
    T[5][6] = -cost * sinp;
    T[6][6] = cost * cosp;
    T[7][6] = -sint * sin2p;
    T[8][6] = sint * cos2p;
    T[4][7] = s3 * sint * sint * 0.5;
    T[5][7] = -sin2t * cosp * 0.5;
    T[6][7] = -sin2t * sinp * 0.5;
    T[7][7] = (1.0 + cost * cost) * cos2p * 0.5;
    T[8][7] = (1.0 + cost * cost) * sin2p * 0.5;
    T[4][8] = 0.0;
    T[5][8] = sint * sinp;
    T[6][8] = -sint * cosp;
    T[7][8] = -cost * sin2p;
    T[8][8] = cost * cos2p;
}

}  // namespace indo
}  // namespace semiempirical
}  // namespace vibeqc
