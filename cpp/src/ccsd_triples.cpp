#include "vibeqc/ccsd_triples.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibeqc {

namespace {

// Accessors for flat-packed arrays.
inline double t1_at(const double* t1, int n_occ, int a, int i) {
    return t1[a * n_occ + i];
}
inline double t2_at(const double* t2, int n_vir, int n_occ,
                    int a, int b, int i, int j) {
    return t2[((a * n_vir + b) * n_occ + i) * n_occ + j];
}
inline double eri_at(const double* eri, int nso,
                     int p, int q, int r, int s) {
    return eri[((p * nso + q) * nso + r) * nso + s];
}

} // anonymous namespace

double ccsd_t_triples(
    const double* t1, const double* t2,
    const double* eri_as,
    const double* eps_o, const double* eps_v,
    int n_occ, int n_vir, int nso) {

    double et = 0.0;

    const double inv36 = 1.0 / 36.0;

    auto connected_raw = [&](int a, int b, int c,
                             int i, int j, int k) {
        // t3c0[abc,ijk] =
        //   sum_e t2[ae,jk] <ei||bc> - sum_m t2[bc,im] <ma||jk>
        double val = 0.0;
        for (int e = 0; e < n_vir; ++e) {
            val += t2_at(t2, n_vir, n_occ, a, e, j, k)
                 * eri_at(eri_as, nso,
                          n_occ + e, i,
                          n_occ + b, n_occ + c);
        }
        for (int m = 0; m < n_occ; ++m) {
            val -= t2_at(t2, n_vir, n_occ, b, c, i, m)
                 * eri_at(eri_as, nso,
                          m, n_occ + a,
                          j, k);
        }
        return val;
    };

    auto disconnected_raw = [&](int a, int b, int c,
                                int i, int j, int k) {
        // t3d0[abc,ijk] = t1[a,i] <jk||bc>
        return t1_at(t1, n_occ, a, i)
             * eri_at(eri_as, nso,
                      j, k,
                      n_occ + b, n_occ + c);
    };

    auto connected_asym_ijk = [&](int a, int b, int c,
                                  int i, int j, int k) {
        return connected_raw(a, b, c, i, j, k)
             - connected_raw(a, b, c, j, i, k)
             - connected_raw(a, b, c, k, j, i);
    };

    auto disconnected_asym_ijk = [&](int a, int b, int c,
                                     int i, int j, int k) {
        return disconnected_raw(a, b, c, i, j, k)
             - disconnected_raw(a, b, c, j, i, k)
             - disconnected_raw(a, b, c, k, j, i);
    };

    auto connected_asym = [&](int a, int b, int c,
                              int i, int j, int k) {
        return connected_asym_ijk(a, b, c, i, j, k)
             - connected_asym_ijk(b, a, c, i, j, k)
             - connected_asym_ijk(c, b, a, i, j, k);
    };

    auto disconnected_asym = [&](int a, int b, int c,
                                 int i, int j, int k) {
        return disconnected_asym_ijk(a, b, c, i, j, k)
             - disconnected_asym_ijk(b, a, c, i, j, k)
             - disconnected_asym_ijk(c, b, a, i, j, k);
    };

    // Flatten the canonical (i<=j<=k) occupied triples into a work list so the
    // OpenMP parallel-for has O(n_occ^3/6) units instead of just n_occ. The
    // previous outer-i-only schedule left most cores idle for the small-
    // occupied cells that drive ccsd_t_triples (e.g. periodic CCM CCSD(T)),
    // and the triangular j>=i,k>=j made per-i work wildly uneven. This mirrors
    // the molecular closed-shell (T) work-list in ccsd.cpp.
    std::vector<std::array<int, 3>> ijk_work;
    ijk_work.reserve(static_cast<std::size_t>(n_occ) *
                     static_cast<std::size_t>(n_occ + 1) *
                     static_cast<std::size_t>(n_occ + 2) / 6);
    for (int i = 0; i < n_occ; ++i)
        for (int j = i; j < n_occ; ++j)
            for (int k = j; k < n_occ; ++k)
                ijk_work.push_back({i, j, k});

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:et) schedule(dynamic)
#endif
    for (std::size_t w = 0; w < ijk_work.size(); ++w) {
        const int i = ijk_work[w][0];
        const int j = ijk_work[w][1];
        const int k = ijk_work[w][2];
        const double ei = eps_o[i];
        const double ej = eps_o[j];
        const double ek = eps_o[k];

        for (int a = 0; a < n_vir; ++a) {
            const double ea = eps_v[a];
            for (int b = a; b < n_vir; ++b) {  // b >= a
                const double eb = eps_v[b];
                for (int c = b; c < n_vir; ++c) { // c >= b
                    const double ec = eps_v[c];

                    // Denominator.
                    const double D = ei + ej + ek - ea - eb - ec;
                    if (std::abs(D) < 1e-15) continue;
                    const double invD = 1.0 / D;

                    const double t3c_val = connected_asym(a, b, c, i, j, k);
                    const double t3d_val = disconnected_asym(a, b, c, i, j, k);

                    // --- Energy contribution ---
                    // Canonical ordering (i<=j<=k, a<=b<=c) covers a
                    // symmetry-reduced subset of the full 6-index space.
                    // Each unique canonical entry contributes with
                    // multiplicity n_perm_ijk x n_perm_abc to the full
                    // unrestricted sum.
                    int mult_ijk = 1;
                    if (i == j && j == k) {
                        mult_ijk = 1;
                    } else if (i == j || j == k) {
                        mult_ijk = 3;
                    } else {
                        mult_ijk = 6;
                    }
                    int mult_abc = 1;
                    if (a == b && b == c) {
                        mult_abc = 1;
                    } else if (a == b || b == c) {
                        mult_abc = 3;
                    } else {
                        mult_abc = 6;
                    }

                    et += inv36 * mult_ijk * mult_abc * t3c_val
                        * (t3c_val + t3d_val) * invD;
                }
            }
        }
    }

    return et;
}

}  // namespace vibeqc
