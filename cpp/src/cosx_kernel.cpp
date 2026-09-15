// Custom nuclear-attraction kernel implementation. See
// ``cpp/include/vibeqc/cosx_kernel.hpp`` for the math, the literature
// anchors, and the API contract.
//
// This first landing (commit 1 of the kernel arc) implements:
//
//   * The Boys-function table — tabulation by series + asymptotic for
//     the top-order column, downward recursion for lower orders;
//     6th-order Taylor interpolation in T for lookup.
//
//   * The per-shell-pair primitive-pair cache.
//
//   * ``cosx_nuclear_pair`` for the (s, s) sector only. For shells of
//     higher total angular momentum the function throws — wiring into
//     the hot loop waits for commit 2 (general Obara-Saika).
//
// All math is from the references in the header; no proprietary
// QC-code source was consulted.

#include "vibeqc/cosx_kernel.hpp"

#include <libint2/solidharmonics.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vibeqc {

namespace {

// (2k − 1)!! = 1 · 3 · 5 · … · (2k − 1) with the convention (−1)!! = 1.
constexpr double kSqrtPi = 1.7724538509055159;            // √π
constexpr double kTwoPi  = 6.283185307179586;             // 2π

double double_factorial_odd(int two_k_minus_one) {
    if (two_k_minus_one <= 0) return 1.0;
    double r = 1.0;
    for (int k = 1; k <= two_k_minus_one; k += 2) {
        r *= static_cast<double>(k);
    }
    return r;
}

// Boys function F_n(T) by the positive-term series obtained from the
// lower-incomplete-gamma identity
//
//   F_n(T) = γ(n + 1/2, T) / (2 · T^{n + 1/2})
//          = e^{−T} · Σ_{k=0}^{∞} (2T)^k
//                                / ((2n + 1)(2n + 3) · · · (2n + 2k + 1))
//
// (Helgaker / Jørgensen / Olsen Ch. 9; equivalent to expanding
// γ via its standard ascending series). All terms are positive
// for T ≥ 0, so there is *no* alternating-sign cancellation —
// the naive alternating form F_n(T) = Σ_k (−T)^k / (k! (2n + 2k + 1))
// loses ~14 digits at T ~ 10 and is unsafe for table-build purposes.
//
// Successive-term ratio is r_k = 2T / (2n + 2k + 1); the series
// crosses into geometric decay at k ~ T, so 500 iterations supports
// T up to about 500. Past that, use the asymptotic form below.
//
// Truncation: stop when ``term < 1e-18 · sum`` (both positive).
double boys_series(int n, double T) {
    const double exp_neg_T = std::exp(-T);
    double term = 1.0 / static_cast<double>(2 * n + 1);
    double sum = term;
    for (int k = 1; k < 500; ++k) {
        term *= 2.0 * T / static_cast<double>(2 * n + 2 * k + 1);
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return exp_neg_T * sum;
}

// Boys function F_n(T) by asymptotic for large T (T > kBoysAsymThreshold):
//
//   F_n(T) ≈ (2n − 1)!! / (2^{n+1}) · √(π / T^{2n+1})
//          = (2n − 1)!! · √π / (2 (2T)^n √T) · ½     ... (simplified)
//
// We expand and use the form that minimises catastrophic cancellation:
//   F_n(T) = (2n - 1)!! · √(π) / (2^(n+1) T^(n + 1/2)).
double boys_asymptotic(int n, double T) {
    const double sqrt_T = std::sqrt(T);
    const double inv_2T = 0.5 / T;
    double r = 0.5 * kSqrtPi / sqrt_T;       // F_0 leading
    for (int k = 0; k < n; ++k) {
        r *= static_cast<double>(2 * k + 1) * inv_2T;
    }
    return r;
}

// Boys function F_n(T) — pick the form valid at this T.
//
// The positive-term series converges geometrically once k > T + n, so
// it is safe (and accurate to FP roundoff) for any T up to a few
// hundred. The leading asymptotic only reaches 1e-13 relative
// accuracy past T ≈ 30 (the next-order correction is O(exp(−T)/T)).
// We therefore use the series for T below ``kSeriesUpper`` and the
// asymptotic above; ``kSeriesUpper = 50`` keeps us safely inside the
// regime where the asymptotic is accurate without leaving the series
// to do unnecessary work at huge T.
double boys_eval_direct(int n, double T) {
    constexpr double kSeriesUpper = 50.0;
    if (T < kSeriesUpper) return boys_series(n, T);
    return boys_asymptotic(n, T);
}

// Downward Boys recursion fills lower orders from a known F_n(T):
//   F_{n-1}(T) = (2T · F_n(T) + exp(−T)) / (2n − 1).
// Numerically stable for any T ≥ 0 because the addition of the
// non-negative exp(−T) term and division by the odd denominator
// keeps cancellation away.
void boys_recurse_down(int n_top, double T, double* F_out, int n_to_fill) {
    const double exp_neg_T = std::exp(-T);
    for (int n = n_top; n > 0 && (n_top - n + 1) <= n_to_fill; --n) {
        F_out[n - 1] = (2.0 * T * F_out[n] + exp_neg_T)
                       / static_cast<double>(2 * n - 1);
    }
}

}  // namespace

double BoysTable::eval(int n, double T) const {
    // Argument check — release builds drop this, debug keeps it.
    // Out-of-range n indicates a bug in the consumer; out-of-range T
    // (negative) likewise — both should never reach here in a working
    // build.

    if (T <= t_max) {
        // 6th-order Taylor interpolation around the nearest table
        // index (rounding down, so h = T − idx · dt ∈ [0, dt)).
        // F_n(T + h) = Σ_{k=0}^{6} (−h)^k / k! · F_{n+k}(T_grid).
        const int idx = static_cast<int>(T / dt);
        const double T_grid = static_cast<double>(idx) * dt;
        const double h = T - T_grid;
        const double* row = &values[static_cast<std::size_t>(idx)
                                    * static_cast<std::size_t>(n_max + 7)];

        // Horner-form 6th-order Taylor:
        //   F(T+h) = a0 − h (a1 − h/2 (a2 − h/3 (a3 − h/4 (a4 − h/5
        //                  (a5 − h/6 · a6)))))
        // expanding to a0 − h·a1 + h²/2!·a2 − h³/3!·a3 + … − h⁶/6!·a6
        // with the alternating-sign Taylor coefficients from
        // F_n(T+h) = Σ_k (−h)^k / k! · F_{n+k}(T).
        const double a0 = row[n + 0];
        const double a1 = row[n + 1];
        const double a2 = row[n + 2];
        const double a3 = row[n + 3];
        const double a4 = row[n + 4];
        const double a5 = row[n + 5];
        const double a6 = row[n + 6];
        return a0 - h * (a1 - 0.5 * h *
                  (a2 - h / 3.0 *
                  (a3 - 0.25 * h *
                  (a4 - 0.2 * h *
                  (a5 - h / 6.0 * a6)))));
    }

    // T > t_max: the leading asymptotic is FP-roundoff accurate at
    // T > 50 for any n (the dropped O(e^{−T}/T) correction underflows;
    // Gill-HG-Pople 1991 confirms this on Table I), so we evaluate
    // F_n(T) directly with no downward recursion + no scratch
    // allocation. Heap-allocation-free hot path.
    return boys_asymptotic(n, T);
}

BoysTable build_boys_table(int max_l) {
    // Top order needed for a Cartesian-OS recursion on bra-ket of total
    // angular momentum 2 · max_l, plus the 6 columns the Taylor lookup
    // consumes.
    BoysTable t;
    t.n_max = 2 * max_l + 6;
    // t_max = 50 is the smallest cutoff where the leading-order
    // asymptotic
    //   F_n(T) ≈ (2n − 1)!! / 2^{n+1} · √(π / T^{2n+1})
    // is accurate to FP roundoff (the dropped correction is
    // O(e^{−T} / T), which is < 2e-22 at T = 50 and decays from
    // there). Below T = 50 the correction is significant
    // (~5e-8 at T = 31) and we must use the table.
    t.t_max = 50.0;
    t.dt = 0.05;
    t.n_grid = static_cast<int>(t.t_max / t.dt) + 1;

    const std::size_t row_stride = static_cast<std::size_t>(t.n_max + 7);
    t.values.assign(
        static_cast<std::size_t>(t.n_grid) * row_stride, 0.0);

    // For each grid T, compute the top-order F_{n_max+6}(T) and fill
    // down by downward Boys recursion. The +6 keeps the Taylor sum
    // valid for any n ≤ n_max.
    const int top_n = t.n_max + 6;
    for (int g = 0; g < t.n_grid; ++g) {
        const double T = static_cast<double>(g) * t.dt;
        double* row = &t.values[static_cast<std::size_t>(g) * row_stride];
        row[top_n] = boys_eval_direct(top_n, T);
        boys_recurse_down(top_n, T, row, top_n);
    }
    return t;
}

// Shared inner loop for the square and two-set cache builders: fill
// the primitive-pair list for one (sh_a, sh_b) contracted shell pair.
static void fill_primitive_pair_list(
    const libint2::Shell& sh_a,
    const libint2::Shell& sh_b,
    std::vector<PrimitivePairData>& pp) {
    const auto& alpha_a = sh_a.alpha;
    // We use simple shells (one contraction per shell) per the
    // ``basis.cpp`` convention — same assumption AO eval relies on.
    const auto& coeff_a = sh_a.contr[0].coeff;
    const auto& A = sh_a.O;
    const auto& alpha_b = sh_b.alpha;
    const auto& coeff_b = sh_b.contr[0].coeff;
    const auto& B = sh_b.O;

    const double dAB_x = A[0] - B[0];
    const double dAB_y = A[1] - B[1];
    const double dAB_z = A[2] - B[2];
    const double AB_sq = dAB_x * dAB_x
                       + dAB_y * dAB_y
                       + dAB_z * dAB_z;

    pp.reserve(alpha_a.size() * alpha_b.size());

    for (std::size_t p = 0; p < alpha_a.size(); ++p) {
        const double ap = alpha_a[p];
        const double cp = coeff_a[p];
        for (std::size_t q = 0; q < alpha_b.size(); ++q) {
            const double aq = alpha_b[q];
            const double cq = coeff_b[q];
            const double a_tot = ap + aq;
            const double inv_a = 1.0 / a_tot;
            const double mu = ap * aq * inv_a;
            const double K_AB = std::exp(-mu * AB_sq);

            PrimitivePairData d;
            d.alpha = a_tot;
            d.inv_alpha = inv_a;
            d.inv_two_alpha = 0.5 * inv_a;
            d.P[0] = (ap * A[0] + aq * B[0]) * inv_a;
            d.P[1] = (ap * A[1] + aq * B[1]) * inv_a;
            d.P[2] = (ap * A[2] + aq * B[2]) * inv_a;
            // Prefactor folds: contraction coefficients
            // (libint already includes per-primitive norm
            // for the (l, 0, 0) Cartesian orientation),
            // 2π/α, and the Gaussian product K_AB. Sign
            // matches libint's ``Operator::nuclear`` output
            // for ``set_params([(−1, C)])`` — that engine
            // implements V(r) = −Σ_A Z_A / |r − R_A| with
            // the standard attractive convention, so Z = −1
            // yields the positive ⟨μ|1/|r − C||ν⟩.
            d.prefactor = cp * cq * kTwoPi * inv_a * K_AB;
            pp.push_back(d);
        }
    }
}

PrimitivePairCache build_primitive_pair_cache(
    const libint2::BasisSet& shells) {
    const int n_shells = static_cast<int>(shells.size());

    PrimitivePairCache cache;
    cache.n_shells = n_shells;
    cache.pairs.assign(
        static_cast<std::size_t>(n_shells) * n_shells,
        std::vector<PrimitivePairData>{});

    for (int s1 = 0; s1 < n_shells; ++s1) {
        for (int s2 = 0; s2 < n_shells; ++s2) {
            fill_primitive_pair_list(
                shells[s1], shells[s2],
                cache.pairs[
                    static_cast<std::size_t>(s1) * n_shells + s2]);
        }
    }
    return cache;
}

PrimitivePairCache build_primitive_pair_cache_two_set(
    const std::vector<libint2::Shell>& shells_a,
    const std::vector<libint2::Shell>& shells_b) {
    const int n_a = static_cast<int>(shells_a.size());
    const int n_b = static_cast<int>(shells_b.size());

    PrimitivePairCache cache;
    // ``n_shells`` doubles as the row stride: pairs[s1 * n_shells + s2]
    // with s1 indexing shells_a and s2 indexing shells_b.
    cache.n_shells = n_b;
    cache.pairs.assign(
        static_cast<std::size_t>(n_a) * n_b,
        std::vector<PrimitivePairData>{});

    for (int s1 = 0; s1 < n_a; ++s1) {
        for (int s2 = 0; s2 < n_b; ++s2) {
            fill_primitive_pair_list(
                shells_a[static_cast<std::size_t>(s1)],
                shells_b[static_cast<std::size_t>(s2)],
                cache.pairs[
                    static_cast<std::size_t>(s1) * n_b + s2]);
        }
    }
    return cache;
}

namespace {

// (The Cartesian-index list, the Cartesian-to-pure transform, and
// the Cartesian-relative-norm factor are all baked into the
// per-thread ``CosxKernelWorkspace`` by ``reserve`` — built once,
// reused across millions of pair calls in the hot loop.)

}  // namespace

void CosxKernelWorkspace::reserve(int max_l_a, int max_l_b) {
    if (max_l_a <= reserved_max_l_a && max_l_b <= reserved_max_l_b) {
        return;
    }
    reserved_max_l_a = std::max(max_l_a, reserved_max_l_a);
    reserved_max_l_b = std::max(max_l_b, reserved_max_l_b);
    const int L_total = reserved_max_l_a + reserved_max_l_b;
    const int Lp1 = L_total + 1;
    const int Lbp1 = reserved_max_l_b + 1;
    bra_aux.assign(
        static_cast<std::size_t>(Lp1) * Lp1 * Lp1 * Lp1, 0.0);
    ket_aux.assign(
        static_cast<std::size_t>(Lp1) * Lp1 * Lp1
            * Lbp1 * Lbp1 * Lbp1, 0.0);

    // Output-side scratch sizing. K_cart accumulates contracted
    // Cartesian (n_cart_a × n_cart_b) blocks; K_mid holds the
    // post-a-side result (which is at most max(n_cart_a, n_pure_a)
    // rows × n_cart_b cols).
    const int max_n_cart_a =
        (reserved_max_l_a + 1) * (reserved_max_l_a + 2) / 2;
    const int max_n_cart_b =
        (reserved_max_l_b + 1) * (reserved_max_l_b + 2) / 2;
    const int max_n_pure_a = 2 * reserved_max_l_a + 1;
    const int max_rows_after_a = std::max(max_n_cart_a, max_n_pure_a);
    K_cart_stride = max_n_cart_b;
    K_mid_stride  = max_n_cart_b;
    K_cart.assign(
        static_cast<std::size_t>(max_n_cart_a) * K_cart_stride, 0.0);
    K_mid.assign(
        static_cast<std::size_t>(max_rows_after_a) * K_mid_stride, 0.0);

    // Cartesian-index + transform + relative-norm tables. Built once
    // per l ∈ [0, max(max_l_a, max_l_b)]. Each call indexes by the
    // shell's l directly.
    const int max_l = std::max(reserved_max_l_a, reserved_max_l_b);
    cart_lx.assign(static_cast<std::size_t>(max_l + 1), {});
    cart_ly.assign(static_cast<std::size_t>(max_l + 1), {});
    cart_lz.assign(static_cast<std::size_t>(max_l + 1), {});
    T_pure  .assign(static_cast<std::size_t>(max_l + 1), {});
    using libint2::solidharmonics::SolidHarmonicsCoefficients;
    for (int l = 0; l <= max_l; ++l) {
        const int n_cart = (l + 1) * (l + 2) / 2;
        cart_lx[l].reserve(static_cast<std::size_t>(n_cart));
        cart_ly[l].reserve(static_cast<std::size_t>(n_cart));
        cart_lz[l].reserve(static_cast<std::size_t>(n_cart));
        for (int lx = l; lx >= 0; --lx) {
            for (int ly = l - lx; ly >= 0; --ly) {
                const int lz = l - lx - ly;
                cart_lx[l].push_back(lx);
                cart_ly[l].push_back(ly);
                cart_lz[l].push_back(lz);
            }
        }
        const int n_pure = 2 * l + 1;
        T_pure[l].assign(
            static_cast<std::size_t>(n_pure) * n_cart, 0.0);
        for (int m = -l; m <= l; ++m) {
            for (int i = 0; i < n_cart; ++i) {
                T_pure[l][static_cast<std::size_t>(m + l) * n_cart + i]
                    = SolidHarmonicsCoefficients<double>::coeff(
                          l, m,
                          cart_lx[l][static_cast<std::size_t>(i)],
                          cart_ly[l][static_cast<std::size_t>(i)],
                          cart_lz[l][static_cast<std::size_t>(i)]);
            }
        }
    }
}

void cosx_nuclear_pair_into(
    const libint2::Shell& shell_a,
    const libint2::Shell& shell_b,
    const std::vector<PrimitivePairData>& pair_data,
    const std::array<double, 3>& C,
    const BoysTable& boys,
    CosxKernelWorkspace& ws,
    double* out,
    double omega) {
    const int l_a = shell_a.contr[0].l;
    const int l_b = shell_b.contr[0].l;
    const int L_total = l_a + l_b;
    const auto& A = shell_a.O;
    const auto& B = shell_b.O;

    // All scratch comes from the workspace — zero heap allocations
    // in the hot loop. The caller is expected to have called
    // ``ws.reserve(max_l_a, max_l_b)`` with bounds that envelop this
    // call's (l_a, l_b).
    const int n_cart_a = (l_a + 1) * (l_a + 2) / 2;
    const int n_cart_b = (l_b + 1) * (l_b + 2) / 2;
    const int* cart_a_lx = ws.cart_lx[l_a].data();
    const int* cart_a_ly = ws.cart_ly[l_a].data();
    const int* cart_a_lz = ws.cart_lz[l_a].data();
    const int* cart_b_lx = ws.cart_lx[l_b].data();
    const int* cart_b_ly = ws.cart_ly[l_b].data();
    const int* cart_b_lz = ws.cart_lz[l_b].data();

    // Cartesian accumulator: row-major (n_cart_a × n_cart_b) at the
    // workspace's reserved stride. Zero the prefix we'll touch.
    const int K_stride = ws.K_cart_stride;
    double* K_cart_buf = ws.K_cart.data();
    for (int i = 0; i < n_cart_a; ++i) {
        double* row = K_cart_buf + static_cast<std::size_t>(i) * K_stride;
        for (int j = 0; j < n_cart_b; ++j) row[j] = 0.0;
    }

    const std::array<double, 3> AB = {A[0] - B[0],
                                      A[1] - B[1],
                                      A[2] - B[2]};

    // Workspace indexing. The workspace was sized using the consumer's
    // declared ``max_l_a`` / ``max_l_b``; the strides match those.
    // ``bra_aux[lx, ly, lz, m]`` holds [a | V_C | 0]^(m) for a with
    // |a| ≤ L_total and m ∈ [0, L_total].
    const int L_total_max = ws.reserved_max_l_a + ws.reserved_max_l_b;
    const int Lp1 = L_total_max + 1;
    auto bra_at = [&](int lx, int ly, int lz, int m) -> double& {
        return ws.bra_aux[(((static_cast<std::size_t>(lx) * Lp1)
                            + ly) * Lp1 + lz) * Lp1 + m];
    };

    // ``ket_aux[ax, ay, az, bx, by, bz]`` holds [a | V_C | b]^(0)
    // for |a| ≤ L_total, |b| ≤ l_b. Strides match the workspace's
    // reserved sizes.
    const int Lap1 = L_total_max + 1;
    const int Lbp1 = ws.reserved_max_l_b + 1;
    auto ket_at = [&](int ax, int ay, int az,
                      int bx, int by, int bz) -> double& {
        return ws.ket_aux[(((((static_cast<std::size_t>(ax) * Lap1)
                              + ay) * Lap1 + az) * Lbp1 + bx) * Lbp1
                           + by) * Lbp1 + bz];
    };

    for (const auto& d : pair_data) {
        const double dx = d.P[0] - C[0];
        const double dy = d.P[1] - C[1];
        const double dz = d.P[2] - C[2];
        const double T = d.alpha * (dx * dx + dy * dy + dz * dz);
        const std::array<double, 3> PA = {d.P[0] - A[0],
                                          d.P[1] - A[1],
                                          d.P[2] - A[2]};
        const std::array<double, 3> PC = {dx, dy, dz};
        const double inv_2a = d.inv_two_alpha;

        // 1. Seed [0 | V | 0]^(m) for m ∈ [0, L_total]. All
        //    higher-|a| entries are filled by the recursion below; we
        //    don't need to zero them because each gets explicitly
        //    written. (bra_aux is zeroed at allocation but reused
        //    across primitive pairs — the recursion's write pattern
        //    always covers the in-use cells.)
        //
        //    Full Coulomb (omega == 0): prefactor · F_m(T).
        //
        //    erfc-attenuated short-range kernel (omega > 0) — the
        //    range-separated-hybrid auxiliary family:
        //
        //      [0 | erfc(ω r)/r | 0]^(m)
        //          = prefactor · [ F_m(T) − ρ^{m+1/2} · F_m(ρT) ],
        //      ρ = ω² / (ω² + α)
        //
        //    (the erf part is the standard attenuated-Coulomb result,
        //    e.g. Heyd-Scuseria-Ernzerhof 2003, J. Chem. Phys. 118,
        //    8207, Appendix; same convention as libint's
        //    ``Operator::erfc_nuclear``, pinned at ULP by
        //    tests/test_cosx_kernel.py). The erf-attenuated family
        //    G_m = ρ^{m+1/2} F_m(ρT) obeys the same derivative
        //    relation dG_m/dT = −G_{m+1} as F_m, so the Obara-Saika
        //    recursion below is untouched — only the seeds change.
        if (omega > 0.0) {
            const double rho =
                omega * omega / (omega * omega + d.alpha);
            const double T_lr = rho * T;
            double rho_pow = std::sqrt(rho);  // ρ^{m+1/2} at m = 0
            for (int m = 0; m <= L_total; ++m) {
                bra_at(0, 0, 0, m) = d.prefactor
                    * (boys.eval(m, T) - rho_pow * boys.eval(m, T_lr));
                rho_pow *= rho;
            }
        } else {
            for (int m = 0; m <= L_total; ++m) {
                bra_at(0, 0, 0, m) = d.prefactor * boys.eval(m, T);
            }
        }

        // 2. Bra-side Obara-Saika recursion (Obara-Saika 1986).
        //
        //   [a + 1_i | V_C | 0]^(m)
        //       = (P_i − A_i) · [a | V_C | 0]^(m)
        //       − (P_i − C_i) · [a | V_C | 0]^(m + 1)
        //       + N_i(a) / (2α)
        //           · ([a − 1_i | V_C | 0]^(m)
        //              − [a − 1_i | V_C | 0]^(m + 1))
        //
        // We iterate by total |a| ascending; at each |a|, pick the
        // axis to lift greedily (largest non-zero component for
        // stable accumulation). The recursion drops m's upper bound
        // by 1 each step; the loop honours that via ``m_max``.
        for (int total_a = 1; total_a <= L_total; ++total_a) {
            for (int ax = total_a; ax >= 0; --ax) {
                for (int ay = total_a - ax; ay >= 0; --ay) {
                    const int az = total_a - ax - ay;
                    int axis;
                    if (ax >= ay && ax >= az && ax > 0) axis = 0;
                    else if (ay >= az && ay > 0)        axis = 1;
                    else                                axis = 2;
                    int ap_x = ax, ap_y = ay, ap_z = az;
                    if (axis == 0)      --ap_x;
                    else if (axis == 1) --ap_y;
                    else                --ap_z;
                    const double pa = PA[axis];
                    const double pc = PC[axis];
                    const int N_ap = (axis == 0 ? ap_x
                                     : axis == 1 ? ap_y : ap_z);
                    int app_x = ap_x, app_y = ap_y, app_z = ap_z;
                    if (axis == 0)      --app_x;
                    else if (axis == 1) --app_y;
                    else                --app_z;
                    const int m_max = L_total - total_a;
                    for (int m = 0; m <= m_max; ++m) {
                        double v = pa * bra_at(ap_x, ap_y, ap_z, m)
                                 - pc * bra_at(ap_x, ap_y, ap_z, m + 1);
                        if (N_ap > 0) {
                            v += static_cast<double>(N_ap) * inv_2a
                                 * (bra_at(app_x, app_y, app_z, m)
                                    - bra_at(app_x, app_y, app_z, m + 1));
                        }
                        bra_at(ax, ay, az, m) = v;
                    }
                }
            }
        }

        // 3. Bra-ket transfer relation (Head-Gordon-Pople 1988 § 2):
        //
        //   [a | V_C | b + 1_i]^(0)
        //       = [a + 1_i | V_C | b]^(0) + (A_i − B_i) · [a | V_C | b]^(0)
        //
        // This is purely algebraic — no Boys-function recurrence. We
        // seed ket_aux[..., (0, 0, 0)] from the bra-only m = 0 slice
        // and lift the ket angular momentum one axis at a time.

        // Seed: ket_aux[a, (0, 0, 0)] = bra_aux[a, 0] for all
        // |a| ≤ L_total.
        for (int total_a = 0; total_a <= L_total; ++total_a) {
            for (int ax = total_a; ax >= 0; --ax) {
                for (int ay = total_a - ax; ay >= 0; --ay) {
                    const int az = total_a - ax - ay;
                    ket_at(ax, ay, az, 0, 0, 0) = bra_at(ax, ay, az, 0);
                }
            }
        }

        // Lift ket along axis 0 (x). Target: ket_aux[a, (bx, 0, 0)]
        // for bx in [1, l_b], for all a with |a| ≤ L_total − bx.
        for (int bx = 1; bx <= l_b; ++bx) {
            const double dAB = AB[0];
            const int a_total_max = L_total - bx;
            for (int total_a = 0; total_a <= a_total_max; ++total_a) {
                for (int ax = total_a; ax >= 0; --ax) {
                    for (int ay = total_a - ax; ay >= 0; --ay) {
                        const int az = total_a - ax - ay;
                        ket_at(ax, ay, az, bx, 0, 0) =
                            ket_at(ax + 1, ay, az, bx - 1, 0, 0)
                            + dAB * ket_at(ax, ay, az, bx - 1, 0, 0);
                    }
                }
            }
        }

        // Lift ket along axis 1 (y). For each (bx, by) with bx + by
        // ≤ l_b and by ≥ 1.
        for (int bx = 0; bx <= l_b; ++bx) {
            for (int by = 1; by <= l_b - bx; ++by) {
                const double dAB = AB[1];
                const int a_total_max = L_total - bx - by;
                for (int total_a = 0; total_a <= a_total_max; ++total_a) {
                    for (int ax = total_a; ax >= 0; --ax) {
                        for (int ay = total_a - ax; ay >= 0; --ay) {
                            const int az = total_a - ax - ay;
                            ket_at(ax, ay, az, bx, by, 0) =
                                ket_at(ax, ay + 1, az, bx, by - 1, 0)
                                + dAB * ket_at(ax, ay, az, bx, by - 1, 0);
                        }
                    }
                }
            }
        }

        // Lift ket along axis 2 (z). For each (bx, by, bz) with
        // bx + by + bz ≤ l_b and bz ≥ 1.
        for (int bx = 0; bx <= l_b; ++bx) {
            for (int by = 0; by <= l_b - bx; ++by) {
                for (int bz = 1; bz <= l_b - bx - by; ++bz) {
                    const double dAB = AB[2];
                    const int a_total_max = L_total - bx - by - bz;
                    for (int total_a = 0; total_a <= a_total_max; ++total_a) {
                        for (int ax = total_a; ax >= 0; --ax) {
                            for (int ay = total_a - ax; ay >= 0; --ay) {
                                const int az = total_a - ax - ay;
                                ket_at(ax, ay, az, bx, by, bz) =
                                    ket_at(ax, ay, az + 1, bx, by, bz - 1)
                                    + dAB * ket_at(ax, ay, az, bx, by, bz - 1);
                            }
                        }
                    }
                }
            }
        }

        // 4. Extract the (|a| = l_a, |b| = l_b) Cartesian block and
        //    accumulate into the contracted block. Indices follow
        //    libint's enumeration order. Direct pointer indexing —
        //    no Eigen ops in the inner loop.
        for (int i = 0; i < n_cart_a; ++i) {
            double* row = K_cart_buf
                + static_cast<std::size_t>(i) * K_stride;
            const int axi_lx = cart_a_lx[i];
            const int axi_ly = cart_a_ly[i];
            const int axi_lz = cart_a_lz[i];
            for (int j = 0; j < n_cart_b; ++j) {
                row[j] += ket_at(axi_lx, axi_ly, axi_lz,
                                 cart_b_lx[j], cart_b_ly[j],
                                 cart_b_lz[j]);
            }
        }
    }

    // Post-loop: apply the Cartesian → output transform per shell-side.
    //
    // libint's ``SolidHarmonicsCoefficients::coeff`` (which our
    // workspace baked into ``T_pure[l]``) consumes Cartesian
    // monomials in libint's pre-normalised convention. Obara-Saika
    // produces integrals that match that convention directly, so
    // the transform applies as-is for pure shells -- and for
    // Cartesian shells there is nothing to apply at all: libint
    // emits ``pure = false`` shells in that same convention.
    //
    // This used to multiply the Cartesian rows/columns by a
    // per-component ``cart_norm[l]`` factor
    // sqrt((2l-1)!!/((2lx-1)!!(2ly-1)!!(2lz-1)!!)), on the stated
    // premise that it "recovers the convention libint's engine
    // emits". It does not: libint normalises a shell with ONE factor
    // from the total l, so <xy|xy> = 1/3 on a d shell, not 1 (checked
    // against a raw libint2::Engine in
    // tests/test_ao_convention_invariants.py). The factor put COSX's
    // analytic half in a different basis from its grid half, which
    // comes from ``evaluate_ao``, and K for a Cartesian shell came out
    // wrong by tens of percent against exact ERI. See
    // tests/test_cosx_cartesian_parity.py.
    //
    // Pipeline:
    //   K_cart  (n_cart_a × n_cart_b)
    //   → K_mid (n_a × n_cart_b)   via a-side transform (pure only)
    //   → out   (n_a × n_b)        via b-side transform (pure only)
    //
    // All three buffers live on ``ws``; we read/write through raw
    // pointers to avoid Eigen's bookkeeping on these tiny blocks.
    const bool pure_a = shell_a.contr[0].pure;
    const bool pure_b = shell_b.contr[0].pure;
    const int n_a = pure_a ? (2 * l_a + 1) : n_cart_a;
    const int n_b = pure_b ? (2 * l_b + 1) : n_cart_b;
    const int M_stride = ws.K_mid_stride;
    double* K_mid_buf = ws.K_mid.data();

    // a-side: K_mid (n_a × n_cart_b) ← transform · K_cart.
    if (pure_a) {
        const double* Ta = ws.T_pure[l_a].data();
        for (int k = 0; k < n_a; ++k) {
            const double* Trow = Ta
                + static_cast<std::size_t>(k) * n_cart_a;
            double* Mrow = K_mid_buf
                + static_cast<std::size_t>(k) * M_stride;
            for (int j = 0; j < n_cart_b; ++j) Mrow[j] = 0.0;
            for (int i = 0; i < n_cart_a; ++i) {
                const double t = Trow[i];
                if (t == 0.0) continue;
                const double* Krow = K_cart_buf
                    + static_cast<std::size_t>(i) * K_stride;
                for (int j = 0; j < n_cart_b; ++j) {
                    Mrow[j] += t * Krow[j];
                }
            }
        }
    } else {
        for (int i = 0; i < n_a; ++i) {
            const double* Krow = K_cart_buf
                + static_cast<std::size_t>(i) * K_stride;
            double* Mrow = K_mid_buf
                + static_cast<std::size_t>(i) * M_stride;
            for (int j = 0; j < n_cart_b; ++j) Mrow[j] = Krow[j];
        }
    }

    // b-side: out (n_a × n_b) ← K_mid · transformᵀ. The output
    // buffer is row-major with row stride n_b (caller-provided).
    if (pure_b) {
        const double* Tb = ws.T_pure[l_b].data();
        for (int k = 0; k < n_a; ++k) {
            const double* Mrow = K_mid_buf
                + static_cast<std::size_t>(k) * M_stride;
            double* Orow = out
                + static_cast<std::size_t>(k) * n_b;
            for (int m = 0; m < n_b; ++m) {
                const double* Trow = Tb
                    + static_cast<std::size_t>(m) * n_cart_b;
                double v = 0.0;
                for (int j = 0; j < n_cart_b; ++j) {
                    v += Mrow[j] * Trow[j];
                }
                Orow[m] = v;
            }
        }
    } else {
        for (int k = 0; k < n_a; ++k) {
            const double* Mrow = K_mid_buf
                + static_cast<std::size_t>(k) * M_stride;
            double* Orow = out
                + static_cast<std::size_t>(k) * n_b;
            for (int m = 0; m < n_b; ++m) {
                Orow[m] = Mrow[m];
            }
        }
    }
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
cosx_nuclear_pair(const libint2::Shell& shell_a,
                  const libint2::Shell& shell_b,
                  const std::vector<PrimitivePairData>& pair_data,
                  const std::array<double, 3>& C,
                  const BoysTable& boys,
                  double omega) {
    const int l_a = shell_a.contr[0].l;
    const int l_b = shell_b.contr[0].l;
    CosxKernelWorkspace ws;
    ws.reserve(l_a, l_b);

    const int n_a = static_cast<int>(shell_a.size());
    const int n_b = static_cast<int>(shell_b.size());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        out(n_a, n_b);
    cosx_nuclear_pair_into(shell_a, shell_b, pair_data, C, boys,
                           ws, out.data(), omega);
    return out;
}

}  // namespace vibeqc
