// Multipole interaction tensor implementation.
//
// Computes the spherical-harmonic interaction tensor from exact
// Cartesian derivatives of 1/|R|, using the Stone real-solid-harmonic
// convention consistent with the BIPOLE spherical moment buffer.
//
// References: Pisani-Dovesi-Roetti (1988), Ch. II.4c, for the periodic
// quartet expansion; Saunders (1992), Sec. 5.3, Eqs. (90)-(92), for radial
// derivatives; Jackson, Classical Electrodynamics, Sec. 4.1, for the bare
// Cartesian Coulomb derivatives.

#include "vibeqc/bipole_multipole.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace vibeqc {

// Hard clamp on the total Cartesian order |alpha| + |beta| retained in the
// bipolar Taylor expansion of the two-body kernel.  Every tensor built here
// -- bare, erf-screened, erfc-screened, and their gradients -- must share
// it: the short-range tensor is the difference of the bare and the erf
// tensor, so a clamp that differs between them leaves the excess block
// holding an unmatched -T_erf.
//
// The tensor needs kernel derivatives to this order, its gradient to one
// order higher. Raising it to total order 4 (within the truncation scheme
// discussed by Pisani-Dovesi-Roetti 1988, p. 51) therefore also
// requires the order-5 erf derivative, which erf_cartesian_derivative
// currently refuses.  Keep in lock-step with _MAX_BIPOLAR_TOTAL_ORDER in
// python/vibeqc/bipole_multipole.py; tests/test_bipole_contractor_parity.py
// pins the two contractors against each other.
constexpr int kMaxBipolarTotalOrder = 3;

// ============================================================================
// Cartesian component indices
// ============================================================================

std::vector<int> multipole_cartesian_indices(int L_max) {
    std::vector<int> result;
    for (int l = 0; l <= L_max; ++l) {
        for (int i = l; i >= 0; --i) {
            for (int j = l - i; j >= 0; --j) {
                int k = l - i - j;
                result.push_back(i);
                result.push_back(j);
                result.push_back(k);
            }
        }
    }
    return result;
}

// ============================================================================
// Cartesian → spherical conversion matrix
// ============================================================================

Eigen::MatrixXd multipole_cartesian_to_spherical_matrix(int L_max) {
    if (L_max > 4) {
        throw std::runtime_error(
            "cartesian_to_spherical_matrix: L_max > 4 not supported");
    }

    const int n_cart = multipole_n_cart_components(L_max);
    const int n_sph = multipole_n_components(L_max);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_sph, n_cart);

    auto idx_flat = multipole_cartesian_indices(L_max);

    // Helper: find the flat index of a Cartesian component (i,j,k).
    auto cart_pos = [&](int ti, int tj, int tk) -> int {
        for (int p = 0; p < n_cart; ++p) {
            if (idx_flat[3*p] == ti && idx_flat[3*p+1] == tj &&
                idx_flat[3*p+2] == tk) {
                return p;
            }
        }
        return -1;
    };

    auto set = [&](int l, int m, int i, int j, int k, double coeff) {
        int row = multipole_lm_index(l, m);
        int col = cart_pos(i, j, k);
        if (col >= 0) C(row, col) = coeff;
    };

    const double sqrt2 = std::sqrt(2.0);
    const double sqrt3 = std::sqrt(3.0);

    // L=0: monopole Z_00 = 1
    set(0, 0, 0, 0, 0, 1.0);

    if (L_max >= 1) {
        // Z_1,-1 = -√2·y
        set(1, -1, 0, 1, 0, -sqrt2);
        // Z_1,0  = z
        set(1, 0, 0, 0, 1, 1.0);
        // Z_1,+1 = √2·x
        set(1, 1, 1, 0, 0, sqrt2);
    }

    if (L_max >= 2) {
        double inv_2sqrt2 = 1.0 / (2.0 * sqrt2);
        double half_sqrt3 = sqrt3 / 2.0;
        // Z_2,-2 = √3·xy
        set(2, -2, 1, 1, 0, sqrt3);
        // Z_2,-1 = √3·yz
        set(2, -1, 0, 1, 1, sqrt3);
        // Z_2,0  = (2zz-xx-yy)/(2√2)
        set(2, 0, 2, 0, 0, -inv_2sqrt2);
        set(2, 0, 0, 2, 0, -inv_2sqrt2);
        set(2, 0, 0, 0, 2, 2.0 * inv_2sqrt2);
        // Z_2,+1 = √3·xz
        set(2, 1, 1, 0, 1, sqrt3);
        // Z_2,+2 = (√3/2)(xx-yy)
        set(2, 2, 2, 0, 0, half_sqrt3);
        set(2, 2, 0, 2, 0, -half_sqrt3);
    }

    if (L_max >= 3) {
        double inv_sqrt3 = 1.0 / std::sqrt(3.0);
        double half = 0.5;
        // Z_3,-3 = (3/2)xxy - (1/2)yyy
        set(3, -3, 2, 1, 0, 1.5);
        set(3, -3, 0, 3, 0, -0.5);
        // Z_3,-2 = xyz
        set(3, -2, 1, 1, 1, 1.0);
        // Z_3,-1 = y(4zz-xx-yy)/2
        set(3, -1, 2, 1, 0, -half);
        set(3, -1, 0, 3, 0, -half);
        set(3, -1, 0, 1, 2, 2.0);
        // Z_3,0  = (zzz - 1.5xxz - 1.5yyz)/√3
        set(3, 0, 2, 0, 1, -1.5 * inv_sqrt3);
        set(3, 0, 0, 2, 1, -1.5 * inv_sqrt3);
        set(3, 0, 0, 0, 3, inv_sqrt3);
        // Z_3,+1 = x(4zz-xx-yy)/2
        set(3, 1, 3, 0, 0, -half);
        set(3, 1, 1, 2, 0, -half);
        set(3, 1, 1, 0, 2, 2.0);
        // Z_3,+2 = z(xx-yy)/2
        set(3, 2, 2, 0, 1, half);
        set(3, 2, 0, 2, 1, -half);
        // Z_3,+3 = x(xx-3yy)/2
        set(3, 3, 3, 0, 0, half);
        set(3, 3, 1, 2, 0, -1.5);
    }

    if (L_max >= 4) {
        // L=4 conversion (hexadecapole).  Coefficients from the Python
        // _cart_to_sph module, following the Stone convention
        // (Schmidt-semi-normalised, sqrt(4pi/(2l+1)) factor absorbed).
        // The leading factor for |m| > 0 is fac = sqrt(l+1) / 2^{l-1}
        // = sqrt(5)/8 for l=4.
        double sqrt5 = std::sqrt(5.0);
        double fac = sqrt5 / 8.0;

        // Z_{4,-4} = 4*fac * xy*(x^2 - y^2) = 4*fac*(x^3 y - x y^3)
        set(4, -4, 3, 1, 0,  4.0 * fac);
        set(4, -4, 1, 3, 0, -4.0 * fac);

        // Z_{4,-3} = 4*fac * yz*(3x^2 - y^2) = 12*fac*x^2 y z - 4*fac*y^3 z
        set(4, -3, 2, 1, 1, 12.0 * fac);
        set(4, -3, 0, 3, 1, -4.0 * fac);

        // Z_{4,-2} = 2*fac * xy*(6z^2 - x^2 - y^2)
        set(4, -2, 1, 1, 2, 12.0 * fac);
        set(4, -2, 3, 1, 0, -2.0 * fac);
        set(4, -2, 1, 3, 0, -2.0 * fac);

        // Z_{4,-1} = 4*fac * y*(4z^3 - 3z(x^2+y^2))
        set(4, -1, 0, 1, 3, 16.0 * fac);
        set(4, -1, 2, 1, 1, -12.0 * fac);
        set(4, -1, 0, 3, 1, -12.0 * fac);

        // Z_{4,0} = (8z^4 - 24z^2(x^2+y^2) + 3(x^4+2x^2y^2+y^4)) / 8
        //         = z^4 - 3z^2 x^2 - 3z^2 y^2 + 3/8 x^4 + 6/8 x^2 y^2 + 3/8 y^4
        set(4, 0, 0, 0, 4,  1.0);
        set(4, 0, 2, 0, 2, -3.0);
        set(4, 0, 0, 2, 2, -3.0);
        set(4, 0, 4, 0, 0,  0.375);
        set(4, 0, 2, 2, 0,  0.75);
        set(4, 0, 0, 4, 0,  0.375);

        // Z_{4,+1} = 4*fac * x*(4z^3 - 3z(x^2+y^2))
        set(4, 1, 1, 0, 3, 16.0 * fac);
        set(4, 1, 3, 0, 1, -12.0 * fac);
        set(4, 1, 1, 2, 1, -12.0 * fac);

        // Z_{4,+2} = 2*fac * (x^2-y^2)*(6z^2-x^2-y^2)
        set(4, 2, 2, 0, 2,  12.0 * fac);
        set(4, 2, 0, 2, 2, -12.0 * fac);
        set(4, 2, 4, 0, 0,  -2.0 * fac);
        set(4, 2, 0, 4, 0,   2.0 * fac);

        // Z_{4,+3} = 4*fac * xz*(x^2 - 3y^2)
        set(4, 3, 3, 0, 1,   4.0 * fac);
        set(4, 3, 1, 2, 1, -12.0 * fac);

        // Z_{4,+4} = 2*fac * (x^4 - 6x^2y^2 + y^4)
        set(4, 4, 4, 0, 0,   2.0 * fac);
        set(4, 4, 2, 2, 0, -12.0 * fac);
        set(4, 4, 0, 4, 0,   2.0 * fac);
    }

    return C;
}

// ============================================================================
// Cartesian derivatives of 1/|R|
// ============================================================================

double multipole_cartesian_derivative_inverse_r(
    int ix, int iy, int iz,
    double rx, double ry, double rz) {

    if (ix < 0 || iy < 0 || iz < 0) {
        throw std::invalid_argument(
            "cartesian_derivative_inverse_r: derivative orders must be "
            "nonnegative");
    }
    // Validate each component before summing so the total-order expression
    // cannot overflow.
    if (ix > 5 || iy > 5 || iz > 5 || ix + iy + iz > 5) {
        throw std::invalid_argument(
            "cartesian_derivative_inverse_r: total derivative order > 5 "
            "not supported");
    }

    int order = ix + iy + iz;
    double r2 = rx*rx + ry*ry + rz*rz;
    double r = std::sqrt(r2);
    if (r < 1e-30) return 0.0; // guarded by caller

    // Expand multi-index into list of axes.
    // e.g. (2,0,1) -> [0,0,2] where 0=x, 1=y, 2=z
    int axes[5];
    int n = 0;
    for (int t = 0; t < ix; ++t) axes[n++] = 0;
    for (int t = 0; t < iy; ++t) axes[n++] = 1;
    for (int t = 0; t < iz; ++t) axes[n++] = 2;

    auto kronecker = [](int i, int j) -> double {
        return (i == j) ? 1.0 : 0.0;
    };

    double R[3] = {rx, ry, rz};

    if (order == 0) {
        return 1.0 / r;
    }
    if (order == 1) {
        int a = axes[0];
        return -R[a] / (r * r2);
    }
    if (order == 2) {
        int a = axes[0], b = axes[1];
        return (3.0 * R[a] * R[b] - kronecker(a, b) * r2) / (r2 * r2 * r);  // / r^5
    }
    if (order == 3) {
        int a = axes[0], b = axes[1], c = axes[2];
        double num = -15.0 * R[a] * R[b] * R[c]
            + 3.0 * r2 * (kronecker(a, b) * R[c]
                        + kronecker(a, c) * R[b]
                        + kronecker(b, c) * R[a]);
        return num / (r2 * r2 * r2 * r);  // / r^7
    }
    if (order == 4) {
        int a = axes[0], b = axes[1], c = axes[2], d = axes[3];
        // All 6 pairwise terms
        double sum15 = 0.0;
        int pairs[6][4] = {
            {a,b,c,d}, {a,c,b,d}, {a,d,b,c},
            {b,c,a,d}, {b,d,a,c}, {c,d,a,b}
        };
        for (int p = 0; p < 6; ++p) {
            sum15 += kronecker(pairs[p][0], pairs[p][1])
                   * R[pairs[p][2]] * R[pairs[p][3]];
        }
        double sum3 =
            kronecker(a, b) * kronecker(c, d)
          + kronecker(a, c) * kronecker(b, d)
          + kronecker(a, d) * kronecker(b, c);

        double num = 105.0 * R[a] * R[b] * R[c] * R[d]
                   - 15.0 * r2 * sum15
                   + 3.0 * r2 * r2 * sum3;
        return num / (r2 * r2 * r2 * r2 * r);  // / r^9
    }
    if (order == 5) {
        // Order-5 derivative needed for the gradient of the L=4
        // interaction tensor.  The general isotropic harmonic tensor
        // form (Jackson, Sec. 4.1) extended to 5th order:
        // d_abcde (1/r) = [-945 R_a R_b R_c R_d R_e
        //   + 105 r^2 Σ(δ_ij R_k R_l R_m)  — 10 terms
        //   - 15 r^4 Σ(δ_ij δ_kl R_m)       — 15 terms] / r^11
        int a = axes[0], b = axes[1], c = axes[2], d = axes[3], e = axes[4];
        double r4 = r2 * r2;

        // Ten terms: choose the Kronecker-delta pair; multiply the three
        // remaining Cartesian components.
        double sum105 = 0.0;
        int delta_remaining[10][5] = {
            {a,b,c,d,e}, {a,c,b,d,e}, {a,d,b,c,e}, {a,e,b,c,d},
            {b,c,a,d,e}, {b,d,a,c,e}, {b,e,a,c,d},
            {c,d,a,b,e}, {c,e,a,b,d}, {d,e,a,b,c}
        };
        for (int t = 0; t < 10; ++t) {
            sum105 += kronecker(
                          delta_remaining[t][0], delta_remaining[t][1])
                    * R[delta_remaining[t][2]]
                    * R[delta_remaining[t][3]]
                    * R[delta_remaining[t][4]];
        }

        // Fifteen terms: choose two disjoint delta pairs and multiply the
        // one remaining Cartesian component.
        double sum15_all = 0.0;
        int pair_pairs_remaining[15][5] = {
            {a,b,c,d,e}, {a,b,c,e,d}, {a,b,d,e,c},
            {a,c,b,d,e}, {a,c,b,e,d}, {a,c,d,e,b},
            {a,d,b,c,e}, {a,d,b,e,c}, {a,d,c,e,b},
            {a,e,b,c,d}, {a,e,b,d,c}, {a,e,c,d,b},
            {b,c,d,e,a}, {b,d,c,e,a}, {b,e,c,d,a}
        };
        for (int t = 0; t < 15; ++t) {
            sum15_all += kronecker(
                             pair_pairs_remaining[t][0],
                             pair_pairs_remaining[t][1])
                       * kronecker(
                             pair_pairs_remaining[t][2],
                             pair_pairs_remaining[t][3])
                       * R[pair_pairs_remaining[t][4]];
        }

        double num = -945.0 * R[a] * R[b] * R[c] * R[d] * R[e]
                   + 105.0 * r2 * sum105
                   - 15.0 * r4 * sum15_all;
        return num / (r2 * r2 * r2 * r2 * r2 * r);  // / r^11
    }

    throw std::runtime_error(
        "cartesian_derivative_inverse_r: order > 5 not supported");
}

// ============================================================================
// Boys function F_n(x) = ∫_0^1 t^{2n} exp(-x t^2) dt
// ============================================================================

namespace {

// Stable downward recurrence (McMurchie & Davidson 1978):
//   F_n(x) = [(2n-1)·F_{n-1}(x) − exp(−x)] / (2x)
// with F_0(x) = √(π/(4x)) · erf(√x)
double boys_function(int n, double x) {
    if (x < 1e-14) {
        return 1.0 / (2.0 * n + 1.0);
    }
    double sqrt_x = std::sqrt(x);
    double f0 = std::sqrt(M_PI / (4.0 * x)) * std::erf(sqrt_x);
    if (n == 0) return f0;

    double f = f0;
    for (int k = 1; k <= n; ++k) {
        f = ((2.0 * k - 1.0) * f - std::exp(-x)) / (2.0 * x);
    }
    return f;
}

// McMurchie-Davidson Hermite ladder for the erf-screened kernel.
//
// R^n_{000} = (-2μ)^n · 2√(μ/π) · F_n(μ·r²)
// R^n_{t+1,u,v} = t · R^{n+1}_{t-1,u,v} + R_x · R^{n+1}_{t,u,v}
// (cyclically for Y, Z).
//
// The Cartesian derivative d^{t,u,v} [erf(√μ·r)/r] = R^0_{t,u,v}.
//
// Reference: McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978),
// Eq. (4.6).  Also Saunders 1992, Eqs. (A8)-(A10).

// We use a memoisation table for R^n_{t,u,v} up to the needed order.
// Key: (t, u, v, n) packed as a 64-bit int for map lookup.

struct MDHermiteKey {
    int t, u, v, n;
    bool operator==(const MDHermiteKey& o) const {
        return t == o.t && u == o.u && v == o.v && n == o.n;
    }
};

struct MDHermiteKeyHash {
    std::size_t operator()(const MDHermiteKey& k) const {
        return (static_cast<std::uint64_t>(k.t + 8) << 40)
             | (static_cast<std::uint64_t>(k.u + 8) << 28)
             | (static_cast<std::uint64_t>(k.v + 8) << 16)
             | static_cast<std::uint64_t>(k.n);
    }
};

// R^n_{t,u,v} evaluator using memoised recurrence.
double md_hermite_R(
    int t, int u, int v, int n,
    double rx, double ry, double rz,
    double mu, double prefactor,
    std::unordered_map<MDHermiteKey, double, MDHermiteKeyHash>& memo) {

    if (t < 0 || u < 0 || v < 0) return 0.0;

    MDHermiteKey key{t, u, v, n};
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;

    double val;

    // Base case: R^n_{000} = (-2μ)^n · prefactor · F_n(μ·r²)
    if (t == 0 && u == 0 && v == 0) {
        double r2 = rx*rx + ry*ry + rz*rz;
        double mu_r2 = mu * r2;
        double factor = std::pow(-2.0 * mu, n);
        val = factor * prefactor * boys_function(n, mu_r2);
    } else {
        // Recurrence: reduce the highest nonzero index.
        if (t > 0) {
            val = (t - 1) * md_hermite_R(t - 2, u, v, n + 1,
                                          rx, ry, rz, mu, prefactor, memo)
                + rx * md_hermite_R(t - 1, u, v, n + 1,
                                    rx, ry, rz, mu, prefactor, memo);
        } else if (u > 0) {
            val = (u - 1) * md_hermite_R(t, u - 2, v, n + 1,
                                          rx, ry, rz, mu, prefactor, memo)
                + ry * md_hermite_R(t, u - 1, v, n + 1,
                                    rx, ry, rz, mu, prefactor, memo);
        } else {
            val = (v - 1) * md_hermite_R(t, u, v - 2, n + 1,
                                          rx, ry, rz, mu, prefactor, memo)
                + rz * md_hermite_R(t, u, v - 1, n + 1,
                                    rx, ry, rz, mu, prefactor, memo);
        }
    }

    memo[key] = val;
    return val;
}

// Cartesian derivative of erf(√μ·r)/r via the M-D Hermite ladder.
double erf_cartesian_derivative(
    int ix, int iy, int iz,
    double rx, double ry, double rz, double mu) {

    if (mu <= 0.0) return 0.0;

    int order = ix + iy + iz;
    if (order > 4) {
        throw std::runtime_error(
            "erf_cartesian_derivative: |gamma| > 4 not supported");
    }

    double r2 = rx*rx + ry*ry + rz*rz;
    if (r2 < 1e-30) {
        // At R=0, the erf-screened kernel and its derivatives have
        // finite limits: erf(√μ·r)/r → 2√(μ/π).
        // For derivatives, only even-order traces are nonzero.
        // For the multipole expansion, R=0 is guarded by the caller.
        return 0.0;
    }

    double prefactor = 2.0 * std::sqrt(mu / M_PI);
    std::unordered_map<MDHermiteKey, double, MDHermiteKeyHash> memo;
    memo.reserve(256);

    return md_hermite_R(ix, iy, iz, 0,
                        rx, ry, rz, mu, prefactor, memo);
}

}  // namespace

// ============================================================================
// Cartesian derivatives of erfc(sqrt(mu)*r)/r
// ============================================================================

double multipole_cartesian_derivative_erfc_over_r(
    int ix, int iy, int iz,
    double rx, double ry, double rz, double mu) {

    if (mu <= 0.0) {
        return multipole_cartesian_derivative_inverse_r(ix, iy, iz, rx, ry, rz);
    }

    // erfc(√μ·r)/r = 1/r - erf(√μ·r)/r
    double bare_val = multipole_cartesian_derivative_inverse_r(
        ix, iy, iz, rx, ry, rz);
    double erf_val = erf_cartesian_derivative(ix, iy, iz, rx, ry, rz, mu);
    return bare_val - erf_val;
}

// ============================================================================
// Spherical interaction tensor assembly
// ============================================================================

Eigen::MatrixXd multipole_interaction_tensor(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz) {

    double r2 = rx*rx + ry*ry + rz*rz;
    double r = std::sqrt(r2);
    if (r < 1e-30) {
        throw std::runtime_error(
            "multipole_interaction_tensor: |R| is zero");
    }

    int L_max = std::max(L_max_A, L_max_B);
    if (L_max > 4) {
        throw std::runtime_error(
            "multipole_interaction_tensor: L_max > 4 not supported");
    }

    auto indices = multipole_cartesian_indices(L_max);
    int n_cart = multipole_n_cart_components(L_max);
    Eigen::MatrixXd T_cart = Eigen::MatrixXd::Zero(n_cart, n_cart);

    static const double fact[10] = {
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0
    };

    for (int ia = 0; ia < n_cart; ++ia) {
        int i1 = indices[3*ia], j1 = indices[3*ia+1], k1 = indices[3*ia+2];
        for (int ib = 0; ib < n_cart; ++ib) {
            int i2 = indices[3*ib], j2 = indices[3*ib+1], k2 = indices[3*ib+2];
            int L = i1 + i2 + j1 + j2 + k1 + k2;
            if (L > L_max || L > kMaxBipolarTotalOrder) continue;

            double D_val = multipole_cartesian_derivative_inverse_r(
                i1 + i2, j1 + j2, k1 + k2, rx, ry, rz);

            int abs_alpha = i1 + j1 + k1;
            double sign = (abs_alpha % 2 == 0) ? 1.0 : -1.0;
            double denom = fact[i1] * fact[j1] * fact[k1]
                         * fact[i2] * fact[j2] * fact[k2];
            T_cart(ia, ib) = sign * D_val / denom;
        }
    }

    // Convert to spherical via pseudoinverse.
    // The C matrix and its pseudoinverse depend only on L_max, so cache them.
    // Use a static map keyed by L_max to avoid recomputing the SVD.
    static std::unordered_map<int,
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> pinv_cache;
    // (C, C_pinv) pairs keyed by L_max

    Eigen::MatrixXd C, C_pinv;
    auto cache_it = pinv_cache.find(L_max);
    if (cache_it != pinv_cache.end()) {
        C = cache_it->second.first;
        C_pinv = cache_it->second.second;
    } else {
        C = multipole_cartesian_to_spherical_matrix(L_max);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            C, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tol = std::max(C.rows(), C.cols())
                     * svd.singularValues()(0)
                     * std::numeric_limits<double>::epsilon();
        C_pinv = svd.matrixV()
            * svd.singularValues()
                  .unaryExpr([tol](double s) {
                      return s > tol ? 1.0/s : 0.0;
                  })
                  .asDiagonal()
            * svd.matrixU().transpose();
        pinv_cache[L_max] = {C, C_pinv};
    }

    Eigen::MatrixXd T_sph = C_pinv.transpose() * T_cart * C_pinv;

    // Slice to requested dimensions.
    int n_A = multipole_n_components(L_max_A);
    int n_B = multipole_n_components(L_max_B);
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n_A, n_B);
    int n_A_full = multipole_n_components(L_max);
    int n_B_full = multipole_n_components(L_max);
    T.topLeftCorner(std::min(n_A, n_A_full),
                    std::min(n_B, n_B_full))
        = T_sph.topLeftCorner(std::min(n_A, n_A_full),
                              std::min(n_B, n_B_full));
    return T;
}

// ============================================================================
// Generic screened interaction tensor builder
// ============================================================================

namespace {

// Build the spherical interaction tensor for a screened kernel.
// The derivative function `deriv_fn(ix, iy, iz, rx, ry, rz, mu)` computes
// the Cartesian derivative d^{ix,iy,iz} of the screened kernel.
using ScreenedDerivativeFn = double (*)(int, int, int, double, double, double, double);

Eigen::MatrixXd build_screened_interaction_tensor(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz, double mu,
    ScreenedDerivativeFn deriv_fn) {

    double r2 = rx*rx + ry*ry + rz*rz;
    double r = std::sqrt(r2);
    if (r < 1e-30) {
        throw std::runtime_error(
            "screened_interaction_tensor: |R| is zero");
    }

    int L_max = std::max(L_max_A, L_max_B);
    if (L_max > 4) {
        throw std::runtime_error(
            "screened_interaction_tensor: L_max > 4 not supported");
    }

    auto indices = multipole_cartesian_indices(L_max);
    int n_cart = multipole_n_cart_components(L_max);
    Eigen::MatrixXd T_cart = Eigen::MatrixXd::Zero(n_cart, n_cart);

    static const double fact[10] = {
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0
    };

    for (int ia = 0; ia < n_cart; ++ia) {
        int i1 = indices[3*ia], j1 = indices[3*ia+1], k1 = indices[3*ia+2];
        for (int ib = 0; ib < n_cart; ++ib) {
            int i2 = indices[3*ib], j2 = indices[3*ib+1], k2 = indices[3*ib+2];
            int L = i1 + i2 + j1 + j2 + k1 + k2;
            if (L > L_max || L > kMaxBipolarTotalOrder) continue;

            double D_val = deriv_fn(
                i1 + i2, j1 + j2, k1 + k2, rx, ry, rz, mu);

            int abs_alpha = i1 + j1 + k1;
            double sign = (abs_alpha % 2 == 0) ? 1.0 : -1.0;
            double denom = fact[i1] * fact[j1] * fact[k1]
                         * fact[i2] * fact[j2] * fact[k2];
            T_cart(ia, ib) = sign * D_val / denom;
        }
    }

    // Convert to spherical via pseudoinverse.
    static std::unordered_map<int,
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> pinv_cache;

    Eigen::MatrixXd C, C_pinv;
    auto cache_it = pinv_cache.find(L_max);
    if (cache_it != pinv_cache.end()) {
        C = cache_it->second.first;
        C_pinv = cache_it->second.second;
    } else {
        C = multipole_cartesian_to_spherical_matrix(L_max);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            C, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tol = std::max(C.rows(), C.cols())
                     * svd.singularValues()(0)
                     * std::numeric_limits<double>::epsilon();
        C_pinv = svd.matrixV()
            * svd.singularValues()
                  .unaryExpr([tol](double s) {
                      return s > tol ? 1.0/s : 0.0;
                  })
                  .asDiagonal()
            * svd.matrixU().transpose();
        pinv_cache[L_max] = {C, C_pinv};
    }

    Eigen::MatrixXd T_sph = C_pinv.transpose() * T_cart * C_pinv;

    // Slice to requested dimensions.
    int n_A = multipole_n_components(L_max_A);
    int n_B = multipole_n_components(L_max_B);
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n_A, n_B);
    int n_A_full = multipole_n_components(L_max);
    int n_B_full = multipole_n_components(L_max);
    T.topLeftCorner(std::min(n_A, n_A_full),
                    std::min(n_B, n_B_full))
        = T_sph.topLeftCorner(std::min(n_A, n_A_full),
                              std::min(n_B, n_B_full));
    return T;
}

// Wrapper for erfc derivative matching the ScreenedDerivativeFn signature.
double erfc_deriv_wrapper(int ix, int iy, int iz,
                          double rx, double ry, double rz, double mu) {
    return multipole_cartesian_derivative_erfc_over_r(ix, iy, iz, rx, ry, rz, mu);
}

// Wrapper for erf derivative.
double erf_deriv_wrapper(int ix, int iy, int iz,
                         double rx, double ry, double rz, double mu) {
    return erf_cartesian_derivative(ix, iy, iz, rx, ry, rz, mu);
}

}  // namespace

Eigen::MatrixXd multipole_erfc_interaction_tensor(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz, double mu) {

    if (mu <= 0.0) {
        return multipole_interaction_tensor(L_max_A, L_max_B, rx, ry, rz);
    }

    return build_screened_interaction_tensor(
        L_max_A, L_max_B, rx, ry, rz, mu, erfc_deriv_wrapper);
}

Eigen::MatrixXd multipole_erf_interaction_tensor(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz, double mu) {

    if (mu <= 0.0) {
        return Eigen::MatrixXd::Zero(
            multipole_n_components(L_max_A),
            multipole_n_components(L_max_B));
    }

    return build_screened_interaction_tensor(
        L_max_A, L_max_B, rx, ry, rz, mu, erf_deriv_wrapper);
}

// ============================================================================
// Gradient of the spherical interaction tensor
// ============================================================================

std::vector<Eigen::MatrixXd> multipole_interaction_tensor_gradient(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz) {

    double r2 = rx*rx + ry*ry + rz*rz;
    double r = std::sqrt(r2);
    if (r < 1e-30) {
        throw std::runtime_error(
            "multipole_interaction_tensor_gradient: |R| is zero");
    }

    int L_max = std::max(L_max_A, L_max_B);
    if (L_max > 4) {
        throw std::runtime_error(
            "multipole_interaction_tensor_gradient: L_max > 4 not supported");
    }

    // The gradient ∂T/∂R_axis is the same as T but with the Cartesian
    // derivative replaced by its higher-order counterpart:
    //   ∂/∂R_axis d^{i,j,k}(1/r) = d^{i+δ_{axis}}(1/r)
    // So we build the same Cartesian tensor loop but with derivatives
    // incremented by 1 along the requested axis.

    auto indices = multipole_cartesian_indices(L_max);
    int n_cart = multipole_n_cart_components(L_max);

    static const double fact[10] = {
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0
    };

    // Cached pseudoinverse.
    static std::unordered_map<int,
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> pinv_cache;

    Eigen::MatrixXd C, C_pinv;
    auto cache_it = pinv_cache.find(L_max);
    if (cache_it != pinv_cache.end()) {
        C = cache_it->second.first;
        C_pinv = cache_it->second.second;
    } else {
        C = multipole_cartesian_to_spherical_matrix(L_max);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            C, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tol = std::max(C.rows(), C.cols())
                     * svd.singularValues()(0)
                     * std::numeric_limits<double>::epsilon();
        C_pinv = svd.matrixV()
            * svd.singularValues()
                  .unaryExpr([tol](double s) {
                      return s > tol ? 1.0/s : 0.0;
                  })
                  .asDiagonal()
            * svd.matrixU().transpose();
        pinv_cache[L_max] = {C, C_pinv};
    }

    std::vector<Eigen::MatrixXd> gradient(3);
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::MatrixXd T_cart = Eigen::MatrixXd::Zero(n_cart, n_cart);

        for (int ia = 0; ia < n_cart; ++ia) {
            int i1 = indices[3*ia], j1 = indices[3*ia+1], k1 = indices[3*ia+2];
            for (int ib = 0; ib < n_cart; ++ib) {
                int i2 = indices[3*ib], j2 = indices[3*ib+1], k2 = indices[3*ib+2];
                int L = i1 + i2 + j1 + j2 + k1 + k2;
                if (L > L_max || L > kMaxBipolarTotalOrder) continue;

                // Gradient: increment derivative index by 1 on this axis.
                int gi = i1 + i2 + (axis == 0 ? 1 : 0);
                int gj = j1 + j2 + (axis == 1 ? 1 : 0);
                int gk = k1 + k2 + (axis == 2 ? 1 : 0);

                // ∂/∂R_axis d^{i,j,k}(1/r) = d^{i+δ_axis, j+δ_axis, k+δ_axis}(1/r)
                double D_val = multipole_cartesian_derivative_inverse_r(
                    gi, gj, gk, rx, ry, rz);

                int abs_alpha = i1 + j1 + k1;
                double sign = (abs_alpha % 2 == 0) ? 1.0 : -1.0;
                double denom = fact[i1] * fact[j1] * fact[k1]
                             * fact[i2] * fact[j2] * fact[k2];
                T_cart(ia, ib) = sign * D_val / denom;
            }
        }

        Eigen::MatrixXd T_sph = C_pinv.transpose() * T_cart * C_pinv;

        int n_A = multipole_n_components(L_max_A);
        int n_B = multipole_n_components(L_max_B);
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n_A, n_B);
        int n_A_full = multipole_n_components(L_max);
        int n_B_full = multipole_n_components(L_max);
        T.topLeftCorner(std::min(n_A, n_A_full),
                        std::min(n_B, n_B_full))
            = T_sph.topLeftCorner(std::min(n_A, n_A_full),
                                  std::min(n_B, n_B_full));
        gradient[axis] = T;
    }

    return gradient;
}

// ---------------------------------------------------------------------------
// erfc-screened interaction tensor gradient
// ---------------------------------------------------------------------------

std::vector<Eigen::MatrixXd> multipole_erfc_interaction_tensor_gradient(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz, double mu) {

    if (mu <= 0.0) {
        return multipole_interaction_tensor_gradient(L_max_A, L_max_B, rx, ry, rz);
    }

    double r2 = rx*rx + ry*ry + rz*rz;
    double r = std::sqrt(r2);
    if (r < 1e-30) {
        throw std::runtime_error(
            "multipole_erfc_interaction_tensor_gradient: |R| is zero");
    }

    int L_max = std::max(L_max_A, L_max_B);
    if (L_max > 3) {
        throw std::runtime_error(
            "multipole_erfc_interaction_tensor_gradient: L_max > 3 not supported");
    }

    auto indices = multipole_cartesian_indices(L_max);
    int n_cart = multipole_n_cart_components(L_max);

    static const double fact[10] = {
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0
    };

    // Reuse the pseudoinverse cache from the bare gradient.
    static std::unordered_map<int,
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> pinv_cache;

    Eigen::MatrixXd C, C_pinv;
    auto cache_it = pinv_cache.find(L_max);
    if (cache_it != pinv_cache.end()) {
        C = cache_it->second.first;
        C_pinv = cache_it->second.second;
    } else {
        C = multipole_cartesian_to_spherical_matrix(L_max);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            C, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tol = std::max(C.rows(), C.cols())
                     * svd.singularValues()(0)
                     * std::numeric_limits<double>::epsilon();
        C_pinv = svd.matrixV()
            * svd.singularValues()
                  .unaryExpr([tol](double s) {
                      return s > tol ? 1.0/s : 0.0;
                  })
                  .asDiagonal()
            * svd.matrixU().transpose();
        pinv_cache[L_max] = {C, C_pinv};
    }

    std::vector<Eigen::MatrixXd> gradient(3);
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::MatrixXd T_cart = Eigen::MatrixXd::Zero(n_cart, n_cart);

        for (int ia = 0; ia < n_cart; ++ia) {
            int i1 = indices[3*ia], j1 = indices[3*ia+1], k1 = indices[3*ia+2];
            for (int ib = 0; ib < n_cart; ++ib) {
                int i2 = indices[3*ib], j2 = indices[3*ib+1], k2 = indices[3*ib+2];
                int L = i1 + i2 + j1 + j2 + k1 + k2;
                if (L > L_max || L > kMaxBipolarTotalOrder) continue;

                int gi = i1 + i2 + (axis == 0 ? 1 : 0);
                int gj = j1 + j2 + (axis == 1 ? 1 : 0);
                int gk = k1 + k2 + (axis == 2 ? 1 : 0);

                double D_val = multipole_cartesian_derivative_erfc_over_r(
                    gi, gj, gk, rx, ry, rz, mu);

                int abs_alpha = i1 + j1 + k1;
                double sign = (abs_alpha % 2 == 0) ? 1.0 : -1.0;
                double denom = fact[i1] * fact[j1] * fact[k1]
                             * fact[i2] * fact[j2] * fact[k2];
                T_cart(ia, ib) = sign * D_val / denom;
            }
        }

        Eigen::MatrixXd T_sph = C_pinv.transpose() * T_cart * C_pinv;

        int n_A = multipole_n_components(L_max_A);
        int n_B = multipole_n_components(L_max_B);
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n_A, n_B);
        int n_A_full = multipole_n_components(L_max);
        int n_B_full = multipole_n_components(L_max);
        T.topLeftCorner(std::min(n_A, n_A_full),
                        std::min(n_B, n_B_full))
            = T_sph.topLeftCorner(std::min(n_A, n_A_full),
                                  std::min(n_B, n_B_full));
        gradient[axis] = T;
    }

    return gradient;
}

}  // namespace vibeqc
