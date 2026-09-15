// POLIPO — Native C++ Gaussian multipole-moment integral engine.
//
// Implements the McMurchie-Davidson Hermite expansion for Cartesian
// multipole moments of Gaussian shell pairs.  For each shell pair
// (a, b) at cell translation R_cell, the product distribution is
// expanded around the adjoined-Gaussian centre P:
//
//   φ_a(r) φ_b(r - R_cell) = K_ab * Σ_{tuv} E_t^{i,a} E_u^{j,b} E_v^{k,b}
//                            * exp(-p |r - P|^2) * (r_x - P_x)^t (r_y - P_y)^u (r_z - P_z)^v
//
// where K_ab = exp(-a*b/p * |A - B - R|^2), p = a + b,
// P = (a*A + b*(B+R))/p, and E_t^{i,a} are the Hermite expansion
// coefficients for primitive i of shell a.
//
// The Cartesian moment around *origin* is then:
//   M_{ijk} = K_ab * Σ_{tuv} E_t E_u E_v * H_{i+t, j+u, k+v}(p, P - origin)
//
// where H_{ijk}(p, C) = ∫ exp(-p r^2) r_x^i r_y^j r_z^k dr
//                    = H_i(p, C_x) * H_j(p, C_y) * H_k(p, C_z)
//
// and H_n(p, c) = ∫_{-∞}^{∞} exp(-p x^2) (x + c)^n dx
//                = Σ_{m=0}^{n} binom(n, m) c^{n-m} * ∫ exp(-p x^2) x^m dx
//
// The 1D Hermite integral has the closed form:
//   ∫_{-∞}^{∞} exp(-p x^2) x^{2m} dx = (2m-1)!! / (2p)^m * √(π/p)
//   ∫_{-∞}^{∞} exp(-p x^2) x^{2m+1} dx = 0
//
// References and provenance
// -------------------------
// Helgaker, Jørgensen & Olsen (2000), Ch. 9.
// McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978).
// Pisani-Dovesi-Roetti (1988), Ch. II.4c, for the periodic product-
// distribution multipole context. Pisani-Dovesi (1980), Sec. 4, for the
// adjoined diffuse s-Gaussian convention. The engine assembly is
// implementation-specific.

#include "vibeqc/polipo_moments.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vibeqc {

namespace {

// ---------------------------------------------------------------------------
//  Factorial and double factorial
// ---------------------------------------------------------------------------

double factorial(int n) {
    static const double cache[] = {
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0,
        362880.0, 3628800.0, 39916800.0, 479001600.0, 6227020800.0,
        87178291200.0, 1307674368000.0, 20922789888000.0,
        355687428096000.0, 6402373705728000.0, 121645100408832000.0
    };
    if (n < 0) return 1.0;
    if (n < 20) return cache[n];
    double val = cache[19];
    for (int i = 20; i <= n; ++i) val *= i;
    return val;
}

double double_factorial(int n) {
    if (n <= 1) return 1.0;
    double val = 1.0;
    for (int i = n; i > 1; i -= 2) val *= i;
    return val;
}

double binomial(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    return factorial(n) / (factorial(k) * factorial(n - k));
}

// ---------------------------------------------------------------------------
//  1D Hermite integral: H_n(p, c) = ∫ exp(-p x^2) (x + c)^n dx
// ---------------------------------------------------------------------------

double hermite_1d(int n, double p, double c) {
    // H_n(p, c) = Σ_{m=0}^{n} binom(n, m) c^{n-m} * ∫ exp(-p x^2) x^m dx
    double sqrt_pi_over_p = std::sqrt(M_PI / p);
    double result = 0.0;
    for (int m = 0; m <= n; ++m) {
        double integral;
        if (m % 2 == 1) {
            integral = 0.0;  // odd powers integrate to 0
        } else {
            integral = double_factorial(m - 1) / std::pow(2.0 * p, m / 2) * sqrt_pi_over_p;
        }
        result += binomial(n, m) * std::pow(c, n - m) * integral;
    }
    return result;
}

// ---------------------------------------------------------------------------
//  3D Hermite integral
// ---------------------------------------------------------------------------

double hermite_3d(int i, int j, int k, double p,
                  double cx, double cy, double cz) {
    return hermite_1d(i, p, cx) * hermite_1d(j, p, cy) * hermite_1d(k, p, cz);
}

// ---------------------------------------------------------------------------
//  Cartesian component index
// ---------------------------------------------------------------------------

int cartesian_component_index(int i, int j, int k) {
    // Cartesian component ordering matching libint's emultipole buffer
    // layout.  For each total angular momentum L = i+j+k, the components
    // are enumerated as:  i = L..0,  j = L-i..0,  k = L-i-j.
    //
    // Example (L=2): (200), (110), (101), (020), (011), (002)
    //          i.e.   xx     xy     xz     yy     yz     zz
    //
    // This matches the hardcoded ``_CARTESIAN_LAYOUT_BY_L_MAX`` table in
    // ``vibeqc.bipole_cell_moments`` and is the order in which libint's
    // ``emultipole{1,2,3}`` engine writes its result buffer.
    int L = i + j + k;
    int offset = L * (L + 1) * (L + 2) / 6;
    int inner = 0;
    for (int ii = L; ii >= 0; --ii) {
        for (int jj = L - ii; jj >= 0; --jj) {
            int kk = L - ii - jj;
            if (ii == i && jj == j && kk == k) {
                return offset + inner;
            }
            ++inner;
        }
    }
    return -1;  // unreachable for valid (i,j,k)
}

// ---------------------------------------------------------------------------
//  Contracted Gaussian normalisation
// ---------------------------------------------------------------------------

/// Normalisation constant for a Cartesian Gaussian primitive —
/// Helgaker, Jørgensen & Olsen, *Molecular Electronic-Structure
/// Theory* (2000), Eq. (6.30):
///
///   N(a, i, j, k) = (2a/π)^{3/4} √[ (4a)^{i+j+k} / ((2i-1)!! (2j-1)!! (2k-1)!!) ]
///
/// This is the standard normalisation such that
///   ∫ φ(r)² d³r = 1  for  φ(r) = N x^i y^j z^k exp(−a r²).
///
/// Note: the ``coefficients`` attribute of ``ShellInfo`` / ``PolipoShellInfo``
/// already includes this factor (they are "pre-normalized" per libint
/// convention).  Callers multiplying shell-pair primitive integrals should
/// therefore use the coefficient *as stored* — do not divide by N and then
/// re-multiply; that strips and re-applies normalisation, cancelling it.
double cartesian_normalisation(double a, int i, int j, int k) {
    double prefactor = std::pow(2.0 * a / M_PI, 0.75);
    double numerator = std::pow(4.0 * a, i + j + k);
    double df_i = (i <= 1) ? 1.0 : double_factorial(2*i - 1);
    double df_j = (j <= 1) ? 1.0 : double_factorial(2*j - 1);
    double df_k = (k <= 1) ? 1.0 : double_factorial(2*k - 1);
    return prefactor * std::sqrt(numerator / (df_i * df_j * df_k));
}

// ---------------------------------------------------------------------------
//  Per-primitive shell-pair moment kernel
// ---------------------------------------------------------------------------

/// Compute Cartesian multipole moments for ONE primitive pair (a_prim, b_prim)
/// expanded around *origin*.
///
/// Returns a vector of length n_comp = (L_max+1)(L_max+2)(L_max+3)/6,
/// indexed by the Cartesian component order.
std::vector<double> primitive_moments_kernel(
    int l_a, double a_exp, double A_x, double A_y, double A_z,
    int i_a, int j_a, int k_a,
    int l_b, double b_exp, double B_x, double B_y, double B_z,
    int i_b, int j_b, int k_b,
    double ox, double oy, double oz,
    int L_max) {

    double p = a_exp + b_exp;
    double P_x = (a_exp * A_x + b_exp * B_x) / p;
    double P_y = (a_exp * A_y + b_exp * B_y) / p;
    double P_z = (a_exp * A_z + b_exp * B_z) / p;

    double AB2 = (A_x - B_x)*(A_x - B_x) + (A_y - B_y)*(A_y - B_y) + (A_z - B_z)*(A_z - B_z);
    double prefactor = std::exp(-a_exp * b_exp / p * AB2);

    int n_comp = (L_max + 1) * (L_max + 2) * (L_max + 3) / 6;
    std::vector<double> moments(n_comp, 0.0);

    double dPA_x = P_x - A_x, dPA_y = P_y - A_y, dPA_z = P_z - A_z;
    double dPB_x = P_x - B_x, dPB_y = P_y - B_y, dPB_z = P_z - B_z;
    double dPO_x = P_x - ox, dPO_y = P_y - oy, dPO_z = P_z - oz;

    // --- Precompute Hermite integrals H_n(p, 0) once for all n ---
    // Maximum Hermite order needed: L_max + max angular momentum from
    // bra + ket.  The bra has t <= i_a, ket has u <= i_b, so
    // total = t+u <= i_a+i_b.  Same for y, z.
    int max_n = L_max + i_a + i_b + j_a + j_b + k_a + k_b;
    std::vector<double> H_all(max_n + 1);
    H_all[0] = std::sqrt(M_PI / p);  // H_0 = sqrt(pi/p)
    // Recurrence: H_{n+2} = (n+1)/(2p) * H_n  (odd terms are zero)
    for (int n = 1; n <= max_n; ++n) {
        if (n % 2 == 1) {
            H_all[n] = 0.0;
        } else {
            H_all[n] = H_all[n - 2] * (n - 1) / (2.0 * p);
        }
    }

    // --- Precompute C coefficients for x, y, z separately ---
    // C_x[t][u] = binom(i_a,t)*dPA_x^(i_a-t) * binom(i_b,u)*dPB_x^(i_b-u)
    int nx_t = i_a + 1, nx_u = i_b + 1;
    std::vector<std::vector<double>> Cx(nx_t, std::vector<double>(nx_u, 0.0));
    for (int t = 0; t <= i_a; ++t) {
        double ca = binomial(i_a, t) * std::pow(dPA_x, i_a - t);
        for (int u = 0; u <= i_b; ++u) {
            Cx[t][u] = ca * binomial(i_b, u) * std::pow(dPB_x, i_b - u);
        }
    }
    int ny_t = j_a + 1, ny_u = j_b + 1;
    std::vector<std::vector<double>> Cy(ny_t, std::vector<double>(ny_u, 0.0));
    for (int t = 0; t <= j_a; ++t) {
        double ca = binomial(j_a, t) * std::pow(dPA_y, j_a - t);
        for (int u = 0; u <= j_b; ++u) {
            Cy[t][u] = ca * binomial(j_b, u) * std::pow(dPB_y, j_b - u);
        }
    }
    int nz_t = k_a + 1, nz_u = k_b + 1;
    std::vector<std::vector<double>> Cz(nz_t, std::vector<double>(nz_u, 0.0));
    for (int t = 0; t <= k_a; ++t) {
        double ca = binomial(k_a, t) * std::pow(dPA_z, k_a - t);
        for (int u = 0; u <= k_b; ++u) {
            Cz[t][u] = ca * binomial(k_b, u) * std::pow(dPB_z, k_b - u);
        }
    }

    // --- Precompute origin-shift binomials for each (L,ii,jj,kk) ---
    // For each Cartesian component (ii,jj,kk), precompute
    // shift_x[a] = binom(ii,a)*dPO_x^(ii-a)  [a=0..ii]
    // These are reused across all (t,u) combinations.
    std::vector<std::vector<double>> shift_x(L_max + 1);
    std::vector<std::vector<double>> shift_y(L_max + 1);
    std::vector<std::vector<double>> shift_z(L_max + 1);
    for (int n = 0; n <= L_max; ++n) {
        shift_x[n].resize(n + 1);
        shift_y[n].resize(n + 1);
        shift_z[n].resize(n + 1);
        for (int a = 0; a <= n; ++a) {
            shift_x[n][a] = binomial(n, a) * std::pow(dPO_x, n - a);
            shift_y[n][a] = binomial(n, a) * std::pow(dPO_y, n - a);
            shift_z[n][a] = binomial(n, a) * std::pow(dPO_z, n - a);
        }
    }

    // --- Main loop over expansion coefficients ---
    for (int t_x = 0; t_x <= i_a; ++t_x) {
        for (int u_x = 0; u_x <= i_b; ++u_x) {
            double cx = Cx[t_x][u_x];
            int total_x = t_x + u_x;
            if (cx == 0.0) continue;

            for (int t_y = 0; t_y <= j_a; ++t_y) {
                for (int u_y = 0; u_y <= j_b; ++u_y) {
                    double cxy = cx * Cy[t_y][u_y];
                    int total_y = t_y + u_y;
                    if (cxy == 0.0) continue;

                    for (int t_z = 0; t_z <= k_a; ++t_z) {
                        for (int u_z = 0; u_z <= k_b; ++u_z) {
                            double c_total = cxy * Cz[t_z][u_z];
                            int total_z = t_z + u_z;
                            if (c_total == 0.0) continue;

                            // Accumulate into moments for all (ii,jj,kk)
                            for (int L = 0; L <= L_max; ++L) {
                                for (int kk = 0; kk <= L; ++kk) {
                                    for (int jj = 0; jj <= L - kk; ++jj) {
                                        int ii = L - jj - kk;
                                        int idx = cartesian_component_index(ii, jj, kk);

                                        double sum = 0.0;
                                        int nx = total_x;
                                        for (int a = 0; a <= ii; ++a, ++nx) {
                                            double bx = shift_x[ii][a];
                                            int ny = total_y;
                                            for (int b = 0; b <= jj; ++b, ++ny) {
                                                double bxy = bx * shift_y[jj][b];
                                                int nz = total_z;
                                                for (int c = 0; c <= kk; ++c, ++nz) {
                                                    sum += bxy * shift_z[kk][c]
                                                         * H_all[nx] * H_all[ny] * H_all[nz];
                                                }
                                            }
                                        }
                                        moments[idx] += prefactor * c_total * sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return moments;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  Public API — compute shell-pair moments for one pair
// ---------------------------------------------------------------------------

Eigen::Tensor<double, 3> compute_polipo_shell_pair_moments(
    const PolipoShellInfo& sh_a,
    const PolipoShellInfo& sh_b,
    const std::array<double, 3>& R_cell,
    const std::array<double, 3>& origin,
    int L_max) {

    double Bx = sh_b.origin[0] + R_cell[0];
    double By = sh_b.origin[1] + R_cell[1];
    double Bz = sh_b.origin[2] + R_cell[2];

    int n_comp = (L_max + 1) * (L_max + 2) * (L_max + 3) / 6;

    // Number of basis functions per shell
    int n_bf_a = sh_a.n_bf;
    int n_bf_b = sh_b.n_bf;

    Eigen::Tensor<double, 3> result(n_comp, n_bf_a, n_bf_b);
    result.setZero();

    // Iterate over angular components for shell a
    // For Cartesian shells, enumerate (i,j,k) with i+j+k = l_a
    std::vector<std::array<int, 3>> comps_a;
    for (int k = 0; k <= sh_a.l; ++k) {
        for (int j = 0; j <= sh_a.l - k; ++j) {
            int i = sh_a.l - j - k;
            comps_a.push_back({i, j, k});
        }
    }
    // For shell b
    std::vector<std::array<int, 3>> comps_b;
    for (int k = 0; k <= sh_b.l; ++k) {
        for (int j = 0; j <= sh_b.l - k; ++j) {
            int i = sh_b.l - j - k;
            comps_b.push_back({i, j, k});
        }
    }

    for (size_t idx_a = 0; idx_a < comps_a.size(); ++idx_a) {
        int i_a = comps_a[idx_a][0];
        int j_a = comps_a[idx_a][1];
        int k_a = comps_a[idx_a][2];

        for (size_t idx_b = 0; idx_b < comps_b.size(); ++idx_b) {
            int i_b = comps_b[idx_b][0];
            int j_b = comps_b[idx_b][1];
            int k_b = comps_b[idx_b][2];

            // Sum over primitives.
            // PolipoShellInfo.coefficients are contraction coefficients
            // that ALREADY include the primitive normalisation N(a,i,j,k)
            // (the same convention as ShellInfo / libint — see bindings.cpp
            // ShellInfo::coefficients docstring).
            // The primitive_moments_kernel returns the integral of
            // UNNORMALIZED primitives, so we simply multiply by the
            // pre-normalized coefficients — no extra normalisation step.
            //
            // Reference: Helgaker, Jørgensen & Olsen (2000), Eq. (6.30),
            // for the primitive normalisation.
            std::vector<double> total_moments(n_comp, 0.0);
            for (size_t p_a = 0; p_a < sh_a.exponents.size(); ++p_a) {
                double a_exp = sh_a.exponents[p_a];
                double coeff_a = sh_a.coeffs[p_a];

                for (size_t p_b = 0; p_b < sh_b.exponents.size(); ++p_b) {
                    double b_exp = sh_b.exponents[p_b];
                    double coeff_b = sh_b.coeffs[p_b];

                    auto prim_mom = primitive_moments_kernel(
                        sh_a.l, a_exp, sh_a.origin[0], sh_a.origin[1], sh_a.origin[2],
                        i_a, j_a, k_a,
                        sh_b.l, b_exp, Bx, By, Bz,
                        i_b, j_b, k_b,
                        origin[0], origin[1], origin[2],
                        L_max);

                    double c_total = coeff_a * coeff_b;
                    for (int c = 0; c < n_comp; ++c) {
                        total_moments[c] += c_total * prim_mom[c];
                    }
                }
            }

            for (int c = 0; c < n_comp; ++c) {
                result(c, static_cast<int>(idx_a), static_cast<int>(idx_b)) = total_moments[c];
            }
        }
    }

    // ---- Apply pure-shell AO permutation (l=1 only) ----
    // For pure (spherical) p shells, libint orders AOs as (py, pz, px)
    // [m=-1,0,+1], while the Cartesian component enumeration gives
    // (px, py, pz).  Permute only the affected rows or columns.
    // For l=0 (s) the permutation is identity (no-op).
    // For l>=2 the full Cartesian->spherical transformation is needed
    // (not yet implemented — these shells fall through unchanged).
    int perm_p[3] = {1, 2, 0};  // cart index -> sph position
    if (sh_a.pure && sh_a.l == 1 && sh_b.pure && sh_b.l == 1) {
        // Both pure p: permute both rows and columns
        Eigen::Tensor<double, 3> tmp = result;
        int n_comp = static_cast<int>(result.dimension(0));
        for (int c = 0; c < n_comp; ++c)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    result(c, a, b) = tmp(c, perm_p[a], perm_p[b]);
    } else if (sh_a.pure && sh_a.l == 1) {
        // Only shell a is pure p: permute rows only
        Eigen::Tensor<double, 3> tmp = result;
        int n_comp = static_cast<int>(result.dimension(0));
        for (int c = 0; c < n_comp; ++c)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < n_bf_b; ++b)
                    result(c, a, b) = tmp(c, perm_p[a], b);
    } else if (sh_b.pure && sh_b.l == 1) {
        // Only shell b is pure p: permute columns only
        Eigen::Tensor<double, 3> tmp = result;
        int n_comp = static_cast<int>(result.dimension(0));
        for (int c = 0; c < n_comp; ++c)
            for (int a = 0; a < n_bf_a; ++a)
                for (int b = 0; b < 3; ++b)
                    result(c, a, b) = tmp(c, a, perm_p[b]);
    }

    return result;
}

// ---------------------------------------------------------------------------
//  Public API — lattice-wide moment computation
// ---------------------------------------------------------------------------

PolipoMultipoleSet compute_polipo_moments_lattice(
    const std::vector<PolipoShellInfo>& shells,
    const std::vector<PolipoCellInfo>& cells,
    const PolipoOptions& opts) {

    PolipoMultipoleSet result;
    result.nbf = 0;
    for (const auto& sh : shells) result.nbf += sh.n_bf;
    result.L_max = opts.L_max;
    result.spherical = opts.spherical;
    result.cells = cells;
    result.origin = {0.0, 0.0, 0.0};

    int n_comp = (opts.L_max + 1) * (opts.L_max + 2) * (opts.L_max + 3) / 6;
    result.blocks.resize(cells.size());

    int n_shells = static_cast<int>(shells.size());
    int n_cells = static_cast<int>(cells.size());

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        for (int s2 = 0; s2 < n_shells; ++s2) {
            for (int c = 0; c < n_cells; ++c) {
                std::array<double, 3> R_cell = cells[c].r_cart;
                std::array<double, 3> origin_arr = {0.0, 0.0, 0.0};

                auto moments = compute_polipo_shell_pair_moments(
                    shells[s1], shells[s2], R_cell, origin_arr, opts.L_max);

                #pragma omp critical
                {
                    if (result.blocks[c].empty()) {
                        result.blocks[c].resize(n_comp);
                        for (int comp = 0; comp < n_comp; ++comp) {
                            result.blocks[c][comp] = Eigen::MatrixXd::Zero(result.nbf, result.nbf);
                        }
                    }
                    int bf_a = shells[s1].bf_offset;
                    int bf_b = shells[s2].bf_offset;
                    int n_a = shells[s1].n_bf;
                    int n_b = shells[s2].n_bf;
                    for (int comp = 0; comp < n_comp; ++comp) {
                        for (int a = 0; a < n_a; ++a) {
                            for (int b = 0; b < n_b; ++b) {
                                result.blocks[c][comp](bf_a + a, bf_b + b) +=
                                    moments(comp, a, b);
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

} // namespace vibeqc
