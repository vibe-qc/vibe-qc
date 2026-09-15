// Bloch-summed AO-pair FT — s-only fast path.
//
// See cpp/include/vibeqc/aopair_ft.hpp for the contract and the
// algorithm derivation.

#include "vibeqc/aopair_ft.hpp"
#include "aopair_ft_internal.hpp"
#include "vibeqc/cart_to_sph_data.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vibeqc {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Per-shell index → first AO offset (for s-only every shell = 1 AO).
// Kept as a helper so the same scaffolding extends naturally to the
// general-L kernel (Item 1b in the integrals-chat handover).
std::vector<std::size_t> shell_to_bf_offsets_ss(
        const std::vector<ShellInfo>& shells)
{
    std::vector<std::size_t> off;
    off.reserve(shells.size());
    std::size_t cursor = 0;
    for (const auto& sh : shells) {
        off.push_back(cursor);
        if (sh.l != 0) {
            throw std::invalid_argument(
                "ao_pair_fourier_transform_bloch_ss: shell with L="
                + std::to_string(sh.l)
                + " encountered. This entry point handles s-shells "
                  "only; the general-L McMurchie-Davidson kernel "
                  "lands in a subsequent milestone (Item 1b).");
        }
        cursor += 1;  // s = 1 AO
    }
    return off;
}

}  // namespace

AOPairFTTensor ao_pair_fourier_transform_bloch_ss(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Vector3d& k_cart)
{
    const std::vector<ShellInfo> shells = basis.shells();
    const auto bf_offsets = shell_to_bf_offsets_ss(shells);

    const std::size_t n_orb = basis.nbasis();
    const std::size_t n_G = static_cast<std::size_t>(G_vectors.rows());
    const std::size_t n_g = static_cast<std::size_t>(R_g_list.rows());
    const std::size_t n_shells = shells.size();

    AOPairFTTensor out;
    out.n_orb = n_orb;
    out.n_G = n_G;
    out.data.assign(n_orb * n_orb * n_G, std::complex<double>(0.0, 0.0));

    if (n_G == 0 || n_g == 0 || n_orb == 0) {
        return out;
    }

    // Precompute G^2 per reciprocal-space sample — reused for every
    // primitive pair via the (π/γ)^{3/2} exp(−G²/4γ) damping.
    std::vector<double> G2(n_G);
    for (std::size_t k = 0; k < n_G; ++k) {
        const double gx = G_vectors(static_cast<Eigen::Index>(k), 0);
        const double gy = G_vectors(static_cast<Eigen::Index>(k), 1);
        const double gz = G_vectors(static_cast<Eigen::Index>(k), 2);
        G2[k] = gx * gx + gy * gy + gz * gz;
    }

    // Precompute Bloch phases exp(+i k · R_g) — one complex per cell.
    std::vector<std::complex<double>> bloch_phase(n_g);
    for (std::size_t g = 0; g < n_g; ++g) {
        const double kr =
              k_cart(0) * R_g_list(static_cast<Eigen::Index>(g), 0)
            + k_cart(1) * R_g_list(static_cast<Eigen::Index>(g), 1)
            + k_cart(2) * R_g_list(static_cast<Eigen::Index>(g), 2);
        bloch_phase[g] = std::complex<double>(std::cos(kr), std::sin(kr));
    }

    // Parallelize the (sM, sN) shell-pair loop. Each iteration writes
    // to a disjoint (μ, ν) slice of ``out``, so there is no inter-
    // thread contention — the inner accumulation runs sequentially on
    // a single thread.
    //
    // For s-shells nMshells × nNshells == n_orb × n_orb, which is
    // typically 25-2500 for realistic systems — plenty of parallel
    // work and no need to nest OMP into the inner loops.
#pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t sM = 0; sM < static_cast<std::ptrdiff_t>(n_shells); ++sM) {
        for (std::ptrdiff_t sN = 0; sN < static_cast<std::ptrdiff_t>(n_shells); ++sN) {
            const ShellInfo& shM = shells[static_cast<std::size_t>(sM)];
            const ShellInfo& shN = shells[static_cast<std::size_t>(sN)];
            const std::size_t bfM = bf_offsets[static_cast<std::size_t>(sM)];
            const std::size_t bfN = bf_offsets[static_cast<std::size_t>(sN)];

            const double Ax = shM.origin[0], Ay = shM.origin[1], Az = shM.origin[2];
            const double Bx0 = shN.origin[0], By0 = shN.origin[1], Bz0 = shN.origin[2];

            const auto& es_M = shM.exponents;
            const auto& cs_M = shM.coefficients;
            const auto& es_N = shN.exponents;
            const auto& cs_N = shN.coefficients;
            const std::size_t nM = es_M.size();
            const std::size_t nN = es_N.size();

            // Accumulator for this (μ, ν) pair across all cells and
            // all G samples. Lives on the stack per shell-pair
            // iteration; the n_G size is the only non-trivial cost.
            std::vector<std::complex<double>> acc(
                n_G, std::complex<double>(0.0, 0.0));

            for (std::size_t g = 0; g < n_g; ++g) {
                const double Rx = R_g_list(static_cast<Eigen::Index>(g), 0);
                const double Ry = R_g_list(static_cast<Eigen::Index>(g), 1);
                const double Rz = R_g_list(static_cast<Eigen::Index>(g), 2);
                const double Bx = Bx0 + Rx;
                const double By = By0 + Ry;
                const double Bz = Bz0 + Rz;

                const double ABx = Ax - Bx;
                const double ABy = Ay - By;
                const double ABz = Az - Bz;
                const double AB_sq = ABx * ABx + ABy * ABy + ABz * ABz;

                const std::complex<double> phase_R = bloch_phase[g];

                for (std::size_t ip = 0; ip < nM; ++ip) {
                    const double alpha = es_M[ip];
                    const double c_a = cs_M[ip];
                    for (std::size_t iq = 0; iq < nN; ++iq) {
                        const double beta = es_N[iq];
                        const double c_b = cs_N[iq];
                        const double gamma = alpha + beta;
                        const double inv_gamma = 1.0 / gamma;

                        // Gaussian-product overlap-K prefactor and
                        // (π/γ)^{3/2} radial weight.
                        const double K_overlap =
                            c_a * c_b * std::exp(-alpha * beta * inv_gamma * AB_sq);
                        const double radial_prefactor =
                            K_overlap * std::pow(kPi * inv_gamma, 1.5);

                        // Gaussian-product centre P_pq^g.
                        const double Px = (alpha * Ax + beta * Bx) * inv_gamma;
                        const double Py = (alpha * Ay + beta * By) * inv_gamma;
                        const double Pz = (alpha * Az + beta * Bz) * inv_gamma;

                        // (Bloch phase) × (overlap K) × (radial damping
                        // factor) folded together so the inner G loop
                        // does only one complex multiply per sample
                        // plus the exp(−iG·P).
                        const std::complex<double> prefactor =
                            phase_R * radial_prefactor;
                        const double quarter_inv_gamma = 0.25 * inv_gamma;

                        for (std::size_t k = 0; k < n_G; ++k) {
                            const double gx =
                                G_vectors(static_cast<Eigen::Index>(k), 0);
                            const double gy =
                                G_vectors(static_cast<Eigen::Index>(k), 1);
                            const double gz =
                                G_vectors(static_cast<Eigen::Index>(k), 2);
                            const double radial =
                                std::exp(-G2[k] * quarter_inv_gamma);
                            const double gp = gx * Px + gy * Py + gz * Pz;
                            const std::complex<double> phase_G(
                                std::cos(gp), -std::sin(gp));  // exp(−iG·P)
                            acc[k] += prefactor * radial * phase_G;
                        }
                    }
                }
            }

            // Drop the accumulated slice into the dense output. With
            // s-shells szM = szN = 1 so this is a single (μ, ν) row.
            const std::size_t base = (bfM * n_orb + bfN) * n_G;
            for (std::size_t k = 0; k < n_G; ++k) {
                out.data[base + k] = acc[k];
            }
        }
    }

    return out;
}

// ============================================================
// General-L Bloch-summed AO-pair FT — McMurchie-Davidson chain.
// ============================================================
//
// See Sun 2017, *J. Chem. Phys.* **147**, 164119 for the periodic
// formulation that PySCF's ``ft_aopair`` ships; the molecular core
// is McMurchie-Davidson 1978 + Helgaker, Jørgensen & Olsen 2000
// §9.5. The mirror Python implementation lives in
// ``python/vibeqc/_aopair_ft.py``.

namespace {

// One-axis McMurchie-Davidson Hermite-expansion coefficients
// E^{i, j}_t for i in [0, la], j in [0, lb], t in [0, la+lb].
//
// Layout: row-major ``e[(i * (lb + 1) + j) * (la + lb + 1) + t]``.
// Returned as a flat std::vector that the caller indexes through
// the inline ``idx`` lambda below — avoids the per-shell-pair
// allocation of three std::vector<double>'s with shape-templated
// strides.
void md_e_coefficients_1d_buffer(
        int la, int lb, double gamma,
        double P_minus_A, double P_minus_B,
        double* out)
{
    const int t_max = la + lb;
    const std::size_t n_la = static_cast<std::size_t>(la + 1);
    const std::size_t n_lb = static_cast<std::size_t>(lb + 1);
    const std::size_t n_t = static_cast<std::size_t>(t_max + 1);
    std::fill_n(out, n_la * n_lb * n_t, 0.0);

    const auto idx = [&](int i, int j, int t) -> std::size_t {
        return (static_cast<std::size_t>(i) * n_lb
                + static_cast<std::size_t>(j)) * n_t
               + static_cast<std::size_t>(t);
    };

    out[idx(0, 0, 0)] = 1.0;
    const double inv2g = 0.5 / gamma;

    // Walk the i-axis: E[i+1, 0, t] in terms of E[i, 0, *].
    for (int i = 0; i < la; ++i) {
        for (int t = 0; t <= t_max; ++t) {
            double term = P_minus_A * out[idx(i, 0, t)];
            if (t > 0) {
                term += inv2g * out[idx(i, 0, t - 1)];
            }
            if (t + 1 <= t_max) {
                term += static_cast<double>(t + 1) * out[idx(i, 0, t + 1)];
            }
            out[idx(i + 1, 0, t)] = term;
        }
    }

    // Walk the j-axis from each (i, 0): E[i, j+1, t] from E[i, j, *].
    for (int j = 0; j < lb; ++j) {
        for (int i = 0; i <= la; ++i) {
            for (int t = 0; t <= t_max; ++t) {
                double term = P_minus_B * out[idx(i, j, t)];
                if (t > 0) {
                    term += inv2g * out[idx(i, j, t - 1)];
                }
                if (t + 1 <= t_max) {
                    term += static_cast<double>(t + 1) *
                            out[idx(i, j, t + 1)];
                }
                out[idx(i, j + 1, t)] = term;
            }
        }
    }
}

void md_e_coefficients_1d(
        int la, int lb, double gamma,
        double P_minus_A, double P_minus_B,
        std::vector<double>& out)
{
    out.resize(static_cast<std::size_t>(la + 1)
               * static_cast<std::size_t>(lb + 1)
               * static_cast<std::size_t>(la + lb + 1));
    md_e_coefficients_1d_buffer(
        la, lb, gamma, P_minus_A, P_minus_B, out.data());
}

// Contract one Cartesian Gaussian-product FT line directly against the
// caller's reciprocal weights. The MD coefficient tables may have been
// built with one slot of raised ket angular momentum; ``cart_A`` and
// ``cart_B`` select the desired original/raised/lowered component.
double contract_md_ft_line(
        const std::array<int, 3>& cart_A,
        const std::array<int, 3>& cart_B,
        const std::vector<double>& Ex,
        const std::vector<double>& Ey,
        const std::vector<double>& Ez,
        std::size_t e_dim_lb,
        std::size_t e_dim_t,
        const std::vector<std::complex<double>>& Gx_pow,
        const std::vector<std::complex<double>>& Gy_pow,
        const std::vector<std::complex<double>>& Gz_pow,
        const std::vector<std::complex<double>>& prefactor_per_G,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                             Eigen::RowMajor>>& G_vectors,
        const Eigen::Ref<const Eigen::VectorXcd>& reciprocal_weights,
        double scale,
        int translation_axis = -1)
{
    if (scale == 0.0) {
        return 0.0;
    }
    const std::size_t n_G = prefactor_per_G.size();
    const auto e_idx = [&](int i, int j, int t) {
        return (static_cast<std::size_t>(i) * e_dim_lb
                + static_cast<std::size_t>(j)) * e_dim_t
               + static_cast<std::size_t>(t);
    };

    double contracted = 0.0;
    for (int t = 0; t <= cart_A[0] + cart_B[0]; ++t) {
        const double et = Ex[e_idx(cart_A[0], cart_B[0], t)];
        if (et == 0.0) {
            continue;
        }
        const auto* gx_t = &Gx_pow[static_cast<std::size_t>(t) * n_G];
        for (int u = 0; u <= cart_A[1] + cart_B[1]; ++u) {
            const double eu = Ey[e_idx(cart_A[1], cart_B[1], u)];
            if (eu == 0.0) {
                continue;
            }
            const auto* gy_u = &Gy_pow[static_cast<std::size_t>(u) * n_G];
            const double etu = scale * et * eu;
            for (int v = 0; v <= cart_A[2] + cart_B[2]; ++v) {
                const double ev = Ez[e_idx(cart_A[2], cart_B[2], v)];
                if (ev == 0.0) {
                    continue;
                }
                const auto* gz_v = &Gz_pow[static_cast<std::size_t>(v) * n_G];
                const double etuv = etu * ev;
                for (std::size_t k = 0; k < n_G; ++k) {
                    const std::complex<double> polynomial =
                        gx_t[k] * gy_u[k] * gz_v[k];
                    std::complex<double> derivative =
                        prefactor_per_G[k] * (etuv * polynomial);
                    if (translation_axis >= 0) {
                        const double component = G_vectors(
                            static_cast<Eigen::Index>(k), translation_axis);
                        derivative *= std::complex<double>(0.0, -component);
                    }
                    contracted += std::real(
                        reciprocal_weights(static_cast<Eigen::Index>(k))
                        * std::conj(derivative));
                }
            }
        }
    }
    return contracted;
}

// G-resolved-weight variant of ``contract_md_ft_line``. The caller's
// reciprocal weight is already folded into the complex per-G pair
// weight row ``q_per_G`` (one Cartesian (a, b) row of Q_cart), so the
// reduction reads Re( q_per_G[k] * conj(derivative_k) ) and only the
// real recurrence factors (2β, -n_axis) remain in ``scale``.
double contract_md_ft_line_gweighted(
        const std::array<int, 3>& cart_A,
        const std::array<int, 3>& cart_B,
        const std::vector<double>& Ex,
        const std::vector<double>& Ey,
        const std::vector<double>& Ez,
        std::size_t e_dim_lb,
        std::size_t e_dim_t,
        const std::vector<std::complex<double>>& Gx_pow,
        const std::vector<std::complex<double>>& Gy_pow,
        const std::vector<std::complex<double>>& Gz_pow,
        const std::vector<std::complex<double>>& prefactor_per_G,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                             Eigen::RowMajor>>& G_vectors,
        const std::complex<double>* q_per_G,
        double scale,
        int translation_axis = -1)
{
    if (scale == 0.0) {
        return 0.0;
    }
    const std::size_t n_G = prefactor_per_G.size();
    const auto e_idx = [&](int i, int j, int t) {
        return (static_cast<std::size_t>(i) * e_dim_lb
                + static_cast<std::size_t>(j)) * e_dim_t
               + static_cast<std::size_t>(t);
    };

    double contracted = 0.0;
    for (int t = 0; t <= cart_A[0] + cart_B[0]; ++t) {
        const double et = Ex[e_idx(cart_A[0], cart_B[0], t)];
        if (et == 0.0) {
            continue;
        }
        const auto* gx_t = &Gx_pow[static_cast<std::size_t>(t) * n_G];
        for (int u = 0; u <= cart_A[1] + cart_B[1]; ++u) {
            const double eu = Ey[e_idx(cart_A[1], cart_B[1], u)];
            if (eu == 0.0) {
                continue;
            }
            const auto* gy_u = &Gy_pow[static_cast<std::size_t>(u) * n_G];
            const double etu = scale * et * eu;
            for (int v = 0; v <= cart_A[2] + cart_B[2]; ++v) {
                const double ev = Ez[e_idx(cart_A[2], cart_B[2], v)];
                if (ev == 0.0) {
                    continue;
                }
                const auto* gz_v = &Gz_pow[static_cast<std::size_t>(v) * n_G];
                const double etuv = etu * ev;
                for (std::size_t k = 0; k < n_G; ++k) {
                    const std::complex<double> polynomial =
                        gx_t[k] * gy_u[k] * gz_v[k];
                    std::complex<double> derivative =
                        prefactor_per_G[k] * (etuv * polynomial);
                    if (translation_axis >= 0) {
                        const double component = G_vectors(
                            static_cast<Eigen::Index>(k), translation_axis);
                        derivative *= std::complex<double>(0.0, -component);
                    }
                    contracted += std::real(
                        q_per_G[k] * std::conj(derivative));
                }
            }
        }
    }
    return contracted;
}

// Per-shell AO offset, accounting for spherical-shell sizes
// (2L + 1). Throws on a Cartesian (non-pure) shell with L > 0 — the
// general kernel only handles the pure-spherical convention that
// vibe-qc defaults to.
std::vector<std::size_t> shell_to_bf_offsets_pure(
        const std::vector<ShellInfo>& shells)
{
    std::vector<std::size_t> off;
    off.reserve(shells.size());
    std::size_t cursor = 0;
    for (const auto& sh : shells) {
        if (sh.l < 0 || sh.l > cart_to_sph_data::kMaxL) {
            throw std::invalid_argument(
                "ao_pair_fourier_transform_bloch: shell with L="
                + std::to_string(sh.l) + " exceeds the cart-to-sph "
                  "table coverage (0.." + std::to_string(cart_to_sph_data::kMaxL)
                + "). Regenerate the table via "
                  "scripts/codegen_cart_to_sph.py and bump the bound.");
        }
        if (!sh.pure && sh.l > 0) {
            throw std::invalid_argument(
                "ao_pair_fourier_transform_bloch: Cartesian "
                "(non-pure) shell with L=" + std::to_string(sh.l)
                + " not supported. vibe-qc forces pure spherical for "
                  "L >= 2 by default; this should not arise in practice.");
        }
        off.push_back(cursor);
        cursor += static_cast<std::size_t>(2 * sh.l + 1);
    }
    return off;
}

double shell_pair_cell_ft_bound(
    const ShellInfo& shM,
    const ShellInfo& shN,
    double Bx,
    double By,
    double Bz,
    double min_G2,
    double max_G_norm)
{
    const double Ax = shM.origin[0], Ay = shM.origin[1], Az = shM.origin[2];
    const double ABx = Ax - Bx;
    const double ABy = Ay - By;
    const double ABz = Az - Bz;
    const double AB_sq = ABx * ABx + ABy * ABy + ABz * ABz;

    double radial_bound = 0.0;
    for (std::size_t ip = 0; ip < shM.exponents.size(); ++ip) {
        const double alpha = shM.exponents[ip];
        const double c_a = std::abs(shM.coefficients[ip]);
        for (std::size_t iq = 0; iq < shN.exponents.size(); ++iq) {
            const double beta = shN.exponents[iq];
            const double c_b = std::abs(shN.coefficients[iq]);
            const double gamma = alpha + beta;
            const double inv_gamma = 1.0 / gamma;
            const double exponent =
                -alpha * beta * inv_gamma * AB_sq
                -0.25 * min_G2 * inv_gamma;
            if (exponent < -745.0) {
                continue;
            }
            radial_bound += c_a * c_b * std::pow(M_PI * inv_gamma, 1.5)
                            * std::exp(exponent);
        }
    }
    if (radial_bound == 0.0) {
        return 0.0;
    }

    const int l_sum = std::max(0, shM.l + shN.l);
    if (l_sum == 0) {
        return radial_bound;
    }
    // The MD polynomial can add powers of G and displacement. This is a
    // deliberately conservative overestimate; it only screens cells whose
    // high-|G| contribution is already exponentially dead.
    const double disp = std::sqrt(AB_sq);
    const double poly_bound = std::pow(1.0 + max_G_norm + disp, l_sum);
    const double cart_factor =
        static_cast<double>(cart_to_sph_data::n_cart_for_l(shM.l))
        * static_cast<double>(cart_to_sph_data::n_cart_for_l(shN.l));
    return radial_bound * poly_bound * cart_factor;
}

bool cell_list_is_inversion_symmetric(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list)
{
    constexpr double tol = 1.0e-9;
    const Eigen::Index n_g = R_g_list.rows();
    for (Eigen::Index g = 0; g < n_g; ++g) {
        bool found = false;
        for (Eigen::Index h = 0; h < n_g; ++h) {
            const double sx = R_g_list(g, 0) + R_g_list(h, 0);
            const double sy = R_g_list(g, 1) + R_g_list(h, 1);
            const double sz = R_g_list(g, 2) + R_g_list(h, 2);
            if (std::abs(sx) <= tol && std::abs(sy) <= tol
                    && std::abs(sz) <= tol) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

bool eval_frequencies_are_lattice_reciprocal(
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list)
{
    // exp(i G.R) == 1 for every (eval frequency, cell vector) pair, i.e.
    // every eval frequency is a reciprocal-lattice vector of the R lattice.
    // This is the third condition (besides k == 0 and the inversion-
    // symmetric cell list) that the mirror identity
    // FT_mu,nu(G) == FT_nu,mu(G) requires: on a momentum-shifted
    // mesh G+q (the Bloch-pair GDF fit, q = k_ket - k_bra != 0) the
    // swap picks up exp(i (G+q).R) phases on the R != 0 inter-cell terms
    // and the identity fails (measured 1.9 four-center error on LiH/sto-3g
    // (2,1,1) exchange pairs, 2026-07-10). Early-exits on the first
    // violating pair, so the q != 0 case costs one row of dot products.
    constexpr double tol = 1.0e-8;
    constexpr double two_pi = 2.0 * M_PI;
    const Eigen::Index n_g = R_g_list.rows();
    const Eigen::Index n_G = G_vectors.rows();
    for (Eigen::Index g = 0; g < n_g; ++g) {
        const double rx = R_g_list(g, 0);
        const double ry = R_g_list(g, 1);
        const double rz = R_g_list(g, 2);
        if (std::abs(rx) <= tol && std::abs(ry) <= tol
                && std::abs(rz) <= tol) {
            continue;  // home cell: G.R == 0 trivially
        }
        for (Eigen::Index k = 0; k < n_G; ++k) {
            const double d = G_vectors(k, 0) * rx + G_vectors(k, 1) * ry
                + G_vectors(k, 2) * rz;
            if (std::abs(std::remainder(d, two_pi)) > tol) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

AOPairFTTensor ao_pair_fourier_transform_bloch(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Vector3d& k_cart,
    double screen_tol)
{
    const std::vector<ShellInfo> shells = basis.shells();
    const auto bf_offsets = shell_to_bf_offsets_pure(shells);

    const std::size_t n_orb = basis.nbasis();
    const std::size_t n_G = static_cast<std::size_t>(G_vectors.rows());
    const std::size_t n_g = static_cast<std::size_t>(R_g_list.rows());
    const std::size_t n_shells = shells.size();

    AOPairFTTensor out;
    out.n_orb = n_orb;
    out.n_G = n_G;
    out.data.assign(n_orb * n_orb * n_G, std::complex<double>(0.0, 0.0));

    if (n_G == 0 || n_g == 0 || n_orb == 0) {
        return out;
    }

    // Precompute G² and Bloch phases (shared across shell pairs).
    std::vector<double> G2(n_G);
    double min_G2 = std::numeric_limits<double>::infinity();
    double max_G2 = 0.0;
    for (std::size_t k = 0; k < n_G; ++k) {
        const double gx = G_vectors(static_cast<Eigen::Index>(k), 0);
        const double gy = G_vectors(static_cast<Eigen::Index>(k), 1);
        const double gz = G_vectors(static_cast<Eigen::Index>(k), 2);
        G2[k] = gx * gx + gy * gy + gz * gz;
        min_G2 = std::min(min_G2, G2[k]);
        max_G2 = std::max(max_G2, G2[k]);
    }
    const double max_G_norm = std::sqrt(max_G2);

    std::vector<std::complex<double>> bloch_phase(n_g);
    for (std::size_t g = 0; g < n_g; ++g) {
        const double kr =
              k_cart(0) * R_g_list(static_cast<Eigen::Index>(g), 0)
            + k_cart(1) * R_g_list(static_cast<Eigen::Index>(g), 1)
            + k_cart(2) * R_g_list(static_cast<Eigen::Index>(g), 2);
        bloch_phase[g] = std::complex<double>(std::cos(kr), std::sin(kr));
    }

    // The mirror identity FT_mu,nu(G) == FT_nu,mu(G) needs THREE
    // conditions: k == 0 (no Bloch phase), an inversion-symmetric cell
    // list (the r -> r+R, R -> -R substitution), AND exp(i G.R) == 1 for
    // every eval frequency (the substitution leaves an exp(i G.R) phase
    // on each R != 0 term). The last holds for the plain reciprocal-mesh
    // Gamma GDF tail this shortcut was built for, but NOT for the
    // momentum-shifted meshes G+q of the Bloch-pair GDF fit
    // (build_lpq_bloch_native_fft calls with k_cart = k_ket = 0 for every
    // (k_bra != Gamma, k_ket = Gamma) exchange pair while the mesh is
    // shifted by q = -k_bra): mirroring there corrupted the multi-k
    // exchange by 2.0e-2 Ha/cell on LiH/sto-3g (2,1,1) (2026-07-10,
    // handovers/HANDOVER_AICCM_DIRECT_TORUS.md Finding 2 addendum).
    const bool gamma_pair_symmetric =
        k_cart.squaredNorm() <= 1.0e-28
        && cell_list_is_inversion_symmetric(R_g_list)
        && eval_frequencies_are_lattice_reciprocal(G_vectors, R_g_list);

    // Parallelize over (shell pair, G-block). At Gamma with an inversion-
    // symmetric cell list, FT_mu,nu(G) == FT_nu,mu(G), so the high-|G| GDF
    // tail computes only the upper shell-pair triangle and mirrors it.
    //
    // Shell pairs alone is far too coarse: LiH/STO-3G has 4 shells = 16
    // tasks of very uneven cost, and this kernel measured flat from 4
    // threads upward (3.32 s -> 1.27 s, then nothing) because the critical
    // path is a single long task. The cell loop below is a Bloch SUM and
    // cannot be split without a reduction, but the output tensor is
    // indexed by G, so a G-block is an exclusively-owned output slice.
    // Each output element's accumulation stays entirely inside one task in
    // the original order, so this is bit-identical, not merely
    // reassociation-equivalent.
    std::size_t g_block = n_G;
    {
        int n_workers = 1;
#ifdef _OPENMP
        n_workers = std::max(1, omp_get_max_threads());
#endif
        if (n_workers > 1) {
            const std::size_t target_blocks =
                static_cast<std::size_t>(n_workers) * 4u;
            g_block = std::max<std::size_t>(
                64u, (n_G + target_blocks - 1u) / target_blocks);
        }
    }
    const std::size_t n_g_blocks = (n_G + g_block - 1u) / g_block;
    const std::ptrdiff_t n_tasks =
        static_cast<std::ptrdiff_t>(n_shells * n_shells * n_g_blocks);

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t task = 0; task < n_tasks; ++task) {
        {
            const std::size_t blk =
                static_cast<std::size_t>(task) % n_g_blocks;
            const std::size_t pair_index =
                static_cast<std::size_t>(task) / n_g_blocks;
            const std::ptrdiff_t sM =
                static_cast<std::ptrdiff_t>(pair_index / n_shells);
            const std::ptrdiff_t sN =
                static_cast<std::ptrdiff_t>(pair_index % n_shells);
            if (gamma_pair_symmetric && sN < sM) {
                continue;
            }
            // This task owns G indices [g_lo, g_lo + n_G_blk).
            const std::size_t g_lo = blk * g_block;
            const std::size_t n_G_blk = std::min(g_block, n_G - g_lo);
            const ShellInfo& shM = shells[static_cast<std::size_t>(sM)];
            const ShellInfo& shN = shells[static_cast<std::size_t>(sN)];
            const int lM = shM.l;
            const int lN = shN.l;
            const std::size_t bfM = bf_offsets[static_cast<std::size_t>(sM)];
            const std::size_t bfN = bf_offsets[static_cast<std::size_t>(sN)];

            const std::size_t n_cart_M = cart_to_sph_data::n_cart_for_l(lM);
            const std::size_t n_cart_N = cart_to_sph_data::n_cart_for_l(lN);
            const std::size_t n_sph_M = cart_to_sph_data::n_sph_for_l(lM);
            const std::size_t n_sph_N = cart_to_sph_data::n_sph_for_l(lN);
            const cart_to_sph_data::CartIdx* cart_M =
                cart_to_sph_data::cart_table_for_l(lM);
            const cart_to_sph_data::CartIdx* cart_N =
                cart_to_sph_data::cart_table_for_l(lN);
            const double* Csph_M = cart_to_sph_data::sph_table_for_l(lM);
            const double* Csph_N = cart_to_sph_data::sph_table_for_l(lN);

            const double Ax = shM.origin[0], Ay = shM.origin[1], Az = shM.origin[2];
            const double Bx0 = shN.origin[0], By0 = shN.origin[1], Bz0 = shN.origin[2];

            const auto& es_M = shM.exponents;
            const auto& cs_M = shM.coefficients;
            const auto& es_N = shN.exponents;
            const auto& cs_N = shN.coefficients;
            const std::size_t nM = es_M.size();
            const std::size_t nN = es_N.size();

            // Per-axis max power = lM + lN; precompute (-iGq)^t for q ∈ {x,y,z}
            // and t ∈ [0, t_max_axis]. Shape (t_max_axis + 1, n_G_blk) --
            // this task's G slice only.
            const int t_max_axis = lM + lN;
            const std::size_t T1 = static_cast<std::size_t>(t_max_axis + 1);
            std::vector<std::complex<double>> Gx_pow(T1 * n_G_blk);
            std::vector<std::complex<double>> Gy_pow(T1 * n_G_blk);
            std::vector<std::complex<double>> Gz_pow(T1 * n_G_blk);
            for (std::size_t k = 0; k < n_G_blk; ++k) {
                const auto kg = static_cast<Eigen::Index>(k + g_lo);
                std::complex<double> ix(0.0, -G_vectors(kg, 0));
                std::complex<double> iy(0.0, -G_vectors(kg, 1));
                std::complex<double> iz(0.0, -G_vectors(kg, 2));
                std::complex<double> px(1.0, 0.0), py(1.0, 0.0), pz(1.0, 0.0);
                for (std::size_t t = 0; t < T1; ++t) {
                    Gx_pow[t * n_G_blk + k] = px; px *= ix;
                    Gy_pow[t * n_G_blk + k] = py; py *= iy;
                    Gz_pow[t * n_G_blk + k] = pz; pz *= iz;
                }
            }

            // Cartesian accumulator for the current (sM, sN, R_g):
            // shape (n_cart_M, n_cart_N, n_G_blk). Re-zeroed per cell.
            std::vector<std::complex<double>> ft_cart_g(
                n_cart_M * n_cart_N * n_G_blk, std::complex<double>(0.0, 0.0));
            // Spherical accumulator over all cells.
            std::vector<std::complex<double>> ft_sph_block(
                n_sph_M * n_sph_N * n_G_blk, std::complex<double>(0.0, 0.0));

            // Scratch for the MD coefficients per primitive pair.
            std::vector<double> Ex, Ey, Ez;
            // Scratch for the per-G "prefactor times exp(-iG·P)" line.
            std::vector<std::complex<double>> prefactor_per_G(n_G_blk);

            for (std::size_t g = 0; g < n_g; ++g) {
                const double Rx = R_g_list(static_cast<Eigen::Index>(g), 0);
                const double Ry = R_g_list(static_cast<Eigen::Index>(g), 1);
                const double Rz = R_g_list(static_cast<Eigen::Index>(g), 2);
                const double Bx = Bx0 + Rx;
                const double By = By0 + Ry;
                const double Bz = Bz0 + Rz;
                const double ABx = Ax - Bx;
                const double ABy = Ay - By;
                const double ABz = Az - Bz;
                const double AB_sq = ABx * ABx + ABy * ABy + ABz * ABz;
                const std::complex<double> phase_R = bloch_phase[g];
                if (screen_tol > 0.0) {
                    const double bound = shell_pair_cell_ft_bound(
                        shM, shN, Bx, By, Bz, min_G2, max_G_norm);
                    if (bound < screen_tol) {
                        continue;
                    }
                }

                // Zero per-cell Cartesian accumulator.
                std::fill(ft_cart_g.begin(), ft_cart_g.end(),
                          std::complex<double>(0.0, 0.0));

                for (std::size_t ip = 0; ip < nM; ++ip) {
                    const double alpha = es_M[ip];
                    const double c_a = cs_M[ip];
                    for (std::size_t iq = 0; iq < nN; ++iq) {
                        const double beta = es_N[iq];
                        const double c_b = cs_N[iq];
                        const double cc = c_a * c_b;
                        const double gamma = alpha + beta;
                        const double inv_gamma = 1.0 / gamma;
                        const double K_overlap = std::exp(
                            -alpha * beta * inv_gamma * AB_sq);
                        const double radial_prefactor = cc * K_overlap *
                            std::pow(M_PI * inv_gamma, 1.5);
                        const double Px = (alpha * Ax + beta * Bx) * inv_gamma;
                        const double Py = (alpha * Ay + beta * By) * inv_gamma;
                        const double Pz = (alpha * Az + beta * Bz) * inv_gamma;
                        const double PAx = Px - Ax;
                        const double PAy = Py - Ay;
                        const double PAz = Pz - Az;
                        const double PBx = Px - Bx;
                        const double PBy = Py - By;
                        const double PBz = Pz - Bz;

                        // McMurchie-Davidson coefficients per axis,
                        // built once per primitive pair.
                        md_e_coefficients_1d(lM, lN, gamma, PAx, PBx, Ex);
                        md_e_coefficients_1d(lM, lN, gamma, PAy, PBy, Ey);
                        md_e_coefficients_1d(lM, lN, gamma, PAz, PBz, Ez);

                        // E layout: row-major ((lM+1) * (lN+1)) * (lM+lN+1).
                        const std::size_t e_dim_t =
                            static_cast<std::size_t>(t_max_axis + 1);
                        const std::size_t e_dim_lb =
                            static_cast<std::size_t>(lN + 1);
                        const auto e_idx = [&](int i, int j, int t) {
                            return (static_cast<std::size_t>(i) * e_dim_lb
                                    + static_cast<std::size_t>(j)) * e_dim_t
                                   + static_cast<std::size_t>(t);
                        };

                        // Per-G radial-and-phase prefactor — the
                        // K * (π/γ)^{3/2} * exp(-G²/4γ) * exp(-iG·P)
                        // line, vectorised over G.
                        const double quarter_inv_gamma = 0.25 * inv_gamma;
                        for (std::size_t k = 0; k < n_G_blk; ++k) {
                            const auto kg =
                                static_cast<Eigen::Index>(k + g_lo);
                            const double gx = G_vectors(kg, 0);
                            const double gy = G_vectors(kg, 1);
                            const double gz = G_vectors(kg, 2);
                            const double radial =
                                std::exp(-G2[k + g_lo] * quarter_inv_gamma);
                            const double gp = gx * Px + gy * Py + gz * Pz;
                            const std::complex<double> phase_G(
                                std::cos(gp), -std::sin(gp));  // exp(-iG·P)
                            prefactor_per_G[k] =
                                radial_prefactor * radial * phase_G;
                        }

                        // Walk the Cartesian-component grid and
                        // accumulate into ft_cart_g.
                        for (std::size_t a = 0; a < n_cart_M; ++a) {
                            const int ixA = cart_M[a].i;
                            const int jyA = cart_M[a].j;
                            const int kzA = cart_M[a].k;
                            for (std::size_t b = 0; b < n_cart_N; ++b) {
                                const int ixB = cart_N[b].i;
                                const int jyB = cart_N[b].j;
                                const int kzB = cart_N[b].k;
                                const int Tmax = ixA + ixB;
                                const int Umax = jyA + jyB;
                                const int Vmax = kzA + kzB;

                                // Tight per-G loop:
                                //   poly_ft(G) = Σ_{tuv} E_x[ixA,ixB,t]
                                //                       · E_y[jyA,jyB,u]
                                //                       · E_z[kzA,kzB,v]
                                //                       · Gx^t Gy^u Gz^v
                                // followed by accumulating into
                                // ft_cart_g[a, b, k].
                                //
                                // Inner re-association: hoist t over
                                // (u, v), and u over v, so we only pay
                                // one complex-mul-by-precomputed-power
                                // per (t, u, v, k) iteration.
                                const std::size_t cart_ab_base =
                                    (a * n_cart_N + b) * n_G_blk;
                                for (int t = 0; t <= Tmax; ++t) {
                                    const double et = Ex[e_idx(ixA, ixB, t)];
                                    if (et == 0.0) {
                                        continue;
                                    }
                                    for (int u = 0; u <= Umax; ++u) {
                                        const double eu = Ey[e_idx(jyA, jyB, u)];
                                        if (eu == 0.0) {
                                            continue;
                                        }
                                        const double etu = et * eu;
                                        for (int v = 0; v <= Vmax; ++v) {
                                            const double ev =
                                                Ez[e_idx(kzA, kzB, v)];
                                            if (ev == 0.0) {
                                                continue;
                                            }
                                            const double etuv = etu * ev;
                                            const std::complex<double>* gx_t =
                                                &Gx_pow[static_cast<std::size_t>(t) * n_G_blk];
                                            const std::complex<double>* gy_u =
                                                &Gy_pow[static_cast<std::size_t>(u) * n_G_blk];
                                            const std::complex<double>* gz_v =
                                                &Gz_pow[static_cast<std::size_t>(v) * n_G_blk];
                                            for (std::size_t k = 0; k < n_G_blk; ++k) {
                                                const std::complex<double> pow_g =
                                                    gx_t[k] * gy_u[k] * gz_v[k];
                                                ft_cart_g[cart_ab_base + k] +=
                                                    prefactor_per_G[k] *
                                                    (etuv * pow_g);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }  // iq
                }  // ip

                // Apply Bloch phase and cart→sph transform on both
                // axes, accumulate into the per-shell-pair spherical
                // block.
                //
                //   ft_sph_block[m, n, :] += phase_R · Σ_ab
                //       Csph_M[m, a] · Csph_N[n, b] · ft_cart_g[a, b, :]
                for (std::size_t m = 0; m < n_sph_M; ++m) {
                    for (std::size_t n = 0; n < n_sph_N; ++n) {
                        const std::size_t sph_base =
                            (m * n_sph_N + n) * n_G_blk;
                        for (std::size_t a = 0; a < n_cart_M; ++a) {
                            const double cma = Csph_M[m * n_cart_M + a];
                            if (cma == 0.0) {
                                continue;
                            }
                            for (std::size_t b = 0; b < n_cart_N; ++b) {
                                const double cnb = Csph_N[n * n_cart_N + b];
                                if (cnb == 0.0) {
                                    continue;
                                }
                                const std::complex<double> weight =
                                    phase_R * (cma * cnb);
                                const std::size_t cart_ab_base =
                                    (a * n_cart_N + b) * n_G_blk;
                                for (std::size_t k = 0; k < n_G_blk; ++k) {
                                    ft_sph_block[sph_base + k] +=
                                        weight * ft_cart_g[cart_ab_base + k];
                                }
                            }
                        }
                    }
                }
            }  // g

            // Drop the spherical block into the dense output.
            for (std::size_t m = 0; m < n_sph_M; ++m) {
                for (std::size_t n = 0; n < n_sph_N; ++n) {
                    const std::size_t out_base =
                        ((bfM + m) * n_orb + (bfN + n)) * n_G + g_lo;
                    const std::size_t sph_base =
                        (m * n_sph_N + n) * n_G_blk;
                    for (std::size_t k = 0; k < n_G_blk; ++k) {
                        out.data[out_base + k] = ft_sph_block[sph_base + k];
                    }
                }
            }
            if (gamma_pair_symmetric && sM != sN) {
                for (std::size_t m = 0; m < n_sph_M; ++m) {
                    for (std::size_t n = 0; n < n_sph_N; ++n) {
                        const std::size_t out_base =
                            ((bfN + n) * n_orb + (bfM + m)) * n_G + g_lo;
                        const std::size_t sph_base =
                            (m * n_sph_N + n) * n_G_blk;
                        for (std::size_t k = 0; k < n_G_blk; ++k) {
                            out.data[out_base + k] =
                                ft_sph_block[sph_base + k];
                        }
                    }
                }
            }
        }
    }  // (shell pair, G-block) task

    return out;
}

std::vector<double> ao_pair_fourier_transform_weighted_per_cell(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Ref<const Eigen::VectorXcd>& reciprocal_weights,
    double screen_tol)
{
    // Per-cell reciprocal-weight-contracted AO-pair FT:
    //
    //   out[g][m,n] = Re( sum_G w(G) . conj( FT_mn(G; R_g) ) )
    //
    // This is the analytic reciprocal half of the Ewald-split periodic
    // nuclear attraction (Sun, Berkelbach, McClain & Chan, J. Chem. Phys.
    // 147, 164119 (2017), doi:10.1063/1.4998644; Lippert, Hutter &
    // Parrinello, Mol. Phys. 92, 477 (1997)) as consumed by
    // vibeqc.periodic_v_ne.compute_v_ne_ewald_3d_ft_lattice, where
    // w(G) = v_long(G).
    //
    // Two structural differences from the Bloch kernels above, both
    // deliberate:
    //
    //  * The weight contraction happens INSIDE the Hermite loop, so the
    //    dense (n_orb, n_orb, n_G) pair-FT tensor never materialises --
    //    the caller previously built it per cell only to contract it
    //    away. Because every remaining factor (the Hermite E
    //    coefficients, the cart-to-sph tables) is real, Re() commutes to
    //    the innermost reduction and the accumulator is a real scalar.
    //  * Output is PER CELL (no Bloch sum), so (shell-pair, cell) tasks
    //    write disjoint blocks and the whole flattened task space
    //    parallelises with no reduction. The Bloch kernels can only
    //    parallelise over shell pairs -- their cell loop is a sum -- which
    //    on a small basis with many image cells starves the threads
    //    (LiH/STO-3G, 4 shells = 16 shell pairs, 135 cells: the per-cell
    //    driver loop saturated at 2.97x on 8 threads).
    //
    // The Hermite (-iG)^t power tables depend only on G and the maximum
    // angular momentum, never on the cell or the shell pair, so one
    // read-only copy is shared by every worker rather than rebuilt per
    // task (same pattern as the gradient kernel below).
    const std::vector<ShellInfo> shells = basis.shells();
    const auto bf_offsets = shell_to_bf_offsets_pure(shells);

    const std::size_t n_orb = basis.nbasis();
    const std::size_t n_G = static_cast<std::size_t>(G_vectors.rows());
    const std::size_t n_g = static_cast<std::size_t>(R_g_list.rows());
    const std::size_t n_shells = shells.size();

    std::vector<double> out(n_g * n_orb * n_orb, 0.0);
    if (n_G == 0 || n_g == 0 || n_orb == 0) {
        return out;
    }
    if (reciprocal_weights.size() != static_cast<Eigen::Index>(n_G)) {
        throw std::invalid_argument(
            "ao_pair_fourier_transform_weighted_per_cell: "
            "reciprocal_weights must have shape (n_G,)");
    }

    std::vector<double> G2(n_G);
    double min_G2 = std::numeric_limits<double>::infinity();
    double max_G2 = 0.0;
    for (std::size_t k = 0; k < n_G; ++k) {
        const double gx = G_vectors(static_cast<Eigen::Index>(k), 0);
        const double gy = G_vectors(static_cast<Eigen::Index>(k), 1);
        const double gz = G_vectors(static_cast<Eigen::Index>(k), 2);
        G2[k] = gx * gx + gy * gy + gz * gz;
        min_G2 = std::min(min_G2, G2[k]);
        max_G2 = std::max(max_G2, G2[k]);
    }
    const double max_G_norm = std::sqrt(max_G2);

    int max_l = 0;
    for (const auto& shell : shells) {
        max_l = std::max(max_l, shell.l);
    }
    const std::size_t n_powers = static_cast<std::size_t>(2 * max_l + 1);
    std::vector<std::complex<double>> Gx_pow(n_powers * n_G);
    std::vector<std::complex<double>> Gy_pow(n_powers * n_G);
    std::vector<std::complex<double>> Gz_pow(n_powers * n_G);
    for (std::size_t k = 0; k < n_G; ++k) {
        const std::complex<double> ix(
            0.0, -G_vectors(static_cast<Eigen::Index>(k), 0));
        const std::complex<double> iy(
            0.0, -G_vectors(static_cast<Eigen::Index>(k), 1));
        const std::complex<double> iz(
            0.0, -G_vectors(static_cast<Eigen::Index>(k), 2));
        std::complex<double> px(1.0, 0.0), py(1.0, 0.0), pz(1.0, 0.0);
        for (std::size_t t = 0; t < n_powers; ++t) {
            Gx_pow[t * n_G + k] = px; px *= ix;
            Gy_pow[t * n_G + k] = py; py *= iy;
            Gz_pow[t * n_G + k] = pz; pz *= iz;
        }
    }

    const std::ptrdiff_t n_tasks =
        static_cast<std::ptrdiff_t>(n_shells * n_shells * n_g);

#pragma omp parallel
    {
        std::vector<double> Ex, Ey, Ez;
        std::vector<std::complex<double>> prefactor_per_G(n_G);
        // (-iGx)^t (-iGy)^u hoisted out of the v loop. The old per-cell
        // path evaluated gx*gy*gz left-to-right, so caching (gx*gy) is
        // the identical association -- no numerical change, one complex
        // multiply saved per (v, G).
        std::vector<std::complex<double>> gxy_scratch(n_G);
        std::vector<double> cart_acc;

#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t task = 0; task < n_tasks; ++task) {
            const std::size_t g =
                static_cast<std::size_t>(task) % n_g;
            const std::size_t pair_index =
                static_cast<std::size_t>(task) / n_g;
            const std::size_t sM = pair_index / n_shells;
            const std::size_t sN = pair_index % n_shells;

            const ShellInfo& shM = shells[sM];
            const ShellInfo& shN = shells[sN];
            const int lM = shM.l;
            const int lN = shN.l;
            const std::size_t bfM = bf_offsets[sM];
            const std::size_t bfN = bf_offsets[sN];

            const std::size_t n_cart_M = cart_to_sph_data::n_cart_for_l(lM);
            const std::size_t n_cart_N = cart_to_sph_data::n_cart_for_l(lN);
            const std::size_t n_sph_M = cart_to_sph_data::n_sph_for_l(lM);
            const std::size_t n_sph_N = cart_to_sph_data::n_sph_for_l(lN);
            const cart_to_sph_data::CartIdx* cart_M =
                cart_to_sph_data::cart_table_for_l(lM);
            const cart_to_sph_data::CartIdx* cart_N =
                cart_to_sph_data::cart_table_for_l(lN);
            const double* Csph_M = cart_to_sph_data::sph_table_for_l(lM);
            const double* Csph_N = cart_to_sph_data::sph_table_for_l(lN);

            const double Ax = shM.origin[0];
            const double Ay = shM.origin[1];
            const double Az = shM.origin[2];
            const double Bx =
                shN.origin[0] + R_g_list(static_cast<Eigen::Index>(g), 0);
            const double By =
                shN.origin[1] + R_g_list(static_cast<Eigen::Index>(g), 1);
            const double Bz =
                shN.origin[2] + R_g_list(static_cast<Eigen::Index>(g), 2);

            if (screen_tol > 0.0) {
                const double bound = shell_pair_cell_ft_bound(
                    shM, shN, Bx, By, Bz, min_G2, max_G_norm);
                if (bound < screen_tol) {
                    continue;
                }
            }

            const double ABx = Ax - Bx;
            const double ABy = Ay - By;
            const double ABz = Az - Bz;
            const double AB_sq = ABx * ABx + ABy * ABy + ABz * ABz;

            cart_acc.assign(n_cart_M * n_cart_N, 0.0);

            const int t_max_axis = lM + lN;
            const std::size_t e_dim_t =
                static_cast<std::size_t>(t_max_axis + 1);
            const std::size_t e_dim_lb = static_cast<std::size_t>(lN + 1);
            const auto e_idx = [&](int i, int j, int t) {
                return (static_cast<std::size_t>(i) * e_dim_lb
                        + static_cast<std::size_t>(j)) * e_dim_t
                       + static_cast<std::size_t>(t);
            };

            for (std::size_t ip = 0; ip < shM.exponents.size(); ++ip) {
                const double alpha = shM.exponents[ip];
                const double c_a = shM.coefficients[ip];
                for (std::size_t iq = 0; iq < shN.exponents.size(); ++iq) {
                    const double beta = shN.exponents[iq];
                    const double cc = c_a * shN.coefficients[iq];
                    const double gamma = alpha + beta;
                    const double inv_gamma = 1.0 / gamma;
                    const double radial_prefactor =
                        cc * std::exp(-alpha * beta * inv_gamma * AB_sq)
                        * std::pow(M_PI * inv_gamma, 1.5);
                    const double Px = (alpha * Ax + beta * Bx) * inv_gamma;
                    const double Py = (alpha * Ay + beta * By) * inv_gamma;
                    const double Pz = (alpha * Az + beta * Bz) * inv_gamma;

                    md_e_coefficients_1d(lM, lN, gamma, Px - Ax, Px - Bx, Ex);
                    md_e_coefficients_1d(lM, lN, gamma, Py - Ay, Py - By, Ey);
                    md_e_coefficients_1d(lM, lN, gamma, Pz - Az, Pz - Bz, Ez);

                    // w(G) . conj(radial . exp(-iG.P)) folded once per
                    // primitive pair; the remaining conj(pow) factor is
                    // applied inside the (t,u,v) reduction below.
                    const double quarter_inv_gamma = 0.25 * inv_gamma;
                    for (std::size_t k = 0; k < n_G; ++k) {
                        const double gx =
                            G_vectors(static_cast<Eigen::Index>(k), 0);
                        const double gy =
                            G_vectors(static_cast<Eigen::Index>(k), 1);
                        const double gz =
                            G_vectors(static_cast<Eigen::Index>(k), 2);
                        const double radial =
                            std::exp(-G2[k] * quarter_inv_gamma);
                        const double gp = gx * Px + gy * Py + gz * Pz;
                        // conj(exp(-iG.P)) = exp(+iG.P)
                        const std::complex<double> conj_phase(
                            std::cos(gp), std::sin(gp));
                        prefactor_per_G[k] =
                            reciprocal_weights(static_cast<Eigen::Index>(k))
                            * (radial_prefactor * radial) * conj_phase;
                    }

                    for (std::size_t a = 0; a < n_cart_M; ++a) {
                        const int ixA = cart_M[a].i;
                        const int jyA = cart_M[a].j;
                        const int kzA = cart_M[a].k;
                        for (std::size_t b = 0; b < n_cart_N; ++b) {
                            const int ixB = cart_N[b].i;
                            const int jyB = cart_N[b].j;
                            const int kzB = cart_N[b].k;
                            double acc = 0.0;
                            for (int t = 0; t <= ixA + ixB; ++t) {
                                const double et = Ex[e_idx(ixA, ixB, t)];
                                if (et == 0.0) {
                                    continue;
                                }
                                const std::complex<double>* gx_t =
                                    &Gx_pow[static_cast<std::size_t>(t) * n_G];
                                for (int u = 0; u <= jyA + jyB; ++u) {
                                    const double eu = Ey[e_idx(jyA, jyB, u)];
                                    if (eu == 0.0) {
                                        continue;
                                    }
                                    const std::complex<double>* gy_u =
                                        &Gy_pow[static_cast<std::size_t>(u)
                                                * n_G];
                                    const double etu = et * eu;
                                    const int Vmax = kzA + kzB;
                                    // Caching (gx*gy) only pays when the v
                                    // loop reuses it; at Vmax == 0 the
                                    // single pass would write and re-read
                                    // n_G complex for nothing.
                                    if (Vmax > 0) {
                                        for (std::size_t k = 0; k < n_G; ++k) {
                                            gxy_scratch[k] =
                                                gx_t[k] * gy_u[k];
                                        }
                                    }
                                    for (int v = 0; v <= Vmax; ++v) {
                                        const double ev =
                                            Ez[e_idx(kzA, kzB, v)];
                                        if (ev == 0.0) {
                                            continue;
                                        }
                                        const double etuv = etu * ev;
                                        // Re(w . conj(pw)) written as the
                                        // real dot product: the complex
                                        // product would compute an
                                        // imaginary part we discard.
                                        // Re() commutes out to here because
                                        // every outer factor is real.
                                        double s = 0.0;
                                        if (Vmax == 0) {
                                            // (-iGz)^0 == 1 exactly: fold it
                                            // out rather than multiply by it.
                                            for (std::size_t k = 0; k < n_G;
                                                 ++k) {
                                                const std::complex<double> pw =
                                                    gx_t[k] * gy_u[k];
                                                const std::complex<double>& wq =
                                                    prefactor_per_G[k];
                                                s += wq.real() * pw.real()
                                                     + wq.imag() * pw.imag();
                                            }
                                        } else {
                                            const std::complex<double>* gz_v =
                                                &Gz_pow[
                                                    static_cast<std::size_t>(v)
                                                    * n_G];
                                            for (std::size_t k = 0; k < n_G;
                                                 ++k) {
                                                const std::complex<double> pw =
                                                    gxy_scratch[k] * gz_v[k];
                                                const std::complex<double>& wq =
                                                    prefactor_per_G[k];
                                                s += wq.real() * pw.real()
                                                     + wq.imag() * pw.imag();
                                            }
                                        }
                                        acc += etuv * s;
                                    }
                                }
                            }
                            cart_acc[a * n_cart_N + b] += acc;
                        }
                    }
                }  // iq
            }  // ip

            double* out_cell = &out[g * n_orb * n_orb];
            for (std::size_t m = 0; m < n_sph_M; ++m) {
                for (std::size_t n = 0; n < n_sph_N; ++n) {
                    double value = 0.0;
                    for (std::size_t a = 0; a < n_cart_M; ++a) {
                        const double cma = Csph_M[m * n_cart_M + a];
                        if (cma == 0.0) {
                            continue;
                        }
                        for (std::size_t b = 0; b < n_cart_N; ++b) {
                            const double cnb = Csph_N[n * n_cart_N + b];
                            if (cnb == 0.0) {
                                continue;
                            }
                            value += cma * cnb * cart_acc[a * n_cart_N + b];
                        }
                    }
                    out_cell[(bfM + m) * n_orb + (bfN + n)] = value;
                }
            }
        }
    }

    return out;
}

std::vector<AOPairFTTensor> ao_pair_fourier_transform_bloch_multi(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& k_carts,
    double screen_tol,
    const std::vector<double>& pair_weights)
{
    // Multi-k batched variant of ao_pair_fourier_transform_bloch: the
    // per-cell Cartesian McMurchie-Davidson pair-FT work (the dominant
    // transcendental cost — Eq. cderi pair density of Sun, Berkelbach,
    // McClain & Chan, J. Chem. Phys. 147, 164119 (2017),
    // doi:10.1063/1.4998644) is k-INDEPENDENT; only the Bloch phase
    // exp(+i k·R) differs between ket momenta. This kernel computes the
    // per-cell Cartesian block once and folds it into every requested
    // k with its own phase, so n_k Bloch FTs on one reciprocal support
    // cost one pair-FT pass plus n_k cheap folds. Numerical parity
    // with n_k separate single-k calls is pinned by
    // tests/test_aopair_ft_parity.py (fold order differs only in
    // floating-point rounding, ~1e-15 relative).
    const std::vector<ShellInfo> shells = basis.shells();
    const auto bf_offsets = shell_to_bf_offsets_pure(shells);

    const std::size_t n_orb = basis.nbasis();
    const std::size_t n_G = static_cast<std::size_t>(G_vectors.rows());
    const std::size_t n_g = static_cast<std::size_t>(R_g_list.rows());
    const std::size_t n_shells = shells.size();
    const std::size_t n_k = static_cast<std::size_t>(k_carts.rows());

    // Optional per-AO-pair store weight (see the header note): empty
    // leaves every store byte-unchanged.
    const bool weighted = !pair_weights.empty();
    if (weighted && pair_weights.size() != n_orb * n_orb) {
        throw std::invalid_argument(
            "ao_pair_fourier_transform_bloch_multi: pair_weights must "
            "have n_orb * n_orb entries");
    }
    const double* pair_w = weighted ? pair_weights.data() : nullptr;

    std::vector<AOPairFTTensor> outs(n_k);
    for (std::size_t ik = 0; ik < n_k; ++ik) {
        outs[ik].n_orb = n_orb;
        outs[ik].n_G = n_G;
        outs[ik].data.assign(n_orb * n_orb * n_G,
                             std::complex<double>(0.0, 0.0));
    }
    if (n_G == 0 || n_g == 0 || n_orb == 0 || n_k == 0) {
        return outs;
    }

    std::vector<double> G2(n_G);
    double min_G2 = std::numeric_limits<double>::infinity();
    double max_G2 = 0.0;
    for (std::size_t k = 0; k < n_G; ++k) {
        const double gx = G_vectors(static_cast<Eigen::Index>(k), 0);
        const double gy = G_vectors(static_cast<Eigen::Index>(k), 1);
        const double gz = G_vectors(static_cast<Eigen::Index>(k), 2);
        G2[k] = gx * gx + gy * gy + gz * gz;
        min_G2 = std::min(min_G2, G2[k]);
        max_G2 = std::max(max_G2, G2[k]);
    }
    const double max_G_norm = std::sqrt(max_G2);

    // Bloch phases exp(+i k·R) per (k, cell), flat (n_k, n_g).
    std::vector<std::complex<double>> bloch_phase(n_k * n_g);
    bool all_k_zero = true;
    for (std::size_t ik = 0; ik < n_k; ++ik) {
        const double kx = k_carts(static_cast<Eigen::Index>(ik), 0);
        const double ky = k_carts(static_cast<Eigen::Index>(ik), 1);
        const double kz = k_carts(static_cast<Eigen::Index>(ik), 2);
        if (kx * kx + ky * ky + kz * kz > 1.0e-28) {
            all_k_zero = false;
        }
        for (std::size_t g = 0; g < n_g; ++g) {
            const double kr =
                  kx * R_g_list(static_cast<Eigen::Index>(g), 0)
                + ky * R_g_list(static_cast<Eigen::Index>(g), 1)
                + kz * R_g_list(static_cast<Eigen::Index>(g), 2);
            bloch_phase[ik * n_g + g] =
                std::complex<double>(std::cos(kr), std::sin(kr));
        }
    }

    // The mirror shortcut needs every ket momentum at Gamma (see the
    // three-condition note in the single-k kernel above).
    const bool gamma_pair_symmetric =
        all_k_zero
        && cell_list_is_inversion_symmetric(R_g_list)
        && eval_frequencies_are_lattice_reciprocal(G_vectors, R_g_list);

    // Task space is (shell pair, G-block), not shell pairs alone. The
    // cell loop below is a Bloch SUM, so cells cannot be handed to
    // separate workers without a reduction -- but the output tensor is
    // indexed by G, so a G-block is an exclusively-owned output slice
    // and costs no reduction at all. Shell pairs alone is far too coarse
    // for a small basis: LiH/STO-3G has 4 shells = 16 tasks of very
    // uneven cost (an (l=1,l=1) pair dwarfs (s,s)), which measured flat
    // from 4 threads upward -- the critical path is one long task, so
    // extra cores bought nothing.
    std::size_t g_block = n_G;
    {
        int n_workers = 1;
#ifdef _OPENMP
        n_workers = std::max(1, omp_get_max_threads());
#endif
        if (n_workers > 1) {
            // ~4 blocks per worker so dynamic scheduling can even out the
            // per-shell-pair cost spread; floored so each block still
            // amortises its own (-iG)^t table build and vectorises well.
            const std::size_t target_blocks =
                static_cast<std::size_t>(n_workers) * 4u;
            g_block = std::max<std::size_t>(
                64u, (n_G + target_blocks - 1u) / target_blocks);
        }
    }
    const std::size_t n_g_blocks = (n_G + g_block - 1u) / g_block;
    const std::ptrdiff_t n_tasks =
        static_cast<std::ptrdiff_t>(n_shells * n_shells * n_g_blocks);

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t task = 0; task < n_tasks; ++task) {
        {
            const std::size_t blk =
                static_cast<std::size_t>(task) % n_g_blocks;
            const std::size_t pair_index =
                static_cast<std::size_t>(task) / n_g_blocks;
            const std::ptrdiff_t sM =
                static_cast<std::ptrdiff_t>(pair_index / n_shells);
            const std::ptrdiff_t sN =
                static_cast<std::ptrdiff_t>(pair_index % n_shells);
            if (gamma_pair_symmetric && sN < sM) {
                continue;
            }
            // This task owns G indices [g_lo, g_lo + n_G_blk).
            const std::size_t g_lo = blk * g_block;
            const std::size_t n_G_blk = std::min(g_block, n_G - g_lo);
            const ShellInfo& shM = shells[static_cast<std::size_t>(sM)];
            const ShellInfo& shN = shells[static_cast<std::size_t>(sN)];
            const int lM = shM.l;
            const int lN = shN.l;
            const std::size_t bfM = bf_offsets[static_cast<std::size_t>(sM)];
            const std::size_t bfN = bf_offsets[static_cast<std::size_t>(sN)];

            const std::size_t n_cart_M = cart_to_sph_data::n_cart_for_l(lM);
            const std::size_t n_cart_N = cart_to_sph_data::n_cart_for_l(lN);
            const std::size_t n_sph_M = cart_to_sph_data::n_sph_for_l(lM);
            const std::size_t n_sph_N = cart_to_sph_data::n_sph_for_l(lN);
            const cart_to_sph_data::CartIdx* cart_M =
                cart_to_sph_data::cart_table_for_l(lM);
            const cart_to_sph_data::CartIdx* cart_N =
                cart_to_sph_data::cart_table_for_l(lN);
            const double* Csph_M = cart_to_sph_data::sph_table_for_l(lM);
            const double* Csph_N = cart_to_sph_data::sph_table_for_l(lN);

            const double Ax = shM.origin[0], Ay = shM.origin[1], Az = shM.origin[2];
            const double Bx0 = shN.origin[0], By0 = shN.origin[1], Bz0 = shN.origin[2];

            const auto& es_M = shM.exponents;
            const auto& cs_M = shM.coefficients;
            const auto& es_N = shN.exponents;
            const auto& cs_N = shN.coefficients;
            const std::size_t nM = es_M.size();
            const std::size_t nN = es_N.size();

            const int t_max_axis = lM + lN;
            const std::size_t T1 = static_cast<std::size_t>(t_max_axis + 1);
            std::vector<std::complex<double>> Gx_pow(T1 * n_G_blk);
            std::vector<std::complex<double>> Gy_pow(T1 * n_G_blk);
            std::vector<std::complex<double>> Gz_pow(T1 * n_G_blk);
            for (std::size_t k = 0; k < n_G_blk; ++k) {
                const auto kg = static_cast<Eigen::Index>(k + g_lo);
                std::complex<double> ix(0.0, -G_vectors(kg, 0));
                std::complex<double> iy(0.0, -G_vectors(kg, 1));
                std::complex<double> iz(0.0, -G_vectors(kg, 2));
                std::complex<double> px(1.0, 0.0), py(1.0, 0.0), pz(1.0, 0.0);
                for (std::size_t t = 0; t < T1; ++t) {
                    Gx_pow[t * n_G_blk + k] = px; px *= ix;
                    Gy_pow[t * n_G_blk + k] = py; py *= iy;
                    Gz_pow[t * n_G_blk + k] = pz; pz *= iz;
                }
            }

            // Per-cell Cartesian accumulator (k-independent, re-zeroed
            // per cell) and its phase-free spherical transform. All of
            // these are now block-sized rather than full-mesh, so the
            // per-worker footprint shrinks with the block count.
            std::vector<std::complex<double>> ft_cart_g(
                n_cart_M * n_cart_N * n_G_blk, std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> sph_cell(
                n_sph_M * n_sph_N * n_G_blk, std::complex<double>(0.0, 0.0));
            // Per-k spherical accumulators over all cells, flat
            // (n_k, n_sph_M * n_sph_N * n_G_blk).
            const std::size_t sph_sz = n_sph_M * n_sph_N * n_G_blk;
            std::vector<std::complex<double>> ft_sph_blocks(
                n_k * sph_sz, std::complex<double>(0.0, 0.0));

            std::vector<double> Ex, Ey, Ez;
            std::vector<std::complex<double>> prefactor_per_G(n_G_blk);

            for (std::size_t g = 0; g < n_g; ++g) {
                const double Rx = R_g_list(static_cast<Eigen::Index>(g), 0);
                const double Ry = R_g_list(static_cast<Eigen::Index>(g), 1);
                const double Rz = R_g_list(static_cast<Eigen::Index>(g), 2);
                const double Bx = Bx0 + Rx;
                const double By = By0 + Ry;
                const double Bz = Bz0 + Rz;
                const double ABx = Ax - Bx;
                const double ABy = Ay - By;
                const double ABz = Az - Bz;
                const double AB_sq = ABx * ABx + ABy * ABy + ABz * ABz;
                if (screen_tol > 0.0) {
                    const double bound = shell_pair_cell_ft_bound(
                        shM, shN, Bx, By, Bz, min_G2, max_G_norm);
                    if (bound < screen_tol) {
                        continue;
                    }
                }

                std::fill(ft_cart_g.begin(), ft_cart_g.end(),
                          std::complex<double>(0.0, 0.0));

                for (std::size_t ip = 0; ip < nM; ++ip) {
                    const double alpha = es_M[ip];
                    const double c_a = cs_M[ip];
                    for (std::size_t iq = 0; iq < nN; ++iq) {
                        const double beta = es_N[iq];
                        const double c_b = cs_N[iq];
                        const double cc = c_a * c_b;
                        const double gamma = alpha + beta;
                        const double inv_gamma = 1.0 / gamma;
                        const double K_overlap = std::exp(
                            -alpha * beta * inv_gamma * AB_sq);
                        const double radial_prefactor = cc * K_overlap *
                            std::pow(M_PI * inv_gamma, 1.5);
                        const double Px = (alpha * Ax + beta * Bx) * inv_gamma;
                        const double Py = (alpha * Ay + beta * By) * inv_gamma;
                        const double Pz = (alpha * Az + beta * Bz) * inv_gamma;
                        const double PAx = Px - Ax;
                        const double PAy = Py - Ay;
                        const double PAz = Pz - Az;
                        const double PBx = Px - Bx;
                        const double PBy = Py - By;
                        const double PBz = Pz - Bz;

                        md_e_coefficients_1d(lM, lN, gamma, PAx, PBx, Ex);
                        md_e_coefficients_1d(lM, lN, gamma, PAy, PBy, Ey);
                        md_e_coefficients_1d(lM, lN, gamma, PAz, PBz, Ez);

                        const std::size_t e_dim_t =
                            static_cast<std::size_t>(t_max_axis + 1);
                        const std::size_t e_dim_lb =
                            static_cast<std::size_t>(lN + 1);
                        const auto e_idx = [&](int i, int j, int t) {
                            return (static_cast<std::size_t>(i) * e_dim_lb
                                    + static_cast<std::size_t>(j)) * e_dim_t
                                   + static_cast<std::size_t>(t);
                        };

                        const double quarter_inv_gamma = 0.25 * inv_gamma;
                        for (std::size_t k = 0; k < n_G_blk; ++k) {
                            const auto kg =
                                static_cast<Eigen::Index>(k + g_lo);
                            const double gx = G_vectors(kg, 0);
                            const double gy = G_vectors(kg, 1);
                            const double gz = G_vectors(kg, 2);
                            const double radial =
                                std::exp(-G2[k + g_lo] * quarter_inv_gamma);
                            const double gp = gx * Px + gy * Py + gz * Pz;
                            const std::complex<double> phase_G(
                                std::cos(gp), -std::sin(gp));  // exp(-iG·P)
                            prefactor_per_G[k] =
                                radial_prefactor * radial * phase_G;
                        }

                        for (std::size_t a = 0; a < n_cart_M; ++a) {
                            const int ixA = cart_M[a].i;
                            const int jyA = cart_M[a].j;
                            const int kzA = cart_M[a].k;
                            for (std::size_t b = 0; b < n_cart_N; ++b) {
                                const int ixB = cart_N[b].i;
                                const int jyB = cart_N[b].j;
                                const int kzB = cart_N[b].k;
                                const int Tmax = ixA + ixB;
                                const int Umax = jyA + jyB;
                                const int Vmax = kzA + kzB;
                                const std::size_t cart_ab_base =
                                    (a * n_cart_N + b) * n_G_blk;
                                for (int t = 0; t <= Tmax; ++t) {
                                    const double et = Ex[e_idx(ixA, ixB, t)];
                                    if (et == 0.0) {
                                        continue;
                                    }
                                    for (int u = 0; u <= Umax; ++u) {
                                        const double eu = Ey[e_idx(jyA, jyB, u)];
                                        if (eu == 0.0) {
                                            continue;
                                        }
                                        const double etu = et * eu;
                                        for (int v = 0; v <= Vmax; ++v) {
                                            const double ev =
                                                Ez[e_idx(kzA, kzB, v)];
                                            if (ev == 0.0) {
                                                continue;
                                            }
                                            const double etuv = etu * ev;
                                            const std::complex<double>* gx_t =
                                                &Gx_pow[static_cast<std::size_t>(t) * n_G_blk];
                                            const std::complex<double>* gy_u =
                                                &Gy_pow[static_cast<std::size_t>(u) * n_G_blk];
                                            const std::complex<double>* gz_v =
                                                &Gz_pow[static_cast<std::size_t>(v) * n_G_blk];
                                            for (std::size_t k = 0; k < n_G_blk; ++k) {
                                                const std::complex<double> pow_g =
                                                    gx_t[k] * gy_u[k] * gz_v[k];
                                                ft_cart_g[cart_ab_base + k] +=
                                                    prefactor_per_G[k] *
                                                    (etuv * pow_g);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }  // iq
                }  // ip

                // Phase-free cart→sph transform for this cell, then one
                // cheap phase-weighted axpy per requested k. This is the
                // whole point of the batch: everything above this line is
                // shared across the n_k ket momenta.
                std::fill(sph_cell.begin(), sph_cell.end(),
                          std::complex<double>(0.0, 0.0));
                for (std::size_t m = 0; m < n_sph_M; ++m) {
                    for (std::size_t n = 0; n < n_sph_N; ++n) {
                        const std::size_t sph_base =
                            (m * n_sph_N + n) * n_G_blk;
                        for (std::size_t a = 0; a < n_cart_M; ++a) {
                            const double cma = Csph_M[m * n_cart_M + a];
                            if (cma == 0.0) {
                                continue;
                            }
                            for (std::size_t b = 0; b < n_cart_N; ++b) {
                                const double cnb = Csph_N[n * n_cart_N + b];
                                if (cnb == 0.0) {
                                    continue;
                                }
                                const double weight = cma * cnb;
                                const std::size_t cart_ab_base =
                                    (a * n_cart_N + b) * n_G_blk;
                                for (std::size_t k = 0; k < n_G_blk; ++k) {
                                    sph_cell[sph_base + k] +=
                                        weight * ft_cart_g[cart_ab_base + k];
                                }
                            }
                        }
                    }
                }
                for (std::size_t ik = 0; ik < n_k; ++ik) {
                    const std::complex<double> phase_R =
                        bloch_phase[ik * n_g + g];
                    std::complex<double>* dst =
                        &ft_sph_blocks[ik * sph_sz];
                    for (std::size_t idx = 0; idx < sph_sz; ++idx) {
                        dst[idx] += phase_R * sph_cell[idx];
                    }
                }
            }  // g

            // out_base keeps the FULL-mesh stride (out.data is indexed
            // over all n_G); only the offset into it is block-local.
            //
            // The optional pair weight is folded into THIS store rather
            // than left to the caller: the caller's version is a NumPy
            // pass over the whole (n_k, n_orb, n_orb, n_G) tensor that
            // runs single-threaded while these workers saturate memory
            // bandwidth, whereas here it rides an existing store inside
            // the parallel region. `store_block` reproduces the NumPy
            // complex-times-real product exactly (header note).
            const auto store_block =
                [&](std::complex<double>* dst,
                    const std::complex<double>* src_block,
                    double w) {
                    if (!weighted) {
                        for (std::size_t k = 0; k < n_G_blk; ++k) {
                            dst[k] = src_block[k];
                        }
                    } else if (w == 0.0) {
                        // Masked pair: a true +0.0, matching the NumPy
                        // masked ASSIGNMENT this replaces (a multiply
                        // by 0.0 would keep the operand's zero sign).
                        for (std::size_t k = 0; k < n_G_blk; ++k) {
                            dst[k] = std::complex<double>(0.0, 0.0);
                        }
                    } else {
                        for (std::size_t k = 0; k < n_G_blk; ++k) {
                            const double vr = src_block[k].real();
                            const double vi = src_block[k].imag();
                            dst[k] = std::complex<double>(
                                vr * w - vi * 0.0, vr * 0.0 + vi * w);
                        }
                    }
                };

            for (std::size_t ik = 0; ik < n_k; ++ik) {
                const std::complex<double>* src =
                    &ft_sph_blocks[ik * sph_sz];
                auto& out = outs[ik];
                for (std::size_t m = 0; m < n_sph_M; ++m) {
                    for (std::size_t n = 0; n < n_sph_N; ++n) {
                        const std::size_t out_base =
                            ((bfM + m) * n_orb + (bfN + n)) * n_G + g_lo;
                        const std::size_t sph_base =
                            (m * n_sph_N + n) * n_G_blk;
                        store_block(
                            &out.data[out_base], &src[sph_base],
                            weighted
                                ? pair_w[(bfM + m) * n_orb + (bfN + n)]
                                : 0.0);
                    }
                }
                if (gamma_pair_symmetric && sM != sN) {
                    for (std::size_t m = 0; m < n_sph_M; ++m) {
                        for (std::size_t n = 0; n < n_sph_N; ++n) {
                            const std::size_t out_base =
                                ((bfN + n) * n_orb + (bfM + m)) * n_G + g_lo;
                            const std::size_t sph_base =
                                (m * n_sph_N + n) * n_G_blk;
                            // The mirrored block is the (n, m) AO pair,
                            // so it takes the TRANSPOSED weight; a fit
                            // screen need not be symmetric.
                            store_block(
                                &out.data[out_base], &src[sph_base],
                                weighted
                                    ? pair_w[(bfN + n) * n_orb + (bfM + m)]
                                    : 0.0);
                        }
                    }
                }
            }
        }
    }  // (shell pair, G-block) task

    return outs;
}

Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
ao_pair_fourier_transform_gamma_gradient_weighted(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic,
                                         Eigen::Dynamic, Eigen::RowMajor>>&
        pair_weights,
    const Eigen::Ref<const Eigen::VectorXcd>& reciprocal_weights,
    std::size_t n_atoms)
{
    using GradientMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    const std::vector<ShellInfo> shells = basis.shells();
    const auto bf_offsets = shell_to_bf_offsets_pure(shells);
    const std::size_t n_orb = basis.nbasis();
    const std::size_t n_G = static_cast<std::size_t>(G_vectors.rows());
    const std::size_t n_g = static_cast<std::size_t>(R_g_list.rows());
    const std::size_t n_shells = shells.size();

    if (pair_weights.rows() != static_cast<Eigen::Index>(n_orb)
            || pair_weights.cols() != static_cast<Eigen::Index>(n_orb)) {
        throw std::invalid_argument(
            "ao_pair_fourier_transform_gamma_gradient_weighted: "
            "pair_weights must have shape (n_orb, n_orb)");
    }
    if (reciprocal_weights.size() != static_cast<Eigen::Index>(n_G)) {
        throw std::invalid_argument(
            "ao_pair_fourier_transform_gamma_gradient_weighted: "
            "reciprocal_weights must have shape (n_G,)");
    }
    for (const auto& shell : shells) {
        if (shell.atom_index < 0
                || static_cast<std::size_t>(shell.atom_index) >= n_atoms) {
            throw std::invalid_argument(
                "ao_pair_fourier_transform_gamma_gradient_weighted: "
                "shell atom_index is outside [0, n_atoms)");
        }
    }

    GradientMatrix gradient = GradientMatrix::Zero(
        static_cast<Eigen::Index>(n_atoms), 3);
    if (n_G == 0 || n_g == 0 || n_orb == 0) {
        return gradient;
    }

    std::vector<double> G2(n_G);
    for (std::size_t k = 0; k < n_G; ++k) {
        const double gx = G_vectors(static_cast<Eigen::Index>(k), 0);
        const double gy = G_vectors(static_cast<Eigen::Index>(k), 1);
        const double gz = G_vectors(static_cast<Eigen::Index>(k), 2);
        G2[k] = gx * gx + gy * gy + gz * gz;
    }

    // The Cartesian MD branch needs powers of (-iG) through one raised
    // angular-momentum slot. These tables depend only on G, so share one
    // read-only copy across all OpenMP shell-pair workers rather than
    // allocating O(n_threads) copies. The s-only fast path needs none.
    int max_l = 0;
    bool has_general_l = false;
    for (const auto& shell : shells) {
        max_l = std::max(max_l, shell.l);
        has_general_l = has_general_l || shell.l > 0;
    }
    std::vector<std::complex<double>> Gx_pow;
    std::vector<std::complex<double>> Gy_pow;
    std::vector<std::complex<double>> Gz_pow;
    if (has_general_l) {
        const std::size_t n_powers =
            static_cast<std::size_t>(2 * max_l + 2);
        Gx_pow.resize(n_powers * n_G);
        Gy_pow.resize(n_powers * n_G);
        Gz_pow.resize(n_powers * n_G);
        for (std::size_t k = 0; k < n_G; ++k) {
            std::complex<double> px(1.0, 0.0);
            std::complex<double> py(1.0, 0.0);
            std::complex<double> pz(1.0, 0.0);
            const std::complex<double> ix(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 0));
            const std::complex<double> iy(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 1));
            const std::complex<double> iz(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 2));
            for (std::size_t t = 0; t < n_powers; ++t) {
                Gx_pow[t * n_G + k] = px;
                Gy_pow[t * n_G + k] = py;
                Gz_pow[t * n_G + k] = pz;
                px *= ix;
                py *= iy;
                pz *= iz;
            }
        }
    }

    // Each ordered shell pair owns one six-vector: bra xyz, then ket xyz.
    // This avoids atom-level races inside the OpenMP shell-pair loop.
    std::vector<std::array<double, 6>> pair_contributions(
        n_shells * n_shells);
    for (auto& contribution : pair_contributions) {
        contribution.fill(0.0);
    }

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::ptrdiff_t sM = 0;
         sM < static_cast<std::ptrdiff_t>(n_shells); ++sM) {
        for (std::ptrdiff_t sN = 0;
             sN < static_cast<std::ptrdiff_t>(n_shells); ++sN) {
            const auto sm = static_cast<std::size_t>(sM);
            const auto sn = static_cast<std::size_t>(sN);
            const ShellInfo& shM = shells[sm];
            const ShellInfo& shN = shells[sn];
            const int lM = shM.l;
            const int lN = shN.l;
            const std::size_t bfM = bf_offsets[sm];
            const std::size_t bfN = bf_offsets[sn];
            auto& contribution = pair_contributions[sm * n_shells + sn];

            const double Ax = shM.origin[0];
            const double Ay = shM.origin[1];
            const double Az = shM.origin[2];
            const double Bx0 = shN.origin[0];
            const double By0 = shN.origin[1];
            const double Bz0 = shN.origin[2];
            const auto& es_M = shM.exponents;
            const auto& cs_M = shM.coefficients;
            const auto& es_N = shN.exponents;
            const auto& cs_N = shN.coefficients;

            // Closed-form s-s branch. This is the common H/He fast path and
            // avoids running seven shifted-index MD contractions per pair.
            if (lM == 0 && lN == 0) {
                const double weight = pair_weights(
                    static_cast<Eigen::Index>(bfM),
                    static_cast<Eigen::Index>(bfN));
                if (weight == 0.0) {
                    continue;
                }
                for (std::size_t g = 0; g < n_g; ++g) {
                    const double Bx = Bx0 + R_g_list(
                        static_cast<Eigen::Index>(g), 0);
                    const double By = By0 + R_g_list(
                        static_cast<Eigen::Index>(g), 1);
                    const double Bz = Bz0 + R_g_list(
                        static_cast<Eigen::Index>(g), 2);
                    const std::array<double, 3> AB = {
                        Ax - Bx, Ay - By, Az - Bz};
                    const double AB_sq = AB[0] * AB[0] + AB[1] * AB[1]
                        + AB[2] * AB[2];
                    for (std::size_t ip = 0; ip < es_M.size(); ++ip) {
                        const double alpha = es_M[ip];
                        for (std::size_t iq = 0; iq < es_N.size(); ++iq) {
                            const double beta = es_N[iq];
                            const double gamma = alpha + beta;
                            const double inv_gamma = 1.0 / gamma;
                            const double radial_prefactor =
                                weight * cs_M[ip] * cs_N[iq]
                                * std::exp(-alpha * beta * inv_gamma * AB_sq)
                                * std::pow(kPi * inv_gamma, 1.5);
                            const std::array<double, 3> P = {
                                (alpha * Ax + beta * Bx) * inv_gamma,
                                (alpha * Ay + beta * By) * inv_gamma,
                                (alpha * Az + beta * Bz) * inv_gamma};
                            const double displacement_scale =
                                2.0 * alpha * beta * inv_gamma;
                            for (std::size_t k = 0; k < n_G; ++k) {
                                const double gx = G_vectors(
                                    static_cast<Eigen::Index>(k), 0);
                                const double gy = G_vectors(
                                    static_cast<Eigen::Index>(k), 1);
                                const double gz = G_vectors(
                                    static_cast<Eigen::Index>(k), 2);
                                const double gp = gx * P[0] + gy * P[1]
                                    + gz * P[2];
                                const std::complex<double> ft =
                                    radial_prefactor
                                    * std::exp(-0.25 * G2[k] * inv_gamma)
                                    * std::complex<double>(
                                        std::cos(gp), -std::sin(gp));
                                const std::array<double, 3> G = {gx, gy, gz};
                                const std::complex<double> q =
                                    reciprocal_weights(
                                        static_cast<Eigen::Index>(k));
                                for (int axis = 0; axis < 3; ++axis) {
                                    const std::complex<double> dA = ft *
                                        std::complex<double>(
                                            -displacement_scale * AB[axis],
                                            -alpha * inv_gamma * G[axis]);
                                    const std::complex<double> dB = ft *
                                        std::complex<double>(
                                            displacement_scale * AB[axis],
                                            -beta * inv_gamma * G[axis]);
                                    contribution[static_cast<std::size_t>(axis)]
                                        += std::real(q * std::conj(dA));
                                    contribution[
                                        static_cast<std::size_t>(axis + 3)]
                                        += std::real(q * std::conj(dB));
                                }
                            }
                        }
                    }
                }
                continue;
            }

            const std::size_t n_cart_M =
                cart_to_sph_data::n_cart_for_l(lM);
            const std::size_t n_cart_N =
                cart_to_sph_data::n_cart_for_l(lN);
            const std::size_t n_sph_M =
                cart_to_sph_data::n_sph_for_l(lM);
            const std::size_t n_sph_N =
                cart_to_sph_data::n_sph_for_l(lN);
            const auto* cart_M = cart_to_sph_data::cart_table_for_l(lM);
            const auto* cart_N = cart_to_sph_data::cart_table_for_l(lN);
            const double* Csph_M = cart_to_sph_data::sph_table_for_l(lM);
            const double* Csph_N = cart_to_sph_data::sph_table_for_l(lN);

            // W_cart = C_M^T W_sph C_N. Raised/lowered derivative indices
            // retain this ORIGINAL-shell transform and normalization.
            std::vector<double> W_cart(n_cart_M * n_cart_N, 0.0);
            for (std::size_t a = 0; a < n_cart_M; ++a) {
                for (std::size_t b = 0; b < n_cart_N; ++b) {
                    double value = 0.0;
                    for (std::size_t m = 0; m < n_sph_M; ++m) {
                        const double cma = Csph_M[m * n_cart_M + a];
                        if (cma == 0.0) {
                            continue;
                        }
                        for (std::size_t n = 0; n < n_sph_N; ++n) {
                            value += cma
                                * pair_weights(
                                    static_cast<Eigen::Index>(bfM + m),
                                    static_cast<Eigen::Index>(bfN + n))
                                * Csph_N[n * n_cart_N + b];
                        }
                    }
                    W_cart[a * n_cart_N + b] = value;
                }
            }

            if (std::all_of(W_cart.begin(), W_cart.end(),
                            [](double weight) { return weight == 0.0; })) {
                continue;
            }

            std::array<double, 3> translation = {0.0, 0.0, 0.0};
            std::array<double, 3> ket = {0.0, 0.0, 0.0};
            std::vector<double> Ex, Ey, Ez;
            std::vector<std::complex<double>> prefactor_per_G(n_G);
            const std::size_t e_dim_lb = static_cast<std::size_t>(lN + 2);
            const std::size_t e_dim_t =
                static_cast<std::size_t>(lM + lN + 2);

            for (std::size_t g = 0; g < n_g; ++g) {
                const double Bx = Bx0 + R_g_list(
                    static_cast<Eigen::Index>(g), 0);
                const double By = By0 + R_g_list(
                    static_cast<Eigen::Index>(g), 1);
                const double Bz = Bz0 + R_g_list(
                    static_cast<Eigen::Index>(g), 2);
                const double ABx = Ax - Bx;
                const double ABy = Ay - By;
                const double ABz = Az - Bz;
                const double AB_sq = ABx * ABx + ABy * ABy + ABz * ABz;

                for (std::size_t ip = 0; ip < es_M.size(); ++ip) {
                    const double alpha = es_M[ip];
                    for (std::size_t iq = 0; iq < es_N.size(); ++iq) {
                        const double beta = es_N[iq];
                        const double gamma = alpha + beta;
                        const double inv_gamma = 1.0 / gamma;
                        const double Px =
                            (alpha * Ax + beta * Bx) * inv_gamma;
                        const double Py =
                            (alpha * Ay + beta * By) * inv_gamma;
                        const double Pz =
                            (alpha * Az + beta * Bz) * inv_gamma;
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Px - Ax, Px - Bx, Ex);
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Py - Ay, Py - By, Ey);
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Pz - Az, Pz - Bz, Ez);

                        const double radial_prefactor =
                            cs_M[ip] * cs_N[iq]
                            * std::exp(-alpha * beta * inv_gamma * AB_sq)
                            * std::pow(kPi * inv_gamma, 1.5);
                        for (std::size_t k = 0; k < n_G; ++k) {
                            const double gx = G_vectors(
                                static_cast<Eigen::Index>(k), 0);
                            const double gy = G_vectors(
                                static_cast<Eigen::Index>(k), 1);
                            const double gz = G_vectors(
                                static_cast<Eigen::Index>(k), 2);
                            const double gp = gx * Px + gy * Py + gz * Pz;
                            prefactor_per_G[k] =
                                radial_prefactor
                                * std::exp(-0.25 * G2[k] * inv_gamma)
                                * std::complex<double>(
                                    std::cos(gp), -std::sin(gp));
                        }

                        for (std::size_t a = 0; a < n_cart_M; ++a) {
                            const std::array<int, 3> cart_A = {
                                cart_M[a].i, cart_M[a].j, cart_M[a].k};
                            for (std::size_t b = 0; b < n_cart_N; ++b) {
                                const double weight =
                                    W_cart[a * n_cart_N + b];
                                if (weight == 0.0) {
                                    continue;
                                }
                                const std::array<int, 3> cart_B = {
                                    cart_N[b].i, cart_N[b].j, cart_N[b].k};
                                for (int axis = 0; axis < 3; ++axis) {
                                    translation[static_cast<std::size_t>(axis)]
                                        += contract_md_ft_line(
                                            cart_A, cart_B, Ex, Ey, Ez,
                                            e_dim_lb, e_dim_t, Gx_pow,
                                            Gy_pow, Gz_pow, prefactor_per_G,
                                            G_vectors, reciprocal_weights,
                                            weight, axis);
                                    if (shM.atom_index == shN.atom_index) {
                                        continue;
                                    }
                                    auto raised = cart_B;
                                    raised[static_cast<std::size_t>(axis)] += 1;
                                    ket[static_cast<std::size_t>(axis)]
                                        += contract_md_ft_line(
                                            cart_A, raised, Ex, Ey, Ez,
                                            e_dim_lb, e_dim_t, Gx_pow,
                                            Gy_pow, Gz_pow, prefactor_per_G,
                                            G_vectors, reciprocal_weights,
                                            weight * 2.0 * beta);
                                    if (cart_B[
                                            static_cast<std::size_t>(axis)] > 0) {
                                        auto lowered = cart_B;
                                        lowered[
                                            static_cast<std::size_t>(axis)] -= 1;
                                        ket[static_cast<std::size_t>(axis)]
                                            += contract_md_ft_line(
                                                cart_A, lowered, Ex, Ey, Ez,
                                                e_dim_lb, e_dim_t, Gx_pow,
                                                Gy_pow, Gz_pow,
                                                prefactor_per_G, G_vectors,
                                                reciprocal_weights,
                                                -weight * cart_B[
                                                    static_cast<std::size_t>(
                                                        axis)]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (int axis = 0; axis < 3; ++axis) {
                const auto idx = static_cast<std::size_t>(axis);
                if (shM.atom_index == shN.atom_index) {
                    contribution[idx] = translation[idx];
                } else {
                    contribution[idx] = translation[idx] - ket[idx];
                    contribution[idx + 3] = ket[idx];
                }
            }
        }
    }

    for (std::size_t sM = 0; sM < n_shells; ++sM) {
        for (std::size_t sN = 0; sN < n_shells; ++sN) {
            const auto& contribution =
                pair_contributions[sM * n_shells + sN];
            const auto atom_M = static_cast<Eigen::Index>(
                shells[sM].atom_index);
            const auto atom_N = static_cast<Eigen::Index>(
                shells[sN].atom_index);
            for (int axis = 0; axis < 3; ++axis) {
                gradient(atom_M, axis) +=
                    contribution[static_cast<std::size_t>(axis)];
                gradient(atom_N, axis) +=
                    contribution[static_cast<std::size_t>(axis + 3)];
            }
        }
    }
    return gradient;
}

Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
ao_pair_fourier_transform_gamma_gradient_gweighted(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const std::complex<double>* pair_weights_g,
    std::size_t n_atoms)
{
    using GradientMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    const std::vector<ShellInfo> shells = basis.shells();
    const auto bf_offsets = shell_to_bf_offsets_pure(shells);
    const std::size_t n_orb = basis.nbasis();
    const std::size_t n_G = static_cast<std::size_t>(G_vectors.rows());
    const std::size_t n_g = static_cast<std::size_t>(R_g_list.rows());
    const std::size_t n_shells = shells.size();

    if (pair_weights_g == nullptr) {
        throw std::invalid_argument(
            "ao_pair_fourier_transform_gamma_gradient_gweighted: "
            "pair_weights_g must not be null");
    }
    for (const auto& shell : shells) {
        if (shell.atom_index < 0
                || static_cast<std::size_t>(shell.atom_index) >= n_atoms) {
            throw std::invalid_argument(
                "ao_pair_fourier_transform_gamma_gradient_gweighted: "
                "shell atom_index is outside [0, n_atoms)");
        }
    }

    GradientMatrix gradient = GradientMatrix::Zero(
        static_cast<Eigen::Index>(n_atoms), 3);
    if (n_G == 0 || n_g == 0 || n_orb == 0) {
        return gradient;
    }

    std::vector<double> G2(n_G);
    for (std::size_t k = 0; k < n_G; ++k) {
        const double gx = G_vectors(static_cast<Eigen::Index>(k), 0);
        const double gy = G_vectors(static_cast<Eigen::Index>(k), 1);
        const double gz = G_vectors(static_cast<Eigen::Index>(k), 2);
        G2[k] = gx * gx + gy * gy + gz * gz;
    }

    // The Cartesian MD branch needs powers of (-iG) through one raised
    // angular-momentum slot. These tables depend only on G, so share one
    // read-only copy across all OpenMP shell-pair workers rather than
    // allocating O(n_threads) copies. The s-only fast path needs none.
    int max_l = 0;
    bool has_general_l = false;
    for (const auto& shell : shells) {
        max_l = std::max(max_l, shell.l);
        has_general_l = has_general_l || shell.l > 0;
    }
    std::vector<std::complex<double>> Gx_pow;
    std::vector<std::complex<double>> Gy_pow;
    std::vector<std::complex<double>> Gz_pow;
    if (has_general_l) {
        const std::size_t n_powers =
            static_cast<std::size_t>(2 * max_l + 2);
        Gx_pow.resize(n_powers * n_G);
        Gy_pow.resize(n_powers * n_G);
        Gz_pow.resize(n_powers * n_G);
        for (std::size_t k = 0; k < n_G; ++k) {
            std::complex<double> px(1.0, 0.0);
            std::complex<double> py(1.0, 0.0);
            std::complex<double> pz(1.0, 0.0);
            const std::complex<double> ix(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 0));
            const std::complex<double> iy(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 1));
            const std::complex<double> iz(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 2));
            for (std::size_t t = 0; t < n_powers; ++t) {
                Gx_pow[t * n_G + k] = px;
                Gy_pow[t * n_G + k] = py;
                Gz_pow[t * n_G + k] = pz;
                px *= ix;
                py *= iy;
                pz *= iz;
            }
        }
    }

    // Each ordered shell pair owns one six-vector: bra xyz, then ket xyz.
    // This avoids atom-level races inside the OpenMP shell-pair loop.
    // Q is read shared (never copied per worker); each worker builds only
    // its own per-pair Cartesian fold. The (pair, G-block) task space of
    // the value kernels is a later perf option for this kernel too.
    std::vector<std::array<double, 6>> pair_contributions(
        n_shells * n_shells);
    for (auto& contribution : pair_contributions) {
        contribution.fill(0.0);
    }

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::ptrdiff_t sM = 0;
         sM < static_cast<std::ptrdiff_t>(n_shells); ++sM) {
        for (std::ptrdiff_t sN = 0;
             sN < static_cast<std::ptrdiff_t>(n_shells); ++sN) {
            const auto sm = static_cast<std::size_t>(sM);
            const auto sn = static_cast<std::size_t>(sN);
            const ShellInfo& shM = shells[sm];
            const ShellInfo& shN = shells[sn];
            const int lM = shM.l;
            const int lN = shN.l;
            const std::size_t bfM = bf_offsets[sm];
            const std::size_t bfN = bf_offsets[sn];
            auto& contribution = pair_contributions[sm * n_shells + sn];

            const double Ax = shM.origin[0];
            const double Ay = shM.origin[1];
            const double Az = shM.origin[2];
            const double Bx0 = shN.origin[0];
            const double By0 = shN.origin[1];
            const double Bz0 = shN.origin[2];
            const auto& es_M = shM.exponents;
            const auto& cs_M = shM.coefficients;
            const auto& es_N = shN.exponents;
            const auto& cs_N = shN.coefficients;

            // Closed-form s-s branch. This is the common H/He fast path and
            // avoids running seven shifted-index MD contractions per pair.
            if (lM == 0 && lN == 0) {
                const std::complex<double>* q_row =
                    pair_weights_g + (bfM * n_orb + bfN) * n_G;
                if (std::all_of(q_row, q_row + n_G,
                                [](const std::complex<double>& q) {
                                    return q == std::complex<double>(0.0, 0.0);
                                })) {
                    continue;
                }
                for (std::size_t g = 0; g < n_g; ++g) {
                    const double Bx = Bx0 + R_g_list(
                        static_cast<Eigen::Index>(g), 0);
                    const double By = By0 + R_g_list(
                        static_cast<Eigen::Index>(g), 1);
                    const double Bz = Bz0 + R_g_list(
                        static_cast<Eigen::Index>(g), 2);
                    const std::array<double, 3> AB = {
                        Ax - Bx, Ay - By, Az - Bz};
                    const double AB_sq = AB[0] * AB[0] + AB[1] * AB[1]
                        + AB[2] * AB[2];
                    for (std::size_t ip = 0; ip < es_M.size(); ++ip) {
                        const double alpha = es_M[ip];
                        for (std::size_t iq = 0; iq < es_N.size(); ++iq) {
                            const double beta = es_N[iq];
                            const double gamma = alpha + beta;
                            const double inv_gamma = 1.0 / gamma;
                            const double radial_prefactor =
                                cs_M[ip] * cs_N[iq]
                                * std::exp(-alpha * beta * inv_gamma * AB_sq)
                                * std::pow(kPi * inv_gamma, 1.5);
                            const std::array<double, 3> P = {
                                (alpha * Ax + beta * Bx) * inv_gamma,
                                (alpha * Ay + beta * By) * inv_gamma,
                                (alpha * Az + beta * Bz) * inv_gamma};
                            const double displacement_scale =
                                2.0 * alpha * beta * inv_gamma;
                            for (std::size_t k = 0; k < n_G; ++k) {
                                const double gx = G_vectors(
                                    static_cast<Eigen::Index>(k), 0);
                                const double gy = G_vectors(
                                    static_cast<Eigen::Index>(k), 1);
                                const double gz = G_vectors(
                                    static_cast<Eigen::Index>(k), 2);
                                const double gp = gx * P[0] + gy * P[1]
                                    + gz * P[2];
                                const std::complex<double> ft =
                                    radial_prefactor
                                    * std::exp(-0.25 * G2[k] * inv_gamma)
                                    * std::complex<double>(
                                        std::cos(gp), -std::sin(gp));
                                const std::array<double, 3> G = {gx, gy, gz};
                                const std::complex<double> q = q_row[k];
                                for (int axis = 0; axis < 3; ++axis) {
                                    const std::complex<double> dA = ft *
                                        std::complex<double>(
                                            -displacement_scale * AB[axis],
                                            -alpha * inv_gamma * G[axis]);
                                    const std::complex<double> dB = ft *
                                        std::complex<double>(
                                            displacement_scale * AB[axis],
                                            -beta * inv_gamma * G[axis]);
                                    contribution[static_cast<std::size_t>(axis)]
                                        += std::real(q * std::conj(dA));
                                    contribution[
                                        static_cast<std::size_t>(axis + 3)]
                                        += std::real(q * std::conj(dB));
                                }
                            }
                        }
                    }
                }
                continue;
            }

            const std::size_t n_cart_M =
                cart_to_sph_data::n_cart_for_l(lM);
            const std::size_t n_cart_N =
                cart_to_sph_data::n_cart_for_l(lN);
            const std::size_t n_sph_M =
                cart_to_sph_data::n_sph_for_l(lM);
            const std::size_t n_sph_N =
                cart_to_sph_data::n_sph_for_l(lN);
            const auto* cart_M = cart_to_sph_data::cart_table_for_l(lM);
            const auto* cart_N = cart_to_sph_data::cart_table_for_l(lN);
            const double* Csph_M = cart_to_sph_data::sph_table_for_l(lM);
            const double* Csph_N = cart_to_sph_data::sph_table_for_l(lN);

            // Q_cart[(a, b), k] = Σ_mn C_M[m,a] Q(bfM+m, bfN+n, k) C_N[n,b].
            // With G-resolved weights the scalar kernel's G-independent
            // W_cart = C_M^T W_sph C_N fold becomes per-G; it is done ONCE
            // per shell pair for all G before the cell/primitive loops.
            // Worker memory is n_cart_M·n_cart_N·n_G complex — fine for
            // L <= 3 at the meshes used here; chunking over G is a later
            // perf option. Raised/lowered derivative indices retain this
            // ORIGINAL-shell transform and normalization.
            std::vector<std::complex<double>> Q_cart(
                n_cart_M * n_cart_N * n_G, std::complex<double>(0.0, 0.0));
            std::vector<char> q_nonzero(n_cart_M * n_cart_N, 0);
            for (std::size_t a = 0; a < n_cart_M; ++a) {
                for (std::size_t b = 0; b < n_cart_N; ++b) {
                    std::complex<double>* dst =
                        &Q_cart[(a * n_cart_N + b) * n_G];
                    for (std::size_t m = 0; m < n_sph_M; ++m) {
                        const double cma = Csph_M[m * n_cart_M + a];
                        if (cma == 0.0) {
                            continue;
                        }
                        for (std::size_t n = 0; n < n_sph_N; ++n) {
                            const double c_mn =
                                cma * Csph_N[n * n_cart_N + b];
                            if (c_mn == 0.0) {
                                continue;
                            }
                            const std::complex<double>* src = pair_weights_g
                                + ((bfM + m) * n_orb + (bfN + n)) * n_G;
                            for (std::size_t k = 0; k < n_G; ++k) {
                                dst[k] += c_mn * src[k];
                            }
                        }
                    }
                    for (std::size_t k = 0; k < n_G; ++k) {
                        if (dst[k] != std::complex<double>(0.0, 0.0)) {
                            q_nonzero[a * n_cart_N + b] = 1;
                            break;
                        }
                    }
                }
            }

            if (std::all_of(q_nonzero.begin(), q_nonzero.end(),
                            [](char nz) { return nz == 0; })) {
                continue;
            }

            std::array<double, 3> translation = {0.0, 0.0, 0.0};
            std::array<double, 3> ket = {0.0, 0.0, 0.0};
            std::vector<double> Ex, Ey, Ez;
            std::vector<std::complex<double>> prefactor_per_G(n_G);
            const std::size_t e_dim_lb = static_cast<std::size_t>(lN + 2);
            const std::size_t e_dim_t =
                static_cast<std::size_t>(lM + lN + 2);

            for (std::size_t g = 0; g < n_g; ++g) {
                const double Bx = Bx0 + R_g_list(
                    static_cast<Eigen::Index>(g), 0);
                const double By = By0 + R_g_list(
                    static_cast<Eigen::Index>(g), 1);
                const double Bz = Bz0 + R_g_list(
                    static_cast<Eigen::Index>(g), 2);
                const double ABx = Ax - Bx;
                const double ABy = Ay - By;
                const double ABz = Az - Bz;
                const double AB_sq = ABx * ABx + ABy * ABy + ABz * ABz;

                for (std::size_t ip = 0; ip < es_M.size(); ++ip) {
                    const double alpha = es_M[ip];
                    for (std::size_t iq = 0; iq < es_N.size(); ++iq) {
                        const double beta = es_N[iq];
                        const double gamma = alpha + beta;
                        const double inv_gamma = 1.0 / gamma;
                        const double Px =
                            (alpha * Ax + beta * Bx) * inv_gamma;
                        const double Py =
                            (alpha * Ay + beta * By) * inv_gamma;
                        const double Pz =
                            (alpha * Az + beta * Bz) * inv_gamma;
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Px - Ax, Px - Bx, Ex);
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Py - Ay, Py - By, Ey);
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Pz - Az, Pz - Bz, Ez);

                        const double radial_prefactor =
                            cs_M[ip] * cs_N[iq]
                            * std::exp(-alpha * beta * inv_gamma * AB_sq)
                            * std::pow(kPi * inv_gamma, 1.5);
                        for (std::size_t k = 0; k < n_G; ++k) {
                            const double gx = G_vectors(
                                static_cast<Eigen::Index>(k), 0);
                            const double gy = G_vectors(
                                static_cast<Eigen::Index>(k), 1);
                            const double gz = G_vectors(
                                static_cast<Eigen::Index>(k), 2);
                            const double gp = gx * Px + gy * Py + gz * Pz;
                            prefactor_per_G[k] =
                                radial_prefactor
                                * std::exp(-0.25 * G2[k] * inv_gamma)
                                * std::complex<double>(
                                    std::cos(gp), -std::sin(gp));
                        }

                        for (std::size_t a = 0; a < n_cart_M; ++a) {
                            const std::array<int, 3> cart_A = {
                                cart_M[a].i, cart_M[a].j, cart_M[a].k};
                            for (std::size_t b = 0; b < n_cart_N; ++b) {
                                if (q_nonzero[a * n_cart_N + b] == 0) {
                                    continue;
                                }
                                const std::complex<double>* q_ab =
                                    &Q_cart[(a * n_cart_N + b) * n_G];
                                const std::array<int, 3> cart_B = {
                                    cart_N[b].i, cart_N[b].j, cart_N[b].k};
                                for (int axis = 0; axis < 3; ++axis) {
                                    translation[static_cast<std::size_t>(axis)]
                                        += contract_md_ft_line_gweighted(
                                            cart_A, cart_B, Ex, Ey, Ez,
                                            e_dim_lb, e_dim_t, Gx_pow,
                                            Gy_pow, Gz_pow, prefactor_per_G,
                                            G_vectors, q_ab, 1.0, axis);
                                    if (shM.atom_index == shN.atom_index) {
                                        continue;
                                    }
                                    auto raised = cart_B;
                                    raised[static_cast<std::size_t>(axis)] += 1;
                                    ket[static_cast<std::size_t>(axis)]
                                        += contract_md_ft_line_gweighted(
                                            cart_A, raised, Ex, Ey, Ez,
                                            e_dim_lb, e_dim_t, Gx_pow,
                                            Gy_pow, Gz_pow, prefactor_per_G,
                                            G_vectors, q_ab, 2.0 * beta);
                                    if (cart_B[
                                            static_cast<std::size_t>(axis)] > 0) {
                                        auto lowered = cart_B;
                                        lowered[
                                            static_cast<std::size_t>(axis)] -= 1;
                                        ket[static_cast<std::size_t>(axis)]
                                            += contract_md_ft_line_gweighted(
                                                cart_A, lowered, Ex, Ey, Ez,
                                                e_dim_lb, e_dim_t, Gx_pow,
                                                Gy_pow, Gz_pow,
                                                prefactor_per_G, G_vectors,
                                                q_ab,
                                                -static_cast<double>(cart_B[
                                                    static_cast<std::size_t>(
                                                        axis)]));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (int axis = 0; axis < 3; ++axis) {
                const auto idx = static_cast<std::size_t>(axis);
                if (shM.atom_index == shN.atom_index) {
                    contribution[idx] = translation[idx];
                } else {
                    contribution[idx] = translation[idx] - ket[idx];
                    contribution[idx + 3] = ket[idx];
                }
            }
        }
    }

    for (std::size_t sM = 0; sM < n_shells; ++sM) {
        for (std::size_t sN = 0; sN < n_shells; ++sN) {
            const auto& contribution =
                pair_contributions[sM * n_shells + sN];
            const auto atom_M = static_cast<Eigen::Index>(
                shells[sM].atom_index);
            const auto atom_N = static_cast<Eigen::Index>(
                shells[sN].atom_index);
            for (int axis = 0; axis < 3; ++axis) {
                gradient(atom_M, axis) +=
                    contribution[static_cast<std::size_t>(axis)];
                gradient(atom_N, axis) +=
                    contribution[static_cast<std::size_t>(axis + 3)];
            }
        }
    }
    return gradient;
}

Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
ao_pair_fourier_transform_bloch_gradient_gweighted(
    const BasisSet& basis,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& G_vectors,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                         Eigen::RowMajor>>& R_g_list,
    const Eigen::Vector3d& k_cart,
    const std::complex<double>* pair_weights_g,
    std::size_t n_atoms)
{
    // Bloch generalization of ao_pair_fourier_transform_gamma_gradient_
    // gweighted above (HANDOVER_GDF_GRADIENT_DEFERRED.md § 4, rung 1).
    // The kernels must stay in lockstep: the ONLY difference is the
    // per-cell ket phase exp(+i k.R_g) multiplied onto the per-cell FT
    // factor (ft / prefactor_per_G) before the shared conj-reduction.
    // The phase carries no atom-position dependence, so every
    // derivative recurrence — the angular-momentum shift on the ket and
    // the translational covariance d_A FT + d_B FT = -iG FT — holds
    // per cell with the phase riding along. At k = 0 the phase is
    // exactly (1, 0) and IEEE complex multiply by (1, 0) is identity,
    // so this kernel reduces to the Gamma kernel bit-for-bit.
    using GradientMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    const std::vector<ShellInfo> shells = basis.shells();
    const auto bf_offsets = shell_to_bf_offsets_pure(shells);
    const std::size_t n_orb = basis.nbasis();
    const std::size_t n_G = static_cast<std::size_t>(G_vectors.rows());
    const std::size_t n_g = static_cast<std::size_t>(R_g_list.rows());
    const std::size_t n_shells = shells.size();

    if (pair_weights_g == nullptr) {
        throw std::invalid_argument(
            "ao_pair_fourier_transform_bloch_gradient_gweighted: "
            "pair_weights_g must not be null");
    }
    for (const auto& shell : shells) {
        if (shell.atom_index < 0
                || static_cast<std::size_t>(shell.atom_index) >= n_atoms) {
            throw std::invalid_argument(
                "ao_pair_fourier_transform_bloch_gradient_gweighted: "
                "shell atom_index is outside [0, n_atoms)");
        }
    }

    GradientMatrix gradient = GradientMatrix::Zero(
        static_cast<Eigen::Index>(n_atoms), 3);
    if (n_G == 0 || n_g == 0 || n_orb == 0) {
        return gradient;
    }

    // Ket Bloch phase exp(+i k.R_g), same sign convention as the value
    // kernel ao_pair_fourier_transform_bloch; the objective conjugates
    // the Bloch-summed FT, so the phase enters the reduction as
    // exp(-i k.R_g) via the shared conj.
    std::vector<std::complex<double>> bloch_phase(n_g);
    for (std::size_t g = 0; g < n_g; ++g) {
        const double kr =
            k_cart[0] * R_g_list(static_cast<Eigen::Index>(g), 0)
            + k_cart[1] * R_g_list(static_cast<Eigen::Index>(g), 1)
            + k_cart[2] * R_g_list(static_cast<Eigen::Index>(g), 2);
        bloch_phase[g] = std::complex<double>(std::cos(kr), std::sin(kr));
    }

    std::vector<double> G2(n_G);
    for (std::size_t k = 0; k < n_G; ++k) {
        const double gx = G_vectors(static_cast<Eigen::Index>(k), 0);
        const double gy = G_vectors(static_cast<Eigen::Index>(k), 1);
        const double gz = G_vectors(static_cast<Eigen::Index>(k), 2);
        G2[k] = gx * gx + gy * gy + gz * gz;
    }

    // The Cartesian MD branch needs powers of (-iG) through one raised
    // angular-momentum slot. These tables depend only on G, so share one
    // read-only copy across all OpenMP shell-pair workers rather than
    // allocating O(n_threads) copies. The s-only fast path needs none.
    int max_l = 0;
    bool has_general_l = false;
    for (const auto& shell : shells) {
        max_l = std::max(max_l, shell.l);
        has_general_l = has_general_l || shell.l > 0;
    }
    std::vector<std::complex<double>> Gx_pow;
    std::vector<std::complex<double>> Gy_pow;
    std::vector<std::complex<double>> Gz_pow;
    if (has_general_l) {
        const std::size_t n_powers =
            static_cast<std::size_t>(2 * max_l + 2);
        Gx_pow.resize(n_powers * n_G);
        Gy_pow.resize(n_powers * n_G);
        Gz_pow.resize(n_powers * n_G);
        for (std::size_t k = 0; k < n_G; ++k) {
            std::complex<double> px(1.0, 0.0);
            std::complex<double> py(1.0, 0.0);
            std::complex<double> pz(1.0, 0.0);
            const std::complex<double> ix(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 0));
            const std::complex<double> iy(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 1));
            const std::complex<double> iz(
                0.0, -G_vectors(static_cast<Eigen::Index>(k), 2));
            for (std::size_t t = 0; t < n_powers; ++t) {
                Gx_pow[t * n_G + k] = px;
                Gy_pow[t * n_G + k] = py;
                Gz_pow[t * n_G + k] = pz;
                px *= ix;
                py *= iy;
                pz *= iz;
            }
        }
    }

    // Each ordered shell pair owns one six-vector: bra xyz, then ket xyz.
    // This avoids atom-level races inside the OpenMP shell-pair loop.
    // Q is read shared (never copied per worker); each worker builds only
    // its own per-pair Cartesian fold. The (pair, G-block) task space of
    // the value kernels is a later perf option for this kernel too.
    std::vector<std::array<double, 6>> pair_contributions(
        n_shells * n_shells);
    for (auto& contribution : pair_contributions) {
        contribution.fill(0.0);
    }

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::ptrdiff_t sM = 0;
         sM < static_cast<std::ptrdiff_t>(n_shells); ++sM) {
        for (std::ptrdiff_t sN = 0;
             sN < static_cast<std::ptrdiff_t>(n_shells); ++sN) {
            const auto sm = static_cast<std::size_t>(sM);
            const auto sn = static_cast<std::size_t>(sN);
            const ShellInfo& shM = shells[sm];
            const ShellInfo& shN = shells[sn];
            const int lM = shM.l;
            const int lN = shN.l;
            const std::size_t bfM = bf_offsets[sm];
            const std::size_t bfN = bf_offsets[sn];
            auto& contribution = pair_contributions[sm * n_shells + sn];

            const double Ax = shM.origin[0];
            const double Ay = shM.origin[1];
            const double Az = shM.origin[2];
            const double Bx0 = shN.origin[0];
            const double By0 = shN.origin[1];
            const double Bz0 = shN.origin[2];
            const auto& es_M = shM.exponents;
            const auto& cs_M = shM.coefficients;
            const auto& es_N = shN.exponents;
            const auto& cs_N = shN.coefficients;

            // Closed-form s-s branch. This is the common H/He fast path and
            // avoids running seven shifted-index MD contractions per pair.
            if (lM == 0 && lN == 0) {
                const std::complex<double>* q_row =
                    pair_weights_g + (bfM * n_orb + bfN) * n_G;
                if (std::all_of(q_row, q_row + n_G,
                                [](const std::complex<double>& q) {
                                    return q == std::complex<double>(0.0, 0.0);
                                })) {
                    continue;
                }
                for (std::size_t g = 0; g < n_g; ++g) {
                    const std::complex<double> phase_R = bloch_phase[g];
                    const double Bx = Bx0 + R_g_list(
                        static_cast<Eigen::Index>(g), 0);
                    const double By = By0 + R_g_list(
                        static_cast<Eigen::Index>(g), 1);
                    const double Bz = Bz0 + R_g_list(
                        static_cast<Eigen::Index>(g), 2);
                    const std::array<double, 3> AB = {
                        Ax - Bx, Ay - By, Az - Bz};
                    const double AB_sq = AB[0] * AB[0] + AB[1] * AB[1]
                        + AB[2] * AB[2];
                    for (std::size_t ip = 0; ip < es_M.size(); ++ip) {
                        const double alpha = es_M[ip];
                        for (std::size_t iq = 0; iq < es_N.size(); ++iq) {
                            const double beta = es_N[iq];
                            const double gamma = alpha + beta;
                            const double inv_gamma = 1.0 / gamma;
                            const double radial_prefactor =
                                cs_M[ip] * cs_N[iq]
                                * std::exp(-alpha * beta * inv_gamma * AB_sq)
                                * std::pow(kPi * inv_gamma, 1.5);
                            const std::array<double, 3> P = {
                                (alpha * Ax + beta * Bx) * inv_gamma,
                                (alpha * Ay + beta * By) * inv_gamma,
                                (alpha * Az + beta * Bz) * inv_gamma};
                            const double displacement_scale =
                                2.0 * alpha * beta * inv_gamma;
                            for (std::size_t k = 0; k < n_G; ++k) {
                                const double gx = G_vectors(
                                    static_cast<Eigen::Index>(k), 0);
                                const double gy = G_vectors(
                                    static_cast<Eigen::Index>(k), 1);
                                const double gz = G_vectors(
                                    static_cast<Eigen::Index>(k), 2);
                                const double gp = gx * P[0] + gy * P[1]
                                    + gz * P[2];
                                const std::complex<double> ft =
                                    phase_R
                                    * (radial_prefactor
                                       * std::exp(-0.25 * G2[k] * inv_gamma)
                                       * std::complex<double>(
                                           std::cos(gp), -std::sin(gp)));
                                const std::array<double, 3> G = {gx, gy, gz};
                                const std::complex<double> q = q_row[k];
                                for (int axis = 0; axis < 3; ++axis) {
                                    const std::complex<double> dA = ft *
                                        std::complex<double>(
                                            -displacement_scale * AB[axis],
                                            -alpha * inv_gamma * G[axis]);
                                    const std::complex<double> dB = ft *
                                        std::complex<double>(
                                            displacement_scale * AB[axis],
                                            -beta * inv_gamma * G[axis]);
                                    contribution[static_cast<std::size_t>(axis)]
                                        += std::real(q * std::conj(dA));
                                    contribution[
                                        static_cast<std::size_t>(axis + 3)]
                                        += std::real(q * std::conj(dB));
                                }
                            }
                        }
                    }
                }
                continue;
            }

            const std::size_t n_cart_M =
                cart_to_sph_data::n_cart_for_l(lM);
            const std::size_t n_cart_N =
                cart_to_sph_data::n_cart_for_l(lN);
            const std::size_t n_sph_M =
                cart_to_sph_data::n_sph_for_l(lM);
            const std::size_t n_sph_N =
                cart_to_sph_data::n_sph_for_l(lN);
            const auto* cart_M = cart_to_sph_data::cart_table_for_l(lM);
            const auto* cart_N = cart_to_sph_data::cart_table_for_l(lN);
            const double* Csph_M = cart_to_sph_data::sph_table_for_l(lM);
            const double* Csph_N = cart_to_sph_data::sph_table_for_l(lN);

            // Q_cart[(a, b), k] = Σ_mn C_M[m,a] Q(bfM+m, bfN+n, k) C_N[n,b].
            // With G-resolved weights the scalar kernel's G-independent
            // W_cart = C_M^T W_sph C_N fold becomes per-G; it is done ONCE
            // per shell pair for all G before the cell/primitive loops.
            // Worker memory is n_cart_M·n_cart_N·n_G complex — fine for
            // L <= 3 at the meshes used here; chunking over G is a later
            // perf option. Raised/lowered derivative indices retain this
            // ORIGINAL-shell transform and normalization.
            std::vector<std::complex<double>> Q_cart(
                n_cart_M * n_cart_N * n_G, std::complex<double>(0.0, 0.0));
            std::vector<char> q_nonzero(n_cart_M * n_cart_N, 0);
            for (std::size_t a = 0; a < n_cart_M; ++a) {
                for (std::size_t b = 0; b < n_cart_N; ++b) {
                    std::complex<double>* dst =
                        &Q_cart[(a * n_cart_N + b) * n_G];
                    for (std::size_t m = 0; m < n_sph_M; ++m) {
                        const double cma = Csph_M[m * n_cart_M + a];
                        if (cma == 0.0) {
                            continue;
                        }
                        for (std::size_t n = 0; n < n_sph_N; ++n) {
                            const double c_mn =
                                cma * Csph_N[n * n_cart_N + b];
                            if (c_mn == 0.0) {
                                continue;
                            }
                            const std::complex<double>* src = pair_weights_g
                                + ((bfM + m) * n_orb + (bfN + n)) * n_G;
                            for (std::size_t k = 0; k < n_G; ++k) {
                                dst[k] += c_mn * src[k];
                            }
                        }
                    }
                    for (std::size_t k = 0; k < n_G; ++k) {
                        if (dst[k] != std::complex<double>(0.0, 0.0)) {
                            q_nonzero[a * n_cart_N + b] = 1;
                            break;
                        }
                    }
                }
            }

            if (std::all_of(q_nonzero.begin(), q_nonzero.end(),
                            [](char nz) { return nz == 0; })) {
                continue;
            }

            std::array<double, 3> translation = {0.0, 0.0, 0.0};
            std::array<double, 3> ket = {0.0, 0.0, 0.0};
            std::vector<double> Ex, Ey, Ez;
            std::vector<std::complex<double>> prefactor_per_G(n_G);
            const std::size_t e_dim_lb = static_cast<std::size_t>(lN + 2);
            const std::size_t e_dim_t =
                static_cast<std::size_t>(lM + lN + 2);

            for (std::size_t g = 0; g < n_g; ++g) {
                const std::complex<double> phase_R = bloch_phase[g];
                const double Bx = Bx0 + R_g_list(
                    static_cast<Eigen::Index>(g), 0);
                const double By = By0 + R_g_list(
                    static_cast<Eigen::Index>(g), 1);
                const double Bz = Bz0 + R_g_list(
                    static_cast<Eigen::Index>(g), 2);
                const double ABx = Ax - Bx;
                const double ABy = Ay - By;
                const double ABz = Az - Bz;
                const double AB_sq = ABx * ABx + ABy * ABy + ABz * ABz;

                for (std::size_t ip = 0; ip < es_M.size(); ++ip) {
                    const double alpha = es_M[ip];
                    for (std::size_t iq = 0; iq < es_N.size(); ++iq) {
                        const double beta = es_N[iq];
                        const double gamma = alpha + beta;
                        const double inv_gamma = 1.0 / gamma;
                        const double Px =
                            (alpha * Ax + beta * Bx) * inv_gamma;
                        const double Py =
                            (alpha * Ay + beta * By) * inv_gamma;
                        const double Pz =
                            (alpha * Az + beta * Bz) * inv_gamma;
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Px - Ax, Px - Bx, Ex);
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Py - Ay, Py - By, Ey);
                        md_e_coefficients_1d(
                            lM, lN + 1, gamma, Pz - Az, Pz - Bz, Ez);

                        const double radial_prefactor =
                            cs_M[ip] * cs_N[iq]
                            * std::exp(-alpha * beta * inv_gamma * AB_sq)
                            * std::pow(kPi * inv_gamma, 1.5);
                        for (std::size_t k = 0; k < n_G; ++k) {
                            const double gx = G_vectors(
                                static_cast<Eigen::Index>(k), 0);
                            const double gy = G_vectors(
                                static_cast<Eigen::Index>(k), 1);
                            const double gz = G_vectors(
                                static_cast<Eigen::Index>(k), 2);
                            const double gp = gx * Px + gy * Py + gz * Pz;
                            prefactor_per_G[k] =
                                phase_R
                                * (radial_prefactor
                                   * std::exp(-0.25 * G2[k] * inv_gamma)
                                   * std::complex<double>(
                                       std::cos(gp), -std::sin(gp)));
                        }

                        for (std::size_t a = 0; a < n_cart_M; ++a) {
                            const std::array<int, 3> cart_A = {
                                cart_M[a].i, cart_M[a].j, cart_M[a].k};
                            for (std::size_t b = 0; b < n_cart_N; ++b) {
                                if (q_nonzero[a * n_cart_N + b] == 0) {
                                    continue;
                                }
                                const std::complex<double>* q_ab =
                                    &Q_cart[(a * n_cart_N + b) * n_G];
                                const std::array<int, 3> cart_B = {
                                    cart_N[b].i, cart_N[b].j, cart_N[b].k};
                                for (int axis = 0; axis < 3; ++axis) {
                                    translation[static_cast<std::size_t>(axis)]
                                        += contract_md_ft_line_gweighted(
                                            cart_A, cart_B, Ex, Ey, Ez,
                                            e_dim_lb, e_dim_t, Gx_pow,
                                            Gy_pow, Gz_pow, prefactor_per_G,
                                            G_vectors, q_ab, 1.0, axis);
                                    if (shM.atom_index == shN.atom_index) {
                                        continue;
                                    }
                                    auto raised = cart_B;
                                    raised[static_cast<std::size_t>(axis)] += 1;
                                    ket[static_cast<std::size_t>(axis)]
                                        += contract_md_ft_line_gweighted(
                                            cart_A, raised, Ex, Ey, Ez,
                                            e_dim_lb, e_dim_t, Gx_pow,
                                            Gy_pow, Gz_pow, prefactor_per_G,
                                            G_vectors, q_ab, 2.0 * beta);
                                    if (cart_B[
                                            static_cast<std::size_t>(axis)] > 0) {
                                        auto lowered = cart_B;
                                        lowered[
                                            static_cast<std::size_t>(axis)] -= 1;
                                        ket[static_cast<std::size_t>(axis)]
                                            += contract_md_ft_line_gweighted(
                                                cart_A, lowered, Ex, Ey, Ez,
                                                e_dim_lb, e_dim_t, Gx_pow,
                                                Gy_pow, Gz_pow,
                                                prefactor_per_G, G_vectors,
                                                q_ab,
                                                -static_cast<double>(cart_B[
                                                    static_cast<std::size_t>(
                                                        axis)]));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (int axis = 0; axis < 3; ++axis) {
                const auto idx = static_cast<std::size_t>(axis);
                if (shM.atom_index == shN.atom_index) {
                    contribution[idx] = translation[idx];
                } else {
                    contribution[idx] = translation[idx] - ket[idx];
                    contribution[idx + 3] = ket[idx];
                }
            }
        }
    }

    for (std::size_t sM = 0; sM < n_shells; ++sM) {
        for (std::size_t sN = 0; sN < n_shells; ++sN) {
            const auto& contribution =
                pair_contributions[sM * n_shells + sN];
            const auto atom_M = static_cast<Eigen::Index>(
                shells[sM].atom_index);
            const auto atom_N = static_cast<Eigen::Index>(
                shells[sN].atom_index);
            for (int axis = 0; axis < 3; ++axis) {
                gradient(atom_M, axis) +=
                    contribution[static_cast<std::size_t>(axis)];
                gradient(atom_N, axis) +=
                    contribution[static_cast<std::size_t>(axis + 3)];
            }
        }
    }
    return gradient;
}

namespace {

constexpr std::size_t kPairPanelMdExtent =
    (cart_to_sph_data::kMaxL + 1) * (cart_to_sph_data::kMaxL + 1)
    * (2 * cart_to_sph_data::kMaxL + 1);
constexpr std::size_t kPairPanelPowerExtent = 2 * cart_to_sph_data::kMaxL + 1;

struct PairPanelNumericWorkspace {
    std::array<double, 3 * kPairPanelMdExtent> md;
    std::array<std::complex<double>, 3 * kPairPanelPowerExtent> powers;
};

// Fixed-size reciprocal tiles amortize image and primitive preparation without
// retaining any image list or changing an individual vector's sum order.
constexpr std::size_t kPairPanelVectorTile = 32;
struct PairPanelVectorState {
    Eigen::Vector3d p;
    double p2;
    std::array<std::complex<double>, 3 * kPairPanelPowerExtent> powers;
    double real_sum, imag_sum, real_correction, imag_correction;
};
using PairPanelVectorWorkspace = std::array<PairPanelVectorState, kPairPanelVectorTile>;

constexpr std::size_t kPairDerivativeMdExtent =
    (cart_to_sph_data::kMaxL + 1) * (cart_to_sph_data::kMaxL + 2)
    * (2 * cart_to_sph_data::kMaxL + 2);
constexpr std::size_t kPairDerivativePowerExtent = 2 * cart_to_sph_data::kMaxL + 2;
struct PairDerivativeNumericWorkspace {
    std::array<double, 3 * kPairDerivativeMdExtent> md;
    std::array<std::complex<double>, 3 * kPairDerivativePowerExtent> powers;
    std::array<std::complex<double>, 7> sums{}, corrections{};
};

static_assert(sizeof(double) == 8 && sizeof(std::complex<double>) == 16,
              "AO-pair panel accounting requires binary64/complex128 storage");

using PairPanelAo = aopair_ft_detail::Ao;
using PairPanelImageBox = aopair_ft_detail::ImageBox;

std::uint64_t pair_panel_product(std::uint64_t a, std::uint64_t b) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::length_error("AO-pair Fourier panel extent overflows uint64");
    }
    return a * b;
}

double pair_panel_norm2(double x, double y, double z) {
    return std::fma(x, x, std::fma(y, y, std::fma(z, z, 0.0)));
}

void pair_panel_require_finite(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("AO-pair Fourier panel requires finite values");
    }
}

PairPanelAo pair_panel_ao(const BasisSet& basis, std::uint64_t ao) {
    std::uint64_t offset = 0;
    for (const auto& shell : basis.libint()) {
        for (const auto& contraction : shell.contr) {
            const auto size = static_cast<std::uint64_t>(2 * contraction.l + 1);
            if (ao >= offset && ao - offset < size) {
                return {&shell, &contraction, static_cast<std::size_t>(ao - offset)};
            }
            offset += size;
        }
    }
    throw std::logic_error("AO-pair Fourier panel AO lookup failed");
}

using PairPanelLongMatrix = Eigen::Matrix<long double, 3, 3>;

PairPanelImageBox pair_panel_image_box(
    const PairPanelAo& bra, const PairPanelAo& ket,
    const PairPanelLongMatrix& inverse, double cutoff) {
    Eigen::Matrix<long double, 3, 1> separation;
    for (int d = 0; d < 3; ++d) {
        separation[d] = static_cast<long double>(bra.shell->O[d])
                        - static_cast<long double>(ket.shell->O[d]);
    }
    const Eigen::Matrix<long double, 3, 1> center = inverse * separation;
    PairPanelImageBox box;
    for (int d = 0; d < 3; ++d) {
        const long double radius = static_cast<long double>(cutoff)
                                   * inverse.row(d).norm();
        // A numerical guard band for inverse/center arithmetic, followed by
        // a full label of padding. This is deliberately not advertised as
        // an interval-arithmetic certification of the source enumeration.
        const long double pad = 1.0L + 64.0L
            * std::numeric_limits<long double>::epsilon()
            * (1.0L + std::abs(center[d]) + radius);
        const long double lower = std::floor(center[d] - radius - pad);
        const long double upper = std::ceil(center[d] + radius + pad);
        constexpr long double label_limit = 4503599627370496.0L;
        if (!std::isfinite(lower) || !std::isfinite(upper)
            || lower < -label_limit || upper > label_limit) {
            throw std::length_error("AO-pair Fourier image label exceeds exact binary64 range");
        }
        box.lower[d] = static_cast<std::int64_t>(lower);
        box.upper[d] = static_cast<std::int64_t>(upper);
        box.candidates = pair_panel_product(
            box.candidates,
            static_cast<std::uint64_t>(box.upper[d] - box.lower[d]) + 1U);
    }
    return box;
}

template<class Function>
std::uint64_t pair_panel_walk_images(
    const PairPanelAo& bra, const PairPanelAo& ket,
    const Eigen::Matrix3d& lattice, const PairPanelImageBox& box,
    double cutoff_squared, Function&& function) {
    return aopair_ft_detail::walk_images(bra, ket, lattice, box, cutoff_squared,
        [&](const std::array<std::int64_t, 3>&, const Eigen::Vector3d& translation,
            const Eigen::Vector3d& separation, double squared) {
            function(translation, separation, squared);
        });
}

void pair_panel_add(double value, double& sum, double& correction) {
    const double next = sum + value;
    correction += std::abs(sum) >= std::abs(value)
        ? (sum - next) + value : (value - next) + sum;
    sum = next;
    pair_panel_require_finite(sum);
    pair_panel_require_finite(correction);
}

std::complex<double> pair_panel_polynomial(
    const PairPanelAo& bra, const PairPanelAo& ket,
    const PairPanelNumericWorkspace& workspace,
    const std::complex<double>* vector_powers) {
    const int la = bra.contraction->l;
    const int lb = ket.contraction->l;
    const std::size_t nb = static_cast<std::size_t>(lb + 1);
    const std::size_t nt = static_cast<std::size_t>(la + lb + 1);
    const auto* acart = cart_to_sph_data::cart_table_for_l(la);
    const auto* bcart = cart_to_sph_data::cart_table_for_l(lb);
    const auto na = cart_to_sph_data::n_cart_for_l(la);
    const auto nc = cart_to_sph_data::n_cart_for_l(lb);
    const double* asph = cart_to_sph_data::sph_table_for_l(la)
                         + bra.component * na;
    const double* bsph = cart_to_sph_data::sph_table_for_l(lb)
                         + ket.component * nc;
    std::complex<double> total(0.0, 0.0);
    for (std::size_t a = 0; a < na; ++a) {
        if (asph[a] == 0.0) {
            continue;
        }
        const std::array<int, 3> ap = {acart[a].i, acart[a].j, acart[a].k};
        for (std::size_t b = 0; b < nc; ++b) {
            if (bsph[b] == 0.0) {
                continue;
            }
            const std::array<int, 3> bp = {bcart[b].i, bcart[b].j, bcart[b].k};
            std::complex<double> product(asph[a] * bsph[b], 0.0);
            for (std::size_t d = 0; d < 3; ++d) {
                const double* coefficients = workspace.md.data()
                    + d * kPairPanelMdExtent
                    + (static_cast<std::size_t>(ap[d]) * nb
                       + static_cast<std::size_t>(bp[d])) * nt;
                const auto* powers = vector_powers + d * kPairPanelPowerExtent;
                std::complex<double> axis(0.0, 0.0);
                for (int t = 0; t <= ap[d] + bp[d]; ++t) {
                    axis += coefficients[t] * powers[t];
                }
                product *= axis;
            }
            total += product;
        }
    }
    return total;
}

// Preparation depends on the image and primitive pair, but not on G. Both
// image policies share it and the scalar contribution below.
template<typename Callback>
void pair_panel_prepare_primitives(
    const PairPanelAo& bra, const PairPanelAo& ket,
    const Eigen::Vector3d& separation, PairPanelNumericWorkspace& workspace,
    Callback&& callback) {
    const int la = bra.contraction->l;
    const int lb = ket.contraction->l;
    for (std::size_t ia = 0; ia < bra.shell->alpha.size(); ++ia) {
        const double alpha = bra.shell->alpha[ia];
        for (std::size_t ib = 0; ib < ket.shell->alpha.size(); ++ib) {
            const double beta = ket.shell->alpha[ib];
            const double gamma = alpha + beta;
            pair_panel_require_finite(gamma);
            const double alpha_fraction = alpha / gamma;
            const double beta_fraction = beta / gamma;
            Eigen::Vector3d center;
            for (std::size_t d = 0; d < 3; ++d) {
                const double pa = -beta_fraction * separation[d];
                const double pb = alpha_fraction * separation[d];
                center[d] = bra.shell->O[d] + pa;
                md_e_coefficients_1d_buffer(
                    la, lb, gamma, pa, pb,
                    workspace.md.data() + d * kPairPanelMdExtent);
            }
            callback(ia, ib, gamma, alpha_fraction, beta, center);
        }
    }
}

void pair_panel_accumulate_primitive(
    const PairPanelAo& bra, const PairPanelAo& ket,
    std::size_t ia, std::size_t ib, double gamma, double alpha_fraction, double beta,
    const Eigen::Vector3d& center, const Eigen::Vector3d& p, double p2,
    double scale_a, double scale_b, const std::complex<double>& bloch_phase,
    double separation2, const PairPanelNumericWorkspace& workspace,
    const std::complex<double>* powers, double& real_sum, double& imag_sum,
    double& real_correction, double& imag_correction) {
    const double argument = p.dot(center);
    pair_panel_require_finite(argument);
    // Keep the original combined exponential and multiplication order. Splitting
    // image/G exponentials would change rounding and underflow behavior.
    const double radial = bra.contraction->coeff[ia]
        * ket.contraction->coeff[ib]
        * std::pow(kPi / gamma, 1.5)
        * std::exp(-alpha_fraction * beta * separation2 - p2 * (0.25 / gamma));
    pair_panel_require_finite(radial);
    const std::complex<double> value =
        (scale_a * scale_b * radial) * bloch_phase
        * std::complex<double>(std::cos(argument), -std::sin(argument))
        * pair_panel_polynomial(bra, ket, workspace, powers);
    pair_panel_require_finite(value.real());
    pair_panel_require_finite(value.imag());
    pair_panel_add(value.real(), real_sum, real_correction);
    pair_panel_add(value.imag(), imag_sum, imag_correction);
}

void pair_panel_accumulate_image(
    const PairPanelAo& bra, const PairPanelAo& ket,
    const Eigen::Vector3d& p, double p2, const Eigen::Vector3d& k_cart,
    double scale_a, double scale_b, const Eigen::Vector3d& translation,
    const Eigen::Vector3d& separation, double separation2,
    PairPanelNumericWorkspace& workspace, double& real_sum, double& imag_sum,
    double& real_correction, double& imag_correction) {
    const double kr = k_cart.dot(translation);
    pair_panel_require_finite(kr);
    const std::complex<double> bloch_phase(std::cos(kr), std::sin(kr));
    pair_panel_prepare_primitives(bra, ket, separation, workspace,
        [&](std::size_t ia, std::size_t ib, double gamma, double alpha_fraction,
            double beta, const Eigen::Vector3d& center) {
            pair_panel_accumulate_primitive(bra, ket, ia, ib, gamma, alpha_fraction,
                beta, center, p, p2, scale_a, scale_b, bloch_phase, separation2,
                workspace, workspace.powers.data(), real_sum, imag_sum,
                real_correction, imag_correction);
        });
}

std::uint64_t pair_cell_add(std::uint64_t a, std::uint64_t b) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
        throw std::length_error("AO-pair cell panel census overflows uint64");
    return a + b;
}
void pair_cell_cap(std::uint64_t value, std::uint64_t bound, const char* message) {
    if (value > bound) throw std::length_error(message);
}
void pair_cell_fp() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
    throw std::runtime_error("AO-pair cell panel requires strict finite IEEE arithmetic");
#endif
#if FLT_EVAL_METHOD != 0
    throw std::runtime_error("AO-pair cell panel requires binary64 evaluation");
#endif
    if (!std::numeric_limits<double>::is_iec559 || FLT_RADIX != 2
        || std::numeric_limits<double>::digits != 53
        || std::fegetround() != FE_TONEAREST)
        throw std::runtime_error("AO-pair cell panel requires nearest IEEE binary64");
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0, two = 2.0;
    volatile double normal = std::numeric_limits<double>::min();
    volatile double half = 0.5;
    if (tiny * one != tiny || tiny + tiny != tiny * two
        || std::fma(tiny, one, 0.0) != tiny
        || std::nextafter(0.0, 1.0) != tiny || normal * half == 0.0)
        throw std::runtime_error("AO-pair cell panel requires gradual underflow");
}

Eigen::Vector3d pair_cell_translation(
    const Eigen::Matrix3d& lattice, const std::int64_t* label) {
    Eigen::Vector3d translation;
    for (int d = 0; d < 3; ++d) {
        translation[d] = std::fma(static_cast<double>(label[0]), lattice(d, 0),
            std::fma(static_cast<double>(label[1]), lattice(d, 1),
                     static_cast<double>(label[2]) * lattice(d, 2)));
        pair_panel_require_finite(translation[d]);
    }
    return translation;
}

}  // namespace

namespace aopair_ft_detail {
Ao ao(const BasisSet& basis, std::uint64_t index) {
    return pair_panel_ao(basis, index);
}
ImageBox image_box(const Ao& bra, const Ao& ket, const LongMatrix& inverse, double cutoff) {
    return pair_panel_image_box(bra, ket, inverse, cutoff);
}
void md_coefficients(int la, int lb, double gamma, double pa, double pb, double* output) {
    md_e_coefficients_1d_buffer(la, lb, gamma, pa, pb, output);
}

std::size_t center_derivative_numeric_workspace_bytes() noexcept {
    return sizeof(PairDerivativeNumericWorkspace);
}

std::array<std::complex<double>, 7> value_and_center_derivatives(
    const Ao& bra, const Ao& ket, const Eigen::Vector3d& p,
    const Eigen::Vector3d& k_cart, const Eigen::Vector3d& translation,
    const Eigen::Vector3d& separation, double separation_squared) {
    using Complex = std::complex<double>;
    const int la = bra.contraction->l, lb = ket.contraction->l;
    if (la < 0 || lb < 0 || la > cart_to_sph_data::kMaxL || lb > cart_to_sph_data::kMaxL)
        throw std::invalid_argument("AO-pair center derivative angular momentum is unsupported");
    PairDerivativeNumericWorkspace workspace;
    const auto* acart = cart_to_sph_data::cart_table_for_l(la);
    const auto* bcart = cart_to_sph_data::cart_table_for_l(lb);
    const auto na = cart_to_sph_data::n_cart_for_l(la);
    const auto nb = cart_to_sph_data::n_cart_for_l(lb);
    const auto* asph = cart_to_sph_data::sph_table_for_l(la) + bra.component * na;
    const auto* bsph = cart_to_sph_data::sph_table_for_l(lb) + ket.component * nb;
    const double scale_a = la == 0 ? 1.0 : std::sqrt(4.0 * kPi / (2 * la + 1));
    const double scale_b = lb == 0 ? 1.0 : std::sqrt(4.0 * kPi / (2 * lb + 1));
    const double kr = k_cart.dot(translation);
    pair_panel_require_finite(kr);
    const Complex phase(std::cos(kr), std::sin(kr));
    const double p2 = pair_panel_norm2(p[0], p[1], p[2]);
    pair_panel_require_finite(p2);
    pair_panel_require_finite(separation_squared);
    const std::size_t stride_b = lb + 2, stride_t = la + lb + 2;
    for (int axis = 0; axis < 3; ++axis) {
        auto* powers = workspace.powers.data() + axis * kPairDerivativePowerExtent;
        powers[0] = 1.0;
        for (int t = 1; t <= la + lb + 1; ++t)
            powers[t] = powers[t - 1] * Complex(0.0, -p[axis]);
    }
    for (std::size_t ia = 0; ia < bra.shell->alpha.size(); ++ia)
        for (std::size_t ib = 0; ib < ket.shell->alpha.size(); ++ib) {
            const double alpha = bra.shell->alpha[ia], beta = ket.shell->alpha[ib];
            const double gamma = alpha + beta;
            pair_panel_require_finite(gamma);
            const double fa = alpha / gamma, fb = beta / gamma;
            Eigen::Vector3d center;
            for (int axis = 0; axis < 3; ++axis) {
                const double pa = -fb * separation[axis], pb = fa * separation[axis];
                center[axis] = bra.shell->O[axis] + pa;
                md_e_coefficients_1d_buffer(la, lb + 1, gamma, pa, pb,
                    workspace.md.data() + axis * kPairDerivativeMdExtent);
            }
            const double radial = bra.contraction->coeff[ia] * ket.contraction->coeff[ib]
                * std::pow(kPi / gamma, 1.5)
                * std::exp(-fa * beta * separation_squared - p2 * (0.25 / gamma));
            pair_panel_require_finite(radial);
            const double argument = p.dot(center);
            pair_panel_require_finite(argument);
            const Complex prefactor = scale_a * scale_b * radial * phase
                * Complex(std::cos(argument), -std::sin(argument));
            std::array<Complex, 4> polynomial{};
            for (std::size_t a = 0; a < na; ++a) {
                if (asph[a] == 0.0) continue;
                const int ap[] = {acart[a].i, acart[a].j, acart[a].k};
                for (std::size_t b = 0; b < nb; ++b) {
                    if (bsph[b] == 0.0) continue;
                    const int bp[] = {bcart[b].i, bcart[b].j, bcart[b].k};
                    std::array<Complex, 3> value{}, derivative{};
                    for (int axis = 0; axis < 3; ++axis) {
                        auto line = [&](int ket_power) {
                            Complex result{};
                            if (ket_power < 0) return result;
                            const auto* coefficients = workspace.md.data()
                                + axis * kPairDerivativeMdExtent
                                + (ap[axis] * stride_b + ket_power) * stride_t;
                            const auto* powers = workspace.powers.data()
                                + axis * kPairDerivativePowerExtent;
                            for (int t = 0; t <= ap[axis] + ket_power; ++t)
                                result += coefficients[t] * powers[t];
                            return result;
                        };
                        value[axis] = line(bp[axis]);
                        // Helgaker and Taylor (1992), Eq. (15),
                        // doi:10.1007/BF01132826, applied to the ket center.
                        // Differentiate the original polynomial Gaussian.
                        // Raised/lowered angular integrals keep the original
                        // normalization and solid-harmonic coefficients.
                        derivative[axis] = 2 * beta * line(bp[axis] + 1)
                            - double(bp[axis]) * line(bp[axis] - 1);
                    }
                    const double angular = asph[a] * bsph[b];
                    polynomial[0] += angular * value[0] * value[1] * value[2];
                    for (int axis = 0; axis < 3; ++axis)
                        polynomial[axis + 1] += angular * derivative[axis]
                            * value[(axis + 1) % 3] * value[(axis + 2) % 3];
                }
            }
            std::array<Complex, 7> contribution{};
            contribution[0] = prefactor * polynomial[0];
            for (int axis = 0; axis < 3; ++axis) {
                contribution[axis + 4] = prefactor * polynomial[axis + 1];
                // A rigid translation gives -i*p times the Fourier value.
                contribution[axis + 1] = Complex(0.0, -p[axis]) * contribution[0]
                    - contribution[axis + 4];
            }
            for (std::size_t i = 0; i < contribution.size(); ++i) {
                pair_panel_require_finite(contribution[i].real());
                pair_panel_require_finite(contribution[i].imag());
                double real = workspace.sums[i].real(), imag = workspace.sums[i].imag();
                double cr = workspace.corrections[i].real(), ci = workspace.corrections[i].imag();
                pair_panel_add(contribution[i].real(), real, cr);
                pair_panel_add(contribution[i].imag(), imag, ci);
                workspace.sums[i] = {real, imag};
                workspace.corrections[i] = {cr, ci};
            }
        }
    for (std::size_t i = 0; i < workspace.sums.size(); ++i) {
        workspace.sums[i] += workspace.corrections[i];
        pair_panel_require_finite(workspace.sums[i].real());
        pair_panel_require_finite(workspace.sums[i].imag());
    }
    return workspace.sums;
}
} // namespace aopair_ft_detail

std::uint64_t ao_pair_fourier_fixed_numeric_workspace_bytes() noexcept {
    return sizeof(PairPanelNumericWorkspace) + sizeof(PairPanelVectorWorkspace);
}

AOPairFourierPanel ao_pair_gaussian_fourier_panel(
    const BasisSet& basis, const PeriodicSystem& system,
    AuxiliaryFourierVectorView vectors, const Eigen::Vector3d& k_ket_cart,
    std::uint64_t pair_begin, std::uint64_t pair_count,
    double image_cutoff_bohr, std::uint64_t maximum_image_candidates,
    std::uint64_t output_byte_cap) {
    return ao_pair_gaussian_fourier_panel(
        basis, basis, system, vectors, k_ket_cart, pair_begin, pair_count,
        image_cutoff_bohr, maximum_image_candidates, output_byte_cap);
}

AOPairFourierPanel ao_pair_gaussian_fourier_panel(
    const BasisSet& bra_basis, const BasisSet& ket_basis,
    const PeriodicSystem& system,
    AuxiliaryFourierVectorView vectors, const Eigen::Vector3d& k_ket_cart,
    std::uint64_t pair_begin, std::uint64_t pair_count,
    double image_cutoff_bohr, std::uint64_t maximum_image_candidates,
    std::uint64_t output_byte_cap) {
    const auto n_bra_basis = static_cast<std::uint64_t>(bra_basis.nbasis());
    const auto n_ket_basis = static_cast<std::uint64_t>(ket_basis.nbasis());
    const auto n_pairs = pair_panel_product(n_bra_basis, n_ket_basis);
    if (pair_begin > n_pairs || pair_count > n_pairs - pair_begin) {
        throw std::invalid_argument("AO-pair Fourier interval exceeds basis pair product");
    }
    const auto elements = pair_panel_product(pair_count, vectors.count);
    const auto bytes = pair_panel_product(elements, sizeof(std::complex<double>));
    if (output_byte_cap == 0 || bytes > output_byte_cap
        || elements > std::vector<std::complex<double>>().max_size()
        || pair_count > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("AO-pair Fourier output exceeds byte or address cap");
    }
    if (maximum_image_candidates == 0) {
        throw std::invalid_argument("AO-pair Fourier image candidate cap must be positive");
    }
    if (system.dim != 3 || !system.lattice.allFinite()
        || !k_ket_cart.allFinite() || !std::isfinite(image_cutoff_bohr)
        || image_cutoff_bohr < 0.0) {
        throw std::invalid_argument("AO-pair Fourier panel requires finite 3D geometry and nonnegative cutoff");
    }
    const double cutoff_squared = image_cutoff_bohr * image_cutoff_bohr;
    pair_panel_require_finite(cutoff_squared);
    const PairPanelLongMatrix lattice = system.lattice.cast<long double>();
    const long double determinant = lattice.determinant();
    if (!std::isfinite(determinant) || determinant == 0.0L) {
        throw std::invalid_argument("AO-pair Fourier panel requires a nonsingular lattice");
    }
    const PairPanelLongMatrix inverse = lattice.inverse();
    const long double condition_bound = lattice.norm() * inverse.norm();
    if (!inverse.allFinite() || !std::isfinite(condition_bound)
        || condition_bound > 1.0e10L) {
        throw std::invalid_argument("AO-pair Fourier lattice exceeds numerical conditioning policy");
    }

    const auto validate_basis = [](const BasisSet& basis, std::uint64_t n_basis) {
        std::uint64_t basis_extent = 0;
        for (const auto& shell : basis.libint()) {
            for (double origin : shell.O) {
                pair_panel_require_finite(origin);
            }
            if (shell.alpha.empty()) {
                throw std::invalid_argument("AO-pair Fourier shell has no primitives");
            }
            for (double exponent : shell.alpha) {
                if (!std::isfinite(exponent) || exponent <= 0.0) {
                    throw std::invalid_argument("AO-pair Fourier exponents must be finite and positive");
                }
            }
            for (const auto& contraction : shell.contr) {
                if (contraction.l < 0 || contraction.l > cart_to_sph_data::kMaxL
                    || (!contraction.pure && contraction.l > 0)) {
                    throw std::invalid_argument("AO-pair Fourier supports only pure spherical L<=6 and Cartesian s shells");
                }
                if (contraction.coeff.size() != shell.alpha.size()) {
                    throw std::invalid_argument("AO-pair Fourier contraction extent mismatch");
                }
                for (double coefficient : contraction.coeff) {
                    pair_panel_require_finite(coefficient);
                }
                const auto count = static_cast<std::uint64_t>(2 * contraction.l + 1);
                if (basis_extent > n_basis || count > n_basis - basis_extent) {
                    throw std::invalid_argument("AO-pair Fourier basis extent mismatch");
                }
                basis_extent += count;
            }
        }
        if (basis_extent != n_basis) {
            throw std::invalid_argument("AO-pair Fourier basis extent mismatch");
        }
    };
    validate_basis(bra_basis, n_bra_basis);
    validate_basis(ket_basis, n_ket_basis);
    const std::array<const double*, 3> lanes = {vectors.x, vectors.y, vectors.z};
    const std::array<std::ptrdiff_t, 3> strides = {
        vectors.x_stride, vectors.y_stride, vectors.z_stride};
    for (std::size_t d = 0; d < 3; ++d) {
        if (strides[d] <= 0 || (vectors.count != 0 && lanes[d] == nullptr)
            || (vectors.count > 1
                && vectors.count - 1 > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max() / strides[d]))) {
            throw std::invalid_argument("AO-pair Fourier vector lane or stride is invalid");
        }
    }
    for (std::size_t v = 0; v < vectors.count; ++v) {
        Eigen::Vector3d p;
        for (std::size_t d = 0; d < 3; ++d) {
            p[d] = lanes[d][static_cast<std::ptrdiff_t>(v) * strides[d]];
            pair_panel_require_finite(p[d]);
        }
        pair_panel_require_finite(pair_panel_norm2(p[0], p[1], p[2]));
    }

    AOPairFourierPanel output;
    output.pair_begin = pair_begin;
    output.n_pairs = static_cast<std::size_t>(pair_count);
    output.n_vectors = vectors.count;
    output.output_bytes = bytes;
    output.fixed_numeric_workspace_bytes = ao_pair_fourier_fixed_numeric_workspace_bytes();
    // Count and validate every selected image before allocating the output.
    for (std::uint64_t row = 0; row < pair_count; ++row) {
        const auto bra = pair_panel_ao(bra_basis, (pair_begin + row) / n_ket_basis);
        const auto ket = pair_panel_ao(ket_basis, (pair_begin + row) % n_ket_basis);
        const auto box = pair_panel_image_box(bra, ket, inverse, image_cutoff_bohr);
        if (box.candidates > maximum_image_candidates - output.image_candidate_count) {
            throw std::length_error("AO-pair Fourier image candidates exceed source cap");
        }
        output.image_candidate_count += box.candidates;
        output.retained_pair_image_count += pair_panel_walk_images(
            bra, ket, system.lattice, box, cutoff_squared,
            [](const Eigen::Vector3d&, const Eigen::Vector3d&, double) {});
    }
    output.data.resize(static_cast<std::size_t>(elements));
    PairPanelNumericWorkspace workspace;
    PairPanelVectorWorkspace tile;
    for (std::uint64_t row = 0; row < pair_count; ++row) {
        const auto bra = pair_panel_ao(bra_basis, (pair_begin + row) / n_ket_basis);
        const auto ket = pair_panel_ao(ket_basis, (pair_begin + row) % n_ket_basis);
        const int la = bra.contraction->l;
        const int lb = ket.contraction->l;
        const double scale_a = la == 0 ? 1.0 : std::sqrt(4.0 * kPi / (2 * la + 1));
        const double scale_b = lb == 0 ? 1.0 : std::sqrt(4.0 * kPi / (2 * lb + 1));
        const auto box = pair_panel_image_box(bra, ket, inverse, image_cutoff_bohr);
        for (std::size_t begin = 0; begin < vectors.count;) {
            const auto count = std::min(kPairPanelVectorTile, vectors.count - begin);
            for (std::size_t v = 0; v < count; ++v) {
                auto& state = tile[v];
                for (std::size_t d = 0; d < 3; ++d) {
                    state.p[d] = lanes[d][static_cast<std::ptrdiff_t>(begin + v) * strides[d]];
                    auto* powers = state.powers.data() + d * kPairPanelPowerExtent;
                    powers[0] = {1.0, 0.0};
                    for (int t = 1; t <= la + lb; ++t) {
                        powers[t] = powers[t - 1] * std::complex<double>(0.0, -state.p[d]);
                    }
                }
                state.p2 = pair_panel_norm2(state.p[0], state.p[1], state.p[2]);
                state.real_sum = state.imag_sum = 0.0;
                state.real_correction = state.imag_correction = 0.0;
            }
            pair_panel_walk_images(
                bra, ket, system.lattice, box, cutoff_squared,
                [&](const Eigen::Vector3d& translation,
                    const Eigen::Vector3d& separation, double separation2) {
                    const double kr = k_ket_cart.dot(translation);
                    pair_panel_require_finite(kr);
                    const std::complex<double> bloch_phase(std::cos(kr), std::sin(kr));
                    pair_panel_prepare_primitives(bra, ket, separation, workspace,
                        [&](std::size_t ia, std::size_t ib, double gamma,
                            double alpha_fraction, double beta, const Eigen::Vector3d& center) {
                            for (std::size_t v = 0; v < count; ++v) {
                                auto& state = tile[v];
                                pair_panel_accumulate_primitive(bra, ket, ia, ib, gamma,
                                    alpha_fraction, beta, center, state.p, state.p2,
                                    scale_a, scale_b, bloch_phase, separation2, workspace,
                                    state.powers.data(), state.real_sum, state.imag_sum,
                                    state.real_correction, state.imag_correction);
                            }
                        });
                });
            for (std::size_t v = 0; v < count; ++v) {
                const auto& state = tile[v];
                const double real = state.real_sum + state.real_correction;
                const double imag = state.imag_sum + state.imag_correction;
                pair_panel_require_finite(real);
                pair_panel_require_finite(imag);
                output.data[static_cast<std::size_t>(row) * vectors.count + begin + v] = {
                    real == 0.0 ? 0.0 : real, imag == 0.0 ? 0.0 : imag};
            }
            begin += count;
        }
    }
    return output;
}

// Explicit-cell counterpart of the finite-distance panel. All count-only
// admission precedes floating payload and cell-label reads.
AOPairFourierCellPanelPlan plan_ao_pair_gaussian_fourier_cell_panel(
    const BasisSet& basis, std::uint64_t n_vectors, std::uint64_t pair_begin,
    std::uint64_t pair_count, std::uint64_t cell_count,
    const AOPairFourierCellPanelCaps& caps) {
    using U = std::uint64_t;
    if (!caps.maximum_cells || !caps.maximum_pair_cell_visits
        || !caps.maximum_work_units || !caps.maximum_output_bytes)
        throw std::invalid_argument("AO-pair cell panel requires positive explicit caps");
    AOPairFourierCellPanelPlan p;
    p.n_basis = basis.nbasis(); p.n_vectors = n_vectors;
    p.pair_begin = pair_begin; p.n_pairs = pair_count; p.cell_count = cell_count;
    const U total_pairs = pair_panel_product(p.n_basis, p.n_basis);
    if (pair_begin > total_pairs || pair_count > total_pairs - pair_begin)
        throw std::invalid_argument("AO-pair cell panel interval exceeds basis square");
    pair_cell_cap(cell_count, caps.maximum_cells, "AO-pair cell panel cell cap");
    p.pair_cell_visits = pair_panel_product(pair_count, cell_count);
    pair_cell_cap(p.pair_cell_visits, caps.maximum_pair_cell_visits,
                  "AO-pair cell panel pair-cell visit cap");
    const U elements = pair_panel_product(pair_count, n_vectors);
    p.output_bytes = pair_panel_product(16, elements);
    pair_cell_cap(p.output_bytes, caps.maximum_output_bytes, "AO-pair cell panel output cap");
    p.borrowed_cell_bytes = pair_panel_product(24, cell_count);
    if (elements > std::vector<std::complex<double>>().max_size()
        || n_vectors > std::numeric_limits<std::size_t>::max()
        || pair_count > std::numeric_limits<std::size_t>::max()
        || p.borrowed_cell_bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::length_error("AO-pair cell panel extent exceeds address range");
    p.duplicate_comparisons = cell_count == 0 ? 0 : pair_panel_product(
        cell_count % 2 == 0 ? cell_count / 2 : cell_count,
        cell_count % 2 == 0 ? cell_count - 1 : (cell_count - 1) / 2);
    p.metadata_work_units = 1024;
    U lanes = 0, contraction_count = 0, max_primitives = 0, max_l = 0, counted_ao = 0;
    p.basis_control_storage_bytes = pair_cell_add(sizeof(BasisSet),
        pair_cell_add(basis.name().size(), 1));
    p.basis_control_storage_bytes = pair_cell_add(p.basis_control_storage_bytes,
        pair_panel_product(basis.nshells(), sizeof(libint2::Shell)
            + sizeof(int) + sizeof(std::size_t)));
    // Initial count admission also covers a potentially very large duplicate
    // loop before walking even the first basis descriptor.
    const U fixed_validation = pair_cell_add(2048,
        pair_cell_add(pair_panel_product(32, p.duplicate_comparisons),
            pair_cell_add(pair_panel_product(256, cell_count),
                pair_cell_add(pair_panel_product(128, n_vectors),
                              pair_panel_product(256, p.pair_cell_visits)))));
    const auto admit_census = [&]() {
        pair_cell_cap(pair_cell_add(p.metadata_work_units,
            pair_cell_add(fixed_validation, pair_panel_product(32, lanes))),
            caps.maximum_work_units, "AO-pair cell panel metadata/validation work cap");
    };
    admit_census();
    for (const auto& shell : basis.libint()) {
        p.metadata_work_units = pair_cell_add(p.metadata_work_units, 256);
        admit_census();
        if (shell.alpha.empty() || shell.contr.empty())
            throw std::invalid_argument("AO-pair cell panel empty shell metadata");
        lanes = pair_cell_add(lanes, pair_cell_add(3,
            pair_cell_add(shell.alpha.size(), shell.max_ln_coeff.size())));
        max_primitives = std::max(max_primitives, static_cast<U>(shell.alpha.size()));
        contraction_count = pair_cell_add(contraction_count, shell.contr.size());
        p.basis_control_storage_bytes = pair_cell_add(p.basis_control_storage_bytes,
            pair_panel_product(shell.contr.size(), sizeof(libint2::Shell::Contraction)));
        admit_census();
        for (const auto& c : shell.contr) {
            p.metadata_work_units = pair_cell_add(p.metadata_work_units, 128);
            admit_census();
            if (c.l < 0 || c.l > cart_to_sph_data::kMaxL || (!c.pure && c.l > 0)
                || c.coeff.size() != shell.alpha.size())
                throw std::invalid_argument("AO-pair cell panel requires matched pure L<=6 or Cartesian s contractions");
            lanes = pair_cell_add(lanes, c.coeff.size());
            counted_ao = pair_cell_add(counted_ao, static_cast<U>(2 * c.l + 1));
            max_l = std::max(max_l, static_cast<U>(c.l));
            admit_census();
        }
    }
    if (counted_ao != p.n_basis)
        throw std::invalid_argument("AO-pair cell panel basis extent mismatch");
    p.borrowed_basis_numeric_bytes = pair_panel_product(8, lanes);
    p.validation_work_units = pair_cell_add(fixed_validation, pair_panel_product(32, lanes));
    const U t = 2 * max_l + 1;
    const U cart = (max_l + 1) * (max_l + 2) / 2;
    const U primitive_work = pair_cell_add(512, pair_cell_add(
        pair_panel_product(192, pair_panel_product((max_l + 1) * (max_l + 1), t)),
        pair_panel_product(128, pair_panel_product(cart * cart, 3 * t + 1))));
    const U image_work = pair_cell_add(512,
        pair_panel_product(pair_panel_product(max_primitives, max_primitives), primitive_work));
    const U lookup_work = pair_panel_product(256,
        pair_panel_product(pair_count,
            pair_cell_add(1, pair_cell_add(basis.nshells(), contraction_count))));
    p.contraction_work_units = pair_cell_add(lookup_work,
        pair_cell_add(pair_panel_product(pair_panel_product(p.pair_cell_visits, n_vectors), image_work),
            pair_panel_product(256, pair_panel_product(elements, max_l + 1))));
    p.work_units = pair_cell_add(p.metadata_work_units,
        pair_cell_add(p.validation_work_units, p.contraction_work_units));
    pair_cell_cap(p.work_units, caps.maximum_work_units, "AO-pair cell panel work cap");
    // The explicit-cell evaluator uses one vector at a time. Its exact live
    // workspace excludes the reciprocal tile used by the distance evaluator.
    p.fixed_numeric_workspace_bytes = sizeof(PairPanelNumericWorkspace);
    p.fixed_scalar_numeric_bytes = 1024;
    p.fixed_control_storage_bytes = 8192 + sizeof(AOPairFourierCellPanelPlan)
        + sizeof(AOPairFourierCellPanelCaps) + sizeof(AOPairFourierCellView)
        + sizeof(AOPairFourierPanel) + sizeof(AuxiliaryFourierVectorView);
    return p;
}

AOPairFourierPanel ao_pair_gaussian_fourier_cell_panel(
    const BasisSet& basis, const PeriodicSystem& system, AuxiliaryFourierVectorView vectors,
    const Eigen::Vector3d& k_cart, std::uint64_t pair_begin,
    std::uint64_t pair_count, AOPairFourierCellView cells,
    const AOPairFourierCellPanelCaps& caps) {
    const auto plan = plan_ao_pair_gaussian_fourier_cell_panel(
        basis, vectors.count, pair_begin, pair_count, cells.cell_count, caps);
    pair_cell_fp();
    if (cells.element_count != pair_panel_product(3, cells.cell_count)
        || (cells.element_count && (!cells.indices
            || reinterpret_cast<std::uintptr_t>(cells.indices) % alignof(std::int64_t) != 0)))
        throw std::invalid_argument("AO-pair cell panel integer-cell extent/alignment");
    if (system.dim != 3 || !system.lattice.allFinite() || !k_cart.allFinite())
        throw std::invalid_argument("AO-pair cell panel requires finite original 3D geometry and k");
    const long double determinant = system.lattice.cast<long double>().determinant();
    if (!std::isfinite(determinant) || determinant == 0.0L)
        throw std::invalid_argument("AO-pair cell panel requires nonsingular original lattice");
    for (const auto& shell : basis.libint()) {
        for (double origin : shell.O) pair_panel_require_finite(origin);
        for (double exponent : shell.alpha) {
            if (!std::isfinite(exponent) || exponent <= 0)
                throw std::invalid_argument("AO-pair cell panel exponents must be finite and positive");
        }
        for (const auto& c : shell.contr)
            for (double coefficient : c.coeff) pair_panel_require_finite(coefficient);
    }
    const std::array<const double*, 3> lanes = {vectors.x, vectors.y, vectors.z};
    const std::array<std::ptrdiff_t, 3> strides = {
        vectors.x_stride, vectors.y_stride, vectors.z_stride};
    for (std::size_t d = 0; d < 3; ++d) {
        if (strides[d] <= 0 || (vectors.count && (!lanes[d]
            || reinterpret_cast<std::uintptr_t>(lanes[d]) % alignof(double) != 0))
            || (vectors.count > 1 && vectors.count - 1 > static_cast<std::size_t>(
                (std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double)) / strides[d])))
            throw std::invalid_argument("AO-pair cell panel vector lane/stride/alignment");
    }
    for (std::size_t v = 0; v < vectors.count; ++v) {
        Eigen::Vector3d p;
        for (std::size_t d = 0; d < 3; ++d) {
            p[d] = lanes[d][static_cast<std::ptrdiff_t>(v) * strides[d]];
            pair_panel_require_finite(p[d]);
        }
        pair_panel_require_finite(pair_panel_norm2(p[0], p[1], p[2]));
    }
    constexpr std::int64_t exact_integer_limit = 9007199254740992LL;
    for (std::uint64_t c = 0; c < cells.cell_count; ++c) {
        const auto* label = cells.indices + 3 * c;
        for (int d = 0; d < 3; ++d)
            if (label[d] < -exact_integer_limit || label[d] > exact_integer_limit)
                throw std::invalid_argument("AO-pair cell panel label exceeds exact binary64 integer range");
        for (std::uint64_t other = 0; other < c; ++other) {
            const auto* previous = cells.indices + 3 * other;
            if (label[0] == previous[0] && label[1] == previous[1] && label[2] == previous[2])
                throw std::invalid_argument("AO-pair cell panel duplicate integer cell");
        }
        const auto translation = pair_cell_translation(system.lattice, label);
        pair_panel_require_finite(k_cart.dot(translation));
    }
    // Numerical geometry for every selected pair/cell is checked before the
    // only variable allocation, including when the reciprocal block is empty.
    for (std::uint64_t row = 0; row < pair_count; ++row) {
        const auto bra = pair_panel_ao(basis, (pair_begin + row) / plan.n_basis);
        const auto ket = pair_panel_ao(basis, (pair_begin + row) % plan.n_basis);
        for (std::uint64_t c = 0; c < cells.cell_count; ++c) {
            const auto translation = pair_cell_translation(system.lattice, cells.indices + 3 * c);
            Eigen::Vector3d separation;
            for (int d = 0; d < 3; ++d) {
                separation[d] = bra.shell->O[d] - ket.shell->O[d] - translation[d];
                pair_panel_require_finite(separation[d]);
            }
            pair_panel_require_finite(pair_panel_norm2(separation[0], separation[1], separation[2]));
        }
    }
    AOPairFourierPanel output;
    output.pair_begin = pair_begin;
    output.n_pairs = static_cast<std::size_t>(pair_count);
    output.n_vectors = vectors.count; output.output_bytes = plan.output_bytes;
    output.image_candidate_count = output.retained_pair_image_count = plan.pair_cell_visits;
    output.fixed_numeric_workspace_bytes = pair_cell_add(
        plan.fixed_numeric_workspace_bytes, plan.fixed_scalar_numeric_bytes);
    output.data.resize(static_cast<std::size_t>(plan.output_bytes / 16));
    PairPanelNumericWorkspace workspace;
    for (std::uint64_t row = 0; row < pair_count; ++row) {
        const auto bra = pair_panel_ao(basis, (pair_begin + row) / plan.n_basis);
        const auto ket = pair_panel_ao(basis, (pair_begin + row) % plan.n_basis);
        const int la = bra.contraction->l, lb = ket.contraction->l;
        const double scale_a = la == 0 ? 1.0 : std::sqrt(4.0 * kPi / (2 * la + 1));
        const double scale_b = lb == 0 ? 1.0 : std::sqrt(4.0 * kPi / (2 * lb + 1));
        for (std::size_t v = 0; v < vectors.count; ++v) {
            Eigen::Vector3d p;
            for (std::size_t d = 0; d < 3; ++d) {
                p[d] = lanes[d][static_cast<std::ptrdiff_t>(v) * strides[d]];
                auto* powers = workspace.powers.data() + d * kPairPanelPowerExtent;
                powers[0] = {1.0, 0.0};
                for (int t = 1; t <= la + lb; ++t)
                    powers[t] = powers[t - 1] * std::complex<double>(0.0, -p[d]);
            }
            const double p2 = pair_panel_norm2(p[0], p[1], p[2]);
            double re = 0, im = 0, cre = 0, cim = 0;
            for (std::uint64_t c = 0; c < cells.cell_count; ++c) {
                const auto translation = pair_cell_translation(system.lattice, cells.indices + 3 * c);
                Eigen::Vector3d separation;
                for (int d = 0; d < 3; ++d)
                    separation[d] = bra.shell->O[d] - ket.shell->O[d] - translation[d];
                const double r2 = pair_panel_norm2(separation[0], separation[1], separation[2]);
                pair_panel_accumulate_image(bra, ket, p, p2, k_cart, scale_a, scale_b,
                    translation, separation, r2, workspace, re, im, cre, cim);
            }
            re += cre; im += cim;
            pair_panel_require_finite(re); pair_panel_require_finite(im);
            output.data[static_cast<std::size_t>(row) * vectors.count + v] = {
                re == 0.0 ? 0.0 : re, im == 0.0 ? 0.0 : im};
        }
    }
    return output;
}

}  // namespace vibeqc
