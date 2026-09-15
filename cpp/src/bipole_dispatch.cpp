// Parallel implementation of the prototype geometric quartet classifier.
// The current metric and order map are not derived in the cited literature.

#include "vibeqc/bipole_dispatch.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

namespace vibeqc {

namespace {

// Product-distribution centre: C = (a1*R1 + a2*R2) / (a1 + a2).
inline std::array<double, 3> product_centre(
    const std::array<double, 3>& R1,
    const std::array<double, 3>& R2,
    double a1, double a2) {
    double inv = 1.0 / (a1 + a2);
    return {{
        (a1 * R1[0] + a2 * R2[0]) * inv,
        (a1 * R1[1] + a2 * R2[1]) * inv,
        (a1 * R1[2] + a2 * R2[2]) * inv,
    }};
}

// Product-distribution width: gamma = a1*a2 / (a1 + a2).
inline double product_width(double a1, double a2) {
    return a1 * a2 / (a1 + a2);
}

// Prototype diffuse-primitive overlap estimate; no one-to-one TOLINTEG map.
inline double estimate_overlap(
    double a1, double a2,
    const std::array<double, 3>& R1,
    const std::array<double, 3>& R2) {
    double gamma = product_width(a1, a2);
    double dx = R1[0] - R2[0];
    double dy = R1[1] - R2[1];
    double dz = R1[2] - R2[2];
    double R_sq = dx*dx + dy*dy + dz*dz;
    double prefactor = std::pow(M_PI / (a1 + a2), 1.5);
    return prefactor * std::exp(-gamma * R_sq);
}

// Implementation-specific geometric classifier.
//
// The dimensionless metric D²·γ_bra measures the separation of two
// product distributions relative to their Gaussian width:
//   D²γ < 1   → overlapping (exact ERI needed)
//   D²γ > 10  → well separated (low-order multipole sufficient)
//
// The cell-length scale adapts the threshold to the system size.
inline int truncation_order(
    const std::array<double, 3>& C_bra,
    const std::array<double, 3>& C_ket,
    double gamma_bra,
    const BipoleDispatchParams& p) {
    double dx = C_bra[0] - C_ket[0];
    double dy = C_bra[1] - C_ket[1];
    double dz = C_bra[2] - C_ket[2];
    double D_sq = dx*dx + dy*dy + dz*dz;

    // Dimensionless penetration metric: D² · γ
    double metric = D_sq * gamma_bra;

    // Near-field cutoff: distributions overlap significantly when
    // D²·γ is small.  The cell scale sets the characteristic length.
    // For a typical system: cell_length_scale_inv ≈ 0.1-0.3 bohr⁻¹,
    // γ ≈ 0.05-1.0 bohr⁻².  We want near-field when D is less than
    // ~2-4 bohr (a few times the product width).
    double L_inv = p.cell_length_scale_inv;
    // near_cutoff ≈ 1.0 — D²·γ < 1 means distributions overlap
    double near_cutoff = 1.0 / (p.dispatch_slope * L_inv + 1.0);

    if (metric < near_cutoff) return 0;  // near-field → exact ERI

    // Far-field: multipole order scales with distance.
    // Use log scaling: order decreases as distance increases.
    double ratio = metric / near_cutoff;
    int order_raw = p.max_multipole_order
        - static_cast<int>(std::log2(ratio));
    // When max_order == 0, all quartets are near-field (no far-field).
    // Otherwise, far-field quartets always receive at least L=1 (dipole),
    // because order zero is reserved as the exact-path sentinel.
    if (p.max_multipole_order == 0) return 0;
    if (order_raw < 1) return 1;
    if (order_raw > p.max_multipole_order) return p.max_multipole_order;
    return order_raw;
}

}  // namespace

std::vector<BipolarQuartetEntry> compute_bipolar_penetration_dispatch(
    const std::vector<BipoleShellInfo>& shells,
    const std::vector<BipoleCellInfo>& cells,
    const BipoleDispatchParams& params) {

    const int n_sh = static_cast<int>(shells.size());
    const int n_c = static_cast<int>(cells.size());
    const int n_pairs = n_c * n_c;

    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<BipolarQuartetEntry>> thread_results(n_threads);

    #pragma omp parallel for schedule(dynamic)
    for (int pair_idx = 0; pair_idx < n_pairs; ++pair_idx) {
        const int c_g = pair_idx / n_c;
        const int c_lam = pair_idx % n_c;
        const int tid = omp_get_thread_num();
        auto& local = thread_results[tid];

        const auto& cell_g = cells[c_g];
        const auto& cell_lam = cells[c_lam];

        for (int s1 = 0; s1 < n_sh; ++s1) {
            const auto& sh1 = shells[s1];
            double a1 = sh1.min_exponent;
            for (int s2 = s1; s2 < n_sh; ++s2) {
                const auto& sh2 = shells[s2];
                double a2 = sh2.min_exponent;
                double gamma_bra = product_width(a1, a2);

                // Bra centre: s1 at origin, s2 shifted by cell_g.
                std::array<double, 3> R2_g = {{
                    sh2.origin[0] + cell_g.r_cart[0],
                    sh2.origin[1] + cell_g.r_cart[1],
                    sh2.origin[2] + cell_g.r_cart[2],
                }};
                auto C_bra = product_centre(sh1.origin, R2_g, a1, a2);

                // Prototype screen: skip if bra estimate is below threshold.
                if (params.overlap_threshold > 0.0) {
                    double S_bra = estimate_overlap(
                        a1, a2, sh1.origin, R2_g);
                    if (S_bra < params.overlap_threshold) continue;
                }

                for (int s3 = 0; s3 < n_sh; ++s3) {
                    const auto& sh3 = shells[s3];
                    double a3 = sh3.min_exponent;
                    for (int s4 = s3; s4 < n_sh; ++s4) {
                        const auto& sh4 = shells[s4];
                        double a4 = sh4.min_exponent;
                        double gamma_ket = product_width(a3, a4);

                        // Ket centre: s3 at home, s4 shifted by cell_lam.
                        // This matches the Python path where the ket pair is
                        // (shell at home, shell at cell_lam), not both shifted.
                        std::array<double, 3> R3_lam = {{
                            sh3.origin[0],
                            sh3.origin[1],
                            sh3.origin[2],
                        }};
                        std::array<double, 3> R4_lam = {{
                            sh4.origin[0] + cell_lam.r_cart[0],
                            sh4.origin[1] + cell_lam.r_cart[1],
                            sh4.origin[2] + cell_lam.r_cart[2],
                        }};
                        auto C_ket = product_centre(R3_lam, R4_lam, a3, a4);

                        // Prototype screen: skip if ket estimate is below threshold.
                        if (params.overlap_threshold > 0.0) {
                            double S_ket = estimate_overlap(
                                a3, a4, R3_lam, R4_lam);
                            if (S_ket < params.overlap_threshold) continue;
                        }

                        int order = truncation_order(
                            C_bra, C_ket, gamma_bra, params);
                        if (order > 0) {
                            local.push_back({
                                s1, s2, s3, s4,
                                c_g, c_lam,
                                order,
                                gamma_bra, gamma_ket,
                            });
                        }
                    }
                }
            }
        }
    }

    // Merge thread-local results.
    std::vector<BipolarQuartetEntry> result;
    for (const auto& local : thread_results) {
        result.insert(result.end(), local.begin(), local.end());
    }
    return result;
}

}  // namespace vibeqc
